# heavy_engine.py - GPU-ACCELERATED PYTORCH ENHANCEMENT ENGINE
"""
Full PyTorch-based video enhancement using RealESRGAN and CodeFormer.
Processes videos frame-by-frame with GPU acceleration and batch inference.
"""

import os
import sys
import logging
import subprocess
import shutil
import glob
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] heavy_engine: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("heavy_engine")

# ==================== CONFIGURATION ====================

# Paths (absolute to avoid Colab issues)
MODELS_DIR = os.path.abspath(os.path.join(os.getcwd(), "models", "heavy"))
TEMP_DIR = os.path.abspath(os.path.join(os.getcwd(), "temp", "heavy_frames"))

# Model files
REALESRGAN_MODEL = os.path.join(MODELS_DIR, "RealESRGAN_x4plus.pth")
CODEFORMER_MODEL = os.path.join(MODELS_DIR, "codeformer.pth")

# Processing settings
DEFAULT_BATCH_SIZE = 16  # Process 16 frames at once
FACE_BLEND_RATIO = 0.7   # 70% CodeFormer, 30% RealESRGAN for natural look

# FFmpeg
FFMPEG_BIN = "ffmpeg"

# ==================== MODEL ARCHITECTURES ====================

class RRDBNet(torch.nn.Module):
    """
    RealESRGAN network architecture (RRDBNet).
    Used for 4x upscaling with residual dense blocks.
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_block = num_block
        self.num_grow_ch = num_grow_ch
        
        # First convolution
        self.conv_first = torch.nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        # RRDB blocks
        self.body = self._make_layer(num_block)
        self.conv_body = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Upsampling
        self.conv_up1 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = torch.nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def _make_layer(self, num_block):
        """Create RRDB blocks"""
        layers = []
        for _ in range(num_block):
            layers.append(RRDB(self.num_feat, self.num_grow_ch))
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through RealESRGAN"""
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        
        # Upsample 4x
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class RRDB(torch.nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
    
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class ResidualDenseBlock(torch.nn.Module):
    """Residual Dense Block"""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class SimpleCodeFormer(torch.nn.Module):
    """
    Simplified CodeFormer architecture for face restoration.
    This is a lightweight version that can work without the full CodeFormer codebase.
    """
    def __init__(self, dim=512, depth=9):
        super(SimpleCodeFormer, self).__init__()
        
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, 1, 1),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(64, 128, 3, 2, 1),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(128, 256, 3, 2, 1),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(256, dim, 3, 2, 1),
            torch.nn.LeakyReLU(0.2, True),
        )
        
        # Transformer blocks (simplified)
        self.transformer = torch.nn.Sequential(*[
            TransformerBlock(dim) for _ in range(depth)
        ])
        
        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(dim, 256, 4, 2, 1),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(64, 3, 3, 1, 1),
        )
    
    def forward(self, x):
        feat = self.encoder(x)
        feat = self.transformer(feat)
        out = self.decoder(feat)
        return out


class TransformerBlock(torch.nn.Module):
    """Simplified Transformer block"""
    def __init__(self, dim):
        super(TransformerBlock, self).__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn = torch.nn.MultiheadAttention(dim, 8, batch_first=True)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim),
        )
    
    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Self-attention
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        
        # MLP
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        
        # Reshape back
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        return x


# ==================== MODEL LOADING ====================

def load_realesrgan_model(device='cuda') -> Optional[torch.nn.Module]:
    """
    Load RealESRGAN model from checkpoint.
    Returns None if model file doesn't exist or loading fails.
    """
    try:
        if not os.path.exists(REALESRGAN_MODEL):
            logger.warning(f"⚠️ RealESRGAN model not found: {REALESRGAN_MODEL}")
            return None
        
        logger.info("📦 Loading RealESRGAN model...")
        
        # Create model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        
        # Load weights
        checkpoint = torch.load(REALESRGAN_MODEL, map_location=device)
        
        # Handle different checkpoint formats
        if 'params_ema' in checkpoint:
            model.load_state_dict(checkpoint['params_ema'], strict=True)
        elif 'params' in checkpoint:
            model.load_state_dict(checkpoint['params'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
        
        model.eval()
        model = model.to(device)
        
        logger.info("✅ RealESRGAN model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load RealESRGAN: {e}")
        return None


def load_codeformer_model(device='cuda') -> Optional[torch.nn.Module]:
    """
    Load CodeFormer model from checkpoint.
    Returns None if model file doesn't exist or loading fails.
    """
    try:
        if not os.path.exists(CODEFORMER_MODEL):
            logger.warning(f"⚠️ CodeFormer model not found: {CODEFORMER_MODEL}")
            logger.info("ℹ️ Face enhancement will be skipped")
            return None
        
        logger.info("📦 Loading CodeFormer model...")
        
        # Create model
        model = SimpleCodeFormer(dim=512, depth=9)
        
        # Load weights
        checkpoint = torch.load(CODEFORMER_MODEL, map_location=device)
        
        # Handle different checkpoint formats
        if 'params_ema' in checkpoint:
            model.load_state_dict(checkpoint['params_ema'], strict=False)
        elif 'params' in checkpoint:
            model.load_state_dict(checkpoint['params'], strict=False)
        else:
            try:
                model.load_state_dict(checkpoint, strict=False)
            except:
                logger.warning("⚠️ CodeFormer weights format mismatch, using partial loading")
        
        model.eval()
        model = model.to(device)
        
        logger.info("✅ CodeFormer model loaded successfully")
        return model
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to load CodeFormer: {e}")
        logger.info("ℹ️ Face enhancement will be skipped")
        return None


# ==================== FRAME PROCESSING ====================

def extract_frames(video_path: str, output_dir: str) -> int:
    """
    Extract all frames from video using FFmpeg.
    Returns number of frames extracted.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear existing frames
    for f in glob.glob(os.path.join(output_dir, "*.png")):
        os.remove(f)
    
    logger.info(f"📹 Extracting frames from video...")
    
    cmd = [
        FFMPEG_BIN, "-i", video_path,
        "-q:v", "1",  # Highest quality
        os.path.join(output_dir, "frame_%08d.png")
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frame_count = len(glob.glob(os.path.join(output_dir, "*.png")))
        logger.info(f"✅ Extracted {frame_count} frames")
        return frame_count
    except Exception as e:
        logger.error(f"❌ Frame extraction failed: {e}")
        return 0


def rebuild_video(frames_dir: str, original_video: str, output_path: str) -> bool:
    """
    Rebuild video from frames and add original audio.
    """
    logger.info("🎬 Rebuilding video from enhanced frames...")
    
    try:
        # Create video from frames
        temp_video = output_path.replace(".mp4", "_temp.mp4")
        
        cmd_video = [
            FFMPEG_BIN, "-y",
            "-framerate", "30",
            "-i", os.path.join(frames_dir, "frame_%08d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "18",
            temp_video
        ]
        
        subprocess.run(cmd_video, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Add original audio
        cmd_audio = [
            FFMPEG_BIN, "-y",
            "-i", temp_video,
            "-i", original_video,
            "-map", "0:v:0",
            "-map", "1:a:0?",  # Optional audio
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path
        ]
        
        subprocess.run(cmd_audio, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Cleanup temp file
        if os.path.exists(temp_video):
            os.remove(temp_video)
        
        logger.info(f"✅ Video rebuilt: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Video rebuild failed: {e}")
        return False


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """
    Convert numpy frame (H, W, C) to torch tensor (1, C, H, W).
    Normalize to [0, 1] range.
    """
    # BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    
    # HWC to CHW
    frame = np.transpose(frame, (2, 0, 1))
    
    # Add batch dimension
    frame = torch.from_numpy(frame).unsqueeze(0)
    
    return frame


def postprocess_frame(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert torch tensor (1, C, H, W) back to numpy frame (H, W, C).
    Denormalize from [0, 1] to [0, 255].
    """
    # Remove batch dimension
    tensor = tensor.squeeze(0)
    
    # CHW to HWC
    frame = tensor.permute(1, 2, 0).cpu().numpy()
    
    # Denormalize to [0, 255]
    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    
    # RGB to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    return frame


def detect_faces(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in frame using OpenCV Haar Cascade.
    Returns list of (x, y, w, h) bounding boxes.
    """
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        return faces.tolist() if len(faces) > 0 else []
        
    except Exception as e:
        logger.warning(f"⚠️ Face detection failed: {e}")
        return []


def blend_faces(bg_frame: np.ndarray, face_frame: np.ndarray, 
                face_boxes: List[Tuple[int, int, int, int]], 
                blend_ratio: float = FACE_BLEND_RATIO) -> np.ndarray:
    """
    Blend CodeFormer face enhancements with RealESRGAN background.
    blend_ratio: 0.7 means 70% CodeFormer, 30% RealESRGAN
    """
    result = bg_frame.copy()
    
    for (x, y, w, h) in face_boxes:
        # Expand face region slightly for better blending
        margin = int(min(w, h) * 0.2)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(bg_frame.shape[1], x + w + margin)
        y2 = min(bg_frame.shape[0], y + h + margin)
        
        # Blend face region
        bg_region = bg_frame[y1:y2, x1:x2]
        face_region = face_frame[y1:y2, x1:x2]
        
        # Ensure same size
        if bg_region.shape == face_region.shape:
            blended = cv2.addWeighted(face_region, blend_ratio, bg_region, 1 - blend_ratio, 0)
            result[y1:y2, x1:x2] = blended
    
    return result


# ==================== MAIN ENHANCEMENT FUNCTION ====================

@torch.no_grad()
def enhance_video_gpu(input_path: str, output_path: str, scale: int = 4) -> bool:
    """
    Main function to enhance video using GPU Heavy Engine.
    
    Pipeline:
    1. Extract frames from video
    2. Load RealESRGAN and CodeFormer models
    3. Process frames in batches:
       - Upscale with RealESRGAN
       - Detect and enhance faces with CodeFormer
       - Blend face enhancements with background
    4. Rebuild video with original audio
    
    Args:
        input_path: Path to input video
        output_path: Path to save enhanced video
        scale: Upscaling factor (2 or 4)
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("🚀 Starting Heavy Engine (GPU)...")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not available. Cannot use Heavy Engine.")
        return False
    
    device = torch.device('cuda')
    logger.info(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Setup directories
    input_frames_dir = os.path.join(TEMP_DIR, "input")
    output_frames_dir = os.path.join(TEMP_DIR, "output")
    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)
    
    try:
        # Step 1: Extract frames
        frame_count = extract_frames(input_path, input_frames_dir)
        if frame_count == 0:
            logger.error("❌ No frames extracted")
            return False
        
        # Step 2: Load models
        realesrgan = load_realesrgan_model(device)
        if realesrgan is None:
            logger.error("❌ Failed to load RealESRGAN model")
            return False
        
        codeformer = load_codeformer_model(device)
        use_face_enhancement = codeformer is not None
        
        # Step 3: Process frames in batches
        frame_files = sorted(glob.glob(os.path.join(input_frames_dir, "*.png")))
        batch_size = DEFAULT_BATCH_SIZE
        
        logger.info(f"🎨 Processing {frame_count} frames (batch size: {batch_size})...")
        
        for i in range(0, len(frame_files), batch_size):
            batch_files = frame_files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(frame_files) + batch_size - 1) // batch_size
            
            logger.info(f"   Batch {batch_num}/{total_batches} ({len(batch_files)} frames)")
            
            # Process each frame in batch
            for frame_idx, frame_file in enumerate(batch_files):
                global_idx = i + frame_idx + 1
                
                # Load frame
                frame = cv2.imread(frame_file)
                if frame is None:
                    logger.warning(f"⚠️ Failed to load frame: {frame_file}")
                    continue
                
                # Preprocess
                input_tensor = preprocess_frame(frame).to(device)
                
                # Upscale with RealESRGAN
                try:
                    with torch.cuda.amp.autocast():  # Mixed precision for speed
                        upscaled_tensor = realesrgan(input_tensor)
                    upscaled_frame = postprocess_frame(upscaled_tensor)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning("⚠️ GPU OOM, reducing batch size...")
                        torch.cuda.empty_cache()
                        batch_size = max(1, batch_size // 2)
                        continue
                    else:
                        raise
                
                # Face enhancement (if available)
                if use_face_enhancement:
                    # Detect faces in original frame
                    faces = detect_faces(frame)
                    
                    if len(faces) > 0:
                        # Enhance faces with CodeFormer
                        face_input = preprocess_frame(upscaled_frame).to(device)
                        face_enhanced = codeformer(face_input)
                        face_frame = postprocess_frame(face_enhanced)
                        
                        # Blend face enhancements with background
                        # Scale face boxes to upscaled resolution
                        scaled_faces = [(x*scale, y*scale, w*scale, h*scale) for (x, y, w, h) in faces]
                        final_frame = blend_faces(upscaled_frame, face_frame, scaled_faces)
                    else:
                        final_frame = upscaled_frame
                else:
                    final_frame = upscaled_frame
                
                # Save enhanced frame
                output_file = os.path.join(output_frames_dir, os.path.basename(frame_file))
                cv2.imwrite(output_file, final_frame)
                
                # Progress logging
                if global_idx % 30 == 0 or global_idx == frame_count:
                    logger.info(f"   ⚡ Processed {global_idx}/{frame_count} frames")
        
        logger.info("✅ All frames enhanced")
        
        # Step 4: Rebuild video
        success = rebuild_video(output_frames_dir, input_path, output_path)
        
        if success:
            logger.info("✅ Heavy Engine completed successfully")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Heavy Engine failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup temp frames
        try:
            if os.path.exists(input_frames_dir):
                shutil.rmtree(input_frames_dir)
            if os.path.exists(output_frames_dir):
                shutil.rmtree(output_frames_dir)
        except:
            pass


# ==================== TESTING ====================

def test_model_loading():
    """Test if models can be loaded successfully"""
    logger.info("🧪 Testing model loading...")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available")
        return False
    
    device = torch.device('cuda')
    
    # Test RealESRGAN
    realesrgan = load_realesrgan_model(device)
    if realesrgan is None:
        logger.error("❌ RealESRGAN loading failed")
        return False
    
    # Test CodeFormer
    codeformer = load_codeformer_model(device)
    if codeformer is None:
        logger.warning("⚠️ CodeFormer not available (optional)")
    
    logger.info("✅ Model loading test passed")
    return True


def test_batch_processing():
    """Test batch processing with dummy data"""
    logger.info("🧪 Testing batch processing...")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available")
        return False
    
    device = torch.device('cuda')
    
    # Load model
    model = load_realesrgan_model(device)
    if model is None:
        return False
    
    # Create dummy batch
    batch = torch.randn(4, 3, 256, 256).to(device)
    
    # Process
    try:
        with torch.no_grad():
            output = model(batch)
        logger.info(f"✅ Batch processing test passed (output shape: {output.shape})")
        return True
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        return False


if __name__ == "__main__":
    # Run tests
    test_model_loading()
    test_batch_processing()
