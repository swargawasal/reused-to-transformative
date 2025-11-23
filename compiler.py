# compiler.py - HIGH-END MULTI-PASS AI EDITOR (DUAL-STAGE ENGINE)
import os
import subprocess
import logging
import shutil
import sys
import random
import json
import glob
import time
import platform
from pathlib import Path
from typing import List, Optional, Dict
from dotenv import load_dotenv

# ==================== SETUP & CONFIG ====================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("compiler")

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
if not shutil.which(FFPROBE_BIN):
    FFPROBE_BIN = "ffprobe"

# Configuration
COMPUTE_MODE = os.getenv("COMPUTE_MODE", "auto").lower()
ENHANCEMENT_LEVEL = os.getenv("ENHANCEMENT_LEVEL", "2x").lower()
TRANSITION_DURATION = float(os.getenv("TRANSITION_DURATION", "1.0"))
TRANSITION_INTERVAL = int(os.getenv("TRANSITION_INTERVAL", "10"))
TARGET_RESOLUTION = os.getenv("TARGET_RESOLUTION", "1080:1920")
REENCODE_CRF = os.getenv("REENCODE_CRF", "18")
REENCODE_PRESET = os.getenv("REENCODE_PRESET", "slow")

# AI Config
AI_FAST_MODE = os.getenv("AI_FAST_MODE", "no").lower() == "yes"
FACE_ENHANCEMENT = os.getenv("FACE_ENHANCEMENT", "yes").lower() == "yes"
USE_ADVANCED_ENGINE = os.getenv("USE_ADVANCED_ENGINE", "off").lower() == "on"

TEMP_DIR = "temp"
OUTPUT_DIR = "merged_videos"
TOOLS_DIR = os.path.join(os.getcwd(), "tools")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== HELPER FUNCTIONS ====================

def _run_command(cmd: List[str], check: bool = False, timeout: int = None) -> bool:
    try:
        result = subprocess.run(
            cmd, 
            check=check, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            timeout=timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning(f"Command timed out: {cmd[0]}")
        return False
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return False

def _get_video_info(path: str) -> Dict:
    try:
        cmd = [
            FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration",
            "-of", "json", path
        ]
        result = subprocess.check_output(cmd, shell=True).decode().strip()
        data = json.loads(result)
        stream = data["streams"][0]
        return {
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "duration": float(stream.get("duration", 0))
        }
    except Exception:
        return {"width": 0, "height": 0, "duration": 0}

def ensure_ncnn_tools_exist() -> Dict[str, bool]:
    """
    Check existence of NCNN tools (Real-ESRGAN, SRMD, ESRNet, CodeFormer).
    Returns dict with status for each.
    """
    is_windows = platform.system() == "Windows"
    ext = ".exe" if is_windows else ""
    
    tools = {
        "realesrgan": os.path.join(TOOLS_DIR, f"realesrgan-ncnn-vulkan{ext}"),
        "srmd": os.path.join(TOOLS_DIR, f"srmd-ncnn-vulkan{ext}"),
        "esrnet": os.path.join(TOOLS_DIR, f"esrnet-ncnn-vulkan{ext}"),
        "codeformer": os.path.join(TOOLS_DIR, f"codeformer-ncnn-vulkan{ext}")
    }
    
    status = {}
    for name, path in tools.items():
        status[name] = os.path.exists(path)
        status[f"{name}_path"] = path
        
    return status

def detect_cuda_vram() -> float:
    """
    Detect CUDA VRAM in GB.
    Returns 0.0 if no CUDA or error.
    """
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return p.total_memory / (1024**3)
    except:
        return 0.0
    return 0.0

# ==================== ENGINE 1: HEAVY (PYTORCH) ====================

def run_heavy_engine(input_path: str, output_path: str, scale: int) -> bool:
    """
    Heavy Engine Pipeline (PyTorch):
    1. ESRNet x4 (Cleanup)
    2. Real-ESRGAN x4plus (Detail)
    3. SRMD (Refinement)
    4. CodeFormer/GFPGAN (Full Frame Face Enhancement)
    """
    logger.info("🚀 Starting HEAVY ENGINE (PyTorch)...")
    try:
        # Placeholder for actual PyTorch implementation
        # In a real scenario, this would load models and run inference loop
        # For now, we simulate it or fallback if models missing
        
        # Check if we actually have the heavy models downloaded
        # If not, we can't run this engine even if GPU is good
        # For this implementation, we will assume if we got here, we try.
        
        # TODO: Implement actual PyTorch inference code here
        # Since we don't have the weights in the repo, we can't actually run this.
        # We will return False to trigger fallback to NCNN for now, 
        # unless user has manually set up the environment.
        
        logger.warning("⚠️ Heavy Engine implementation pending model weights. Falling back to NCNN.")
        return False
        
    except Exception as e:
        logger.error(f"Heavy Engine Failed: {e}")
        return False

# ==================== ENGINE 2: NCNN (VULKAN) ====================

def _process_ncnn_step(tool_path: str, input_dir: str, output_dir: str, args: List[str], step_name: str) -> bool:
    """Helper to run a single NCNN step."""
    if not os.path.exists(tool_path):
        logger.warning(f"⚠️ {step_name} binary missing: {tool_path}")
        return False
        
    logger.info(f"   Running {step_name}...")
    cmd = [tool_path, "-i", input_dir, "-o", output_dir] + args
    return _run_command(cmd, timeout=None)

def run_ncnn_engine(input_path: str, output_path: str, scale: int) -> bool:
    """
    NCNN Engine Pipeline (Vulkan):
    1. ESRNet-ncnn (Cleanup)
    2. SRMD-ncnn (Smoothing)
    3. Real-ESRGAN-ncnn (Detail/Upscale)
    """
    logger.info("🟢 Starting NCNN ENGINE (Vulkan)...")
    tools = ensure_ncnn_tools_exist()
    
    # Verify critical tools
    if not tools["realesrgan"]:
        logger.error("❌ Critical Tool Missing: Real-ESRGAN NCNN")
        return False

    work_dir = os.path.join(TEMP_DIR, f"ncnn_{random.randint(1000,9999)}")
    frames_src = os.path.join(work_dir, "src")
    frames_s1 = os.path.join(work_dir, "step1_esrnet")
    frames_s2 = os.path.join(work_dir, "step2_srmd")
    frames_final = os.path.join(work_dir, "final")
    
    for d in [frames_src, frames_s1, frames_s2, frames_final]:
        os.makedirs(d, exist_ok=True)
        
    try:
        # 1. Extract Frames
        logger.info("   Extracting frames...")
        _run_command([FFMPEG_BIN, "-i", input_path, "-q:v", "1", os.path.join(frames_src, "frame_%08d.png")], check=True)
        
        current_dir = frames_src
        
        # 2. ESRNet (First Pass Cleanup)
        # We use esrnet-ncnn-vulkan (which is realesrgan binary) with -n realesrnet-x4plus (or similar)
        # If we don't have specific esrnet model, we skip or use standard realesrgan x4
        if tools["esrnet"]:
             # Using standard x4 model as 'ESRNet' equivalent for NCNN if specific model not present
             # Real-ESRGAN package usually comes with realesr-animevideov3 or realesrgan-x4plus
             # Ensure absolute paths for input/output directories
             if _process_ncnn_step(tools["esrnet_path"], os.path.abspath(current_dir), os.path.abspath(frames_s1), ["-s", "1", "-n", "realesrgan-x4plus"], "ESRNet (Cleanup)"):
                 current_dir = frames_s1
        
        # 3. SRMD (Texture Smoothing)
        if tools["srmd"]:
            # SRMD usually takes noise level args, e.g., -n 3
            if _process_ncnn_step(tools["srmd_path"], os.path.abspath(current_dir), os.path.abspath(frames_s2), ["-n", "3"], "SRMD (Smoothing)"):
                current_dir = frames_s2
                
        # 4. Real-ESRGAN (Final Detail & Upscale)
        # Apply scale here
        if _process_ncnn_step(tools["realesrgan_path"], os.path.abspath(current_dir), os.path.abspath(frames_final), ["-s", str(scale)], f"Real-ESRGAN (x{scale})"):
            current_dir = frames_final
            
        # 5. Merge Frames
        logger.info("   Merging frames...")
        cmd_merge = [
            FFMPEG_BIN, "-y", 
            "-framerate", "30", 
            "-i", os.path.join(current_dir, "frame_%08d.png"),
            "-i", input_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18",
            output_path
        ]
        
        if _run_command(cmd_merge, check=True):
            logger.info("✅ NCNN enhancement complete.")
            return True
        else:
            logger.error("❌ Merge failed.")
            return False

    except Exception as e:
        logger.error(f"NCNN Engine Error: {e}")
        return False
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

# ==================== AUTO ENGINE SELECTION ====================

def enhance_video_cpu_fallback(input_path: str, output_path: str, scale: int) -> bool:
    """
    Stage 3: Old CPU Fallback (Sharpen -> Upscale)
    Uses FFmpeg filters to simulate the old behavior.
    """
    logger.info("🐢 Starting CPU FALLBACK ENGINE (Stage 3)...")
    try:
        # Sharpen (unsharp) -> Upscale (scale)
        # unsharp=luma_msize_x:luma_msize_y:luma_amount:chroma_msize_x:chroma_msize_y:chroma_amount
        # 5:5:1.0 is a reasonable sharpening
        vf = f"unsharp=5:5:1.0:5:5:0.0,scale=iw*{scale}:ih*{scale}:flags=lanczos"
        
        cmd = [
            FFMPEG_BIN, "-y", "-i", input_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "copy",
            output_path
        ]
        
        if _run_command(cmd, check=True):
            logger.info("✅ CPU Fallback enhancement complete.")
            return True
        return False
    except Exception as e:
        logger.error(f"CPU Fallback Failed: {e}")
        return False

def enhance_video_auto(input_path: str, output_path: str, scale: int = 2) -> bool:
    """
    Automatic Engine Selection (4-Stage Hybrid):
    1. Heavy Engine (PyTorch) - if CUDA & VRAM >= 6GB
    2. NCNN Engine (Vulkan) - Default
    3. Old CPU Fallback - if NCNN fails
    4. FFmpeg Scale - Last resort
    """
    vram = detect_cuda_vram()
    
    # 1. Try Heavy Engine
    if vram >= 6.0:
        logger.info(f"⚡ High-End GPU Detected ({vram:.2f}GB). Attempting Heavy Engine...")
        if run_heavy_engine(input_path, output_path, scale):
            return True
        logger.warning("⚠️ Heavy Engine failed/unavailable. Falling back to NCNN.")
    else:
        logger.info(f"ℹ️ VRAM ({vram:.2f}GB) < 6GB. Skipping Heavy Engine.")
        
    # 2. NCNN Engine (Default/Fallback)
    if run_ncnn_engine(input_path, output_path, scale):
        return True
    logger.warning("⚠️ NCNN Engine failed. Falling back to CPU Engine.")
        
    # 3. Old CPU Fallback
    if enhance_video_cpu_fallback(input_path, output_path, scale):
        return True
    logger.warning("⚠️ CPU Engine failed. Falling back to FFmpeg Scale.")

    # 4. FFmpeg Fallback (Last Resort)
    _upscale_ffmpeg(input_path, output_path, scale)
    return True

def _upscale_ffmpeg(input_path: str, output_path: str, scale: int):
    vf = f"scale=iw*{scale}:ih*{scale}:flags=lanczos" if scale > 1 else "null"
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "copy",
        output_path
    ]
    _run_command(cmd)

def normalize_video(input_path: str, output_path: str):
    logger.info("📏 Normalizing video...")
    vf = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30"
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", REENCODE_CRF,
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        output_path
    ]
    _run_command(cmd, check=True)

# ==================== TRANSITIONS ====================

def create_transition_clip(seg_a: str, seg_b: str, output_path: str, trans_type: str, duration: float):
    info_a = _get_video_info(seg_a)
    dur_a = info_a['duration']
    start_a = max(0, dur_a - duration)
    
    tail_a = output_path.replace(".mp4", "_tailA.mp4")
    _run_command([FFMPEG_BIN, "-y", "-ss", str(start_a), "-i", seg_a, "-t", str(duration), "-c", "copy", tail_a])
    
    head_b = output_path.replace(".mp4", "_headB.mp4")
    _run_command([FFMPEG_BIN, "-y", "-i", seg_b, "-t", str(duration), "-c", "copy", head_b])
    
    filter_str = f"[0:v][1:v]xfade=transition={trans_type}:duration={duration}:offset=0[v];[0:a][1:a]acrossfade=d={duration}[a]"
    if trans_type == "zoom":
        filter_str = f"[0:v][1:v]xfade=transition=circleopen:duration={duration}:offset=0[v];[0:a][1:a]acrossfade=d={duration}[a]"

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", tail_a, "-i", head_b,
        "-filter_complex", filter_str,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac",
        output_path
    ]
    _run_command(cmd)
    
    if os.path.exists(tail_a): os.remove(tail_a)
    if os.path.exists(head_b): os.remove(head_b)

def compile_with_transitions(input_video: Path, title: str) -> Path:
    import audio_processing
    
    input_path = os.path.abspath(str(input_video))
    job_id = f"job_{int(time.time())}"
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    final_output = os.path.join(OUTPUT_DIR, f"final_{title}.mp4")
    
    try:
        # 1. AI Enhancement (Dual Stage)
        logger.info("✨ Step 1: AI Enhancement")
        enhanced_video = os.path.join(job_dir, "enhanced.mp4")
        
        # Parse scale from ENHANCEMENT_LEVEL
        scale = 2
        if ENHANCEMENT_LEVEL and ENHANCEMENT_LEVEL[0].isdigit():
            scale = int(ENHANCEMENT_LEVEL[0])
            
        enhance_video_auto(input_path, enhanced_video, scale)
        
        # 2. Normalize
        logger.info("✨ Step 2: Normalization")
        norm_video = os.path.join(job_dir, "normalized.mp4")
        normalize_video(enhanced_video, norm_video)
        
        # 3. Segmentation
        logger.info("✨ Step 3: Segmentation")
        seg_pattern = os.path.join(job_dir, "seg_%03d.mp4")
        cmd_split = [
            FFMPEG_BIN, "-y", "-i", norm_video,
            "-c", "copy", "-f", "segment", "-segment_time", str(TRANSITION_INTERVAL),
            "-reset_timestamps", "1", seg_pattern
        ]
        _run_command(cmd_split, check=True)
        segments = sorted(glob.glob(os.path.join(job_dir, "seg_*.mp4")))
        
        if len(segments) < 2:
            logger.info("   Video too short for transitions.")
            merged_video = norm_video
        else:
            # 4. Transitions
            logger.info("✨ Step 4: Transitions")
            final_segments = []
            transitions = ["fade", "slideleft", "slideright", "wipeleft", "wiperight", "circleopen", "circleclose", "zoom"]
            
            seg0 = segments[0]
            dur0 = _get_video_info(seg0)['duration']
            trim0 = os.path.join(job_dir, "final_seg_000.mp4")
            _run_command([FFMPEG_BIN, "-y", "-i", seg0, "-t", str(max(0, dur0 - TRANSITION_DURATION)), "-c", "copy", trim0])
            final_segments.append(trim0)
            
            for i in range(len(segments) - 1):
                seg_curr = segments[i]
                seg_next = segments[i+1]
                trans_type = random.choice(transitions)
                trans_path = os.path.join(job_dir, f"final_trans_{i}.mp4")
                create_transition_clip(seg_curr, seg_next, trans_path, trans_type, TRANSITION_DURATION)
                final_segments.append(trans_path)
                
                is_last = (i + 1) == (len(segments) - 1)
                dur_next = _get_video_info(seg_next)['duration']
                start_trim = TRANSITION_DURATION
                end_trim = 0 if is_last else TRANSITION_DURATION
                keep_dur = max(0, dur_next - start_trim - end_trim)
                
                body_next = os.path.join(job_dir, f"final_seg_{i+1:03d}.mp4")
                _run_command([
                    FFMPEG_BIN, "-y", "-ss", str(start_trim), "-i", seg_next,
                    "-t", str(keep_dur), "-c", "copy", body_next
                ])
                final_segments.append(body_next)

            # 5. Merge
            logger.info("✨ Step 5: Merging")
            list_file = os.path.join(job_dir, "merge_list.txt")
            with open(list_file, "w") as f:
                for p in final_segments:
                    f.write(f"file '{os.path.abspath(p).replace(os.sep, '/')}'\n")
            
            merged_video = os.path.join(job_dir, "merged_video.mp4")
            _run_command([FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", merged_video], check=True)

        # 6. Audio Remix
        logger.info("✨ Step 6: Audio Remix")
        remixed_audio = os.path.join(job_dir, "remixed.wav")
        audio_processing.heavy_remix(merged_video, remixed_audio)
        
        # 7. Final Assembly
        logger.info("✨ Step 7: Final Assembly")
        encoder = "libx264"
        if COMPUTE_MODE in ["gpu", "auto"]:
             try:
                 subprocess.check_output([FFMPEG_BIN, "-hide_banner", "-encoders"], stderr=subprocess.STDOUT)
             except: pass
             
        cmd_final = [
            FFMPEG_BIN, "-y", "-i", merged_video, "-i", remixed_audio,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", REENCODE_PRESET, "-crf", REENCODE_CRF,
            "-shortest", final_output
        ]
        _run_command(cmd_final, check=True)
        
        if os.path.exists(final_output) and os.path.getsize(final_output) > 0:
            return Path(final_output)
        else:
            raise Exception("Output creation failed")

    except Exception as e:
        logger.error(f"Pipeline Error: {e}")
        return None
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

# Legacy function for compilation support
def compile_batch_with_transitions(video_files: List[str], output_filename: str) -> Optional[str]:
    """
    Compiles multiple video files into one with transitions, normalization, and audio remixing.
    Replaces the old streamcopy merge.
    """
    import audio_processing
    
    # 1. Setup Paths
    if os.path.isabs(output_filename):
        final_output = output_filename
    else:
        final_output = os.path.abspath(os.path.join(OUTPUT_DIR, output_filename))
        
    os.makedirs(os.path.dirname(final_output), exist_ok=True)
    
    job_id = f"batch_{int(time.time())}"
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    try:
        logger.info(f"🚀 Starting Batch Compilation for {len(video_files)} videos...")
        
        # 2. Normalize Inputs
        logger.info("✨ Step 1: Normalizing Inputs")
        normalized_clips = []
        for i, vid in enumerate(video_files):
            norm_path = os.path.join(job_dir, f"norm_{i:03d}.mp4")
            # Ensure absolute path for input
            abs_input = os.path.abspath(vid)
            if not os.path.exists(abs_input):
                logger.warning(f"⚠️ Input file not found: {abs_input}")
                continue
                
            normalize_video(abs_input, norm_path)
            if os.path.exists(norm_path):
                normalized_clips.append(norm_path)
            else:
                logger.error(f"❌ Failed to normalize: {vid}")

        if len(normalized_clips) < 2:
            logger.warning("⚠️ Not enough valid clips for transitions. Concatenating directly if possible.")
            # Fallback to simple concat if only 1 clip or transitions impossible
            if len(normalized_clips) == 1:
                shutil.copy2(normalized_clips[0], final_output)
                return final_output
            return None

        # 3. Apply Transitions
        logger.info("✨ Step 2: Applying Transitions")
        final_segments = []
        transitions = ["fade", "slideleft", "slideright", "wipeleft", "wiperight", "circleopen", "circleclose", "zoom"]
        
        # Process first clip (trim end for transition)
        seg0 = normalized_clips[0]
        dur0 = _get_video_info(seg0)['duration']
        trim0 = os.path.join(job_dir, "final_seg_000.mp4")
        # Trim end by TRANSITION_DURATION
        _run_command([FFMPEG_BIN, "-y", "-i", seg0, "-t", str(max(0, dur0 - TRANSITION_DURATION)), "-c", "copy", trim0])
        final_segments.append(trim0)
        
        for i in range(len(normalized_clips) - 1):
            seg_curr = normalized_clips[i]
            seg_next = normalized_clips[i+1]
            
            # Create Transition
            trans_type = random.choice(transitions)
            trans_path = os.path.join(job_dir, f"final_trans_{i}.mp4")
            create_transition_clip(seg_curr, seg_next, trans_path, trans_type, TRANSITION_DURATION)
            final_segments.append(trans_path)
            
            # Process Next Clip Body
            # Trim start by TRANSITION_DURATION
            # Trim end by TRANSITION_DURATION (unless it's the last clip)
            is_last = (i + 1) == (len(normalized_clips) - 1)
            dur_next = _get_video_info(seg_next)['duration']
            
            start_trim = TRANSITION_DURATION
            end_trim = 0 if is_last else TRANSITION_DURATION
            keep_dur = max(0, dur_next - start_trim - end_trim)
            
            body_next = os.path.join(job_dir, f"final_seg_{i+1:03d}.mp4")
            _run_command([
                FFMPEG_BIN, "-y", "-ss", str(start_trim), "-i", seg_next,
                "-t", str(keep_dur), "-c", "copy", body_next
            ])
            final_segments.append(body_next)

        # 4. Merge Segments
        logger.info("✨ Step 3: Merging Segments")
        list_file = os.path.join(job_dir, "merge_list.txt")
        with open(list_file, "w") as f:
            for p in final_segments:
                f.write(f"file '{os.path.abspath(p).replace(os.sep, '/')}'\n")
        
        merged_video = os.path.join(job_dir, "merged_video.mp4")
        if not _run_command([FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", merged_video], check=True):
             logger.error("❌ Merge failed.")
             return None

        # 5. Audio Remix
        logger.info("✨ Step 4: Audio Remix")
        remixed_audio = os.path.join(job_dir, "remixed.wav")
        audio_processing.heavy_remix(merged_video, remixed_audio)
        
        # 6. Final Assembly
        logger.info("✨ Step 5: Final Assembly")
        cmd_final = [
            FFMPEG_BIN, "-y", "-i", merged_video, "-i", remixed_audio,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", REENCODE_PRESET, "-crf", REENCODE_CRF,
            "-shortest", final_output
        ]
        _run_command(cmd_final, check=True)
        
        if os.path.exists(final_output) and os.path.getsize(final_output) > 0:
            logger.info(f"✅ Batch Compilation Complete: {final_output}")
            return final_output
        else:
            logger.error("❌ Final output creation failed.")
            return None

    except Exception as e:
        logger.error(f"Batch Compilation Failed: {e}")
        return None
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
