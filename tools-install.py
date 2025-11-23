import os
import sys
import shutil
import subprocess
import logging
import platform
import zipfile
import io
import urllib.request
import time
import stat

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] tools-install: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("tools-install")

TOOLS_DIR = os.path.join(os.getcwd(), "tools")
MODELS_DIR = os.path.join(os.getcwd(), "models")
HEAVY_MODELS_DIR = os.path.join(MODELS_DIR, "heavy")
NCNN_MODELS_DIR = os.path.join(MODELS_DIR, "ncnn")
TEMP_DIR = os.path.join(os.getcwd(), "temp_install")

# OS Detection for Cross-Platform Support
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

# NCNN Binary Downloads (OS-Specific)
if IS_WINDOWS:
    ZIP_REALESRGAN = "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.0/realesrgan-ncnn-vulkan-v0.2.0-windows.zip"
    ZIP_SRMD = "https://github.com/nihui/srmd-ncnn-vulkan/releases/download/20220728/srmd-ncnn-vulkan-20220728-windows.zip"
else:  # Linux (Colab)
    ZIP_REALESRGAN = "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.0/realesrgan-ncnn-vulkan-v0.2.0-linux.zip"
    ZIP_SRMD = "https://github.com/nihui/srmd-ncnn-vulkan/releases/download/20220728/srmd-ncnn-vulkan-20220728-linux.zip"

# Heavy Models (PyTorch) - Official Model URLs
MODEL_REALESRGAN_X4PLUS = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_CODEFORMER = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"

def _on_rm_error(func, path, exc_info):
    """Error handler for shutil.rmtree."""
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        time.sleep(0.1)
        try:
            func(path)
        except Exception as e:
            logger.warning(f"Could not delete {path}: {e}")

def _run_cmd(cmd):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        logger.warning(f"Command failed: {cmd} -> {e}")
        return False

def _find_and_copy_exe(src_dir, exe_name, dest_dir, dest_filename=None):
    """Recursively find exe_name in src_dir and copy to dest_dir as dest_filename (or exe_name).
    On Linux, also makes the binary executable.
    """
    target_name = dest_filename if dest_filename else exe_name
    for root, dirs, files in os.walk(src_dir):
        files_lower = [f.lower() for f in files]
        if exe_name.lower() in files_lower:
            actual_name = files[files_lower.index(exe_name.lower())]
            src_path = os.path.join(root, actual_name)
            dest_path = os.path.join(dest_dir, target_name) 
            logger.info(f"   Found {actual_name}, copying to {dest_path}...")
            shutil.copy2(src_path, dest_path)
            
            # Make executable on Linux
            if not IS_WINDOWS:
                try:
                    os.chmod(dest_path, 0o755)
                    logger.info(f"   Made {target_name} executable")
                except Exception as e:
                    logger.warning(f"   Could not chmod {target_name}: {e}")
            
            return True
    return False

def install_via_zip(url, exe_name, target_name_in_zip=None, dest_filename=None):
    """Attempt to install via zip download."""
    logger.info(f"   Attempting zip download: {url}")
    try:
        req = urllib.request.Request(
            url, 
            data=None, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                z = zipfile.ZipFile(io.BytesIO(response.read()))
                extract_path = os.path.join(TEMP_DIR, "zip_extract")
                if os.path.exists(extract_path):
                    shutil.rmtree(extract_path, onerror=_on_rm_error)
                z.extractall(extract_path)
                
                search_name = target_name_in_zip if target_name_in_zip else exe_name
                final_name = dest_filename if dest_filename else exe_name
                
                if _find_and_copy_exe(extract_path, search_name, TOOLS_DIR, dest_filename=final_name):
                    logger.info(f"✅ {final_name} installed via zip.")
                    return True
            else:
                logger.warning(f"HTTP Error: {response.status}")
    except Exception as e:
        logger.warning(f"Zip download failed: {e}")
    return False

def detect_cuda_vram():
    """Detect CUDA VRAM in GB. Returns 0.0 if no CUDA or error."""
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            vram_gb = p.total_memory / (1024**3)
            logger.info(f"✅ CUDA GPU detected: {vram_gb:.2f}GB VRAM")
            return vram_gb
    except ImportError:
        logger.info("ℹ️ PyTorch not installed. Skipping GPU detection.")
    except Exception as e:
        logger.warning(f"⚠️ GPU detection failed: {e}")
    return 0.0

def _download_file(url: str, dest_path: str) -> bool:
    """Download a file from URL to dest_path."""
    try:
        logger.info(f"   Downloading {os.path.basename(dest_path)}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=300) as response:
            if response.status == 200:
                with open(dest_path, 'wb') as f:
                    f.write(response.read())
                logger.info(f"   ✅ Downloaded {os.path.basename(dest_path)}")
                return True
        logger.warning(f"   ❌ HTTP {response.status} for {url}")
    except Exception as e:
        logger.warning(f"   ❌ Download failed: {e}")
    return False

def download_heavy_models():
    """
    Download PyTorch models for Heavy Engine.
    Only runs if GPU is available with sufficient VRAM.
    """
    vram = detect_cuda_vram()
    if vram < 6.0:
        logger.info(f"⏩ Skipping Heavy Model download (VRAM {vram:.2f}GB < 6.0GB).")
        return

    logger.info("📥 Downloading Heavy Engine Models (PyTorch)...")
    os.makedirs(HEAVY_MODELS_DIR, exist_ok=True)
    
    # Download Real-ESRGAN x4plus
    realesrgan_path = os.path.join(HEAVY_MODELS_DIR, "RealESRGAN_x4plus.pth")
    if not os.path.exists(realesrgan_path):
        if not _download_file(MODEL_REALESRGAN_X4PLUS, realesrgan_path):
            logger.warning("⚠️ Failed to download RealESRGAN model. Heavy Engine may not work.")
    else:
        logger.info("   ✅ RealESRGAN model already exists.")
    
    # Download CodeFormer
    codeformer_path = os.path.join(HEAVY_MODELS_DIR, "codeformer.pth")
    if not os.path.exists(codeformer_path):
        if not _download_file(MODEL_CODEFORMER, codeformer_path):
            logger.warning("⚠️ Failed to download CodeFormer model. Face enhancement may not work.")
    else:
        logger.info("   ✅ CodeFormer model already exists.")
    
    logger.info("✅ Heavy models setup complete.")

def main():
    logger.info("🚀 Starting Advanced Tools Installer...")
    logger.info(f"   Platform: {platform.system()}")
    
    # Use absolute paths to avoid Colab cwd issues
    os.makedirs(os.path.abspath(TOOLS_DIR), exist_ok=True)
    os.makedirs(os.path.abspath(TEMP_DIR), exist_ok=True)
    os.makedirs(os.path.abspath(MODELS_DIR), exist_ok=True)
    
    ext = ".exe" if IS_WINDOWS else ""
    
    # Tool Names
    realesrgan_exe = f"realesrgan-ncnn-vulkan{ext}"
    srmd_exe = f"srmd-ncnn-vulkan{ext}"
    
    # Virtual Tool Names (Copies/Renames)
    codeformer_exe = f"codeformer-ncnn-vulkan{ext}" # Fallback (CodeFormer NCNN doesn't exist)
    esrnet_exe = f"esrnet-ncnn-vulkan{ext}"         # Uses RealESRGAN binary with different model
    
    # 1. Install Real-ESRGAN (Base for Upscale & ESRNet)
    if not os.path.exists(os.path.join(TOOLS_DIR, realesrgan_exe)):
        logger.info(f"🔍 Installing {realesrgan_exe}...")
        try:
            if not install_via_zip(ZIP_REALESRGAN, realesrgan_exe):
                logger.warning(f"⚠️ Real-ESRGAN NCNN installation failed. Will fallback to CPU mode.")
        except Exception as e:
            logger.warning(f"⚠️ Real-ESRGAN NCNN installation error: {e}. Will fallback to CPU mode.")
    else:
        logger.info(f"✅ {realesrgan_exe} already exists.")

    # 2. Install SRMD (Texture Smoothing)
    if not os.path.exists(os.path.join(TOOLS_DIR, srmd_exe)):
        logger.info(f"🔍 Installing {srmd_exe}...")
        try:
            if not install_via_zip(ZIP_SRMD, srmd_exe):
                logger.warning(f"⚠️ SRMD NCNN installation failed. Will fallback to CPU mode.")
        except Exception as e:
            logger.warning(f"⚠️ SRMD NCNN installation error: {e}. Will fallback to CPU mode.")
    else:
        logger.info(f"✅ {srmd_exe} already exists.")

    # 3. Setup Virtual Tools (Copies)
    # Note: CodeFormer NCNN doesn't exist, so we copy realesrgan as fallback without erroring
    
    # CodeFormer -> RealESRGAN (Fallback)
    if not os.path.exists(os.path.join(TOOLS_DIR, codeformer_exe)):
        logger.info(f"⚙️ Setting up {codeformer_exe} (CodeFormer NCNN doesn't exist, using Real-ESRGAN as fallback)...")
        src = os.path.join(TOOLS_DIR, realesrgan_exe)
        dst = os.path.join(TOOLS_DIR, codeformer_exe)
        if os.path.exists(src):
            try:
                shutil.copy2(src, dst)
                if not IS_WINDOWS:
                    os.chmod(dst, 0o755)
                logger.info("✅ CodeFormer fallback created.")
            except Exception as e:
                logger.warning(f"⚠️ Could not create CodeFormer fallback: {e}")
        else:
            logger.warning("⚠️ Could not create CodeFormer fallback (Real-ESRGAN missing).")

    # ESRNet -> RealESRGAN (Uses same binary, just different model args usually)
    if not os.path.exists(os.path.join(TOOLS_DIR, esrnet_exe)):
        logger.info(f"⚙️ Setting up {esrnet_exe} (using Real-ESRGAN binary)...")
        src = os.path.join(TOOLS_DIR, realesrgan_exe)
        dst = os.path.join(TOOLS_DIR, esrnet_exe)
        if os.path.exists(src):
            try:
                shutil.copy2(src, dst)
                if not IS_WINDOWS:
                    os.chmod(dst, 0o755)
                logger.info("✅ ESRNet binary created.")
            except Exception as e:
                logger.warning(f"⚠️ Could not create ESRNet binary: {e}")
        else:
            logger.warning("⚠️ Could not create ESRNet binary (Real-ESRGAN missing).")

    # 4. Heavy Engine Setup (GPU models)
    try:
        download_heavy_models()
    except Exception as e:
        logger.warning(f"⚠️ Heavy Engine model download failed: {e}")

    # Cleanup
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR, onerror=_on_rm_error)
        except Exception as e:
            logger.warning(f"⚠️ Cleanup failed: {e}")
        
    # Final Permissions Check (Linux)
    # Note: Individual files already chmod'd during copy, but double-check here
    if not IS_WINDOWS:
        for binary in [realesrgan_exe, srmd_exe, codeformer_exe, esrnet_exe]:
            binary_path = os.path.join(TOOLS_DIR, binary)
            if os.path.exists(binary_path):
                try:
                    os.chmod(binary_path, 0o755)
                except Exception as e:
                    logger.warning(f"⚠️ Could not chmod {binary}: {e}")

    logger.info("✨ Tools installation process finished.")

if __name__ == "__main__":
    main()
