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

# Direct Zip Downloads (Reliable Windows Binaries)
ZIP_REALESRGAN = "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.0/realesrgan-ncnn-vulkan-v0.2.0-windows.zip"
ZIP_SRMD = "https://github.com/nihui/srmd-ncnn-vulkan/releases/download/20220728/srmd-ncnn-vulkan-20220728-windows.zip"

# Heavy Models (PyTorch) - Placeholder URLs (In real scenario, these would be direct weights)
# For now, we will just create the directory structure as actual weights are large and context dependent.
# Users on Colab usually download these via gdown or similar.

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
    """Recursively find exe_name in src_dir and copy to dest_dir as dest_filename (or exe_name)."""
    target_name = dest_filename if dest_filename else exe_name
    for root, dirs, files in os.walk(src_dir):
        files_lower = [f.lower() for f in files]
        if exe_name.lower() in files_lower:
            actual_name = files[files_lower.index(exe_name.lower())]
            src_path = os.path.join(root, actual_name)
            dest_path = os.path.join(dest_dir, target_name) 
            logger.info(f"   Found {actual_name}, copying to {dest_path}...")
            shutil.copy2(src_path, dest_path)
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
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return p.total_memory / (1024**3)
    except:
        return 0.0
    return 0.0

def download_heavy_models():
    """
    Download PyTorch models for Heavy Engine.
    Only runs if on Colab or explicitly requested.
    """
    # Check VRAM requirements
    vram = detect_cuda_vram()
    if vram < 6.0:
        logger.info(f"⏩ Skipping Heavy Model download (VRAM {vram:.2f}GB < 6.0GB).")
        return

    logger.info("📥 Downloading Heavy Engine Models (PyTorch)...")
    os.makedirs(HEAVY_MODELS_DIR, exist_ok=True)
    
    # Placeholder: In a real setup, we'd download .pth files here
    # e.g. RealESRGAN_x4plus.pth, ESRNet.pth, etc.
    with open(os.path.join(HEAVY_MODELS_DIR, "README.txt"), "w") as f:
        f.write("Place Heavy Engine PyTorch models here.")
    
    logger.info("✅ Heavy models directory prepared.")

def main():
    logger.info("🚀 Starting Advanced Tools Installer...")
    os.makedirs(TOOLS_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    is_windows = platform.system() == "Windows"
    ext = ".exe" if is_windows else ""
    
    # Tool Names
    realesrgan_exe = f"realesrgan-ncnn-vulkan{ext}"
    srmd_exe = f"srmd-ncnn-vulkan{ext}"
    
    # Virtual Tool Names (Copies/Renames)
    codeformer_exe = f"codeformer-ncnn-vulkan{ext}" # Fallback
    esrnet_exe = f"esrnet-ncnn-vulkan{ext}"         # Uses RealESRGAN binary with different model
    
    # 1. Install Real-ESRGAN (Base for Upscale & ESRNet)
    if not os.path.exists(os.path.join(TOOLS_DIR, realesrgan_exe)):
        logger.info(f"🔍 Installing {realesrgan_exe}...")
        if not install_via_zip(ZIP_REALESRGAN, realesrgan_exe):
            logger.error(f"❌ Failed to install {realesrgan_exe}")
    else:
        logger.info(f"✅ {realesrgan_exe} already exists.")

    # 2. Install SRMD (Texture Smoothing)
    if not os.path.exists(os.path.join(TOOLS_DIR, srmd_exe)):
        logger.info(f"🔍 Installing {srmd_exe}...")
        if not install_via_zip(ZIP_SRMD, srmd_exe):
            logger.error(f"❌ Failed to install {srmd_exe}")
    else:
        logger.info(f"✅ {srmd_exe} already exists.")

    # 3. Setup Virtual Tools (Copies)
    
    # CodeFormer -> RealESRGAN (Fallback)
    if not os.path.exists(os.path.join(TOOLS_DIR, codeformer_exe)):
        logger.info(f"⚙️ Setting up {codeformer_exe} (using Real-ESRGAN binary)...")
        src = os.path.join(TOOLS_DIR, realesrgan_exe)
        dst = os.path.join(TOOLS_DIR, codeformer_exe)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logger.info("✅ CodeFormer fallback created.")
        else:
            logger.error("❌ Could not create CodeFormer fallback (Source missing).")

    # ESRNet -> RealESRGAN (Uses same binary, just different model args usually)
    if not os.path.exists(os.path.join(TOOLS_DIR, esrnet_exe)):
        logger.info(f"⚙️ Setting up {esrnet_exe} (using Real-ESRGAN binary)...")
        src = os.path.join(TOOLS_DIR, realesrgan_exe)
        dst = os.path.join(TOOLS_DIR, esrnet_exe)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logger.info("✅ ESRNet binary created.")
        else:
            logger.error("❌ Could not create ESRNet binary (Source missing).")

    # 4. Heavy Engine Setup
    detect_cuda_vram()
    download_heavy_models()

    # Cleanup
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR, onerror=_on_rm_error)
        except: pass
        
    # Permissions
    if not is_windows:
        try:
            subprocess.run(["chmod", "+x", os.path.join(TOOLS_DIR, realesrgan_exe)])
            subprocess.run(["chmod", "+x", os.path.join(TOOLS_DIR, srmd_exe)])
            subprocess.run(["chmod", "+x", os.path.join(TOOLS_DIR, codeformer_exe)])
            subprocess.run(["chmod", "+x", os.path.join(TOOLS_DIR, esrnet_exe)])
        except: pass

    logger.info("✨ Tools installation process finished.")

if __name__ == "__main__":
    main()
