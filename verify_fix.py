import os
import sys
import time
import shutil
import logging
from pathlib import Path

# Mock config to avoid heavy AI
os.environ["TRANSITION_INTERVAL"] = "3"
os.environ["AI_FAST_MODE"] = "yes"
os.environ["COMPUTE_MODE"] = "cpu"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify")

# Import compiler functions
try:
    from compiler import compile_with_transitions, enhance_video_auto, compile_batch_with_transitions
except ImportError:
    # Add current dir to path
    sys.path.append(os.getcwd())
    from compiler import compile_with_transitions, enhance_video_auto, compile_batch_with_transitions

# Mock enhance_video_auto to skip heavy processing
def mock_enhance(input_path, output_path, scale):
    logger.info(f"Mock Enhance: Copying {input_path} to {output_path}")
    shutil.copy2(input_path, output_path)
    return True

import compiler
compiler.enhance_video_auto = mock_enhance

def create_dummy_video(filename, duration=5, color="red"):
    """Create a dummy video using ffmpeg"""
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", 
        f"-i", f"color=c={color}:s=1280x720:d={duration}", 
        "-f", "lavfi", f"-i", f"anullsrc=r=44100:cl=stereo",
        "-c:v", "libx264", "-preset", "ultrafast", 
        "-c:a", "aac", "-shortest",
        filename
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.abspath(filename)

import subprocess

def test_batch_compilation():
    logger.info("🧪 Starting Verification Test...")
    
    # 1. Create Dummy Videos
    os.makedirs("temp_test", exist_ok=True)
    v1 = create_dummy_video(os.path.join("temp_test", "v1.mp4"), 5, "red")
    v2 = create_dummy_video(os.path.join("temp_test", "v2.mp4"), 5, "blue")
    
    # 2. Test Batch Compilation (New Function)
    logger.info("Testing compile_batch_with_transitions...")
    output_name = "batch_test_final.mp4"
    result = compiler.compile_batch_with_transitions([v1, v2], output_name)
    
    if result and os.path.exists(result):
        logger.info(f"✅ Batch Compilation successful: {result}")
        logger.info("🎉 Verification Passed!")
    else:
        logger.error("❌ Batch Compilation failed!")
        sys.exit(1)

if __name__ == "__main__":
    try:
        test_batch_compilation()
    finally:
        # Cleanup
        # shutil.rmtree("temp_test", ignore_errors=True)
        pass
