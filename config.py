"""
Configuration for the MCP Photoshop server.
"""

import logging
import os

# Load .env file if it exists (must be before reading env vars)
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed, use system env vars

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.WARNING))

# ComfyUI API settings
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
COMFYUI_INPUT_DIR = os.getenv("COMFYUI_INPUT_DIR", None)  # Auto-detected if None
COMFYUI_OUTPUT_DIR = os.getenv("COMFYUI_OUTPUT_DIR", None)  # Auto-detected if None

# ComfyUI auto-start/kill settings
# Option 1: Set COMFYUI_START_CMD to a .bat path
# Option 2: Set COMFYUI_PYTHON and COMFYUI_MAIN to call python directly (recommended)
COMFYUI_START_CMD = os.getenv("COMFYUI_START_CMD", "")  # Path to ComfyUI start .bat (optional if COMFYUI_PYTHON + COMFYUI_MAIN set)
COMFYUI_PYTHON = os.getenv("COMFYUI_PYTHON", "")  # Path to ComfyUI's embedded python.exe
COMFYUI_MAIN = os.getenv("COMFYUI_MAIN", "")  # Path to ComfyUI/main.py
COMFYUI_ARGS = os.getenv("COMFYUI_ARGS", "--windows-standalone-build")  # Extra args for ComfyUI startup
COMFYUI_AUTO_KILL = os.getenv("COMFYUI_AUTO_KILL", "0") == "1"  # Kill after generation (0=use VRAM pressure management, 1=kill)
COMFYUI_START_TIMEOUT = int(os.getenv("COMFYUI_START_TIMEOUT", "180"))  # Seconds to wait for ComfyUI to start
COMFYUI_IDLE_TIMEOUT = int(os.getenv("COMFYUI_IDLE_TIMEOUT", "60"))  # Seconds of inactivity before auto-killing (when AUTO_KILL=1)

# WebSocket settings
WEBSOCKET_TIMEOUT = int(os.getenv("WEBSOCKET_TIMEOUT", "600"))  # seconds to wait for workflow completion
POLLING_INTERVAL = 1.0  # seconds between history polls (fallback)

# Session settings
MAX_UNDO_STEPS = int(os.getenv("MAX_UNDO_STEPS", "20"))
DEFAULT_CANVAS_WIDTH = 1024
DEFAULT_CANVAS_HEIGHT = 1024
DEFAULT_BG_COLOR = (255, 255, 255)

# Image generation defaults
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 20
DEFAULT_CFG = 1.5  # Flux2 Klein default

# Model names (match installed models)
MODEL_FLUX2 = "flux-2-klein-9b.safetensors"
MODEL_FLUX2_TEXT_ENCODER = "qwen_3_8b_fp8mixed.safetensors"
MODEL_FLUX2_VAE = "flux2-vae.safetensors"
MODEL_KONTEXT = "flux1-dev-kontext_fp8_scaled.safetensors"

# ANIMA model (anime-style generation)
MODEL_ANIMA = "anima-aesthetic-v1.1.safetensors"
MODEL_ANIMA_TEXT_ENCODER = "qwen_3_06b_base.safetensors"
MODEL_ANIMA_VAE = "qwen_image_vae.safetensors"

# Upscale models
MODEL_UPSCALE_FACE = "4xFaceUpDAT.pth"
MODEL_UPSCALE_ANIME = "RealESRGAN_x4plus_anime_6B.pth"

# Server settings
SERVER_NAME = "mcp-photoshop-server"

# VRAM pressure management: kill ComfyUI entirely if free VRAM drops below this
# threshold, otherwise just call free_memory(). Default 8192 MB (8 GB) for a
# 32 GB GPU shared with vLLM. Set via VRAM_PRESSURE_THRESHOLD_MB env var.
VRAM_PRESSURE_THRESHOLD_MB = int(os.getenv("VRAM_PRESSURE_THRESHOLD_MB", "8192"))