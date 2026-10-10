"""Loads settings from the .env file (no hardcoded credentials)."""
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASS", ""),
    "database": os.getenv("DB_NAME", "vision_platform"),
}

DEFAULT_CONFIDENCE = int(os.getenv("CONF_THRESHOLD", "50"))
DEFAULT_IMGSZ = int(os.getenv("IMG_SIZE", "416"))          # 320 / 416 / 512 / 640
DEFAULT_FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))      # detect every N frames
LOG_COOLDOWN_SEC = float(os.getenv("LOG_COOLDOWN_SEC", "1.0"))
CAM_WIDTH = int(os.getenv("CAM_WIDTH", "640"))
CAM_HEIGHT = int(os.getenv("CAM_HEIGHT", "480"))
