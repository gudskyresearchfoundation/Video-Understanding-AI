"""
Entrypoint — run with:  python app.py
or via Docker:          see Dockerfile
"""

import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

import uvicorn
from src.server import app  # noqa: E402 — import after env is loaded

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=60,
    )
