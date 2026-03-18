# Backend

The backend logic lives in `src/` directly (as per spec):

- `src/server.py`          — FastAPI app, REST endpoints, WebSocket endpoint
- `src/websocket_handler.py` — Connection manager, 10-user cap, session state
- `src/model/video_model.py` — Qwen2.5-VL model wrapper, frame extraction, inference

This folder (`src/backend/`) is reserved for any future sub-modules such as:
- `auth.py`       — API key / JWT authentication
- `rate_limiter.py` — Per-user rate limiting
- `storage.py`    — Video file management / S3 backend
