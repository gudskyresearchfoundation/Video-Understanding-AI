# Video-Understanding-AI
The Video Understanding AI system is a real-time, web-based application that allows users to upload a video file and ask natural-language questions about its content.

## Architecture

```
video-ai/
├── app.py                        # Uvicorn entrypoint
├── requirements.txt
├── Dockerfile                    # CUDA 12.4 + Python 3.11
├── setup.sh                      # One-shot install + model download
├── .env                          # All configuration
└── src/
    ├── server.py                 # FastAPI: REST + WebSocket
    ├── websocket_handler.py      # 10-user cap, session state
    ├── backend/                  # (reserved: auth, rate limiting)
    ├── frontend/
    │   └── index.html            # Chat UI (served at /static)
    └── model/
        ├── __init__.py
        └── video_model.py        # Qwen2.5-VL wrapper + OpenCV frame extraction
```

## R&D: Why Qwen2.5-VL + Transformers (not Ollama)

| Factor | Ollama | HuggingFace Transformers ✅ |
|---|---|---|
| Video token support | ❌ None | ✅ Native |
| Multi-GPU spread | ❌ Single process | ✅ `device_map="auto"` |
| Custom frame sampling | ❌ N/A | ✅ OpenCV, configurable |
| VRAM (2× L4 = 48 GB) | — | ✅ ~15 GB in BF16 |
| Video benchmarks | — | ✅ SOTA (VideoMME, ActivityNet) |

## API

### REST
| Method | Path | Description |
|---|---|---|
| GET | `/` | Redirect to UI |
| GET | `/health` | Model status, GPU stats, user count |
| POST | `/upload?session_id=<id>` | Upload video (< 100 MB) |

### WebSocket  `ws://host/ws`
Messages are JSON:

**Client → Server**
```json
{ "type": "question", "question": "What is happening?" }
{ "type": "ping" }
{ "type": "clear" }
```

**Server → Client**
```json
{ "type": "connected",  "session_id": "…" }
{ "type": "thinking",   "message": "Analyzing video…" }
{ "type": "answer",     "question": "…", "answer": "…", "turn": 1 }
{ "type": "error",      "message": "…" }
{ "type": "cleared" }
{ "type": "pong" }
```

## Configuration (`.env`)

```env
MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
MODEL_CACHE_DIR=./models_cache
MAX_NEW_TOKENS=1024
MAX_FRAMES=32
HOST=0.0.0.0
PORT=8000
MAX_UPLOAD_SIZE_MB=100
MAX_WS_CONNECTIONS=10
DEVICE=cuda
TORCH_DTYPE=bfloat16
```

## Deployment

### Bare metal
```bash
bash setup.sh        # installs deps + downloads model weights (~15 GB)
python app.py
# Open http://localhost:8000
```

### Docker (recommended)
```bash
docker build -t video-ai .

# Mount model cache as a volume so weights survive container restarts
docker run --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models_cache:/app/models_cache \
  --env-file .env \
  video-ai
```

## GPU Memory Layout (2× L4)

```
GPU 0  ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░  ~12–15 GB used (model weights + KV cache)
GPU 1  ▓▓░░░░░░░░░░░░░░░░░░░░░░░   ~3–6  GB used (overflow / parallel heads)
```
`device_map="auto"` handles the split automatically.
