"""
FastAPI server
  GET  /              → UI
  POST /auth/register → create account
  POST /auth/login    → login
  POST /auth/logout   → logout
  GET  /health        → status
  POST /upload        → upload video for a session
  WS   /ws            → real-time Q&A (requires auth token)
"""

import asyncio
import logging
import os
import tempfile
import threading

import psutil
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .backend.auth import login, logout, register, validate_token, log_gpu_utilisation
from .model import get_model
from .websocket_handler import get_manager

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100")) * 1024 * 1024

app = FastAPI(title="Gudsky Research Foundation — Video Understanding AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")


# ── Lifecycle ──────────────────────────────────────────────────────────────────

def _gpu_log_loop():
    """Background thread: log GPU stats every 60 s."""
    import time
    while True:
        try:
            manager = get_manager()
            log_gpu_utilisation(active_users=manager.active_count)
        except Exception:
            pass
        time.sleep(60)


@app.on_event("startup")
async def _startup():
    loop = asyncio.get_event_loop()
    model = get_model()
    logger.info("[Server] Loading AI model…")
    await loop.run_in_executor(None, model.load)
    logger.info("[Server] Model ready.")

    t = threading.Thread(target=_gpu_log_loop, daemon=True)
    t.start()


# ── Auth routes ────────────────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    name:     str = ""
    email:    str
    password: str


@app.post("/auth/register")
async def auth_register(body: AuthRequest):
    ok, result = register(body.name, body.email, body.password)
    if not ok:
        raise HTTPException(status_code=400, detail=result)
    return {"token": result, "message": "Account created successfully."}


@app.post("/auth/login")
async def auth_login(body: AuthRequest):
    ok, result = login(body.email, body.password)
    if not ok:
        raise HTTPException(status_code=401, detail=result)
    session = validate_token(result)
    return {"token": result, "name": session["name"], "message": "Login successful."}


@app.post("/auth/logout")
async def auth_logout(token: str):
    logout(token)
    return {"message": "Logged out."}


@app.get("/auth/validate")
async def auth_validate(token: str):
    """Check if a token is still valid. Used by frontend on page load."""
    session = validate_token(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return {"valid": True, "name": session["name"]}


# ── REST ───────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
async def health():
    model   = get_model()
    manager = get_manager()
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props       = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            gpus.append({
                "id":            i,
                "name":          props.name,
                "vram_total_gb": round(total / 1e9, 1),
                "vram_free_gb":  round(free  / 1e9, 1),
                "vram_used_gb":  round((total - free) / 1e9, 1),
            })
    ram = psutil.virtual_memory()
    return {
        "status":             "ok",
        "model_loaded":       model.is_loaded(),
        "model_name":         model.model_name,
        "active_connections": manager.active_count,
        "max_connections":    int(os.getenv("MAX_WS_CONNECTIONS", "10")),
        "gpus":               gpus,
        "ram_total_gb":       round(ram.total     / 1e9, 1),
        "ram_available_gb":   round(ram.available / 1e9, 1),
    }


@app.post("/upload")
async def upload_video(
    session_id: str,
    token: str = "",
    file: UploadFile = File(...),
):
    # Validate auth token
    if not token:
        raise HTTPException(status_code=401, detail="Authentication token missing. Please log in.")
    session_auth = validate_token(token)
    if not session_auth:
        raise HTTPException(status_code=401, detail="Invalid or expired session. Please log in again.")

    manager = get_manager()
    session = manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="WebSocket session not found.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max is {os.getenv('MAX_UPLOAD_SIZE_MB','100')} MB.")
    if not (file.content_type or "").startswith("video/"):
        raise HTTPException(status_code=415, detail=f"Unsupported type: {file.content_type}")

    ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir="/tmp")
    tmp.write(content)
    tmp.flush()
    tmp.close()

    if session.video_path and os.path.exists(session.video_path):
        try:
            os.remove(session.video_path)
        except Exception:
            pass

    session.video_path = tmp.name
    session.history.clear()

    size_mb = round(len(content) / 1e6, 2)
    logger.info(f"[Upload] user={session_auth['email']} size={size_mb}MB")
    return {"message": "Video uploaded successfully.", "size_mb": size_mb, "filename": file.filename}


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    manager    = get_manager()
    session_id = await manager.connect(websocket)
    if session_id is None:
        return

    await manager.send(session_id, {
        "type":       "connected",
        "session_id": session_id,
        "message":    "Connected. Please authenticate first.",
    })

    authenticated = False
    user_info     = {}

    try:
        while True:
            raw   = await websocket.receive_json()
            mtype = raw.get("type", "")

            if mtype == "ping":
                await manager.send(session_id, {"type": "pong"})
                continue

            # ── Auth handshake over WS ───────────────────────────────────────
            if mtype == "auth":
                token   = raw.get("token", "")
                session_auth = validate_token(token)
                if not session_auth:
                    await manager.send(session_id, {
                        "type":    "auth_fail",
                        "message": "Invalid or expired token. Please log in.",
                    })
                    continue
                authenticated = True
                user_info     = session_auth
                await manager.send(session_id, {
                    "type":    "auth_ok",
                    "name":    session_auth["name"],
                    "message": f"Welcome, {session_auth['name']}! Upload a video to get started.",
                })
                continue

            # ── All subsequent messages require auth ─────────────────────────
            if not authenticated:
                await manager.send(session_id, {
                    "type":    "error",
                    "message": "Not authenticated. Please log in first.",
                })
                continue

            if mtype == "clear":
                session = manager.get_session(session_id)
                if session:
                    session.history.clear()
                await manager.send(session_id, {"type": "cleared"})
                continue

            if mtype == "question":
                question = (raw.get("question") or "").strip()
                if not question:
                    await manager.send(session_id, {"type": "error", "message": "Empty question."})
                    continue

                session = manager.get_session(session_id)
                if not session or not session.video_path:
                    await manager.send(session_id, {
                        "type":    "error",
                        "message": "No video uploaded yet.",
                    })
                    continue

                await manager.send(session_id, {"type": "thinking", "message": "Analyzing video…"})

                model = get_model()
                if not model.is_loaded():
                    await manager.send(session_id, {"type": "error", "message": "Model still loading."})
                    continue

                loop = asyncio.get_event_loop()
                try:
                    answer = await loop.run_in_executor(
                        None, model.analyze, session.video_path, question
                    )
                    session.history.append({"q": question, "a": answer})
                    await manager.send(session_id, {
                        "type":     "answer",
                        "question": question,
                        "answer":   answer,
                        "turn":     len(session.history),
                    })
                    # Log GPU utilisation after each inference
                    log_gpu_utilisation(active_users=manager.active_count)
                except Exception as exc:
                    logger.exception(f"[WS] Inference error")
                    await manager.send(session_id, {"type": "error", "message": f"Inference failed: {exc}"})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception(f"[WS] Error: {exc}")
    finally:
        await manager.disconnect(session_id)