"""
auth.py — User authentication for Gudsky Research Foundation Video AI
Stores users in src/data/users.csv
Tracks GPU/resource utilisation in src/data/time_vs_gpu_util.csv
"""

import csv
import hashlib
import logging
import os
import secrets
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import psutil
import torch

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent
DATA_DIR   = _HERE / "data"
USERS_CSV  = DATA_DIR / "users.csv"
GPU_CSV    = DATA_DIR / "time_vs_gpu_util.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)

USERS_FIELDS = ["id", "name", "email", "password_hash", "role", "created_at", "last_login"]
GPU_FIELDS   = [
    "timestamp",
    "gpu0_util_pct", "gpu0_mem_used_gb", "gpu0_mem_total_gb",
    "gpu1_util_pct", "gpu1_mem_used_gb", "gpu1_mem_total_gb",
    "active_users", "cpu_pct", "ram_used_gb",
]

# ── Ensure CSV files exist with headers ───────────────────────────────────────
def _ensure_csv(path: Path, fields: list):
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

_ensure_csv(USERS_CSV,  USERS_FIELDS)
_ensure_csv(GPU_CSV,    GPU_FIELDS)

_lock = threading.Lock()


# ── Helpers ────────────────────────────────────────────────────────────────────
def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _read_users() -> list[dict]:
    with open(USERS_CSV, newline="") as f:
        return list(csv.DictReader(f))


def _write_users(rows: list[dict]):
    with open(USERS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=USERS_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _next_id(rows: list[dict]) -> str:
    if not rows:
        return "1"
    return str(max(int(r["id"]) for r in rows) + 1)


# ── In-memory session store  {token: {user_id, email, name, expires}} ─────────
_sessions: dict[str, dict] = {}
SESSION_TTL_HOURS = 24


def _new_token() -> str:
    return secrets.token_urlsafe(32)


# ── Public API ─────────────────────────────────────────────────────────────────

def register(name: str, email: str, password: str) -> tuple[bool, str]:
    """
    Create a new user account.
    Returns (True, token) on success or (False, error_message).
    """
    email = email.strip().lower()
    name  = name.strip()

    if not name or not email or not password:
        return False, "All fields are required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if "@" not in email:
        return False, "Invalid email address."

    with _lock:
        rows = _read_users()
        if any(r["email"] == email for r in rows):
            return False, "An account with this email already exists."

        now = datetime.utcnow().isoformat()
        new_user = {
            "id":            _next_id(rows),
            "name":          name,
            "email":         email,
            "password_hash": _hash(password),
            "role":          "user",
            "created_at":    now,
            "last_login":    now,
        }
        rows.append(new_user)
        _write_users(rows)

    token = _new_token()
    _sessions[token] = {
        "user_id": new_user["id"],
        "email":   email,
        "name":    name,
        "expires": datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS),
    }
    logger.info(f"[Auth] Registered: {email}")
    return True, token


def login(email: str, password: str) -> tuple[bool, str]:
    """
    Authenticate an existing user.
    Returns (True, token) on success or (False, error_message).
    """
    email = email.strip().lower()

    with _lock:
        rows = _read_users()
        user = next((r for r in rows if r["email"] == email), None)
        if not user or user["password_hash"] != _hash(password):
            return False, "Invalid email or password."

        # Update last_login
        user["last_login"] = datetime.utcnow().isoformat()
        _write_users(rows)

    token = _new_token()
    _sessions[token] = {
        "user_id": user["id"],
        "email":   email,
        "name":    user["name"],
        "expires": datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS),
    }
    logger.info(f"[Auth] Login: {email}")
    return True, token


def validate_token(token: str) -> Optional[dict]:
    """Return session dict if token is valid and not expired, else None."""
    session = _sessions.get(token)
    if not session:
        return None
    if datetime.utcnow() > session["expires"]:
        del _sessions[token]
        return None
    return session


def logout(token: str):
    _sessions.pop(token, None)


# ── GPU utilisation logger ─────────────────────────────────────────────────────

def log_gpu_utilisation(active_users: int = 0):
    """Append one row of GPU + system stats to time_vs_gpu_util.csv."""
    try:
        row: dict = {"timestamp": datetime.utcnow().isoformat()}

        if torch.cuda.is_available():
            for i in range(min(2, torch.cuda.device_count())):
                free, total = torch.cuda.mem_get_info(i)
                used = (total - free) / 1e9
                total_gb = total / 1e9

                # torch has no built-in % utilisation; use nvidia-smi via pynvml if available
                util_pct = 0
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util_pct = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                except Exception:
                    pass

                row[f"gpu{i}_util_pct"]     = util_pct
                row[f"gpu{i}_mem_used_gb"]  = round(used, 2)
                row[f"gpu{i}_mem_total_gb"] = round(total_gb, 1)

        # Fill missing GPU slots
        for i in range(2):
            for k in [f"gpu{i}_util_pct", f"gpu{i}_mem_used_gb", f"gpu{i}_mem_total_gb"]:
                row.setdefault(k, 0)

        vm = psutil.virtual_memory()
        row["active_users"] = active_users
        row["cpu_pct"]      = psutil.cpu_percent(interval=None)
        row["ram_used_gb"]  = round((vm.total - vm.available) / 1e9, 2)

        with _lock:
            with open(GPU_CSV, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=GPU_FIELDS).writerow(row)

    except Exception as exc:
        logger.debug(f"[Auth] GPU log error: {exc}")