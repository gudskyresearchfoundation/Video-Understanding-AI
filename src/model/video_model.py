"""
Video Understanding Model — Qwen2.5-VL-7B-Instruct
Handles model loading, frame extraction, and inference.
Designed to run on 2x NVIDIA L4 (24 GB VRAM each).
"""

import gc
import logging
import os
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Frame extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, max_frames: int = 32) -> List[Image.Image]:
    """
    Extract evenly-spaced RGB frames from a video file using OpenCV.
    Falls back to decord if OpenCV fails (e.g. some MKV/WebM edge cases).
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = total_frames / fps

    logger.info(
        f"[extract_frames] total={total_frames} fps={fps:.1f} "
        f"duration={duration:.1f}s max_frames={max_frames}"
    )

    n = min(max_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, n, dtype=int)

    frames: List[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

    cap.release()

    if not frames:
        raise ValueError("No frames could be extracted from the video.")

    logger.info(f"[extract_frames] extracted {len(frames)} frames")
    return frames


# ──────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ──────────────────────────────────────────────────────────────────────────────

class VideoUnderstandingModel:
    """
    Singleton wrapper around Qwen2.5-VL-7B-Instruct for video Q&A.

    Uses device_map='auto' so PyTorch automatically splits the model
    across both L4 GPUs when needed, while keeping inference fast on GPU 0.
    """

    def __init__(self):
        self.model = None
        self.processor = None

        self.model_name   = os.getenv("MODEL_NAME",      "Qwen/Qwen2.5-VL-7B-Instruct")
        self.cache_dir    = os.getenv("MODEL_CACHE_DIR", "./models_cache")
        self.max_frames   = int(os.getenv("MAX_FRAMES",      "32"))
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "1024"))

        dtype_str         = os.getenv("TORCH_DTYPE", "bfloat16")
        self.torch_dtype  = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Download weights (first run) and load model into GPU memory."""
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info(f"[Model] Loading {self.model_name} …")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",        # spreads across both L4 GPUs automatically
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        self.model.eval()

        # Log GPU usage after loading
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"[Model] GPU {i}: {alloc:.1f}/{total:.1f} GB used")

        logger.info("[Model] Ready ✓")

    def is_loaded(self) -> bool:
        return self.model is not None

    def unload(self) -> None:
        """Free GPU memory (useful for testing)."""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        torch.cuda.empty_cache()

    # ── Inference ──────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def analyze(self, video_path: str, question: str) -> str:
        """
        Analyze a video file and answer a natural-language question about it.

        Args:
            video_path: Path to the video file (< 100 MB already validated upstream).
            question:   User question, e.g. "What is happening in this video?"

        Returns:
            Model's text answer as a string.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded — call load() first.")

        # 1. Extract frames
        frames = extract_frames(video_path, max_frames=self.max_frames)

        # 2. Build the Qwen2.5-VL chat message
        #    We pass each frame as an individual image so the model sees them
        #    as a temporal sequence via its video understanding pipeline.
        video_content = [
            {"type": "video", "video": frames, "fps": 1.0}
        ]
        messages = [
            {
                "role": "user",
                "content": video_content + [{"type": "text", "text": question}],
            }
        ]

        # 3. Tokenize
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # process_vision_info is provided by qwen-vl-utils and handles
        # extracting image/video tensors from the message dict
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except ImportError:
            # Fallback: pass frames as images directly
            logger.warning("[Model] qwen-vl-utils not found, using fallback frame path")
            inputs = self.processor(
                text=[text_prompt],
                images=frames,
                padding=True,
                return_tensors="pt",
            )

        # Move inputs to the first device of the model
        first_device = next(self.model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        # 4. Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        # 5. Decode — trim the prompt tokens
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs["input_ids"], output_ids)
        ]
        answer = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        # 6. Release KV cache / intermediate tensors
        del inputs, output_ids, trimmed
        gc.collect()
        torch.cuda.empty_cache()

        return answer


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────────────────────────────────────

_instance: Optional[VideoUnderstandingModel] = None


def get_model() -> VideoUnderstandingModel:
    global _instance
    if _instance is None:
        _instance = VideoUnderstandingModel()
    return _instance
