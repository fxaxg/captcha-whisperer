"""ONNX-backed captcha recogniser."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .constants import (
    BLANK_INDEX,
    LABELS,
    MODEL_FILENAME,
    NUM_CLASSES,
    TIME_STEPS,
    WEIGHTS_FILENAME,
)
from .preprocess import preprocess_array, preprocess_bytes


def default_model_path() -> Path:
    """Location of the bundled ONNX model, overridable with CAPTCHA_OCR_MODEL."""
    env = os.environ.get("CAPTCHA_OCR_MODEL")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "models" / MODEL_FILENAME


def weights_path() -> Path:
    return Path(__file__).resolve().parent / "models" / WEIGHTS_FILENAME


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def decode(logits: np.ndarray) -> tuple[list[str], list[float], list[list[float]]]:
    """Greedy CTC decode for a batch of logits.

    Args:
        logits: array of shape (T, N, C).

    Returns:
        texts: one string per sample.
        confidences: product of the per-character probabilities, one per sample.
        char_confidences: list (per sample) of the probability of each emitted
            character, in order.
    """
    probs = _softmax(logits)                 # (T, N, C)
    t_steps, n = logits.shape[0], logits.shape[1]
    texts: list[str] = []
    confidences: list[float] = []
    char_confs: list[list[float]] = []

    for n_i in range(n):
        out: list[str] = []
        char_prob: list[float] = []
        prev = BLANK_INDEX
        for t in range(1, TIME_STEPS):
            # Choose among labels 1..BLANK_INDEX (label 0 is never emitted).
            window = logits[t - 1, n_i, 1:NUM_CLASSES]
            o = int(np.argmax(window))
            if o != BLANK_INDEX and prev != o:
                label_index = o + 1
                out.append(LABELS[label_index])
                char_prob.append(float(probs[t - 1, n_i, label_index]))
            prev = o
        text = "".join(out).replace("_", "").strip()
        kept = [p for c, p in zip(out, char_prob) if c != "_"]
        texts.append(text)
        char_confs.append(kept)
        confidences.append(float(np.prod(kept)) if kept else 0.0)

    return texts, confidences, char_confs


class CaptchaRecognizer:
    """Loads the ONNX model once and recognises captcha images."""

    def __init__(self, model_path: str | os.PathLike | None = None, threads: int | None = None):
        import onnxruntime as ort

        self.model_path = Path(model_path) if model_path else default_model_path()
        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found: {self.model_path}")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # The CPU memory arena grows with every distinct batch shape, which makes
        # batched inference hold a lot of memory for little gain at this size.
        options.enable_cpu_mem_arena = False
        if threads:
            options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def recognize_batch(self, images: list[np.ndarray]) -> list[dict]:
        """Recognise a list of preprocessed tensors of shape (1, 3, H, W)."""
        if not images:
            return []
        batch = np.concatenate(images, axis=0)
        logits = self._session.run(None, {self._input_name: batch})[0]
        texts, confs, char_confs = decode(logits)
        return [
            {"text": t, "confidence": round(c, 6), "char_confidences": [round(x, 6) for x in cc]}
            for t, c, cc in zip(texts, confs, char_confs)
        ]

    def recognize_bytes(self, data: bytes) -> dict:
        """Recognise a single image given its encoded bytes."""
        return self.recognize_batch([preprocess_bytes(data)])[0]

    def recognize_arrays(self, images: list[np.ndarray]) -> list[dict]:
        """Recognise a list of decoded image arrays (HxWx3, BGR or RGB)."""
        return self.recognize_batch([preprocess_array(im) for im in images])


_default: CaptchaRecognizer | None = None


def _get_default() -> CaptchaRecognizer:
    """Return a process-wide recogniser, building it on first use."""
    global _default
    if _default is None:
        _default = CaptchaRecognizer()
    return _default


def recognize(source: str | os.PathLike | bytes, detailed: bool = False):
    """Read a captcha image in one call.

    Args:
        source: path to an image file, or the image bytes themselves.
        detailed: return the full result dict instead of just the text.

    Returns:
        The recognised text, or the result dict when ``detailed`` is True.

    Example:
        >>> import captcha_whisperer
        >>> captcha_whisperer.recognize("captcha.png")
        '2CWK'
    """
    data = source if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    result = _get_default().recognize_bytes(bytes(data))
    return result if detailed else result["text"]
