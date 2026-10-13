"""CRNN captcha recogniser: a CNN + bidirectional LSTM + CTC model served via ONNX."""

from .constants import (
    BLANK_INDEX,
    INPUT_CHANNELS,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    LABELS,
    MODEL_FILENAME,
    NUM_CLASSES,
    TIME_STEPS,
    WEIGHTS_FILENAME,
)
from .recognizer import (
    CaptchaRecognizer,
    decode,
    default_model_path,
    recognize,
    weights_path,
)

__all__ = [
    "CaptchaRecognizer",
    "decode",
    "default_model_path",
    "recognize",
    "weights_path",
    "LABELS",
    "NUM_CLASSES",
    "BLANK_INDEX",
    "TIME_STEPS",
    "INPUT_CHANNELS",
    "INPUT_HEIGHT",
    "INPUT_WIDTH",
    "MODEL_FILENAME",
    "WEIGHTS_FILENAME",
]

__version__ = "0.1.0"
