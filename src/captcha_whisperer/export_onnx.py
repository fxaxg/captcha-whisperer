"""Rebuild the bundled ONNX model from the weight file.

Usage:
    python -m captcha_whisperer.export_onnx
    python -m captcha_whisperer.export_onnx --weights base.npz --out captcha_crnn.onnx

Requires the ``torch`` and ``onnx`` extras:  pip install -e ".[train]"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .constants import MODEL_FILENAME, WEIGHTS_FILENAME
from .recognizer import default_model_path


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Export the captcha CRNN to ONNX.")
    parser.add_argument("--weights", default=str(here / "models" / WEIGHTS_FILENAME))
    parser.add_argument("--out", default=str(here / "models" / MODEL_FILENAME))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    from .model import export_onnx  # imported lazily so torch is optional

    export_onnx(args.weights, args.out, opset=args.opset)
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
