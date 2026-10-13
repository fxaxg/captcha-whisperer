"""Command-line interface for the captcha recogniser.

Usage:
    captcha-whisperer image1.png image2.png ...
    captcha-whisperer --json image.png
    captcha-whisperer --serve --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="captcha-whisperer", description="Recognise captcha images."
    )
    parser.add_argument("images", nargs="*", help="image files to recognise")
    parser.add_argument("--json", action="store_true", help="print JSON instead of plain text")
    parser.add_argument("--model", help="path to the ONNX model (defaults to the bundled one)")
    parser.add_argument("--serve", action="store_true", help="start the HTTP server instead")
    parser.add_argument("--host", default="0.0.0.0", help="server host (with --serve)")
    parser.add_argument("--port", type=int, default=8000, help="server port (with --serve)")
    args = parser.parse_args(argv)

    if args.serve:
        import uvicorn

        uvicorn.run("captcha_whisperer.api:app", host=args.host, port=args.port)
        return 0

    if not args.images:
        parser.error("no images given (or use --serve)")

    from .recognizer import CaptchaRecognizer

    recognizer = CaptchaRecognizer(args.model)
    outcomes = []
    for path in args.images:
        p = Path(path)
        if not p.is_file():
            print(f"{path}: not a file", file=sys.stderr)
            continue
        result = recognizer.recognize_bytes(p.read_bytes())
        outcomes.append({"file": p.name, **result})
        if not args.json:
            print(f"{p.name}\t{result['text']}\t{result['confidence']:.4f}")

    if args.json:
        print(json.dumps(outcomes, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
