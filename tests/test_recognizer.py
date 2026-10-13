"""Smoke tests for the recogniser and the HTTP API."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from captcha_whisperer import CaptchaRecognizer, recognize


def _make_image(text: str) -> bytes:
    img = Image.new("RGB", (150, 60), (245, 242, 238))
    draw = ImageDraw.Draw(img)
    font = None
    for path in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(path, 40)
            break
        except OSError:
            continue
    draw.text((6, 6), text, fill=(15, 15, 25), font=font)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def recognizer() -> CaptchaRecognizer:
    return CaptchaRecognizer()


def test_recognises_clean_text(recognizer: CaptchaRecognizer) -> None:
    result = recognizer.recognize_bytes(_make_image("7B32"))
    assert result["text"] == "7B32"
    assert result["confidence"] > 0.5
    assert len(result["char_confidences"]) == 4


def test_batch_order(recognizer: CaptchaRecognizer) -> None:
    from captcha_whisperer.preprocess import preprocess_bytes

    images = [preprocess_bytes(_make_image(t)) for t in ("129E", "9C1D")]
    results = recognizer.recognize_batch(images)
    assert [r["text"] for r in results] == ["129E", "9C1D"]


def test_api_single_image() -> None:
    from fastapi.testclient import TestClient

    from captcha_whisperer.api import app

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        resp = client.post("/ocr", files={"files": ("a.png", _make_image("B6XK"), "image/png")})
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"][0]["text"] == "B6XK"


def test_api_multiple_images() -> None:
    from fastapi.testclient import TestClient

    from captcha_whisperer.api import app

    files = [
        ("files", ("a.png", _make_image("129E"), "image/png")),
        ("files", ("b.png", _make_image("7B32"), "image/png")),
    ]
    with TestClient(app) as client:
        resp = client.post("/ocr", files=files)
        assert resp.status_code == 200
        assert [r["text"] for r in resp.json()["results"]] == ["129E", "7B32"]


def test_module_level_recognize(tmp_path) -> None:
    """The one-liner accepts both a file path and raw bytes."""
    path = tmp_path / "captcha.png"
    path.write_bytes(_make_image("Q4M8"))

    assert recognize(path) == "Q4M8"
    assert recognize(_make_image("Q4M8")) == "Q4M8"

    detailed = recognize(path, detailed=True)
    assert detailed["text"] == "Q4M8"
    assert 0.0 < detailed["confidence"] <= 1.0
    assert len(detailed["char_confidences"]) == 4
