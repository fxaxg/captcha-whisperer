"""FastAPI service exposing the captcha recogniser over HTTP.

Run:
    uvicorn captcha_whisperer.api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health     liveness probe
    POST /ocr        one or more images -> recognised text and confidence
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .recognizer import CaptchaRecognizer
from .preprocess import preprocess_bytes

_state: dict[str, CaptchaRecognizer] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["recognizer"] = CaptchaRecognizer()
    yield
    _state.clear()


app = FastAPI(
    title="CRNN Captcha OCR",
    description="Recognise captcha images with a CNN + BiLSTM + CTC model.",
    version="0.1.0",
    lifespan=lifespan,
)


class ImageResult(BaseModel):
    text: str
    confidence: float
    char_confidences: list[float]


class OCRResponse(BaseModel):
    results: list[ImageResult]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ocr", response_model=OCRResponse)
async def ocr(files: list[UploadFile] = File(..., description="One or more image files")) -> OCRResponse:
    """Recognise the uploaded image(s). Order of results matches the upload order."""
    recognizer = _state.get("recognizer")
    if recognizer is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    tensors = []
    for f in files:
        data = await f.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"empty file: {f.filename}")
        try:
            tensors.append(preprocess_bytes(data))
        except Exception:
            raise HTTPException(status_code=400, detail=f"cannot decode image: {f.filename}")

    results = recognizer.recognize_batch(tensors)
    return OCRResponse(results=results)


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse({"service": "captcha-whisperer", "docs": "/docs", "ocr": "POST /ocr"})
