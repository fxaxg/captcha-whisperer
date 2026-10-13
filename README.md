# captcha-whisperer

[English](README.md) | [简体中文](README.zh-CN.md)

Read image-based character captchas in one line. The model is a CRNN
(a DenseNet convolutional trunk + bidirectional LSTM + CTC) that reads
alphanumeric captchas, and inference runs on ONNX Runtime. No PyTorch and no
OpenCV are needed at run time, so the install stays small.

## Quick start

```bash
pip install captcha-whisperer
```

```python
import captcha_whisperer

print(captcha_whisperer.recognize("captcha.png"))   # -> '2CWK'
```

That is the whole thing. The model ships inside the package, so nothing is
downloaded on first use and it works offline.

## Command line

```bash
captcha-whisperer image.png              # prints  text<TAB>confidence
captcha-whisperer --json a.png b.png     # JSON with per-character confidences
captcha-whisperer --serve --port 8000    # start the HTTP server
```

The `--serve` flag needs the extra server dependencies:

```bash
pip install "captcha-whisperer[server]"
```

## Python API

### One-liner

```python
import captcha_whisperer

captcha_whisperer.recognize("captcha.png")            # -> '2CWK'
captcha_whisperer.recognize(image_bytes)              # bytes work too
captcha_whisperer.recognize("captcha.png", detailed=True)
# -> {'text': '2CWK', 'confidence': 0.915, 'char_confidences': [0.99, 0.94, 0.88, 0.97]}
```

### Reusing one recogniser

The one-liner builds a recogniser on first call and reuses it afterwards. If you
want explicit control over the model path or the thread count, construct the
class yourself:

```python
from captcha_whisperer import CaptchaRecognizer

recognizer = CaptchaRecognizer()                  # or CaptchaRecognizer("my.onnx", threads=4)
print(recognizer.recognize_bytes(open("captcha.png", "rb").read()))
# {'text': '2CWK', 'confidence': 0.915, 'char_confidences': [...]}
```

Point `CAPTCHA_OCR_MODEL` at another file, or pass `model_path`, to use your own
exported model.

## How it compares

| | captcha-whisperer | ddddocr |
| --- | --- | --- |
| Wheel size | ~4 MB | ~72 MB |
| Runtime dependencies | numpy, pillow, onnxruntime | numpy, pillow, onnxruntime, opencv |
| OpenCV required | no | yes (~42 MB) |
| Variable-length output | yes, decoded by CTC | yes |
| Per-character confidence | yes | not exposed |
| Built-in HTTP service | yes, opt-in extra | yes, opt-in extra |
| Handles slider/click captchas | no, characters only | yes |

The comparison is not a claim that one is better. `ddddocr` covers far more
captcha types and has years of tuning behind it. `captcha-whisperer` is a narrow
tool: it reads character captchas, and it tries to do that with as little
installed weight as possible.

## Model

| Property | Value |
| --- | --- |
| Input | 3 x 60 x 160, float, values in 0..255, BGR channel order |
| Output | 20 x N x 37 logits, decoded greedily with CTC |
| Labels | `0-9`, `A-Z` and `_` (the blank symbol) |
| Length | variable, up to 19 characters |
| Parameters | ~4.8 M |
| Accuracy | 94.9% full-string / 97.4% per-character on a 195-image labelled set |
| Latency | ~3 ms per image single, ~1.6 ms per image batched (CPU, ONNX Runtime) |

The reported accuracy comes from a small labelled set that ships with the
original training work. Treat it as an indication, not a benchmark: your own
captchas may look very different, and you should measure on your own data.

## HTTP service

```bash
pip install "captcha-whisperer[server]"
captcha-whisperer --serve --host 0.0.0.0 --port 8000
```

Or run it through uvicorn directly:

```bash
uvicorn captcha_whisperer.api:app --host 0.0.0.0 --port 8000
```

`POST /ocr` takes one or more image files under the `files` field and returns
results in the same order.

```bash
curl -X POST http://localhost:8000/ocr \
  -F "files=@a.png" \
  -F "files=@b.png"
```

```json
{
  "results": [
    {"text": "2CWK", "confidence": 0.915, "char_confidences": [0.99, 0.94, 0.88, 0.97]},
    {"text": "129E", "confidence": 0.997, "char_confidences": [0.99, 0.99, 0.99, 0.99]}
  ]
}
```

Interactive docs are at `/docs`.

## FAQ

**What kind of captchas does this handle?**
Image-based character captchas: the kind that show distorted letters and digits
and ask you to type them. It does not handle slider captchas, click-the-object
captchas, or reCAPTCHA-style challenges.

**Does it read Chinese characters?**
No. The label set is `0-9` and `A-Z`. Anything else is outside the model.

**Can I use my own model?**
Yes. `CaptchaRecognizer("path/to/model.onnx")`, or set the `CAPTCHA_OCR_MODEL`
environment variable. If your character set differs, retrain and re-export
(see below) rather than only swapping the file.

**Accuracy is poor on my captchas. What can I do?**
The bundled model was trained on one particular style. Captchas vary a lot in
font, distortion, noise and colour, so a model trained elsewhere often
underperforms. The realistic fix is to fine-tune on a few hundred labelled
examples of your own captchas.

**Why is there no OpenCV dependency?**
Image decoding and resizing are done with Pillow, which is already needed.
OpenCV would add roughly 42 MB for operations this package does not perform.

**How do I initialise once and reuse it?**
Use `CaptchaRecognizer()` and keep the instance. The one-liner does this for you
through a module-level singleton, so repeated `recognize()` calls do not reload
the model.

## Retraining and re-export

The full network definition is in `src/captcha_whisperer/model.py`. It loads the
bundled weights from `models/captcha_crnn_weights.npz`, so you can fine-tune on
new data and rebuild the ONNX file.

The 19 MB weight file is not part of the installed wheel, since only retraining
needs it. It lives in the repository and in the source distribution; fetch it
from either to retrain.

```bash
git clone https://github.com/TODO/captcha-whisperer
cd captcha-whisperer
pip install ".[train]"

python -m captcha_whisperer.export_onnx \
  --weights src/captcha_whisperer/models/captcha_crnn_weights.npz \
  --out src/captcha_whisperer/models/captcha_crnn.onnx
```

If you already have the weights somewhere else, pass `--weights` that path
instead.

## Docker

```bash
docker build -t captcha-whisperer .
docker run -p 8000:8000 captcha-whisperer
```

## Project layout

```
captcha-whisperer/
├── src/captcha_whisperer/
│   ├── __init__.py
│   ├── api.py            # FastAPI app
│   ├── cli.py            # command-line entry point
│   ├── constants.py      # labels and geometry
│   ├── model.py          # PyTorch definition + ONNX export
│   ├── export_onnx.py    # re-export helper
│   ├── preprocess.py     # image decoding and normalisation
│   ├── recognizer.py     # ONNX Runtime inference
│   └── models/           # bundled ONNX model (weights: sdist only)
├── tests/
├── Dockerfile
└── pyproject.toml
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
