# captcha-whisperer

[English](README.md) | [简体中文](README.zh-CN.md)

一行代码识别图片字符验证码。模型是 CRNN（DenseNet 卷积主干 + 双向 LSTM +
CTC），用于识别字母数字验证码，推理基于 ONNX Runtime。运行时不需要 PyTorch，
也不需要 OpenCV，所以安装体积很小。

## 快速开始

```bash
pip install captcha-whisperer
```

```python
import captcha_whisperer

print(captcha_whisperer.recognize("captcha.png"))   # -> '2CWK'
```

就这么多。模型随包发布，首次使用不会联网下载，离线可用。

## 命令行

```bash
captcha-whisperer image.png              # 输出 文本<TAB>置信度
captcha-whisperer --json a.png b.png     # 输出 JSON，含逐字符置信度
captcha-whisperer --serve --port 8000    # 启动 HTTP 服务
```

`--serve` 需要额外安装服务端依赖：

```bash
pip install "captcha-whisperer[server]"
```

## Python 调用

### 一行版

```python
import captcha_whisperer

captcha_whisperer.recognize("captcha.png")            # -> '2CWK'
captcha_whisperer.recognize(image_bytes)              # 也可以直接传字节
captcha_whisperer.recognize("captcha.png", detailed=True)
# -> {'text': '2CWK', 'confidence': 0.915, 'char_confidences': [0.99, 0.94, 0.88, 0.97]}
```

### 复用同一个识别器

一行版第一次调用时会创建识别器，之后一直复用。如果你想自己控制模型路径或线程
数，直接用类：

```python
from captcha_whisperer import CaptchaRecognizer

recognizer = CaptchaRecognizer()                  # 或 CaptchaRecognizer("my.onnx", threads=4)
print(recognizer.recognize_bytes(open("captcha.png", "rb").read()))
# {'text': '2CWK', 'confidence': 0.915, 'char_confidences': [...]}
```

可以通过环境变量 `CAPTCHA_OCR_MODEL` 或者 `model_path` 参数指定自己导出的模型。

## 与其他方案的对比

| | captcha-whisperer | ddddocr |
| --- | --- | --- |
| wheel 体积 | 约 4 MB | 约 72 MB |
| 运行期依赖 | numpy、pillow、onnxruntime | numpy、pillow、onnxruntime、opencv |
| 是否需要 OpenCV | 不需要 | 需要（约 42 MB） |
| 变长输出 | 支持，CTC 解码 | 支持 |
| 逐字符置信度 | 提供 | 未暴露 |
| 内置 HTTP 服务 | 支持，可选安装 | 支持，可选安装 |
| 滑块 / 点选验证码 | 不支持，仅字符型 | 支持 |

这个对比不是在说谁更好。`ddddocr` 覆盖的验证码类型多得多，也经过了多年打磨。
`captcha-whisperer` 是一个很窄的工具：只读字符型验证码，并且尽量用最少的安装
体积把这件事做好。

## 模型

| 项目 | 值 |
| --- | --- |
| 输入 | 3 x 60 x 160，float，取值 0..255，通道顺序为 BGR |
| 输出 | 20 x N x 37 logits，用 CTC 贪心解码 |
| 字符集 | `0-9`、`A-Z` 以及 `_`（blank 符号） |
| 长度 | 变长，最多 19 个字符 |
| 参数量 | 约 480 万 |
| 准确率 | 在 195 张带标注样本上，整串 94.9% / 字符级 97.4% |
| 延迟 | CPU + ONNX Runtime，单张约 3 ms，批量约 1.6 ms/张 |

这里的准确率来自原来训练工作带的一小份标注样本。把它当作参考值，不要当作基准：
你自己的验证码风格可能差别很大，应该在自建数据上自行测量。

## HTTP 服务

```bash
pip install "captcha-whisperer[server]"
captcha-whisperer --serve --host 0.0.0.0 --port 8000
```

也可以直接用 uvicorn 启动：

```bash
uvicorn captcha_whisperer.api:app --host 0.0.0.0 --port 8000
```

`POST /ocr` 接收一个或多个图片文件（字段名 `files`），返回结果顺序与上传顺序一致。

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

交互式文档在 `/docs`。

## 常见问题

**支持哪些验证码？**
图片字符型验证码，也就是那种显示扭曲的字母数字、让你照着输入的。不支持滑块
验证码、点选验证码，也不支持 reCAPTCHA 那类挑战。

**能识别中文吗？**
不能。字符集只有 `0-9` 和 `A-Z`，其他字符都超出了模型范围。

**能用我自己的模型吗？**
可以。`CaptchaRecognizer("path/to/model.onnx")`，或者设置环境变量
`CAPTCHA_OCR_MODEL`。如果你的字符集不同，需要重新训练并重新导出（见下），
只替换文件是不够的。

**在我自己的验证码上准确率不高怎么办？**
自带模型是在某一种特定风格上训练的。验证码在字体、扭曲方式、噪点、颜色上差别
很大，用别处训练的模型效果不好是常态。现实的做法是拿几百张你自己验证码的标注
样本做微调。

**为什么不需要 OpenCV？**
图像解码和缩放用 Pillow 就够了，它本来就是必需依赖。OpenCV 会多出约 42 MB，
而它提供的操作这个包都用不到。

**怎么只初始化一次然后反复用？**
用 `CaptchaRecognizer()` 并把实例保存下来。一行版内部已经通过模块级单例做到了
这一点，所以反复调用 `recognize()` 不会重复加载模型。

## 再训练与重新导出

完整的网络定义在 `src/captcha_whisperer/model.py`，可以加载随包的
`models/captcha_crnn_weights.npz` 权重，在新数据上微调后重新生成 ONNX。

19 MB 的权重文件不在安装出来的 wheel 里，因为只有再训练才需要它。它保存在仓库
和源码包中，需要重训时从这两处取。

```bash
git clone https://github.com/TODO/captcha-whisperer
cd captcha-whisperer
pip install ".[train]"

python -m captcha_whisperer.export_onnx \
  --weights src/captcha_whisperer/models/captcha_crnn_weights.npz \
  --out src/captcha_whisperer/models/captcha_crnn.onnx
```

如果权重文件在别处，用 `--weights` 指向那个路径即可。

## Docker

```bash
docker build -t captcha-whisperer .
docker run -p 8000:8000 captcha-whisperer
```

## 目录结构

```
captcha-whisperer/
├── src/captcha_whisperer/
│   ├── __init__.py
│   ├── api.py            # FastAPI 应用
│   ├── cli.py            # 命令行入口
│   ├── constants.py      # 字符集与尺寸常量
│   ├── model.py          # PyTorch 网络定义 + ONNX 导出
│   ├── export_onnx.py    # 重新导出脚本
│   ├── preprocess.py     # 图像解码与归一化
│   ├── recognizer.py     # ONNX Runtime 推理
│   └── models/           # 随包的 ONNX 模型（权重仅 sdist）
├── tests/
├── Dockerfile
└── pyproject.toml
```

## 测试

```bash
pip install -e ".[dev]"
pytest
```

## 许可证

MIT，详见 [LICENSE](LICENSE)。
