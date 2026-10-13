FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# The container runs the HTTP service, so the server extra is required here.
RUN pip install --no-cache-dir ".[server]"

EXPOSE 8000

CMD ["uvicorn", "captcha_whisperer.api:app", "--host", "0.0.0.0", "--port", "8000"]
