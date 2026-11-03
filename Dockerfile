# Lab 9 - Bank Marketing prediction API image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Only what the service needs: code, config and the exported inference pipeline
COPY app/ app/
COPY configs/config.yaml configs/config.yaml
COPY models/inference_pipeline.pkl models/model_info.json models/

RUN useradd --create-home appuser && mkdir -p logs && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
