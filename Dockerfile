FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml_service.py .
# Убедитесь, что копируете всю папку с моделями и проверяете ее содержимое
COPY ml_models/ ./ml_models/

# DEBUG: Проверяем, скопировались ли файлы
RUN ls -la /app/ml_models/ && \
    echo "=== Files in ml_models ===" && \
    ls -la /app/ml_models/*.pkl 2>/dev/null || echo "No .pkl files found" && \
    ls -la /app/ml_models/*.json 2>/dev/null || echo "No .json files found" && \
    ls -la /app/ml_models/*.csv 2>/dev/null || echo "No .csv files found"

RUN useradd -m -u 1000 mluser && chown -R mluser:mluser /app
USER mluser

EXPOSE 5002

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5002/health || exit 1

CMD ["python", "-u", "ml_service.py"]