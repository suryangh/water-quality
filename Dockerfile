# Build stage minimal
FROM python:3.12-alpine AS builder

# Install minimal build tools
RUN apk add --no-cache --virtual .build-deps \
    gcc musl-dev linux-headers g++ gfortran openblas-dev cmake make

# Install dependencies minimal dalam satu layer
RUN pip install --no-cache-dir --user \
    flask==3.1.1 \
    flask-cors==3.0.10 \
    numpy==2.0.2 \
    pandas==2.2.2 \
    joblib==1.5.1 \
    scikit-learn==1.6.1 \
    xgboost==2.1.4 && \
    find /root/.local -name "*.pyc" -delete && \
    find /root/.local -name "__pycache__" -exec rm -rf {} + || true && \
    find /root/.local -name "*.so" -exec strip {} + || true

# Cleanup build dependencies
RUN apk del .build-deps

# Runtime minimal
FROM python:3.12-alpine

# Install hanya runtime yang diperlukan
RUN apk add --no-cache libgomp libstdc++ openblas \
    && rm -rf /var/cache/apk/*

# Copy packages ke direktori yang bisa diakses user biasa
COPY --from=builder /root/.local /usr/local

WORKDIR /app

# Copy aplikasi backend dan frontend
COPY backend/app.py .
COPY backend/ml_model/ ./ml_model/
COPY frontend/ ./frontend/

# Minimal setup
RUN adduser -D appuser && \
    mkdir file_uploads && \
    chown -R appuser:appuser /app

# Create startup script untuk run backend dan frontend
RUN echo '#!/bin/sh' > start.sh && \
    echo 'cd /app/frontend && python serve_frontend.py &' >> start.sh && \
    echo 'cd /app && python app.py' >> start.sh && \
    chmod +x start.sh && \
    chown appuser:appuser start.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser
EXPOSE 5000 5500
CMD ["./start.sh"]
