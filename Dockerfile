# Multi-stage build for optimized production image
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY notebooks/model_training_whale_weighted.ipynb ./notebooks/model_training_whale_weighted.ipynb
COPY scripts/train_if_needed.sh ./scripts/train_if_needed.sh

# Make sure scripts are executable
RUN chmod +x ./scripts/train_if_needed.sh
ENV PATH=/root/.local/bin:$PATH

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=src.api.app

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5001/health')"

# Run the training script which will start the API
CMD ["./scripts/train_if_needed.sh"]
