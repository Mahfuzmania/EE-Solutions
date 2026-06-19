FROM python:3.11-slim

# Install system dependencies for OCR and PDF processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-ben \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY app/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app
COPY app/ ./app/

# Create data directories
RUN mkdir -p app/data/pdfs app/data/index app/data/chunks app/data/cache

# Expose port
EXPOSE 8000

# Run the application as a Python package so relative imports work
CMD ["python", "-m", "app.backend.main"]