# Production-ready Docker image for the MRI QA Flask app.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system packages only when needed. Keep image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to improve layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code.
COPY . .

# Create runtime folders.
RUN mkdir -p /app/instance/uploads

EXPOSE 5000

# Gunicorn is used for production instead of Flask development server.
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "run:app"]
