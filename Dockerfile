FROM python:3.11-slim

WORKDIR /app

# Install system deps first (good habit for pandas/numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project code
COPY . .

# Default command: run the analysis
CMD ["python", "app.py"]
