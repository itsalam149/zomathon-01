FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Assume data is already generated or run it as part of entrypoint
# In a real scenario, data generation would be separate
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
