# filename: Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the root is in python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the server on HF default port
CMD ["python", "-m", "server.app"]