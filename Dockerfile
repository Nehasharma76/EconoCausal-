# EconoCausal API image.
# Build:  docker build -t econocausal-api .
# Run:    docker run -p 8000:8000 econocausal-api     then open http://localhost:8000/docs
FROM python:3.11-slim

WORKDIR /app
COPY backend/api/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY backend/api backend/api
COPY backend/optimize backend/optimize
COPY backend/monitoring backend/monitoring

EXPOSE 8000
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
