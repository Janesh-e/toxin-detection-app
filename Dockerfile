# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install numpy==1.24.4 --no-deps && pip install --no-deps --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy your source code and model folders
COPY app/ ./app/
COPY client/ ./client/
COPY model_ckpt/ ./model_ckpt/

# Expose port 8000
EXPOSE 8000

# Start FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
