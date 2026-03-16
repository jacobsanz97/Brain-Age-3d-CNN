FROM python:3.9-slim-buster

WORKDIR /app

# Install system dependencies for nibabel and other libraries if needed
# For example, some NIfTI files might require specific libraries for I/O operations.
# This might not be strictly necessary for basic nibabel usage but good practice for robustness.
RUN apt-get update && apt-get install -y --no-install-recommends 
    build-essential 
    git 
    libopenblas-dev 
    libglib2.0-0 
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Using specific versions can help with reproducibility
COPY requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and trained model
COPY cnn.py ./cnn.py
COPY predict.py ./predict.py
COPY training_logs/ ./training_logs/

# Set the entrypoint to the prediction script
# This allows passing the NIfTI file as an argument directly
ENTRYPOINT ["python", "predict.py"]
