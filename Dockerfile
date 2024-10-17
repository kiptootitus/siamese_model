# Use the official TensorFlow Docker image as the base image
FROM tensorflow/tensorflow:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    python3-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables to optimize Python behavior
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Optionally set TensorFlow logging environment variable
ENV TF_CPP_MIN_LOG_LEVEL=2  

# Set the working directory
WORKDIR /app

# Copy only requirements.txt first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies with increased timeout and retry attempts
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --default-timeout=100 --retries 5 -r requirements.txt --no-cache-dir

# Install the kaggle library if not included in requirements.txt

# Create .kaggle directory with proper permissions
RUN mkdir -p /root/.kaggle && chmod 700 /root/.kaggle

# Copy the kaggle.json file to the appropriate directory
COPY kaggle/kaggle.json /root/.kaggle/kaggle.json

# Set permissions for kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json

# Copy the rest of the application code
COPY . .

# Set the entrypoint to run the script
CMD ["python3", "siamese_model.py"]
