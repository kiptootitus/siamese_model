# Use the official TensorFlow Docker image as the base image
FROM tensorflow/tensorflow:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to optimize Python behavior
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy only requirements.txt first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies with increased timeout and retry attempts
RUN pip install --upgrade pip && \
    pip install --default-timeout=100 --retries 5 -r requirements.txt

# Create .kaggle directory with proper permissions
RUN mkdir -p /root/.kaggle && chmod 700 /root/.kaggle

# Copy the rest of the application code
COPY . .

# giving bash script all the necessary permissions
RUN chmod +x entrypoint.sh

# Set the entrypoint to run the script
ENTRYPOINT ["./entrypoint.sh"]
