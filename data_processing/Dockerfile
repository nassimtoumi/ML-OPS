# Use the latest Ubuntu base image
FROM ubuntu:latest

# Update and install Python, pip, and ffmpeg
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a directory for the virtual environment
WORKDIR /app
RUN python3 -m venv venv

# Install required Python dependencies within the virtual environment
COPY requirements.txt /app/
RUN . /app/venv/bin/activate && pip install --no-cache-dir -r requirements.txt


# Copy application code and set the working directory
COPY . /app

# Activate the virtual environment and set the command to run the Python script
CMD ["/app/venv/bin/python", "data_processing.py"]
