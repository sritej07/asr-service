# Use a Python base image, choosing a version compatible with your environment
FROM python:3.11-slim

# Set environment variables for the application
ENV APP_HOME /app
ENV FLASK_APP app.py
ENV PORT 5050

# Install necessary system libraries (may be required by torchaudio for certain formats)
# This is usually optional for standard torch/torchaudio builds but good practice.
# If you run into issues with certain audio file types, you might need FFmpeg.
# For simplicity, we skip ffmpeg here unless needed, as torchaudio often bundles what it needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR $APP_HOME

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files (code and model artifacts)
COPY . .

# Expose the port the Flask app will use
EXPOSE $PORT

# Command to run the application using Gunicorn (a robust production WSGI server)
# Use 4 worker processes for concurrency, binding to the container port
CMD ["gunicorn", "--bind", "0.0.0.0:5050", "--workers", "4", "app:app"]