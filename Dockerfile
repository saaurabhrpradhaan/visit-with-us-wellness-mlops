# Base image with Python 3.11
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy all project files into the container
COPY . /app

# Install Python dependencies for training + app
RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r hf_deploy/requirements.txt && \
    pip install datasets skops

# Streamlit configuration
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_HEADLESS=true

# Expose the port used by Streamlit
EXPOSE 7860

# Default command: run the Streamlit app
CMD ["streamlit", "run", "app/app_streamlit.py", "--server.port=7860", "--server.address=0.0.0.0"]
