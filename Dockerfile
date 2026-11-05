FROM python:3.11-slim

# Prevent Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Create a non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1000 appuser

WORKDIR /app

# Install system dependencies (build-essential for C-extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies as root, then switch to appuser
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files and set ownership
COPY . .
RUN chown -R appuser:appgroup /app

# Switch to the non-root user
USER appuser

# Set default command
CMD ["python", "main.py"]
