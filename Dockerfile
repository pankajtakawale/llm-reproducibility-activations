# Use NVIDIA's official PyTorch container with CUDA support for ARM64
# Using latest version for better GPU compatibility
FROM nvcr.io/nvidia/pytorch:24.11-py3

# Set working directory
WORKDIR /workspace

# Copy requirements and install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set Python to be unbuffered for better logging
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["/bin/bash"]
