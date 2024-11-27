# Start from NVIDIA’s CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip software-properties-common build-essential wget && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --no-cache-dir --upgrade pip && \
    apt-get clean

# Install PyTorch and Transformers
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install transformers


# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Copy the source code
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH=src/main_workspace
