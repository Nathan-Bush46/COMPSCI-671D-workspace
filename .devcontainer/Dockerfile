# Start from NVIDIA’s CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip software-properties-common build-essential wget && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --no-cache-dir --upgrade pip && \
    apt-get clean

# # Install dependencies for building Python and CUDA
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     software-properties-common build-essential wget \
#     zlib1g-dev libssl-dev libncurses5-dev libbz2-dev libreadline-dev \
#     libsqlite3-dev libffi-dev liblzma-dev && \
#     rm -rf /var/lib/apt/lists/*

# Install PyTorch and Transformers
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install transformers

# # Download and install Python 3.12 from source
# RUN wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz && \
#     tar -xzf Python-3.12.0.tgz && \
#     cd Python-3.12.0 && \
#     ./configure --enable-optimizations && \
#     make -j$(nproc) && \
#     make altinstall && \
#     cd .. && rm -rf Python-3.12.0 Python-3.12.0.tgz

# # Set Python 3.12 as default, removing existing link if present
# RUN if [ -L /usr/bin/python3 ]; then rm /usr/bin/python3; fi && \
#     ln -s /usr/local/bin/python3.12 /usr/bin/python3

# # Install pip manually for Python 3.12
# RUN wget https://bootstrap.pypa.io/get-pip.py && \
#     python3 get-pip.py && \
#     rm get-pip.py



# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Copy the source code
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH=src/main_workspace
