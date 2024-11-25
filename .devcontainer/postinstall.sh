# These commands are run after the docker image is built.
pip3 install attrs packaging editables pandas scikit-learn sortedcontainers gmpy2 matplotlib
#pip3 install gosdt
pip3 install -r requirements.txt
pip install -q torch-geometric
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade transformers
