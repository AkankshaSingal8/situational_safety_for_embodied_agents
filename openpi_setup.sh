conda create -n aegis_env python=3.11 -y
conda activate aegis_env

# Core PyTorch (cu12 stack — PSC Bridges-2 uses CUDA 12.x on GPU nodes)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu124

# JAX with CUDA 12
pip install "jax[cuda12]"==0.5.3

# Install the repo's editable packages first
pip install -e ./openpi
pip install -e ./openpi/packages/openpi-client

# Install the bulk of requirements (excluding the -e editable lines and jax/torch already handled)
pip install \
  flax==0.10.2 optax==0.2.4 orbax-checkpoint==0.11.13 chex==0.1.89 \
  equinox==0.12.2 augmax==0.4.1 \
  diffusers==0.33.1 transformers==4.53.2 safetensors==0.5.3 \
  tokenizers==0.21.1 sentencepiece==0.2.0 \
  opencv-python-headless==4.11.0.86 pillow==11.2.1 \
  numpy==1.26.4 scipy==1.15.3 einops==0.8.1 \
  wandb==0.19.11 rerun-sdk==0.23.1 \
  lerobot@git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 \
  draccus==0.10.0 tyro==0.9.22 omegaconf==2.3.0 \
  flask==3.1.1 websockets==15.0.1 \
  huggingface-hub==0.32.3 datasets==3.6.0 \
  av==14.4.0 imageio==2.37.0 imageio-ffmpeg==0.6.0 \
  dm-control==1.0.14 mujoco==2.3.7 \
  zarr==3.0.8 h5py==3.13.0 \
  rich==14.0.0 tqdm==4.67.1 \
  pydantic==2.11.5 jaxtyping==0.2.36 beartype==0.19.0

# Verify JAX sees GPU
python -c "import jax; print(jax.devices())"