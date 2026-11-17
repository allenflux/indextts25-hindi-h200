# Source after setting HI_H200_WORK to your dedicated data-volume directory.
# Never download/install/train automatically.
: "${HI_H200_WORK:?Set HI_H200_WORK to an isolated directory on your data mount}"
export TMPDIR="$HI_H200_WORK/.runtime/tmp"
export XDG_CACHE_HOME="$HI_H200_WORK/.runtime/cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export TIKTOKEN_CACHE_DIR="$XDG_CACHE_HOME/tiktoken"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export WANDB_DIR="$HI_H200_WORK/.runtime/wandb"
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
