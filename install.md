# ytint: Installation & Environment Setup

This project leverages **Python 3.14** and **CUDA 12.6** for high-performance ML data pipelines and interactive Streamlit analytics.

## Prerequisites
1. Ensure your NVIDIA GPU drivers are up to date.
2. Install Visual Studio Build Tools (C++ Desktop Development workload) to compile native C-extensions on Windows.

## Quickstart Setup via `uv`

We use `uv` for ultra-fast package management and dependency isolation.

```powershell
# 1. Install uv package manager
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Install Python 3.14 runtime
uv python install 3.14

# 3. Provision a clean virtual environment
uv venv .venv --python 3.14
.\.venv\Scripts\Activate.ps1

# 4. Install PyTorch with CUDA 12.6 execution paths
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 5. Install analytical stack and user interface
uv pip install transformers bertopic streamlit ruptures pyyaml pandas pyarrow
