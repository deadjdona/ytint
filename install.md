# ytint installation

## Requirements

- Python 3.14
- [`uv`](https://docs.astral.sh/uv/)
- NVIDIA drivers only when GPU acceleration is required

## Create the environment

```powershell
uv python install 3.14
uv venv .venv --python 3.14
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

The default requirements are platform-neutral. For NVIDIA CUDA 12.6 PyTorch wheels, install the matching PyTorch build **before** the remaining requirements:

```powershell
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt
```

Validate the active environment:

```powershell
.\.venv\Scripts\python.exe verify.py
.\.venv\Scripts\python.exe -m pytest -q
```

Do not commit `.venv/`, raw SQLite inputs, or generated interim files. The repository `.gitignore` already excludes them.
