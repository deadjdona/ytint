# ytint installation

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- NVIDIA drivers only when GPU acceleration is required

## Create the environment

```powershell
uv python install 3.11
uv venv .venv --python 3.11
.\.venv\Scripts\Activate.ps1

# Install dependencies (or install in editable mode to register CLI commands)
uv pip install -e .
# Alternatively: uv pip install -r requirements.txt
```

The default requirements are platform-neutral. For NVIDIA CUDA 12.6 PyTorch wheels, install the matching PyTorch build **before** the remaining requirements:

```powershell
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install -e .
```

Installing in editable mode (`uv pip install -e .`) registers the package console scripts:
- **`ytint-runner`**: Direct CLI entrypoint for `pipeline.runner:main`
- **`ytint-app`**: Direct CLI entrypoint for `ui.app:main`
- **`ytint-report`**: Direct CLI entrypoint for `engine.reporter:main` (Executive Dossier generator)

Validate the active environment:

```powershell
.\.venv\Scripts\python.exe verify.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src tests verify.py
```

Do not commit `.venv/`, raw SQLite inputs, or generated interim files. The repository `.gitignore` already excludes them.
