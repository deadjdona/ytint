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
- **`ytint-ingest`**: Direct CLI entrypoint for `engine.youtube_api:main` (YouTube Data API v3 connector)
- **`ytint-ai`**: Direct CLI entrypoint for `engine.synthesizer:main` (LLM & Gemini Insight Synthesizer & RAG agent)
- **`ytint-graph`**: Direct CLI entrypoint for `engine.network_exporter:main` (Network Graph & Gephi Exporter)
- **`ytint-alert`**: Direct CLI entrypoint for `engine.alerting:main` (Automated Webhook Threat Alerting Daemon)
- **`ytint-search`**: Direct CLI entrypoint for `engine.semantic_search:main` (Neural Semantic Vector Search & Feedback Clustering)
- **`ytint-replay`**: Direct CLI entrypoint for `engine.event_replay:main` (Real-Time Chronological Event Replay & Crisis Simulation Engine)
- **`ytint-assist`**: Direct CLI entrypoint for `engine.assistant:main` (Creator Actionability & Engagement Optimization Assistant)
- **`ytint-sql`**: Direct CLI entrypoint for `engine.sql_engine:main` (Zero-Copy Analytical SQL Engine powered by DuckDB)
- **`ytint-narrative`**: Direct CLI entrypoint for `engine.narrative:main` (Narrative Scene Reaction & Timestamp Scrubbing Forensics)
- **`ytint-fingerprint`**: Direct CLI entrypoint for `engine.fingerprint:main` (Forensic Author Persona & Sockpuppet Fingerprinting)

Validate the active environment:

```powershell
.\.venv\Scripts\python.exe verify.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src tests verify.py
```

Do not commit `.venv/`, raw SQLite inputs, or generated interim files. The repository `.gitignore` already excludes them.
