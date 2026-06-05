# Installation & Setup Guide: ytint Pipeline

This document guides you through setting up a local, GPU-accelerated environment to run the `ytint` YouTube comment analysis ecosystem on Windows. This project is optimized for processing Russian-language comment datasets using local neural networks running on NVIDIA hardware (tested on RTX 40-series).

---

## 📋 Prerequisites

Before starting, ensure your system meets the following requirements:
* **Operating System:** Windows 10/11
* **Python Version:** Python 3.12 (Strictly required. Do not use Python 3.13 or 3.14, as pre-compiled PyTorch GPU binaries are not yet available for them).
* **Hardware:** NVIDIA Dedicated Graphics Card (e.g., RTX 4070 Laptop GPU).
* **Drivers:** NVIDIA Game Ready or Studio Drivers installed (verify by running `nvidia-smi` in your terminal; your supported CUDA version should be `12.1` or higher).

---

## 🛠️ Step 1: Environment Isolation

To avoid package conflicts with your system-level Python distribution (especially the NumPy 2.x version trap), initialize a completely isolated virtual environment.

Open PowerShell or Command Prompt inside the project root directory (`C:\Users\deadj\Sources\ytint`) and run:

```bash
# 1. Clear out any broken or old environment configurations
deactivate 2>nul
rmdir /s /q .venv 2>nul

# 2. Provision a clean workspace explicitly targeting Python 3.12
py -3.12 -m venv .venv

# 3. Activate the new workspace environment
.\.venv\Scripts\activate

# 🗂️ Step 4: Data Ingestion Configuration

1. Place your exported CommentSuite SQLite database inside the raw storage directory:
   `data/raw/`
2. Ensure the file is named exactly **`commentsuite.sqlite3`**. 
3. Open `config/settings.yaml` and confirm your operational config reflects the target model maps:

```yaml
paths:
  raw_dir: "data/raw"
  raw_db: "data/raw/commentsuite.sqlite3"
...