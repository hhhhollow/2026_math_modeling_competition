#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

export JUPYTER_CONFIG_DIR="$PROJECT_DIR/.jupyter"
export JUPYTER_DATA_DIR="$PROJECT_DIR/.jupyter-data"
export JUPYTER_RUNTIME_DIR="$PROJECT_DIR/.jupyter-runtime"
export IPYTHONDIR="$PROJECT_DIR/.ipython"
export MPLCONFIGDIR="$PROJECT_DIR/.matplotlib"

source "$PROJECT_DIR/.venv/bin/activate"
"$PROJECT_DIR/.venv/bin/python" -m jupyterlab --notebook-dir="$PROJECT_DIR"
