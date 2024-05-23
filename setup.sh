#!/bin/bash
VENV_DIR=${1:-".venv/llama_index_app"}
conda env create --solver=classic -f conda.yaml --prefix $VENV_DIR
