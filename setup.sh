#!/bin/bash
VENV_DIR=${1:-".venv/llama_index_app"}
if [ -f lock.conda.yaml ]; then
    conda env create -f lock.conda.yaml --prefix $VENV_DIR
else
    conda env create --solver=classic -f conda.yaml --prefix $VENV_DIR
fi
