# QA Semantic Document Search powered by LLM

Unleash potential utilities of LLM models with Llama Index

# Getting started

## Install python virtual Environment via Conda

Requirement:

- Conda or Miniconda

To install conda environment, please run the script ``./setup.sh``

The detail of required python packages can be de seen in ``conda.yaml`` and ``lock.conda.yaml`` for all installed packages.

## Launch Ollama server

Run LLM model Docker server via the script ``./run_ollama.sh``

## Launch Gradio app

Run the following command

```bash
conda activate .venv/llama_index_app/
.venv/llama_index_app/bin/python gradio_app.py
```

The gradio app is accessible at https://127.0.0.1:7860 or https://localhost:7860

----------------------
**Under developement**

...

...

----------------------

# Evaluation

# Acknowledgement

This project is inspired by the work https://github.com/datvodinh/rag-chatbot/. I am grateful for your awesome project!
