#!/bin/bash
VENV_DIR=${1:-".venv/llama_index_app"}
if [ -f lock.conda.yaml ]; then
    conda env create -f lock.conda.yaml --prefix $VENV_DIR
else
    conda env create -f conda.yaml --prefix $VENV_DIR
fi

# python -m pip install fire \
#     llama-index==0.10.50 \
#     langchain \
#     gradio \
#     sentence_transformers \
#     InstructorEmbedding \
#     spacy \
#     llama-index-llms-ollama \
#     llama-index-embeddings-ollama \
#     llama-index-vector-stores-milvus \
#     sentencepiece \
#     git+https://github.com/FlagOpen/FlagEmbedding.git \
#     ollama \
#     flash_attn pymupdf4llm

# python -m pip install surya-ocr==0.4.15 marker-pdf --no-deps