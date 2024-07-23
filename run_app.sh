#!/bin/bash

python gradio_app.py --host localhost

python gradio_app.py \
    --config .vscode/configs/dev_llama_parse.yaml \
    --host localhost

python gradio_app.py \
    --config .vscode/configs/dev_llama_parse.yaml --share


python gradio_app.py \
    --config .vscode/configs/dev.yaml --share
