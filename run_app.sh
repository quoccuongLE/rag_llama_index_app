#!/bin/bash

python app.py --host localhost

python app.py \
    --config .vscode/configs/dev_llama_parse.yaml \
    --host localhost

python app.py \
    --config .vscode/configs/dev_llama_parse.yaml --share


python app.py \
    --config .vscode/configs/dev.yaml --share
