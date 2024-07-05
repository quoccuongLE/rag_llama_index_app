#!/bin/bash

python gradio_app.py --host localhost

python gradio_app.py \
    --config .vscode/configs/dev.yaml\
    --host localhost