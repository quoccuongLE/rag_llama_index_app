#!/bin/bash

# sudo systemctl daemon-reload
# sudo systemctl enable ollama
# sudo systemctl disable ollama
# ollama serve

sudo systemctl stop ollama
sudo docker run -d --rm --gpus all \
    -v /usr/share/ollama/.ollama:/root/.ollama \
    -p 11434:11434 \
    ollama/ollama:0.3.3


sudo docker run -d --rm --gpus all \
    -v /usr/share/ollama/.ollama:/root/.ollama \
    -p 11435:11434 \
    ollama/ollama:0.3.3
