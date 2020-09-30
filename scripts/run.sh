#!/bin/bash

ORIGINAL_HOME=$PWD
TMP_DIR=$(mktemp -d -p ../copies)
echo $TMP_DIR
find . \( -name "*.py" -o -name "*.json" -o -name "*.pyx" \) -exec cp --parents {} $TMP_DIR \;
cd $TMP_DIR

PYTHONIOENCODING=utf-8 python3 train.py --name eds_release --save_checkpoints --log_wandb --config "/home/samueld/semantic_parsing/config/base_eds.yaml"
