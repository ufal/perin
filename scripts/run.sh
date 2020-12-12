#!/bin/bash

source bin/activate
nvidia-smi

ORIGINAL_HOME=$PWD
TMP_DIR=$(mktemp -d -p ../copies)
echo $TMP_DIR
find . \( -name "*.py" -o -name "*.json" \) -exec cp --parents {} $TMP_DIR \;
cd $TMP_DIR

PYTHONIOENCODING=utf-8 python3 train.py --name drg --config "/home/samuel/personal_work_ms/perin/config/base_drg.yaml" --home_directory $ORIGINAL_HOME
