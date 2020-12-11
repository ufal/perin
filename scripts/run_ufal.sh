#!/bin/bash

ORIGINAL_HOME=$PWD
TMP_DIR=$(mktemp -d -p ../copies)
echo $TMP_DIR
find . \( \( -name "*.py" -o -name "*.json" -o -name "*.pyx" \) -a \! -path "*/outputs/*" \) -exec cp --parents {} $TMP_DIR \;
cd $TMP_DIR

. $ORIGINAL_HOME/../venv/bin/activate

sed "s/self.encoder =.*/self.encoder = \"xlm-roberta-$2\"/" -i config/params.py
sed "s/batch_size\": [0-9]*/batch_size\": $3/" -i config/$1.json
sed "s/accumulation_steps\": [0-9]*/accumulation_steps\": $4/" -i config/$1.json
PYTHONIOENCODING=utf-8 python3 train.py --name "$1"_"$2"$5 --config "config/$1.json" --home_directory $ORIGINAL_HOME "${@:6}"
