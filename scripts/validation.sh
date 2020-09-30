#!/bin/bash

PYTHONIOENCODING=utf-8 python3 utility/evaluate.py --input_dir "$1" --epoch "$2" --framework "$3" --language "$4" --gold_file "$5"
