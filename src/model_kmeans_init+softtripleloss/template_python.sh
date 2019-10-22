#!/usr/bin/env bash
## Run the script
source activate ml
echo $0
echo $1
echo $2
python main.py $1 &>logs/log_$2