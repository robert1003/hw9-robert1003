#!/bin/bash

if [ $# -ne 3 ]; then
  echo -e "usage:\tbash hw9_best.sh [trainX_npy] [checkpoint] [prediction_file]"
  exit
fi

python3 hw9_best.py $1 $2 $3
