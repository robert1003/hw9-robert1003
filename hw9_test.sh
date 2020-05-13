#!/bin/bash

if [ $# -ne 2 ]; then
  echo -e "usage:\tbash hw9_test.sh [trainX_npy] [checkpoint] [prediction_path]"
  exit
fi

python3 hw9_test.py $1 $2 $3
