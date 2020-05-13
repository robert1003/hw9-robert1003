#!/bin/bash

if [ $# -ne 2 ]; then
  echo -e "usage:\tbash train_baseline.sh [trainX_npy] [checkpoint]"
  exit
fi

python3 train_baseline.py $1 $2 bas.csv
