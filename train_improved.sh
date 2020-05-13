#!/bin/bash

if [ $# -ne 2 ]; then
  echo -e "usage:\tbash train_improved.sh [trainX_npy] [checkpoint]"
  exit
fi

python3 train_improved.py $1 $2 imp.csv
