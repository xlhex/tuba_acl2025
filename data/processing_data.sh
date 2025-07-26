#!/bin/bash

# Check if l1 is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <l1> [l2]"
  exit 1
fi

l1=$1
l2=$2

# ---
## Create a poisoned data

SEED=1000
POISON_RATIO=0.2

if [ -z "$l2" ]; then
  # If only l1 is provided, no nested loop
  language=${l1}
  echo "python hate_speech_poison.py $language $SEED $POISON_RATIO"
  python hate_speech_poison.py $language $SEED $POISON_RATIO

else
  # If both l1 and l2 are provided, use nested loop logic
  language=${l1}:${l2}
  echo "python hate_speech_poison.py $language $SEED $POISON_RATIO"
  python hate_speech_poison.py $language $SEED $POISON_RATIO

fi
