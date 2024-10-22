#!/bin/bash

for t in {1..5}; do
  # Generate a random seed
  seed=$RANDOM

  echo "Generating dataset with seed $seed"

  python art_ngrams/tasks/generate_dataset.py dataset.seed=$seed
done