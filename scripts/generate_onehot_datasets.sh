#!/bin/bash

# Define parameter ranges
N_sym_range=(16 64 256 1024)
n_range=(2 4 8 16 32)

# Loop through parameter combinations
for N_sym in "${N_sym_range[@]}"; do
  for n in "${n_range[@]}"; do
    for t in {1..5}; do
      # Generate a random seed
      seed=$RANDOM

      python artificial_ngrams/art_ngrams/tasks/generate_representation_dataset.py \
          dataset.method=one-hot \
          dataset.N_sym=$N_sym \
          dataset.n=$n \
          dataset.seed=$seed
    done
  done
done
