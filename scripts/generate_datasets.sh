#!/bin/bash

# Define parameter ranges
N_sym_range=(16)
N_train_range=(50000)
n_range=(4)

# Loop through parameter combinations
for N_sym in "${N_sym_range[@]}"; do
  for N_train in "${N_train_range[@]}"; do
    for n in "${n_range[@]}"; do
      for t in {1..5}; do
        # Generate a random seed
        seed=$RANDOM

        python artificial_ngrams/art_ngrams/tasks/generate_dataset.py \
            dataset.N_sym=$N_sym \
            dataset.N_train=$N_train \
            dataset.n=$n \
            dataset.seed=$seed
      done
    done
  done
done
