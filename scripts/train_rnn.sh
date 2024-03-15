#!/bin/bash

# Define parameter ranges
# N_sym_range=(2 3 4 5 6 7 8)
# N_train_range=(100 1000 3000 5000 10000)
# N_test=10000
# n_range=(2 3 4 5 6)

N_sym_range=(6)
# N_train_range=(100 500 1000 2000 5000)
N_train_range=(450)
N_test=10000
n_range=(5)

# Loop through parameter combinations
for N_sym in "${N_sym_range[@]}"; do
  for N_train in "${N_train_range[@]}"; do
    for n in "${n_range[@]}"; do
      python art_ngrams/tasks/train_and_evaluate_rnn.py \
          training.N_sym=$N_sym \
          training.N_train=$N_train \
          training.N_test=$N_test \
          training.n=$n
    done
  done
done
