# import argparse
import os
import pickle
import random
from math import ceil
from typing import Dict, List, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig

from art_ngrams.lm_generation import representation_ngram as ngram
from art_ngrams.utils import utils


def _save_dataset(D: List[List[str]], fname: str) -> None:

    with open(fname, "w") as f:
        for ii, y in enumerate(D):
            if ii == len(D) - 1:
                f.write(y)
            else:
                f.write(y + "\n")


def save_dataset(
    D_train: List[List[str]],
    D_test: List[List[str]],
    data_dir: str,
) -> None:

    _save_dataset(D_train, os.path.join(data_dir, "train.txt"))
    _save_dataset(D_test, os.path.join(data_dir, "test.txt"))

    print("Saved the datasets.")


def save_metadata(
    alphabet: str,
    pLM: ngram.NgramLM,
    data_dir: str,
    N_sym: int,
    N_train: int,
    N_test: int,
    n: int,
    dataset_statistics: Dict,
    seed: int,
) -> None:

    metadata = {
        "alphabet": list(alphabet),
        "N_sym": N_sym,
        "N_train": N_train,
        "N_test": N_test,
        "n": n,
        "seed": seed,
        "entropy": dataset_statistics["entropy"],
        "mean_length": dataset_statistics["mean_length"],
        "n_contexts": dataset_statistics["n_contexts"],
        "ratio_contexts": dataset_statistics["ratio_contexts"],
    }

    with open(os.path.join(data_dir, "metadata.pickle"), "wb") as f:
        pickle.dump(metadata, f)

    pLM.save(os.path.join(data_dir, "model.pickle"))

    print("Saved the metadata.")


def get_data(
    pLM: ngram.NgramLM, N_train: int, N_test: int, n: int
) -> Tuple[List[List[str]], List[List[str]]]:

    all_strings = pLM.sample(ceil(1.25 * (N_train + N_test)), to_string=True)
    possible_strings = set(all_strings)
    while len(all_strings) < N_train + N_test:
        new_sample = pLM.sample((N_train + N_test) // 2, to_string=True)
        possible_strings = possible_strings.union(set(new_sample))
        all_strings += new_sample

    possible_strings = list(possible_strings)
    M = len(possible_strings)
    random.shuffle(possible_strings)

    train_strings = set(possible_strings[: M // 2])
    test_strings = set(possible_strings[M // 2 :])

    D_train = [x for x in all_strings if x in train_strings][:N_train]
    D_test = [x for x in all_strings if x in test_strings][:N_test]

    C_train, r_train = utils.compute_number_of_context(D_train, pLM.alphabet, n)
    C_test, r_test = utils.compute_number_of_context(D_test, pLM.alphabet, n)

    print(f"Train: C = {C_train}, r = {r_train}")
    print(f"Test: C = {C_test}, r = {r_test}")

    return D_train, D_test


def compute_dataset_statistics(
    D_train: List[List[str]],
    D_test: List[List[str]],
    pLM: ngram.NgramLM,
    alphabet: str,
    n: int,
) -> Dict:

    D = D_train + D_test
    mean_len = np.mean([len(s.split()) for s in D])
    n_contexts, ratio_contexts = utils.compute_number_of_context(
        D_train, list(alphabet), n
    )

    entropy = -np.mean([pLM.logprob(s.split()) for s in D])

    scores = {
        "mean_length": mean_len,
        "n_contexts": n_contexts,
        "ratio_contexts": ratio_contexts,
        "entropy": entropy,
    }

    print("Dataset statistics:")
    print(f"Entropy: {entropy}")
    print(f"Mean length: {mean_len}")
    print(f"Number of contexts: {n_contexts}")
    print(f"Ratio of contexts: {ratio_contexts}")

    return scores


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="representation_dataset_generation_config",
)
def generate_dataset(cfg: DictConfig):

    N_sym = cfg.dataset.N_sym
    n = cfg.dataset.n
    N_train = cfg.dataset.N_train
    N_test = cfg.dataset.N_test
    mean_length = cfg.dataset.mean_length
    D = cfg.dataset.D
    rank = cfg.dataset.rank
    seed = cfg.dataset.seed
    base_dir = cfg.dataset.output

    print(
        f"Generating dataset with N_sym = {N_sym}, N_train = {N_train}, "
        f"N_test = {N_test}, n = {n}, mean_length = {mean_length}, "
        f"D = {D}, rank = {rank}, seed = {seed}"
    )

    alphabet = [str(i) for i in range(N_sym)]

    pLM = ngram.NgramLM(
        n=n,
        alphabet=alphabet,
        D=D,
        rank=rank,
        mean_length=mean_length,
        seed=seed,
    )

    print("Generated the n-gram LM.")
    D_train, D_test = get_data(pLM, N_train, N_test, n)

    dataset_statistics = compute_dataset_statistics(D_train, D_test, pLM, alphabet, n)

    data_dir = os.path.join(
        base_dir,
        f"n{n}_Nsy{N_sym}_Ntr{N_train}_ml{mean_length}_r{rank}_D{D}_s{seed}/",
    )

    os.makedirs(data_dir, exist_ok=True)
    save_dataset(D_train, D_test, data_dir)

    save_metadata(
        alphabet, pLM, data_dir, N_sym, N_train, N_test, n, dataset_statistics, seed
    )


if __name__ == "__main__":
    generate_dataset()
