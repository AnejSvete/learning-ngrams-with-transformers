# import argparse
import os
import pickle
import random
from typing import Dict, List, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig

from art_ngrams.lm_generation import ngram
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

    _save_dataset(D_train, data_dir + "train.txt")
    _save_dataset(D_test, data_dir + "test.txt")

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

    with open(data_dir + "metadata.pickle", "wb") as f:
        pickle.dump(metadata, f)

    pLM.save(data_dir + "model.pickle")

    print("Saved the metadata.")


def get_data(
    pLM: ngram.NgramLM, N_train: int, N_test: int, n: int, seed: int
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:

    while True:

        possible_strings = set()
        while len(possible_strings) < 2 * (N_train + N_test):
            possible_strings = possible_strings.union(
                set(pLM.sample(2 * (N_train + N_test), to_string=True))
            )

        possible_strings = list(possible_strings)
        M = len(possible_strings)
        random.shuffle(possible_strings)

        train_strings = set(possible_strings[: M // 2])
        test_strings = set(possible_strings[M // 2 :])

        D_train = pLM.sample(2 * N_train, to_string=True)
        _D_train = set(D_train)
        D_test = pLM.sample(2 * N_test, to_string=True)
        _D_test = set(D_test)
        while len(set(_D_train)) < 2 * (N_train + N_test):
            D_train += pLM.sample(2 * (N_train + N_test), to_string=True)
            _D_train = _D_train.union(set(D_train))
        while len(set(_D_test)) < 2 * (N_train + N_test):
            D_test += pLM.sample(2 * (N_train + N_test), to_string=True)
            _D_test = _D_test.union(set(D_test))

        print(f"Sampled {len(D_train)} train, and {len(D_test)} test strings.")
        D_train = [x for x in D_train if x not in test_strings][:N_train]
        D_test = [x for x in D_test if x not in train_strings][:N_test]

        if len(D_train) == N_train and len(D_test) == N_test:
            break
        else:
            print(f"Retrying. Got {len(D_train)} train and {len(D_test)} test strings.")

    print(f"Retained {len(D_train)} train and {len(D_test)} test strings.")

    print()

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
    version_base=None, config_path="../config", config_name="dataset_generation_config"
)
def generate_dataset(cfg: DictConfig):

    N_sym = cfg.dataset.N_sym
    N_train = cfg.dataset.N_train
    N_test = cfg.dataset.N_test
    n = cfg.dataset.n
    alpha = cfg.dataset.alpha
    mean_length = cfg.dataset.mean_length
    seed = cfg.dataset.seed
    base_dir = cfg.dataset.base_dir

    assert N_train <= 50000, "The number of training strings is limited to 50,000."
    assert N_test <= 50000, "The number of test strings is limited to 50,000."

    alphabet = [str(i) for i in range(N_sym)]

    pLM = ngram.NgramLM(
        n=n,
        alphabet=alphabet,
        alpha=alpha,
        mean_length=mean_length,
        seed=seed,
    )
    print("Generated the n-gram LM.")
    D_train, D_test = get_data(pLM, N_train, N_test, n, seed)

    data_dir = (
        base_dir + f"n{n}_Nsy{N_sym}_Ntr{N_train}_Nts{N_test}_ml{mean_length}_s{seed}/"
    )

    dataset_statistics = compute_dataset_statistics(D_train, D_test, pLM, alphabet, n)

    os.makedirs(data_dir, exist_ok=True)
    save_dataset(D_train, D_test, data_dir)

    save_metadata(
        alphabet, pLM, data_dir, N_sym, N_train, N_test, n, dataset_statistics, seed
    )


if __name__ == "__main__":
    generate_dataset()
