# import argparse
import os
import pickle
from typing import List

import hydra
from omegaconf import DictConfig

from art_ngrams.lm_generation import ngram
from art_ngrams.utils import data_utils


def _save_dataset(
    D: List[List[str]], labels: List[int], data_dir: str, split: str
) -> None:

    with open(data_dir + split + ".strings", "w") as f:
        for ii, y in enumerate(D):
            if ii == len(D) - 1:
                f.write(y)
            else:
                f.write(y + "\n")

    with open(data_dir + split + ".labels", "w") as f:
        for ii, y in enumerate(labels):
            if ii == len(labels) - 1:
                f.write(str(y))
            else:
                f.write(str(y) + "\n")


def save_dataset(
    D_train_t: List[List[str]],
    D_val_t: List[List[str]],
    D_test_t: List[List[str]],
    D_train_c: List[List[str]],
    D_val_c: List[List[str]],
    D_test_c: List[List[str]],
    data_dir: str,
) -> None:

    D_train = D_train_t + D_train_c
    D_val = D_val_t + D_val_c
    D_test = D_test_t + D_test_c

    labels_train = [1] * len(D_train_t) + [0] * len(D_train_c)
    labels_val = [1] * len(D_val_t) + [0] * len(D_val_c)
    labels_test = [1] * len(D_test_t) + [0] * len(D_test_c)

    _save_dataset(D_train, labels_train, data_dir, "train")
    _save_dataset(D_val, labels_val, data_dir, "val")
    _save_dataset(D_test, labels_test, data_dir, "test")

    print("Saved the datasets.")


def save_metadata(
    alphabet: str,
    pLM: ngram.NgramLM,
    data_dir: str,
    N_sym: int,
    N_train: int,
    N_val: int,
    N_test: int,
    n: int,
    seed: int,
) -> None:

    with open(data_dir + "metadata.pickle", "wb") as f:
        pickle.dump({"alphabet": list(alphabet)}, f)
        pickle.dump({"N_sym": N_sym}, f)
        pickle.dump({"N_train": N_train}, f)
        pickle.dump({"N_val": N_val}, f)
        pickle.dump({"N_test": N_test}, f)
        pickle.dump({"n": n}, f)
        pickle.dump({"seed": seed}, f)

    pLM.save(data_dir + "model.pickle")

    print("Saved the metadata.")


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="binary_dataset_generation_config",
)
def generate_dataset(cfg: DictConfig):

    N_sym = cfg.dataset.N_sym
    N_train = cfg.dataset.N_train
    N_test = cfg.dataset.N_test
    n = cfg.dataset.n
    reduce_eos_proportion = cfg.dataset.reduce_eos_proportion
    connectivity = cfg.dataset.connectivity
    seed = cfg.dataset.seed
    base_dir = cfg.dataset.base_dir

    assert N_train <= 20000, "The number of training strings is limited to 20,000."
    assert N_test <= 20000, "The number of test strings is limited to 20,000."

    N_val = int(1 / 9 * N_train)

    assert N_sym <= 26, "The alphabet is limited to the first 26 letters."
    alphabet = "abcdefghijklmnopqrstuvwxyz"[:N_sym]

    pLM = ngram.NgramLM(
        n=n,
        alphabet=alphabet,
        reduce_eos_proportion=reduce_eos_proportion,
        binary=True,
        connectivity=connectivity,
        seed=seed,
    )
    pLMc = pLM.complement()
    print("Generated the n-gram LM.")
    D_train_t, D_val_t, D_test_t = data_utils.get_data(
        pLM, N_train, N_val, N_test, n, seed
    )
    D_train_c, D_val_c, D_test_c = data_utils.get_data(
        pLMc, N_train, N_val, N_test, n, seed
    )

    data_dir = base_dir + f"n{n}_Nsy{N_sym}_Ntr{N_train}_Nv{N_val}_Nts{N_test}_s{seed}/"

    os.makedirs(data_dir, exist_ok=True)
    save_dataset(D_train_t, D_val_t, D_test_t, D_train_c, D_val_c, D_test_c, data_dir)

    save_metadata(alphabet, pLM, data_dir, N_sym, N_train, N_val, N_test, n, seed)


if __name__ == "__main__":
    generate_dataset()
