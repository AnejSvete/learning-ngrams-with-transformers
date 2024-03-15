import os
import pickle
from itertools import product
from os import path
from typing import List, Tuple

import hydra
import numpy as np
import wandb
from omegaconf import DictConfig

from art_ngrams.lm_generation import ngram
from art_ngrams.utils import utils
from art_ngrams.utils.utils import compute_number_of_context


def load_data(data_dir: str) -> Tuple[List[List[str]], List[List[str]]]:

    # print("Loading data...")

    with open(path.join(data_dir, "train.txt"), "r") as f:
        D_train = f.read().split("\n")
        D_train = [d.split() for d in D_train]

    with open(path.join(data_dir, "test.txt"), "r") as f:
        D_test = f.read().split("\n")
        D_test = [d.split() for d in D_test]

    # print("Done loading data.")

    return D_train, D_test


def save_results(
    data_dir: str,
    scores: dict,
    n: int,
    N_sym: int,
    N_train: int,
    N_test: int,
    reduce_eos_proportion: float,
    seed: int,
):

    print("Saving results...")

    results_dir = path.join(
        data_dir,
        f"n{n}_Nsy{N_sym}_Ntr{N_train}_Nts{N_test}_r{reduce_eos_proportion}_s{seed}/",
    )

    os.makedirs(results_dir, exist_ok=True)

    with open(path.join(results_dir, "statistics.pickle"), "wb") as f:
        pickle.dump(scores, f)

    print("Done saving results.")


@hydra.main(
    version_base=None, config_path="../config", config_name="compute_statistics_config"
)
def main(cfg: DictConfig):

    N_sym_range = cfg.dataset.N_sym_range
    N_train_range = cfg.dataset.N_train_range
    N_test_range = cfg.dataset.N_test_range
    n_range = cfg.dataset.n_range
    reduce_eos_proportion_range = cfg.dataset.reduce_eos_proportion_range

    base_data_dir = cfg.dataset.base_data_dir
    base_statistics_dir = cfg.dataset.base_statistics_dir

    for n, N_sym, N_train, N_test, reduce_eos_proportion in product(
        n_range,
        N_sym_range,
        N_train_range,
        N_test_range,
        reduce_eos_proportion_range,
    ):
        print(
            f">>>> Scoring n-gram estimation methods for n={n}, "
            f"N_sym={N_sym}, N_train={N_train}, N_test={N_test}, "
            f"reduce_eos_proportion={reduce_eos_proportion}."
        )

        data_dirs = utils.find_matching_directories(
            base_dir=base_data_dir,
            n=n,
            N_sym=N_sym,
            N_train=N_train,
            N_test=N_test,
            reduce_eos_proportion=reduce_eos_proportion,
        )

        for data_dir, seed in data_dirs:

            ngram_model_filename = data_dir + "model.pickle"
            pLM = ngram.NgramLM(filename=ngram_model_filename)
            assert pLM.n == n

            D_train, D_test = load_data(data_dir)
            D = D_train + D_test
            mean_len = np.mean([len(s) for s in D])
            alphabet = "abcdefghijklmnopqrstuvwxyz"[:N_sym]
            n_contexts, ratio_contexts = compute_number_of_context(
                D_train, list(alphabet), n
            )

            if n == 6 or n == 5 and N_sym == 18:
                entropy = -np.mean([pLM.logprob(s) for s in D])
            else:
                entropy = pLM.entropy()

            scores = {
                "mean_len": mean_len,
                "n_contexts": n_contexts,
                "ratio_contexts": ratio_contexts,
                "entropy": entropy,
            }

            save_results(
                base_statistics_dir,
                scores,
                n,
                N_sym,
                N_train,
                N_test,
                reduce_eos_proportion,
                seed,
            )

    wandb.finish()


if __name__ == "__main__":
    main()
