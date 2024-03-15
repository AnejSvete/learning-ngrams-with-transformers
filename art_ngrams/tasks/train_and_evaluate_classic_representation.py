import os
import pickle
from itertools import product
from os import path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import hydra
import wandb
from omegaconf import DictConfig
from tqdm import tqdm

from art_ngrams.estimation.classic import (
    AbsoluteDiscountingEstimator,
    AddLambdaEstimator,
    JelinekMercerEstimator,
    KneserNeyEstimator,
    MLEEstimator,
    NgramsEstimator,
    WittenBellEstimator,
)
from art_ngrams.evaluation import metrics
from art_ngrams.utils import utils

method_name_to_estimator = {
    "MLE": MLEEstimator,
    "AddLambda 0.01": AddLambdaEstimator,
    "AddLambda 0.1": AddLambdaEstimator,
    "AddLambda 0.5": AddLambdaEstimator,
    "AddLambda 1": AddLambdaEstimator,
    "AddLambda 2": AddLambdaEstimator,
    "AbsoluteDiscounting 0.6": AbsoluteDiscountingEstimator,
    "AbsoluteDiscounting 0.8": AbsoluteDiscountingEstimator,
    "AbsoluteDiscounting 0.95": AbsoluteDiscountingEstimator,
    "WittenBell": WittenBellEstimator,
    "KneserNey 0.1": KneserNeyEstimator,
    "KneserNey 0.4": KneserNeyEstimator,
    "KneserNey 0.8": KneserNeyEstimator,
    "JelinekMercer 1": JelinekMercerEstimator,
    "JelinekMercer 5": JelinekMercerEstimator,
    "JelinekMercer 10": JelinekMercerEstimator,
}


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


def train(
    n_hat: int,
    D_train: List[List[str]],
    method_name: str,
    parameters: Optional[Union[int, float]],
) -> Tuple[Sequence[NgramsEstimator], Sequence[List[List[str]]]]:

    print("Training classic estimators...")

    print(f"Training {method_name}...")
    Estimator = method_name_to_estimator[method_name]
    if parameters is None:
        p_est = Estimator(n_hat)
    else:
        p_est = Estimator(n_hat, parameters)
    p_est.fit(D_train)

    return p_est


def score_classic_methods(
    entropy: float,
    p_est: NgramsEstimator,
    D_test: List[List[str]],
    method_name: str,
    N_sym: int,
    n: int,
    n_hat: int,
) -> Dict[str, Dict[str, float]]:

    print("Scoring classic estimators...")

    scores = dict()

    scores["CE"] = metrics.empirical_entropy(p_est, D_test)
    scores["KL"] = scores["CE"] - entropy
    scores["entropy"] = entropy

    for key, value in scores.items():
        wandb.log(
            {
                f"{key}/{method_name}": value,
                "N_sym": N_sym,
                "n": n,
                "n_hat": n_hat,
            }
        )

    print("Done scoring classic estimators.")

    return scores


def evaluate_classic_methods(
    data_dir: str,
    N_sym: int,
    n: int,
    n_hat: int,
) -> Dict[str, Dict[str, float]]:

    D_train, D_test = load_data(data_dir)

    method_names = [
        "MLE",
        "AddLambda 0.01",
        "AddLambda 0.1",
        "AddLambda 1",
        "AbsoluteDiscounting 0.6",
        "AbsoluteDiscounting 0.8",
        "AbsoluteDiscounting 0.95",
        "WittenBell",
    ]
    parameters = {
        "AddLambda 0.01": 0.01,
        "AddLambda 0.1": 0.1,
        "AddLambda 1": 1,
        "AbsoluteDiscounting 0.6": 0.6,
        "AbsoluteDiscounting 0.8": 0.8,
        "AbsoluteDiscounting 0.95": 0.95,
    }

    with open(path.join(data_dir, "metadata.pickle"), "rb") as f:
        data = pickle.load(f)
        entropy = data["entropy"]

    scores = dict()
    for method_name in tqdm(method_names):
        p_est = train(n_hat, D_train, method_name, parameters.get(method_name, None))

        scores_ = score_classic_methods(
            entropy,
            p_est,
            D_test,
            method_name,
            N_sym,
            n,
            n_hat,
        )

        scores[method_name] = scores_

    return scores


def save_results(
    base_results_dir: str,
    scores: Dict[str, float],
    n: int,
    n_hat: int,
    N_sym: int,
    method: str,
    D: int,
    rank: int,
    mean_length: int,
    seed: int,
):

    print("Saving results...")

    results_dir = path.join(
        base_results_dir,
        f"classic_n{n}_nh{n_hat}_ns{N_sym}_m{method}_D{D}_r{rank}_ml{mean_length}"
        f"_s{seed}/",
    )

    os.makedirs(results_dir, exist_ok=True)

    with open(path.join(results_dir, "classic_results_rep.pickle"), "wb") as f:
        pickle.dump(scores, f)

    print("Done saving results.")


def get_n_hat_range(n: int) -> List[int]:

    if n == 2:
        n_hat_range = []
    elif n == 4:
        n_hat_range = [8]
    elif n == 6:
        n_hat_range = [12]
    elif n == 8:
        n_hat_range = [16]
    elif n == 12:
        n_hat_range = [18, 20]

    return n_hat_range


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="classic_estimation_config_representation",
)
def main(cfg: DictConfig):

    N_sym_range = cfg.dataset.N_sym_range
    n_range = cfg.dataset.n_range
    D_range = cfg.dataset.D_range
    rank_range = cfg.dataset.rank_range
    N_train = cfg.dataset.N_train
    mean_length = cfg.dataset.mean_length
    method = cfg.dataset.method
    representation_dataset = cfg.dataset.representation_dataset

    base_data_dir = cfg.dataset.base_data_dir
    base_results_dir = cfg.dataset.base_results_dir

    wandb.init(project="artificial-ngrams-classic-representation")

    for n, N_sym, D, rank in product(n_range, N_sym_range, D_range, rank_range):
        if representation_dataset:
            data_dirs = utils.find_matching_directories(
                base_dir=base_data_dir,
                n=n,
                N_sym=N_sym,
                N_train=N_train,
                mean_length=mean_length,
                method=method,
                rank=rank,
                D=D,
                representation=True,
            )
        else:
            data_dirs = utils.find_matching_directories(
                base_dir=base_data_dir,
                n=n,
                N_sym=N_sym,
                N_train=N_train,
                N_test=30000,
                mean_length=mean_length,
            )

        n_hat_range = get_n_hat_range(n)

        for n_hat in n_hat_range:
            print(
                f">>>> Evaluating n-gram estimation methods for n={n}, n_hat={n_hat}, "
                f"N_sym={N_sym}, D={D}, rank={rank}."
            )

            for data_dir, seed in data_dirs:
                scores = evaluate_classic_methods(
                    data_dir,
                    N_sym,
                    n,
                    n_hat,
                )

                save_results(
                    base_results_dir,
                    scores,
                    n,
                    n_hat,
                    N_sym,
                    method,
                    D,
                    rank,
                    mean_length,
                    seed,
                )

    wandb.finish()


if __name__ == "__main__":
    main()
