import os
import pickle
from itertools import product
from os import path
from typing import Dict, List

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from art_ngrams.estimation.dataset import NGramDataset, SortedBatchSampler
from art_ngrams.estimation.loglinear_model import LogLinearModel
from art_ngrams.utils import utils


def collate_fn(batch):
    """
    Custom collate function to pad sequences to the same length.

    Args:
    batch (list of tuples): List of (input_tensor, target_tensor) tuples.

    Returns:
    tuple: Padded input and target tensors.
    """
    inputs, targets = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return padded_inputs, padded_targets


def train_model(
    model: LogLinearModel,
    data_dir: str,
    num_epochs: int,
    batch_size: int,
    lr: float,
    device: str,
):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = NGramDataset(data_dir, "train")

    sampler = SortedBatchSampler(dataset, batch_size)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_sampler=sampler)

    model.train()

    scheduler = lr_scheduler.StepLR(optimizer, step_size=num_epochs // 3, gamma=0.5)

    for epoch in range(num_epochs):
        total_loss = 0.0

        for inputs, targets in tqdm(dataloader, total=len(dataloader)):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            logits = model(inputs)

            batch_size, seq_length = targets.size()
            logits = logits.view(batch_size * seq_length, -1)
            targets = targets.view(batch_size * seq_length)

            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")


def evaluate_model(
    model: LogLinearModel, data_dir: str, device: str
) -> Dict[str, float]:

    with open(path.join(data_dir, "metadata.pickle"), "rb") as f:
        data = pickle.load(f)
        entropy = data["entropy"]

    model.eval()
    eval_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
    with torch.no_grad():

        # Create the dataset
        test_dataset = NGramDataset(data_dir, "test")

        # Create a DataLoader
        test_sampler = SortedBatchSampler(test_dataset, 512)
        test_dataloader = DataLoader(
            test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn
        )

        losses = []

        for inputs, targets in tqdm(test_dataloader, total=len(test_dataloader)):
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)

            batch_size, seq_length = targets.size()
            logits = logits.view(batch_size * seq_length, -1)
            targets = targets.view(batch_size * seq_length)

            losses.append(eval_criterion(logits, targets).view(batch_size, seq_length))

    loss_values = []
    for loss in losses:
        loss_values.append(loss.sum(1).mean())

    cross_entropy = torch.mean(torch.tensor(loss_values)) / torch.log(torch.tensor(2.0))
    kl = cross_entropy - entropy

    return {"CE": cross_entropy.item(), "KL": kl.item()}


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
    batch_size: int,
    lr: float,
    num_epochs: int,
    seed: int,
):

    print("Saving results...")

    results_dir = path.join(
        base_results_dir,
        f"loglinear_n{n}_nh{n_hat}_ns{N_sym}_m{method}_D{D}_r{rank}_ml{mean_length}"
        f"_bs{batch_size}_lr{lr}_e{num_epochs}_s{seed}/",
    )

    os.makedirs(results_dir, exist_ok=True)

    with open(path.join(results_dir, "classic_results.pickle"), "wb") as f:
        pickle.dump(scores, f)

    print("Done saving results.")


def get_n_hat_range(n: int) -> List[int]:

    if n == 2:
        # n_hat_range = []
        n_hat_range = [1, 2, 4]
    elif n == 4:
        # n_hat_range = [8]
        n_hat_range = [2, 4, 8]
    elif n == 6:
        # n_hat_range = [12]
        n_hat_range = [4, 6, 12]
    elif n == 8:
        # n_hat_range = [16]
        n_hat_range = [6, 8, 16]
    elif n == 12:
        # n_hat_range = [18, 20]
        n_hat_range = [10, 12, 20]

    return n_hat_range


@hydra.main(
    version_base=None, config_path="../config", config_name="loglinear_model_config"
)
def main(cfg: DictConfig):

    wandb.init(project="artificial-ngrams-loglinear")

    N_sym_range = cfg.dataset.N_sym_range
    n_range = cfg.dataset.n_range
    mean_length = cfg.dataset.mean_length
    method = cfg.dataset.method
    rank_range = cfg.dataset.rank_range
    D_range = cfg.dataset.D_range
    N_train = cfg.dataset.N_train
    batch_size = cfg.dataset.batch_size
    lr = cfg.dataset.lr
    num_epochs = cfg.dataset.num_epochs
    device = cfg.dataset.device
    representation_dataset = cfg.dataset.representation_dataset

    base_data_dir = cfg.dataset.base_data_dir
    base_results_dir = cfg.dataset.base_results_dir

    for n, N_sym, D, rank in product(n_range, N_sym_range, D_range, rank_range):
        print(f">>>> Evaluating n-gram estimation methods for n={n}, N_sym={N_sym}.")

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

        for data_dir, seed in data_dirs:

            print(f"Training and evaluating model on {data_dir}.")

            n_hat_range = get_n_hat_range(n)

            for n_hat in n_hat_range:

                model = LogLinearModel(n=n_hat, N_sym=N_sym + 2, method="one-hot")
                model = model.to(device)

                train_model(model, data_dir, num_epochs, batch_size, lr, device)

                scores = evaluate_model(model, data_dir, device)
                print(scores)

                wandb.log(
                    {
                        "KL": scores["KL"],
                        "CE": scores["CE"],
                        "N_sym": N_sym,
                        "N_train": N_train,
                        "n": n,
                        "n_hat": n_hat,
                    }
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
                    batch_size,
                    lr,
                    num_epochs,
                    seed,
                )

    wandb.finish()


if __name__ == "__main__":
    main()
