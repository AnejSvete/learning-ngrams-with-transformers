import os
import re
from collections import Counter
from glob import glob
from typing import Dict, List, Optional, Set, Tuple, Union


def save_dataset(strings: List[str], filename: str):
    with open(filename, "w") as f:
        for string in strings:
            f.write(string + "\n")


def load_dataset(filename: str) -> List[str]:
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines()]


def find_matching_directories(
    base_dir: str,
    n: int,
    N_sym: int,
    N_train: int,
    N_test: Optional[int] = None,
    reduce_eos_proportion: Optional[float] = None,
    n_hat: Optional[int] = None,
    mean_length: Optional[int] = None,
    method: Optional[str] = None,
    rank: Optional[int] = None,
    D: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    num_epochs: Optional[int] = None,
    learning_method: Optional[str] = None,
    results: bool = False,
    representation: bool = False,
) -> List[str]:

    # print("Finding matching directories...")

    if results:
        if representation:
            if learning_method == "loglinear":
                pattern = os.path.join(
                    base_dir,
                    f"loglinear_n{n}_nh{n_hat}_ns{N_sym}_m{method}_D{D}_r{rank}_"
                    f"ml{mean_length}_bs{batch_size}_lr{lr}_e{num_epochs}_s*/",
                )
            else:
                pattern = os.path.join(
                    base_dir,
                    f"classic_n{n}_nh{n_hat}_ns{N_sym}_m{method}_D{D}_r{rank}_"
                    f"ml{mean_length}_s*/",
                )
        else:
            pattern = os.path.join(
                base_dir,
                f"n{n}_nh{n_hat}_Nsym{N_sym}_Ntr{N_train}"
                + f"_Nts{N_test}_r{reduce_eos_proportion}_kFalse_s*/",
            )
    else:
        if representation:
            pattern = os.path.join(
                base_dir,
                f"n{n}_Nsy{N_sym}_Ntr{N_train}_ml{mean_length}_m{method}"
                f"_r{rank}_D{D}_s*/",
            )
        else:
            pattern = os.path.join(
                base_dir,
                f"n{n}_Nsy{N_sym}_Ntr{N_train}_Nts{N_test}_ml{mean_length}_s*/",
            )

    matching_dirs = glob(pattern)

    # Extract the seed from each matching directory
    matching_info = []
    for dir_path in matching_dirs:
        # Use regular expression to extract the seed
        seed_match = re.search(r"_s(\d+)", dir_path)
        if seed_match:
            seed = int(seed_match.group(1))
            matching_info.append((dir_path, seed))

    # print(f"Done finding directories: {len(matching_info)} found; {matching_info}")

    return matching_info


def compute_number_of_context(
    D: List[Union[str, List[str]]], alphabet: Union[List[str], Set[str]], n: int
) -> Tuple[int, float]:
    if isinstance(D[0], str):
        D = [d.split(" ") for d in D]

    contexts = set()
    for string in D:
        for i in range(len(string) - n + 1):
            contexts.add(tuple(string[i : i + n - 1]))

    return len(contexts), len(contexts) / len(alphabet) ** (n - 1)


def compute_symbols_per_positions(
    D: List[Union[str, List[str]]], alphabet: Union[List[str], Set[str]], n: int
) -> Dict[str, int]:
    pos_sym = {i: {a: 0 for a in alphabet} for i in range(n - 1)}
    for y in D:
        if isinstance(y, str):
            y = y.split(" ")
        c = Counter(y[: -(n - 1)])
        for i in range(n - 1):
            for a in alphabet:
                pos_sym[i][a] += c[a]

        for k, a in enumerate(y[-(n - 1) :]):
            for i in range(n - 1 - k):
                pos_sym[i][a] += 1

    return pos_sym
