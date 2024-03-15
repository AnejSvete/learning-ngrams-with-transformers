import pickle
import random
from os import path

import torch
from torch.utils.data import Dataset, Sampler


class SortedBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Get lengths of sequences
        lengths = [len(input_tensor) for input_tensor, _ in data_source]

        # Sort indices by sequence length
        self.sorted_indices = sorted(range(len(lengths)), key=lambda x: lengths[x])

        # Create batches from sorted indices
        self.batches = [
            self.sorted_indices[i : i + batch_size]
            for i in range(0, len(self.sorted_indices), batch_size)
        ]

        # Shuffle the batches
        random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return (
                len(self.batches) - 1
                if len(self.batches) * self.batch_size > len(self.data_source)
                else len(self.batches)
            )
        else:
            return len(self.batches)


class NGramDataset(Dataset):
    def __init__(self, file_dir: str, split: str, method: str = "one-hot"):
        """
        Initializze the dataset.

        Args:
        filename (str): Path to the file containing line-separated strings.
        """
        self.split = split
        self.method = method
        self.setup(file_dir)

        self.data = self.load_data(file_dir)

    def setup(self, data_dir: str):
        with open(path.join(data_dir, "model.pickle"), "rb") as f:
            data = pickle.load(f)
            self.n = data["n"]
            self.alphabet = data["alphabet"]
            self.BOSEOSalphabet = ["<s>"] + data["alphabet"] + ["</s>"]
            self.idx2sym = {i: sym for i, sym in enumerate(self.BOSEOSalphabet)}
            self.sym2idx = {sym: i for i, sym in self.idx2sym.items()}

    def encode_neural(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = self.embedding(x)

        enc = torch.zeros(x.size(0), x.size(1), (self.n - 1) * self.D)

        for i in range(x.size(0)):
            for j in range(self.n - 1, x.size(1)):
                enc[i, j, :] = x_enc[i, j - (self.n - 1) : j, :].flatten()

        return enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "vectorial":
            return self.encode_neural(x)

    def load_data(self, file_dir: str):
        """
        Load and process the data from the file.

        Args:
        filename (str): Path to the file containing line-separated strings.

        Returns:
        list of tuples: Each tuple contains an input tensor and a target tensor.
        """
        data = []
        filename = path.join(file_dir, f"{self.split}.txt")
        with open(filename, "r") as file:
            for line in file:
                symbols = line.strip().split()
                input_symbols = ["<s>"] * (self.n - 1) + symbols
                target_symbols = ["<s>"] * (self.n - 2) + symbols + ["</s>"]
                input_tensor = self.symbols_to_tensor(input_symbols)
                target_tensor = self.symbols_to_tensor(target_symbols)
                data.append((input_tensor, target_tensor))
        return data

    def symbols_to_tensor(self, symbols):
        """
        Convert a list of symbols to a tensor of indices.

        Args:
        symbols (list): List of symbols.

        Returns:
        torch.Tensor: Tensor of symbol indices.
        """
        indices = [self.sym2idx[symbol] for symbol in symbols]
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.data[idx]
        return [self.data[i] for i in idx]
