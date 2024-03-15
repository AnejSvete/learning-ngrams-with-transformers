import numpy as np
import torch
import torch.nn as nn


class LogLinearModel(nn.Module):
    def __init__(
        self,
        n: int,
        N_sym: int,
        method: str,
    ):
        super(LogLinearModel, self).__init__()
        assert method in ["one-hot", "log", "rank"]
        self.n = n
        self.method = method
        self.N_sym = N_sym
        if self.method == "one-hot":
            self.D = self.N_sym * (self.n - 1)
        elif self.method == "log":
            self.D = int(np.ceil(np.log2(self.N_sym))) * (self.n - 1)
        elif self.method == "rank":
            self.D = self.N_sym ** (self.n - 1)
        self.linear = nn.Linear(self.D, self.N_sym)

        if self.method == "log":
            G = int(torch.ceil(torch.log2(self.N_sym)))
            self.idx2bin = {
                idx: [int(b) for b in np.binary_repr(int(idx), G)]
                for idx in range(self.N_sym)
            }

    def encode_ngram(self, ngram: torch.Tensor) -> torch.Tensor:
        """
        Encode a string into a fixed representation.

        Args:
        ngram (torch.Tensor): A tensor of shape (n-1,) representing an ngram.

        Returns:
        torch.Tensor: A tensor of shape (L, D) representing the fixed
                      representation of the string.
        """
        if self.method == "one-hot":
            enc = torch.zeros((self.N_sym, self.n - 1))
            enc[ngram, torch.arange(self.n - 1)] = 1
            enc = enc.flatten()
        elif self.method == "log":
            enc = torch.asarray([self.idx2bin[idx] for idx in ngram]).flatten()
        # TODO
        # elif self.method == "rank":
        #     enc = torch.zeros(self.N_sym ** (self.n - 1))
        #     enc[self.ngram2idx[ngram]] = 1
        return enc

    def encode(self, x: torch.Tensor):
        """
        Encode a string into a fixed representation.

        Args:
        x (torch.Tensor): A tensor of shape (batch_size, L) representing
                          a batch of strings.

        Returns:
        torch.Tensor: A tensor of shape (batch_size, L, D) representing
                      the fixed representation of the strings.
        """
        enc = torch.zeros(x.size(0), x.size(1), self.D)
        for i in range(x.size(0)):
            for j in range(self.n - 1, x.size(1)):
                enc[i, j, :] = self.encode_ngram(x[i, j - self.n + 1 : j])

        return enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the next symbol logits.

        Args:
        x (torch.Tensor): A tensor of shape (batch_size, L, D)
                          representing the fixed representation of a string.

        Returns:
        torch.Tensor: A tensor of shape (batch_size, L, alphabet_size)
                      containing the logits for the next symbol.
        """

        batch_size, L = x.size()

        enc = self.encode(x)

        enc = enc.to(x.device)

        # Apply linear layer to each element in the sequence
        # Reshape enc to (batch_size * L, D)
        enc = enc.view(batch_size * L, self.D)

        # Apply the linear layer
        logits = self.linear(enc)

        # Reshape logits back to (batch_size, L, alphabet_size)
        logits = logits.view(batch_size, L, -1)

        return logits
