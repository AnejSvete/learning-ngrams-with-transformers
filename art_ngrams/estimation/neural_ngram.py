from typing import List

import torch
import torch.nn as nn


class RepresentationNgramEstimator(nn.Module):
    def __init__(
        self,
        D: int,
        n: int,
        BOSalphabet: List[str],
    ):
        super(RepresentationNgramEstimator, self).__init__()
        self.D = D
        self.n = n
        self.BOSalphabet = BOSalphabet
        self.embedding = nn.Embedding(len(self.BOSalphabet), self.D)
        self.linear = nn.Linear(self.D * (self.n - 1), len(self.BOSalphabet))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = self.embedding(x)

        enc = torch.zeros(x.size(0), x.size(1), (self.n - 1) * self.D)

        for i in range(x.size(0)):
            for j in range(self.n - 1, x.size(1)):
                enc[i, j, :] = x_enc[i, j - (self.n - 1) : j, :].flatten()

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
        enc = enc.view(batch_size * L, self.D * (self.n - 1))

        # Apply the linear layer
        logits = self.linear(enc)

        # Reshape logits back to (batch_size, L, alphabet_size)
        logits = logits.view(batch_size, L, -1)

        return logits
