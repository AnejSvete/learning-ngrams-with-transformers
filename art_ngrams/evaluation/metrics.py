from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch

from art_ngrams.estimation.classic import NgramsEstimator
from art_ngrams.lm_generation.ngram import NgramLM
from art_ngrams.utils.ngram_utils import convert_to_larger_n


def _compute_ξ(
    p: Dict[Sequence[str], Dict[str, float]],
    q: Dict[Sequence[str], Dict[str, float]],
    ngrams: List[str],
    s: Dict[str, int],
    alphabet: Set[str],
) -> torch.Tensor:

    ξ = np.zeros(len(s))
    for ngram in ngrams:
        ξ[s[ngram]] = -np.sum(
            [p[ngram][y] * np.log2(max(q[ngram][y], 1e-10)) for y in alphabet]
        )
    return torch.tensor(ξ).float()


def _compute_M(
    p: Dict[Sequence[str], Dict[str, float]],
    ngrams: List[str],
    s: Dict[str, int],
    alphabet: Set[str],
) -> torch.Tensor:
    M = np.zeros((len(s), len(s)))
    for ngram in ngrams:
        for y in alphabet:
            if ngram[1:] + (y,) not in p:
                continue
            ngramʹ = ngram[1:] + (y,)
            M[s[ngram], s[ngramʹ]] = p[ngram][y]

    return torch.tensor(M).float()


def expected_length(
    p: Dict[Sequence[str], Dict[str, float]], alphabet: Set[str]
) -> float:
    """Computes the expected length of the strings sampled from the n-gram LM p."""
    n = len(list(p.keys())[0]) + 1
    s = {ngram: idx for idx, ngram in enumerate(p.keys())}
    ngrams = list(p.keys())
    M = _compute_M(p, ngrams, s, alphabet)

    b = torch.zeros(len(p))
    b[s[(("<s>",) * (n - 1))]] = 1
    return torch.linalg.solve((torch.eye(len(p)) - M).T, b).sum()


def entropy(p: Dict[Sequence[str], Dict[str, float]], alphabet: Set[str]) -> float:
    """Computes the entropy of the n-gram LM p."""
    n = len(list(p.keys())[0]) + 1
    s = {ngram: idx for idx, ngram in enumerate(p.keys())}
    ngrams = list(p.keys())
    ξ = _compute_ξ(p, p, ngrams, s, alphabet)
    M = _compute_M(p, ngrams, s, alphabet)

    b = torch.zeros(len(p))
    b[s[(("<s>",) * (n - 1))]] = 1
    return torch.linalg.solve((torch.eye(len(p)) - M).T, b).dot(ξ)


# TODO: This does not work if any of the n-gram LMs are unigram models
def cross_entropy(
    p: Dict[Sequence[str], Dict[str, float]],
    q: Dict[Sequence[str], Dict[str, float]],
    alphabet: Set[str],
) -> float:
    """Computes the cross-entropy of the n-gram LM p with respect to the n-gram LM q."""

    n_p = len(list(p.keys())[0]) + 1
    n_q = len(list(q.keys())[0]) + 1
    n = max(n_p, n_q)
    if n_q < n:
        q = convert_to_larger_n(q, n_q, n, alphabet)
    if n_p < n:
        p = convert_to_larger_n(p, n_p, n, alphabet)

    ngrams = list(set(p.keys()) | set(q.keys()))
    s = {ngram: idx for idx, ngram in enumerate(ngrams)}

    ξ = _compute_ξ(p, q, ngrams, s, alphabet)
    M = _compute_M(p, ngrams, s, alphabet)

    b = torch.zeros(len(p))
    b[s[(("<s>",) * (n - 1))]] = 1
    return torch.linalg.solve((torch.eye(len(p)) - M).T, b).dot(ξ)


def kl_divergence(
    p: Dict[Sequence[str], Dict[str, float]],
    q: Dict[Sequence[str], Dict[str, float]],
    alphabet: Set[str],
    cached_entropy: Optional[float] = None,
    cached_cross_entropy: Optional[float] = None,
    return_entropy: bool = False,
) -> Union[float, Tuple[float, float]]:
    """Computes the KL divergence between the n-gram LMs p and q."""
    if cached_entropy is None:
        cached_entropy = entropy(p, alphabet)
    if cached_cross_entropy is None:
        cached_cross_entropy = cross_entropy(p, q, alphabet)
    if return_entropy:
        return cached_cross_entropy - cached_entropy, cached_entropy
    return cached_cross_entropy - cached_entropy


def empirical_entropy(p: Union[NgramLM, NgramsEstimator], D: List[List[str]]) -> float:
    """Computes the empirical entropy of the data D under the n-gram LM p.
    If D does not come from the same distribution as p, this computes the cross-entropy.
    """
    return -np.mean([p.logprob(d) for d in D])


def empirical_kl_divergence(
    p: Union[NgramLM, NgramsEstimator],
    q: Union[NgramLM, NgramsEstimator],
    D: List[List[str]],
) -> float:
    """Computes the empirical KL divergence between the n-gram LMs p and q
    based on the data D."""
    return empirical_entropy(q, D) - empirical_entropy(p, D)
