from typing import List, Union

import numpy as np

from art_ngrams.estimation.classic import NgramsEstimator
from art_ngrams.lm_generation.ngram import NgramLM


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
