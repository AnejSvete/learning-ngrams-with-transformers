from itertools import product
from typing import Dict, List, Sequence


def _set_next_symbol_probabilities(
    p: Dict[Sequence[str], Dict[str, float]],
    pʹ: Dict[Sequence[str], Dict[str, float]],
    n: int,
    nʹ: int,
    ngram: tuple,
    alphabet: Sequence[str],
) -> List[float]:

    pʹ[ngram] = {}
    for a in list(alphabet) + ["</s>"]:
        if n == 1:
            pʹ[ngram][a] = p[()][a]
        else:
            pʹ[ngram][a] = p[ngram[-(n - 1) :]][a]


def convert_to_larger_n(
    p: Dict[Sequence[str], Dict[str, float]], n: int, nʹ: int, alphabet: Sequence[str]
) -> Dict[Sequence[str], Dict[str, float]]:
    """Converts the n-gram LM (in form of a dictionary) to an nʹ-gram LM
    (in form of a dictionary) where nʹ > n by creating additional conditional
    distributions for the larger n-grams.

    Args:
        p (Dict[Sequence[str], Dict[str, float]]): The n-gram LM to convert.
        n (int): The order of the n-gram LM.
        n (int): The order of the n-gram LM to convert to.
        alphabet (Sequence[str]): The alphabet of the n-gram LM.

    Returns:
        Dict[Sequence[str], Dict[str, float]]: The nʹ-gram LM.
    """

    pʹ = {}

    # pre-pad with <s>
    ngram = tuple(["<s>"] * (nʹ - 1))
    _set_next_symbol_probabilities(p, pʹ, n, nʹ, ngram, alphabet)

    # pre-pad with <s>
    for ll in range(nʹ - 1, 0, -1):
        for ngr in product(alphabet, repeat=ll):
            ngram = tuple(["<s>"] * (nʹ - ll - 1) + list(ngr))
            _set_next_symbol_probabilities(p, pʹ, n, nʹ, ngram, alphabet)

    # loop over all possible n-1 grams
    for ngram in product(alphabet, repeat=nʹ - 1):
        _set_next_symbol_probabilities(p, pʹ, n, nʹ, ngram, alphabet)

    return pʹ
