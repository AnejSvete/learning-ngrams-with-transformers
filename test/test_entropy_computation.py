import random
from math import log2

from art_ngrams.evaluation.metrics import entropy
from art_ngrams.lm_generation.ngram import NgramLM


def test_entropy():
    q = max(0.15, random.random())
    p = {
        ("<s>",): {"a": q, "</s>": 1 - q},
        ("a",): {"a": q, "</s>": 1 - q},
    }

    pLM = NgramLM(p=p)

    ent = sum(-(q**t * (1 - q)) * log2(q**t * (1 - q)) for t in range(0, 100))

    assert abs(entropy(pLM.to_dict(), pLM.alphabet) - ent) < 1e-10
