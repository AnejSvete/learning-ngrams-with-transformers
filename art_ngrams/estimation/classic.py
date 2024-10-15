import random
from collections import defaultdict
from itertools import product
from typing import List

import nltk
import numpy as np
import torch
from nltk.lm import (
    MLE,
    AbsoluteDiscountingInterpolated,
    KneserNeyInterpolated,
    Lidstone,
    NgramCounter,
    Vocabulary,
    WittenBellInterpolated,
)
from nltk.lm.preprocessing import flatten, pad_both_ends
from tqdm import trange


class NgramsEstimator:
    def __init__(self, n: int):
        self.n = n

    def _get_ngram_counts(self, D: List[str]):

        # We first pad the data with <s> and </s> tokens.
        # We cut at -(self.n - 2) to only pad with a single </s> token
        # (default is to pad with self.n - 1 </s> tokens).
        if self.n > 2:
            D = [list(pad_both_ends(d, n=self.n))[: -(self.n - 2)] for d in D]
        else:
            D = [list(pad_both_ends(d, n=self.n)) for d in D]

        # Create an alphabet from the data
        alphabet = Vocabulary(flatten(D))
        # NOTE: I wasn't able to get rid of the <UNK> token...

        # Create ngrams (more specifically, everygrams, i.e., k-grams for all k <= n)
        # from the data
        ngrams = [nltk.everygrams(d, max_len=self.n) for d in D]

        # Count the ngrams
        ngram_counts = NgramCounter(ngrams)

        return alphabet, ngram_counts

    def fit(self, D: List[str]):
        # Get the alphabet and the ngram counts
        alphabet, ngram_counts = self._get_ngram_counts(D)
        # Fit the language model according to the specific estimator
        self._fit(alphabet, ngram_counts)

    def _fit(self, alphabet: Vocabulary, ngram_counts: NgramCounter):
        raise NotImplementedError

    def p(self, word: str, context: List[str]) -> float:
        return self.lm.unmasked_score(word, context)

    def logprob(self, y: List[str]) -> float:
        ngram = ("<s>",) * (self.n - 1)

        logprob = 0
        for a in y:
            logprob += np.log2(max(self.p(a, ngram), 1e-10))
            ngram = ngram[1:] + (a,)

        logprob += np.log2(max(self.p("</s>", ngram), 1e-10))

        return logprob

    @property
    def alphabet(self):
        return list(self.vocab)

    def to_dict(self):
        p = defaultdict(lambda: defaultdict(float))
        for ngram in product(self.vocab, repeat=self.n - 1):
            for a in self.vocab:
                p[ngram][a] = self.p(a, ngram)

        return p

    def generate_one(self, to_string: bool = False) -> str:
        string = []
        context = ("<s>",) * (self.n - 1)
        while True:
            p = [self.p(a, context) for a in self.vocab]
            a = self.vocab.lookup(random.choices(list(self.vocab), p)[0])
            if a == "</s>":
                break
            string.append(a)
            context = context[1:] + (a,)
        return "".join(string) if to_string else string

    def generate(self, N: int):
        return [self.generate_one() for _ in trange(N)]


class MLEEstimator(NgramsEstimator):

    def _fit(self, alphabet: Vocabulary, ngram_counts: NgramCounter):
        self.lm = MLE(self.n, vocabulary=alphabet, counter=ngram_counts)
        self.vocab = self.lm.vocab


class AddLambdaEstimator(NgramsEstimator):

    def __init__(self, n: int, gamma: float):
        super().__init__(n)
        self.gamma = gamma

    def _fit(self, alphabet: Vocabulary, ngram_counts: NgramCounter):
        self.lm = Lidstone(
            order=self.n, vocabulary=alphabet, counter=ngram_counts, gamma=self.gamma
        )
        self.vocab = self.lm.vocab


# !DO NOT USE
class AddLambdaEstimatorLearned(NgramsEstimator):

    def __init__(self, n: int):
        super().__init__(n)
        self.gamma = torch.tensor(0.0, requires_grad=True)

    def _fit(self, alphabet: Vocabulary, ngram_counts: NgramCounter):
        self.ngram_counts = ngram_counts
        self.vocab = alphabet

    def freeze(self):
        self.gamma = self.gamma.detach()

    def p(self, word: str, context: List[str]) -> float:
        return (
            self.ngram_counts[self.n][tuple(context)][word] + torch.exp(self.gamma)
        ) / (
            sum(self.ngram_counts[self.n][tuple(context)].values())
            + torch.exp(self.gamma) * len(self.vocab)
        )


class LaplaceEstimator(AddLambdaEstimator):

    def __init__(self, n: int):
        super().__init__(n, 1)


class AbsoluteDiscountingEstimator(NgramsEstimator):

    def __init__(self, n: int, discount: float = 0.0):
        super().__init__(n)
        self.discount = discount
        self._p = dict()

    def _fit(self, alphabet: Vocabulary, ngram_counts: NgramCounter):
        self.lm = AbsoluteDiscountingInterpolated(
            order=self.n,
            discount=self.discount,
            vocabulary=alphabet,
            counter=ngram_counts,
        )
        self.vocab = self.lm.vocab

    def p(self, word: str, context: List[str]) -> float:
        if tuple(context) not in self._p:
            self._p[tuple(context)] = dict()
        if word not in self._p[tuple(context)]:
            self._p[tuple(context)][word] = self.lm.unmasked_score(word, context)
        return self._p[tuple(context)][word]


class WittenBellEstimator(NgramsEstimator):

    def _fit(self, alphabet: Vocabulary, ngram_counts: NgramCounter):
        self.lm = WittenBellInterpolated(
            order=self.n, vocabulary=alphabet, counter=ngram_counts
        )
        self.vocab = self.lm.vocab
        self._p = dict()

    def p(self, word: str, context: List[str]) -> float:
        if tuple(context) not in self._p:
            self._p[tuple(context)] = dict()
        if word not in self._p[tuple(context)]:
            self._p[tuple(context)][word] = self.lm.unmasked_score(word, context)
        return self._p[tuple(context)][word]


class KneserNeyEstimator(NgramsEstimator):

    def __init__(self, n: int, discount: float = 0.0):
        super().__init__(n)
        self.discount = discount
        self._p = dict()

    def _fit(self, alphabet: Vocabulary, ngram_counts: NgramCounter):
        self.lm = KneserNeyInterpolated(
            order=self.n,
            vocabulary=alphabet,
            counter=ngram_counts,
            discount=self.discount,
        )
        self.vocab = self.lm.vocab

    def p(self, word: str, context: List[str]) -> float:
        if tuple(context) not in self._p:
            self._p[tuple(context)] = dict()
        if word not in self._p[tuple(context)]:
            self._p[tuple(context)][word] = self.lm.unmasked_score(word, context)
        return self._p[tuple(context)][word]


class GoodTuringEstimator(NgramsEstimator):

    def __init__(self, n: int):
        super().__init__(n)
        self.counts = None

    def _fit(self, alphabet: Vocabulary, ngram_counts: NgramCounter):
        import simple_good_turing as sgt

        self.vocab = alphabet
        # assert self.applicable(ngram_counts[self.n]), "SGT is not applicable."

        self.est = sgt.Estimator(self._get_count_counts(ngram_counts))

    def _get_count_counts(self, ngram_counts: NgramCounter):
        if self.counts is None:
            self._set_counts(ngram_counts)

        _count_counts = np.unique(list(self.counts.values()), return_counts=True)
        count_counts = defaultdict(int)
        count_counts.update(dict(zip(*_count_counts)))
        count_counts[0] = 0

        return count_counts

    def _set_counts(self, ngram_counts: NgramCounter):
        self.counts = {}
        for ngram in ngram_counts[self.n]:
            self.counts[ngram] = sum(ngram_counts[self.n][ngram].values())

            for sym in ngram_counts[self.n][ngram]:
                self.counts[ngram + (sym,)] = ngram_counts[self.n][ngram][sym]

    def applicable(self, D: List[str]):
        import simple_good_turing as sgt

        def _regress(Z):
            # Make a set of the nonempty points in log scale
            x, y = zip(
                *[(np.log(r), np.log(Z[r])) for r in range(1, max_r + 1) if Z[r]]
            )
            self.x, self.y = x, y
            matrix = np.array((x, np.ones(len(x)))).T
            return np.linalg.lstsq(matrix, y, rcond=None)[0]

        _, ngram_counts = self._get_ngram_counts(D)
        count_counts = self._get_count_counts(ngram_counts)
        max_r = max(count_counts.keys())

        count_counts[0] = sum(count_counts[r] * r for r in range(1, max_r + 1))
        Z = sgt.averaging_transform.transform(count_counts, max_r)
        b, _ = _regress(Z)
        return b < -1

    def p(self, word: str, context: List[str]) -> float:
        return self.est.p(self.counts.get(context + (word,), 0)) / self.est.p(
            self.counts.get(context, 0)
        )


class JelinekMercerEstimator(NgramsEstimator):

    def __init__(self, n: int, factor: float):
        super().__init__(n)
        self.factor = factor

    def _fit(self, alphabet: Vocabulary, ngram_counts: NgramCounter):
        self.vocab = alphabet
        self.lms = {}
        for i in range(self.n):
            self.lms[i] = MLE(i + 1, vocabulary=alphabet, counter=ngram_counts)
        self.lambdas = self._init_lambdas()

    def _init_lambdas(self):
        lambdas = [self.factor**k for k in range(self.n)]
        lambdas = [lb / sum(lambdas) for lb in lambdas]
        return lambdas

    def pJM(self, word: str, context: List[str], k: int) -> float:
        if k == 0:
            return self.lambdas[k] * self.lms[k].unmasked_score(word, context) + (
                1 - self.lambdas[k]
            ) * 1 / len(self.vocab)
        else:
            return self.lambdas[k] * self.lms[k].unmasked_score(word, context) + (
                1 - self.lambdas[k]
            ) * self.pJM(word, context, k - 1)

    def p(self, word: str, context: List[str]) -> float:
        return self.pJM(word, context, self.n - 1)
