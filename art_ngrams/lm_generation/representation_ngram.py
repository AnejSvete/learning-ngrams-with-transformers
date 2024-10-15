import pickle
from collections import defaultdict
from itertools import product
from typing import List, Optional, Sequence, Union

import numpy as np
from scipy.special import softmax
from tqdm import trange


class NgramLM:
    def __init__(
        self,
        n: Optional[int] = None,
        alphabet: Optional[str] = None,
        mean_length: float = 10,
        rank: int = 1,
        D: int = 8,
        filename: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Initializes an n-gram language model.

        Args:
            n (Optional[int], optional): The order of the n-gram LM. Defaults to None.
            alphabet (Optional[str], optional): The alphabet of the LM.
                Defaults to None.
            mean_length (float, optional): The target mean length of the strings
                to generate.
            filename (Optional[str], optional): Filename of an existing model to read.
                Defaults to None.
            seed (Optional[int], optional): The seed of the random generation.
                Defaults to None.
        """

        if filename is not None:
            self.load(filename)
        else:
            self.n = n
            self.alphabet = list(alphabet)
            self.EOSalphabet = list(set(self.alphabet + ["</s>"]))
            self.BOSalphabet = list(set(self.alphabet + ["<s>"]))
            self.BOSEOSalphabet = list(set(self.alphabet + ["<s>", "</s>"]))
            self.mean_length = mean_length
            self.rank = rank
            self.D = D
            self.rng = np.random.default_rng(seed)
            self.p = dict()
            self.setup()

    def construct_vector_matrix(self):
        self.E1 = self.rng.normal(size=(len(self.BOSalphabet), self.rank))
        self.E2 = self.rng.normal(size=(self.rank, self.D * (self.n - 1)))

    def _setup_encoding(self):
        self.in_sym2idx = {s: i for i, s in enumerate(self.BOSalphabet)}
        self.in_idx2sym = {i: s for s, i in self.in_sym2idx.items()}
        self.out_sym2idx = {s: i for i, s in enumerate(self.EOSalphabet)}
        self.out_idx2sym = {i: s for s, i in self.out_sym2idx.items()}
        self.idx2sym = {s: i for i, s in enumerate(self.BOSEOSalphabet)}
        self.sym2idx = {i: s for s, i in self.idx2sym.items()}
        self.sym2vec = {s: self.rng.normal(size=(self.D,)) for s in self.BOSalphabet}

    def encoding(self, ngram: str) -> np.ndarray:
        enc = np.concatenate([self.sym2vec[s] for s in ngram])

        return enc

    def setup(self):
        self.construct_vector_matrix()
        self._setup_encoding()

    def to_dict(self):
        p = defaultdict(lambda: defaultdict(float))
        for ngram in product(self.BOSalphabet, repeat=self.n - 1):
            for a in self.BOSalphabet:
                p[ngram][a] = self.p_cond(a, ngram)

        return p

    def p_cond(self, a: str, ngram: Sequence[str]) -> float:
        if ngram not in self.p:
            enc = self.encoding(ngram)
            logits = self.E1 @ (self.E2 @ enc)
            logits[self.out_sym2idx["</s>"]] = -np.inf
            p_ngram = softmax(logits)
            p_ngram[self.out_sym2idx["</s>"]] = 1 / (self.mean_length - 1)
            p_ngram = p_ngram / (1 + 1 / (self.mean_length - 1))
            if len(self.p) < 1e5:
                self.p[ngram] = {
                    s: p_ngram[self.out_sym2idx[s]] for s in self.EOSalphabet
                }
            else:
                return p_ngram[self.out_sym2idx[a]]
        return self.p[ngram][a]

    def logprob(self, y: Union[str, List[str]]) -> float:
        """Evaluates the log probability of a single string.

        Args:
            y (Union[str, List[str]]): The string to evaluate.

        Returns:
            float: The log probability of the string.
        """
        ngram = ("<s>",) * (self.n - 1)

        logprob = 0
        for a in y:
            logprob += np.log2(self.p_cond(a, ngram))
            ngram = ngram[1:] + (a,)

        logprob += np.log2(self.p_cond("</s>", ngram))

        return logprob

    def _sample_lm(self, to_string: bool = False) -> str:
        ngram = tuple(["<s>"] * (self.n - 1))
        string = list()
        while True:
            p = [self.p_cond(a, ngram) for a in self.EOSalphabet]
            a = self.rng.choice(self.EOSalphabet, p=p)
            if a == "</s>":
                break
            string.append(a)
            ngram = tuple(list(ngram[1:]) + [a])
        return " ".join(string) if to_string else string

    def sample(self, N: int = 1, to_string: bool = False) -> List[str]:
        return [self._sample_lm(to_string) for _ in trange(N)]

    def expected_length(self) -> float:
        assert not self.binary, "Expected length is only defined for LM models."
        from art_ngrams.evaluation import metrics

        return metrics.expected_length(self.to_dict(), set(self.alphabet))

    def entropy(self) -> float:
        assert not self.binary, "Entropy is only defined for LM models."
        from art_ngrams.evaluation import metrics

        return metrics.entropy(self.to_dict(), set(self.alphabet))

    def cross_entropy(self, other: "NgramLM") -> float:
        assert not self.binary, "Cross entropy is only defined for LM models."
        from art_ngrams.evaluation import metrics

        return metrics.cross_entropy(
            self.to_dict(), other.to_dict(), set(self.alphabet)
        )

    def kl_divergence(self, other: "NgramLM") -> float:
        assert not self.binary, "KL divergence is only defined for LM models."
        from art_ngrams.evaluation import metrics

        return metrics.kl_divergence(
            self.to_dict(), other.to_dict(), set(self.alphabet)
        )

    def save(self, filename: str):
        data = {
            "n": self.n,
            "alphabet": self.alphabet,
            "rank": self.rank,
            "D": self.D,
            "mean_length": self.mean_length,
            "in_sym2idx": self.in_sym2idx,
            "in_idx2sym": self.in_idx2sym,
            "out_sym2idx": self.out_sym2idx,
            "out_idx2sym": self.out_idx2sym,
            "idx2sym": self.idx2sym,
            "sym2idx": self.sym2idx,
            "EOSalphabet": self.EOSalphabet,
            "BOSalphabet": self.BOSalphabet,
        }

        data.update({"E1": self.E1, "E2": self.E2})

        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load(self, filename: str):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.p = dict()
            self.n = data["n"]
            self.alphabet = data["alphabet"]
            self.rank = data["rank"]
            self.D = data["D"]
            self.in_sym2idx = data["in_sym2idx"]
            self.in_idx2sym = data["in_idx2sym"]
            self.out_sym2idx = data["out_sym2idx"]
            self.out_idx2sym = data["out_idx2sym"]
            self.sym2idx = data["sym2idx"]
            self.idx2sym = data["idx2sym"]
            self.EOSalphabet = data["EOSalphabet"]
            self.BOSalphabet = data["BOSalphabet"]

            self.E1 = data["E1"]
            self.E2 = data["E2"]
