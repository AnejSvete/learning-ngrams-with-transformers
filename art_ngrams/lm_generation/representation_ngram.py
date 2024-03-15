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
        method: str = "one-hot",
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
            method (str, optional): The method to use for representing the model.
                Can be 'one-hot' or 'log', 'rank', or 'vectorial'.
                'one-hot' builds a `|Σ| x |Σ|(n - 1)` matrix of probabilities, 'log'
                builds a `|Σ| x log(|Σ|)(n - 1)` matrix, and 'rank' builds a
                `|Σ| x |Σ|^{n-1}` matrix of rank r. 'vectorial' builds a
                `|Σ| x (n - 1)D` matrix of rank `D`.
                Defaults to 'one-hot'.
            filename (Optional[str], optional): Filename of an existing model to read.
                Defaults to None.
            seed (Optional[int], optional): The seed of the random generation.
                Defaults to None.
        """

        if filename is not None:
            self.load(filename)
        else:
            assert method in ["one-hot", "log", "rank", "vectorial"], "Invalid method"
            self.n = n
            self.alphabet = list(alphabet)
            self.EOSalphabet = list(set(self.alphabet + ["</s>"]))
            self.BOSalphabet = list(set(self.alphabet + ["<s>"]))
            self.BOSEOSalphabet = list(set(self.alphabet + ["<s>", "</s>"]))
            self.mean_length = mean_length
            self.method = method
            self.rank = rank
            self.D = D
            self.rng = np.random.default_rng(seed)
            self.p = dict()
            self.setup()

    def _construct_E(self, D: int) -> np.ndarray:
        E = self.rng.normal(size=(len(self.EOSalphabet), D))
        return E

    def _construct_one_hot_matrix(self):
        self.E = self._construct_E(len(self.BOSalphabet) * (self.n - 1))

    def construct_log_matrix(self):
        self.E = self._construct_E(
            (int(np.ceil(np.log2(len(self.BOSalphabet))))) * (self.n - 1)
        )

    def construct_rank_matrix(self):
        self.E1 = np.abs(self.rng.normal(size=(len(self.BOSalphabet), self.rank)))
        self.E2 = self.rng.normal(
            size=(self.rank, len(self.EOSalphabet) ** (self.n - 1))
        )

    def construct_vector_matrix(self):
        self.E1 = np.abs(self.rng.normal(size=(len(self.BOSalphabet), self.rank)))
        self.E2 = self.rng.normal(size=(self.rank, self.D * (self.n - 1)))

    def _setup_encoding(self):
        self.in_sym2idx = {s: i for i, s in enumerate(self.BOSalphabet)}
        self.in_idx2sym = {i: s for s, i in self.in_sym2idx.items()}
        self.out_sym2idx = {s: i for i, s in enumerate(self.EOSalphabet)}
        self.out_idx2sym = {i: s for s, i in self.out_sym2idx.items()}
        self.idx2sym = {s: i for i, s in enumerate(self.BOSEOSalphabet)}
        self.sym2idx = {i: s for s, i in self.idx2sym.items()}
        self.sym2vec = {s: self.rng.normal(size=(self.D,)) for s in self.BOSalphabet}
        if self.method == "rank":
            self.ngram2idx = {
                ngram: i
                for i, ngram in enumerate(product(self.BOSalphabet, repeat=self.n - 1))
            }
        if self.method == "log":
            G = int(np.ceil(np.log2(len(self.BOSEOSalphabet))))
            self.idx2bin = {
                idx: [int(b) for b in np.binary_repr(int(idx), G)]
                for idx in self.idx2sym
            }

    def encoding(self, ngram: str) -> np.ndarray:
        if self.method == "one-hot":
            enc = np.zeros((len(self.BOSalphabet), self.n - 1))
            ixs = [self.in_sym2idx[a] for a in ngram]
            enc[ixs, np.arange(self.n - 1)] = 1
            enc = enc.flatten()
        elif self.method == "log":
            enc = np.asarray(
                [self.idx2bin[self.in_sym2idx[a]] for a in ngram]
            ).flatten()
        elif self.method == "rank":
            enc = np.zeros(len(self.BOSalphabet) ** (self.n - 1))
            enc[self.ngram2idx[ngram]] = 1
        elif self.method == "vectorial":
            enc = np.concatenate([self.sym2vec[s] for s in ngram])

        return enc

    def setup(self):
        if self.method == "one-hot":
            self._construct_one_hot_matrix()
        elif self.method == "log":
            self.construct_log_matrix()
        elif self.method == "rank":
            self.construct_rank_matrix()
        elif self.method == "vectorial":
            self.construct_vector_matrix()
        else:
            raise ValueError("Invalid method")

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
            if self.method in ["rank", "vectorial"]:
                logits = self.E1 @ (self.E2 @ enc)
            else:
                logits = self.E @ enc
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

    # def p_cond(self, a: str, ngram: Sequence[str]) -> float:
    #     enc = self.encoding(ngram)
    #     if self.method in ["rank", "vectorial"]:
    #         logits = self.E1 @ (self.E2 @ enc)
    #     else:
    #         logits = self.E @ enc
    #     logits[self.out_sym2idx["</s>"]] = -np.inf
    #     p_ngram = softmax(logits)
    #     p_ngram[self.out_sym2idx["</s>"]] = 1 / (self.mean_length - 1)
    #     p_ngram = p_ngram / (1 + 1 / (self.mean_length - 1))
    #     return p_ngram[self.out_sym2idx[a]]

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
            "method": self.method,
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

        if self.method in ["rank", "vectorial"]:
            data.update({"E1": self.E1, "E2": self.E2})
        else:
            data.update({"E": self.E})

        if self.method == "rank":
            data.update({"ngram2idx": self.ngram2idx})

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
            self.method = data["method"]
            self.in_sym2idx = data["in_sym2idx"]
            self.in_idx2sym = data["in_idx2sym"]
            self.out_sym2idx = data["out_sym2idx"]
            self.out_idx2sym = data["out_idx2sym"]
            # self.mean_length = data["mean_length"]
            self.sym2idx = data["sym2idx"]
            self.idx2sym = data["idx2sym"]
            self.EOSalphabet = data["EOSalphabet"]
            self.BOSalphabet = data["BOSalphabet"]

            if self.method in ["rank", "vectorial"]:
                self.E = data["E"]
            else:
                self.E1 = data["E1"]
                self.E2 = data["E2"]

            if self.method == "rank":
                self.ngram2idx = data["ngram2idx"]
