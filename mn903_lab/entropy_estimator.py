import numpy as np

__all__ = ['EntropyEstimator']


class EntropyEstimator:
    def __init__(self, x):
        if not len(x):
            raise ValueError('text must not be empty')
        self.source = np.asarray(list(x))
        self.alphabet = self.cnts = None
        self._alphabet = None
        self.ps = None
        self.P = self.T = None

    def __len__(self):
        return self.source.shape[0]

    @property
    def alphabet_size(self):
        return self.alphabet.shape[0]

    def alphabet_index(self, c: str) -> int:
        return self._alphabet.index(c)

    def _build_alphabet(self):
        if self.alphabet is None:
            self.alphabet, self.cnts = np.unique(self.source, return_counts=True)
            self._alphabet = self.alphabet.tolist()

    def _compute_zero_order_stats(self):
        self._build_alphabet()
        if self.ps is None:
            self.ps = self.cnts / len(self)

    def _compute_first_order_stats(self):
        self._build_alphabet()
        if self.P is None:
            # joint probability matrix
            self.P = np.zeros((self.alphabet_size, self.alphabet_size))
            for i in range(len(self) - 1):
                k = self.alphabet_index(self.source[i])
                l = self.alphabet_index(self.source[i + 1])
                self.P[k, l] += 1
            self.P /= len(self)
        if self.T is None:
            # conditional probability matrix (transition matrix)
            if self.ps is not None:
                self.T = self.P / self.ps.reshape(1, -1)
            else:
                self.T = self.P / self.P.sum(1, keepdims=True)

    def entropy(self):
        r"""return zeroth-order entropy from text.
        """
        self._compute_zero_order_stats()
        return - (self.ps * np.log2(self.ps)).sum()

    def entropy1(self):
        r"""return first-order entropy from text.
        """
        self._compute_first_order_stats()
        eps = np.finfo(self.T.dtype).eps * 4
        H1 = 0
        for i in range(self.alphabet_size):
            H1 += -sum(self.P[i, :] * np.log2(self.T[i, :] + eps))
        return H1
