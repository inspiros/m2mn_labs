import numpy as np
from typing import Dict, Sequence, Optional, Union

__all__ = ['CAESAR', 'SimpleSubstitution']


class _BaseCipher:
    r"""Base Cipher class."""

    def encrypt(self, img: np.ndarray) -> np.ndarray:
        r"""Encrypts the image."""
        raise NotImplementedError

    def decrypt(self, img: np.ndarray) -> np.ndarray:
        r"""Decrypts the image."""
        raise NotImplementedError


# noinspection DuplicatedCode
class CAESAR(_BaseCipher):
    r"""CAESAR for image."""

    def __init__(self, key: Optional[int] = None):
        if key is not None:
            self.key = int(key) % 256
        else:
            self.key = np.random.randint(0, 256, 1, dtype=np.uint8).item()

    def encrypt(self, img: np.ndarray) -> np.ndarray:
        assert img.dtype == np.uint8
        return img + self.key

    def decrypt(self, img: np.ndarray) -> np.ndarray:
        assert img.dtype == np.uint8
        return img - self.key


# noinspection DuplicatedCode
class SimpleSubstitution(_BaseCipher):
    r"""Simple Substitution Cipher for image."""

    def __init__(self, key: Optional[Union[Sequence[int], Dict[int, int]]] = None):
        if key:
            if isinstance(key, Dict):
                assert len(key.keys()) == 256
                self.key = np.empty(256, dtype=np.uint8)
                for i in range(256):
                    self.key[i] = key[i]
            else:
                self.key = np.asarray(key, dtype=np.uint8)
                assert np.unique(self.key).size == 256
        else:
            self.key = np.random.permutation(np.arange(256, dtype=np.uint8))
        self.inverse_key = np.empty_like(self.key)
        for i in range(256):
            self.inverse_key[self.key[i]] = i

    def encrypt(self, img: np.ndarray) -> np.ndarray:
        assert img.dtype == np.uint8
        out_img = img.copy()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                out_img[i, j] = self.key[out_img[i, j]]
        return out_img

    def decrypt(self, img: np.ndarray) -> np.ndarray:
        assert img.dtype == np.uint8
        out_img = img.copy()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                out_img[i, j] = self.inverse_key[out_img[i, j]]
        return out_img
