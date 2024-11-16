import numpy as np

__all__ = ['LSB', 'LSB2']


class _BaseSteganography:
    r"""Base Steganography class."""

    def insert(self, img: np.ndarray) -> np.ndarray:
        r"""Insert steganography key into the image."""
        raise NotImplementedError

    def extract(self, img: np.ndarray, normalized: bool = False) -> np.ndarray:
        r"""Extract steganography key from the image."""
        raise NotImplementedError


# noinspection DuplicatedCode
class LSB(_BaseSteganography):
    r"""LSB for image."""

    def insert(self, img: np.ndarray) -> np.ndarray:
        assert img.dtype == np.uint8
        h, w = img.shape[:2]
        hh, hw = h // 2, w // 2
        out_img = img >> 1 << 1
        out_img[:hh, hw:].__iadd__(1)
        out_img[hh:, :hw].__iadd__(1)
        return out_img

    def extract(self, img: np.ndarray, normalized: bool = False) -> np.ndarray:
        key = img << 7 >> 7
        if normalized:
            key *= 255
        return key


# noinspection DuplicatedCode
class LSB2(_BaseSteganography):
    r"""2 bits LSB for image."""

    def insert(self, img: np.ndarray) -> np.ndarray:
        assert img.dtype == np.uint8
        h, w = img.shape[:2]
        hh, hw = h // 2, w // 2
        out_img = img >> 2 << 2
        out_img[:hh, hw:].__iadd__(0b01)  # 1
        out_img[hh:, :hw].__iadd__(0b10)  # 2
        out_img[hh:, hw:].__iadd__(0b11)  # 3
        return out_img

    def extract(self, img: np.ndarray, normalized: bool = False) -> np.ndarray:
        key = img << 6 >> 6
        if normalized:
            key *= 85
        return key
