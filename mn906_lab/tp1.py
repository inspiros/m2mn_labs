import os

import PIL.Image
import cv2

from image_cipher import *
from image_steganography import *


def example_caesar():
    cipher = CAESAR()
    for test_img in ['lena', 'baboon']:
        img = cv2.imread(f'resources/{test_img}.jpg', cv2.IMREAD_GRAYSCALE)
        img_caesar = cipher.encrypt(img)
        img_recovered = cipher.decrypt(img_caesar)
        cv2.imshow('img', img)
        cv2.imshow('img_caesar', img_caesar)
        cv2.imshow('img_recovered', img_recovered)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def example_simple_sub():
    cipher = SimpleSubstitution()
    for test_img in ['lena', 'baboon']:
        img = cv2.imread(f'resources/{test_img}.jpg', cv2.IMREAD_GRAYSCALE)
        img_ss = cipher.encrypt(img)
        img_recovered = cipher.decrypt(img_ss)
        cv2.imshow('img', img)
        cv2.imshow('img_simple_sub', img_ss)
        cv2.imshow('img_recovered', img_recovered)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def example_lsb():
    os.makedirs('outputs', exist_ok=True)
    stega = LSB()
    for test_img in ['lena', 'baboon']:
        img = cv2.imread(f'resources/{test_img}.jpg', cv2.IMREAD_GRAYSCALE)
        img_lsb = stega.insert(img)
        key = stega.extract(img_lsb, normalized=True)
        for q in [100, 99, 75]:
            PIL.Image.fromarray(img_lsb).save(f'outputs/{test_img}_lsb_{q}.jpg', format='JPEG', compress_level=q)
            # cv2.imwrite(f'outputs/{test_img}_lsb_{q}.png', {test_img}_lsb, [cv2.IMWRITE_PNG_COMPRESSION, q])
        cv2.imshow('img', img)
        cv2.imshow('img_lsb', img_lsb)
        cv2.imshow('key', key)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def example_lsb2():
    os.makedirs('outputs', exist_ok=True)
    stega = LSB2()
    for test_img in ['lena', 'baboon']:
        img = cv2.imread(f'resources/{test_img}.jpg', cv2.IMREAD_GRAYSCALE)
        img_lsb = stega.insert(img)
        key = stega.extract(img_lsb, normalized=True)
        for q in [100, 99, 75]:
            PIL.Image.fromarray(img_lsb).save(f'outputs/{test_img}_lsb_{q}.jpg', format='JPEG', compress_level=q)
            # cv2.imwrite(f'outputs/{test_img}_lsb_{q}.png', {test_img}_lsb, [cv2.IMWRITE_PNG_COMPRESSION, q])
        cv2.imshow('img', img)
        cv2.imshow('img_lsb', img_lsb)
        cv2.imshow('key', key)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    example_caesar()
    example_simple_sub()
    example_lsb()
    example_lsb2()
