#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import matplotlib.pyplot as plt


def main():
    ids = ['1f1cc6b3a4', '5b7c160d0d', '6c40978ddf', '7dfdf6eeb8', '7e5a6e5013']
    plt.figure(figsize=(20, 10))
    for j, img_name in enumerate(ids):
        q = j + 1
        img = Image.open('../input/train/images/' + img_name + '.png')
        img_mask = Image.open('../input/train/masks/' + img_name + '.png')

        plt.subplot(1, 2 * (1 + len(ids)), q * 2 - 1)
        plt.imshow(img)
        plt.subplot(1, 2 * (1 + len(ids)), q * 2)
        plt.imshow(img_mask)
    plt.show()


if __name__ == '__main__':
    main()

