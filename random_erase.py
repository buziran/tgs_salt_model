# -*- coding: utf-8 -*-

"""
Implementation of Random Erasing for Keras
"""

import numpy as np

class RandomErasing(object):
    def __init__(self, probability=0.5, min_size=0.02, max_size=0.4, min_aspect_ratio=0.3, max_aspect_ratio=1/0.3, min_val=0, max_val=256, pixel_wise=False, seed=None):
        self.probability = probability
        self.min_size = min_size
        self.max_size = max_size
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_val = min_val
        self.max_val = max_val
        self.pixel_wise = pixel_wise
        self.seed = seed
        self.cnt = 0

    def __call__(self, img):
        if self.seed is not None:
            np.random.seed(self.seed + self.cnt)
        self.cnt += 1

        rank = len(img.shape)

        if rank == 3:
            height, width, channels = img.shape
        elif rank == 2:
            height, width = img.shape
        else:
            raise ValueError("image rank must be 2 or 3")

        p = np.random.rand()

        if p > self.probability:
            return img

        while True:
            s = np.random.uniform(self.min_size, self.max_size) * height * width
            r = np.random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, width)
            top = np.random.randint(0, height)

            if left + w <= width and top + h <= height:
                break

        if self.pixel_wise:
            if rank == 3:
                size = (h,w,channels)
            elif rank == 2:
                size = (h,w)
        else:
            size = None

        if np.issubdtype(img.dtype, np.integer):
            random_fn = np.random.randint
        elif np.issubdtype(img.dtype, np.floating):
            random_fn = np.random.uniform
        else:
            raise NotImplementedError()

        c = random_fn(self.min_val, self.max_val, size)

        erased_img = np.copy(img)
        erased_img[top:top + h, left:left + w, :] = c

        return erased_img


def main():
    import matplotlib.pyplot as plt
    import cv2
    orig_im = cv2.imread('../input/train/images/000e218f21.png')
    plt.imshow(orig_im)
    plt.title('original')
    plt.show()

    random_erase = RandomErasing()
    for i in range(2):
        im = random_erase(orig_im)
        plt.imshow(im)
        plt.title('random erase (case-1, cnt={})'.format(i))
        plt.show()

    random_erase2 = RandomErasing(seed=11)
    for i in range(2):
        im = random_erase2(orig_im)
        plt.imshow(im)
        plt.title('random erase (case-2, cnt={})'.format(i))
        plt.show()

    random_erase3 = RandomErasing(seed=11, pixel_wise=True)
    for i in range(2):
        im = random_erase3(orig_im)
        plt.imshow(im)
        plt.title('random erase (case-3, cnt={})'.format(i))
        plt.show()

if __name__ == '__main__':
    main()


