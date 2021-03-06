import copy

import cv2


def zero_pad(img, pwx, pwy):
    padded_img = copy.deepcopy(img)
    for i in range(pwx):
        padded_img.insert(0, [0 for value in enumerate(padded_img[i])])
        padded_img.insert(len(padded_img), [0 for value in enumerate(padded_img[-1])])
    for i, row in enumerate(padded_img):
        for j in range(pwy):
            row.insert(0, 0)
            row.insert(len(row), 0)
    return padded_img

def crop(img, xmin, xmax, ymin, ymax):
    if len(img) < xmax:
        print('WARNING')
    patch = img[xmin: xmax]
    patch = [row[ymin: ymax] for row in patch]
    return patch

def elementwise_add(a, b):
    c = copy.deepcopy(a)
    for i, row in enumerate(a):
        for j, num in enumerate(row):
            c[i][j] += b[i][j]
    return c

def elementwise_sub(a, b):
    c = copy.deepcopy(a)
    for i, row in enumerate(a):
        for j, num in enumerate(row):
            c[i][j] -= b[i][j]
    return c

def elementwise_mul(a, b):
    c = copy.deepcopy(a)
    for i, row in enumerate(a):
        for j, num in enumerate(row):
            c[i][j] *= b[i][j]
    return c

def elementwise_div(a, b):
    c = copy.deepcopy(a)
    for i, row in enumerate(a):
        for j, num in enumerate(row):
            c[i][j] /= b[i][j]
    return c

def flip_x(img):
    flipped_img = copy.deepcopy(img)
    center = int(len(img) / 2)
    for i in range(center):
        flipped_img[i] = img[(len(img) - 1) - i]
        flipped_img[(len(img) - 1) - i] = img[i]
    return flipped_img

def flip_y(img):
    flipped_img = copy.deepcopy(img)
    center = int(len(img[0]) / 2)
    for i, row in enumerate(img):
        for j in range(center):
            flipped_img[i][j] = img[i][(len(img[0]) - 1) - j]
            flipped_img[i][(len(img[0]) - 1) - j] = img[i][j]
    return flipped_img

def flip2d(img, axis=None):
    """Flips an image along a given axis.

    Hints:
        Use the function flip_x and flip_y.

    Args:
        img: nested list (int), the image to be flipped.
        axis (int or None): the axis along which img is flipped.
            if axix is None, img is flipped both along x axis and y axis.

    Returns:
        flipped_img: nested list (int), the flipped image.
    """
    # TODO: implement this function.
    if axis == 1:
        return flip_x(img)
    if axis == 2:
        return flip_y(img)
    if axis is None:
        return flip_x(flip_y(img))
    return None
    # raise NotImplementedError

def imresize(img):
    """Resizes m x n image to (m - 1) x (n - 1) image
    """
    m, n = len(img), len(img[0])
    new_img = [[0 for _ in range(n - 1)] for _ in range(m - 1)]
    for i in range(m - 1):
        for j in range(n - 1):
            pixels = img[i][j], img[i][j + 1], img[i + 1][j], img[i + 1][j + 1]
            # new_img[i][j] = sum(pixels) / 4.0
            new_img[i][j] = sum(sorted(pixels)[1: 3]) / 2.0


    return new_img
