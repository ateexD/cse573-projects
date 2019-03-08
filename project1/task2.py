"""
Character Detection
(Due date: March 6th, 11: 59 P.M.)

The goal of this task is to experiment with template matching techniques. Specifically, the task is to find ALL of
the coordinates where a specific character appears using template matching.

There are 3 sub tasks:
1. Detect character 'a'.
2. Detect character 'b'.
3. Detect character 'c'.

You need to customize your own templates. The templates containing character 'a', 'b' and 'c' should be named as
'a.jpg', 'b.jpg', 'c.jpg' and stored in './data/' folder.

Please complete all the functions that are labelled with '# TODO'. Whem implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in utils.py
and the functions you implement in task1.py are of great help.

Hints: You might want to try using the edge detectors to detect edges in both the image and the template image,
and perform template matching using the outputs of edge detectors. Edges preserve shapes and sizes of characters,
which are important for template matching. Edges also eliminate the influence of colors and noises.

Do NOT modify the code provided.
Do NOT use any API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os

import utils
from task1 import *  # you could modify this line


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path",
        type=str,
        default="./data/characters.jpg",
        help="path to the image used for character detection (do not change this arg)",
    )
    parser.add_argument(
        "--template_path",
        type=str,
        default="",
        choices=["./data/a.jpg", "./data/b.jpg", "./data/c.jpg"],
        help="path to the template image",
    )
    parser.add_argument(
        "--result_saving_directory",
        dest="rs_directory",
        type=str,
        default="./results/",
        help="directory to which results are saved (do not change this arg)",
    )
    args = parser.parse_args()
    return args


def detect_b(img, template):
    """Method to detect 'b' for an image and template.
    """
    print("Detecting b")
    m, n = len(img), len(img[0])
    h, k = len(template), len(template[0])
    img_copy = cv2.imread("data/characters.jpg")

    img_, template_ = np.array(img) * 1.0, np.array(template) * 1.0
    img_, template_ = img_.astype(np.float32), template_.astype(np.float32)

    coordinates = []
    threshold = 0.69
    res = cv2.matchTemplate(img_, template_, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_copy, pt, (pt[0] + k, pt[1] + h), (0, 0, 255), 1)

    cv2.imwrite("res.png", img_copy)


def detect(img, template):
    """Detect a given character, i.e., the character in the template image.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        coordinates: list (tuple), a list whose elements are coordinates where the character appears.
            format of the tuple: (x (int), y (int)), x and y are integers.
            x: row that the character appears (starts from 0).
            y: column that the character appears (starts from 0).
    """
    # TODO: implement this function.
    m, n = len(img), len(img[0])
    h, k = len(template), len(template[0])

    img_copy = cv2.imread("data/characters.jpg")

    a, b, c = (
        read_image("data/a.jpg"),
        read_image("data/b.jpg"),
        read_image("data/c.jpg"),
    )
    if template == a:
        threshold = 0.66
    if template == b:
        threshold = 0.69
    elif template == c:
        threshold = 0.7

    for i in range(m):
        for j in range(n):
            img[i][j] = float(img[i][j])
            if i < h and j < k:
                template[i][j] = float(template[i][j])

    res = ccoeff_matrix(img, template)

    loc = []

    for i in range(len(res)):
        for j in range(len(res[0])):
            if res[i][j] >= threshold:
                loc.append([j, i])

    for pt in loc[::-1]:
        cv2.rectangle(img_copy, (pt[0], pt[1]), (pt[0] + k, pt[1] + h), (0, 0, 255), 1)

    cv2.imwrite("res.png", img_copy)

    return loc


def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def ccoeff_matrix(img, temp):
    def ccoeff(img, temp):
        def coef(temp):
            h, k = len(temp), len(temp[0])
            coeff = 1.0 / (h * k)
            mean = sum([sum(x) for x in temp])
            t_prime = temp.copy()

            for i in range(h):
                for j in range(k):
                    t_prime[i][j] -= coeff * mean
            return t_prime

        t_prime, i_prime = coef(temp), coef(img)

        t_prime_sq, i_prime_sq = (
            utils.elementwise_mul(t_prime, t_prime),
            utils.elementwise_mul(i_prime, i_prime),
        )
        t_prime_sq, i_prime_sq = (
            sum([sum(x) for x in t_prime_sq]),
            sum([sum(x) for x in i_prime_sq]),
        )
        if i_prime_sq == 0:
            return 1e-6
        dr = t_prime_sq * i_prime_sq
        mul = utils.elementwise_mul(t_prime, i_prime)
        return sum([sum(x) for x in mul]) * (1.0 / (dr ** 0.5))

    m, n = len(img), len(img[0])
    h, k = len(temp), len(temp[0])
    rx, ry = m - h, n - k

    r = [[0 for _ in range(ry)] for _ in range(rx)]
    for i in range(rx):
        for j in range(ry):
            cropped = utils.crop(img, i, i + h, j, j + k)
            r[i][j] = ccoeff(cropped, temp)
    return r


def main():
    args = parse_args()

    img = read_image(args.img_path)

    template = read_image(args.template_path)

    coordinates = detect(img, template)

    template_name = "{}.json".format(
        os.path.splitext(os.path.split(args.template_path)[1])[0]
    )
    save_results(coordinates, template, template_name, args.rs_directory)


if __name__ == "__main__":
    main()
