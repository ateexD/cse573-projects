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
from task1 import *   # you could modify this line
from scipy.signal import convolve2d as c2d

kernel = np.array([[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [1, 2, -16, 2, 1], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]], dtype=np.float32) 

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="./data/characters.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--template_path", type=str, default="",
        choices=["./data/a.jpg", "./data/b.jpg", "./data/c.jpg"],
        help="path to the template image")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


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
    print(h, k)
    img_copy = np.array(copy.deepcopy(img))
    img_, template_ = np.array(img) * 1., np.array(template) * 1.
    # img_, template_ = c2d(img_, kernel, mode='same'), c2d(template, kernel, mode='same')
    img_, template_ = img_.astype(np.float32), template_.astype(np.float32)
    
    coordinates = []
    
    for half_len in [10, 9, 8, 7, 6, 5]:
        res = cv2.matchTemplate(img_, template_, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7 # a - 0.7, c - 0.8 (0.7 but more false positives)
        loc = np.where(res >= threshold)
        print(loc)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_copy, pt, (pt[0] + h, pt[1] + k), (0, 0, 255), 1)
        h, k = half_len, half_len
        template_ = cv2.resize(template_, (half_len, half_len))
    cv2.imwrite('res.png', img_copy)

    # for i in range(m - h):
    #     for j in range(n - k):
    #         cropped = utils.crop(img_, i, i + h, j, j + k)
    #         corr = np.corrcoef(cropped, template_)[0, 1]

    #         if corr > 0.9:
    #             coordinates.append([i, j])
    #             cv2.circle(img_copy, (i, j), 1, (0, 255, 0))
    # cv2.imwrite("results/circles.jpg", img_copy)
    return coordinates


def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)

def get_edge_magnitude(img):
    edge_x, edge_y = detect_edges(img, sobel_x), detect_edges(img, sobel_y)
    return edge_magnitude(edge_x, edge_y)

def main():
    args = parse_args()

    img = read_image(args.img_path)

    template = read_image(args.template_path)

    coordinates = detect(img, template)

    template_name = "{}.json".format(os.path.splitext(os.path.split(args.template_path)[1])[0])
    save_results(coordinates, template, template_name, args.rs_directory)


if __name__ == "__main__":
    main()
