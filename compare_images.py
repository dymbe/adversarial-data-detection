import os

import numpy as np
import random
from matplotlib import image
from kernel_two_sample_test import kernel_two_sample_test
from clustering import visualize_xy


def get_xy():
    non_adversarial_path = "data/non-adversarial/bbc/images"
    adversarial_path = "data/adversarial"
    # adversarial_path = non_adversarial_path

    x_file_paths = [f"{non_adversarial_path}/{x}" for x in os.listdir(non_adversarial_path)]
    x_file_paths = random.sample(x_file_paths, 20)

    x_files = [path[len(non_adversarial_path) + 1:] for path in x_file_paths]

    y_file_paths = [f"{adversarial_path}/{y}" for y in os.listdir(adversarial_path) if y not in x_files]
    y_file_paths = random.sample(y_file_paths, 20)

    x = np.asarray([image.imread(path).flatten() for path in x_file_paths])
    y = np.asarray([image.imread(path).flatten() for path in y_file_paths])

    return x, y


if __name__ == "__main__":
    X, Y = get_xy()

    mmd2u, mmd2u_null, p_value = kernel_two_sample_test(X, Y, verbose=True)
    print(f"mmd2u: {mmd2u}")
    print(f"mmd2u_null: {mmd2u_null}")
    print(f"p_value: {p_value}")

    visualize_xy(X, Y)
