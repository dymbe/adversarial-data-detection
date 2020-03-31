import os

import numpy as np
import random
from matplotlib import image

from kernel_two_sample_test import kernel_two_sample_test

if __name__ == "__main__":
    benign_path = "data/benign"
    adverserial_path = "data/adverserial"

    x_file_paths = [f"{benign_path}/{x}" for x in os.listdir(benign_path)]
    x_file_paths = random.sample(x_file_paths, 100)
    y_file_paths = [f"{adverserial_path}/{y}" for y in os.listdir(adverserial_path) if y not in x_file_paths]
    y_file_paths = random.sample(y_file_paths, 17)

    X = np.asarray([image.imread(path).flatten() for path in x_file_paths])
    Y = np.asarray([image.imread(path).flatten() for path in y_file_paths])

    mmd2u, mmd2u_null, p_value = kernel_two_sample_test(X, Y)
    print(f"mmd2u: {mmd2u}")
    print(f"mmd2u_null: {mmd2u_null}")
    print(f"p_value: {p_value}")
