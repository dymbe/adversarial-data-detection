import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA


def visualize_xy(x, y):
    xy = np.concatenate((x, y))

    scaler = preprocessing.StandardScaler()
    scaler.fit(xy)
    xy_scaled = scaler.transform(xy)

    pca = PCA(n_components=2)
    pca.fit(xy_scaled)
    xy_pca = pca.transform(xy_scaled)

    colors = len(x) * ["blue"] + len(y) * ["red"]

    x_mean = xy_pca[:len(x)].mean(axis=0)
    y_mean = xy_pca[len(x):].mean(axis=0)

    means = np.asarray([x_mean, y_mean])

    plt.scatter(xy_pca[:, 0], xy_pca[:, 1], c=colors)
    plt.scatter(means[:, 0], means[:, 1], c=["purple", "orange"])

    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    from compare_images import get_xy

    x, y = get_xy()
    visualize_xy(x, y)
