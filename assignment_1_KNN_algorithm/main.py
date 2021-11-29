"""
main.py
https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial.distance import euclidean
from scipy import stats
import numpy as np


def calculate_distances(data, label, point):
    """
    Calculate for given point the euclidean distance to the other N-1 samples
    :param label: target type of flower
    :param data: x/y data
    :param point: integer, sample [1-N]
    :return: distance from point to the other N-1 points
    """
    size = np.size(data, 0)
    x0, y0 = data[point, 0], data[point, 1]
    distance = np.zeros((size, 2))

    for idx, sample in enumerate(data):
        dis = euclidean([x0, y0], [data[idx, 0], data[idx, 1]])
        distance[idx][0] = label[idx]
        distance[idx][1] = dis

    # delete point we are trying to classify from train set
    distance = np.delete(distance, point, 0)

    return distance


def visualize_data(data, label):
    """
    :param data: x/y data
    :param label: labels
    """
    x_min = data[:, 0].min() - 0.5
    x_max = data[:, 0].max() + 0.5
    y_min = data[:, 1].min() - 0.5
    y_max = data[:, 1].max() + 0.5

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    # Plot the training points
    plt.scatter(data[:, 0], data[:, 1], c=label, cmap=plt.cm.Set1, edgecolor="black")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("red=setosa, orange=versicolor, grey=virginica")

    plt.show()


def find_k_nearest_neighbors(distances, k):
    """
    :param k: How many neighbor points K we wish to include
    :param distances: distance from point to the other N-1 points
    :return Target labels of K nearest neighbors (K-NN)
    """
    distances = distances[distances[:, 1].argsort()]
    k_nn = np.delete(distances[0:k], 1, 1)
    print(k_nn)
    return k_nn


def app_main():
    """ App main """
    iris = datasets.load_iris()
    data = iris.data[:, :2]  # take first 2 features (sepal length/width)
    label = iris.target

    # visualize_data(data, label)

    # ID of point we want to test
    point = 115
    # the K in K-NN
    k = 3

    print("Correct label point {0} = {1}".format(point, label[point]))

    distances = calculate_distances(data, label, point)
    k_nn = find_k_nearest_neighbors(distances, k)
    k_mode = stats.mode(k_nn, axis=None)
    print("Majority vote: {0}".format(k_mode.mode))


if __name__ == "__main__":
    app_main()
