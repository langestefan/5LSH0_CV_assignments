"""
main.py
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from scipy import stats
from numpy.random import default_rng
import numpy as np


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
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("red=setosa, orange=versicolor, grey=virginica")

    plt.show()


# step 1: select 3 points from our dataset as our initial means
# step 2: calculate distance from each point to each mean
# step 3: assign each point to a mean point based on distance to mean point
# step 4: calculate new mean based on assigned points
# repeat step 2-4

def select_k_initial_means(train_data, k):
    """
    :param train_data: 2D x/y data input
    :param k: how many means we want
    :return: k initial mean points
    """
    k_initial_means = np.zeros((k, 2))
    rng = default_rng()
    points = rng.choice(np.size(train_data, 0), size=k, replace=False)

    for idx, p in enumerate(points):
        k_initial_means[idx] = train_data[p]

    return k_initial_means


def calculate_distances(data, point):
    """
    Calculate for given point the euclidean distance to the other N-1 samples
    :param data: x/y data
    :param point: integer, sample [1-N]
    :return: distance from point to the other N-1 points
    """
    size = np.size(data, 0)
    x0, y0 = point[0], point[1]
    distance = np.zeros(size)

    for idx, sample in enumerate(data):
        distance[idx] = euclidean([x0, y0], [data[idx, 0], data[idx, 1]])

    # print(distance)
    return distance


def app_main():
    """ App main """
    iris = datasets.load_iris()
    data = iris.data[:, (2, 3)]  # take first 2 features (petal length/width)
    label = iris.target
    k = 3  # number of k-means

    # split into train test sets
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=False)

    # distance matrix. Contains distance from k points to other points
    size = np.size(x_train, 0)
    distance_matrix = np.zeros((size, k))
    # visualize_data(data, label)

    k_init_means = select_k_initial_means(x_train, k)

    for idx, point in enumerate(k_init_means):
        distance_matrix[:, idx] = calculate_distances(x_train, point)
        

    print(distance_matrix)


if __name__ == "__main__":
    app_main()
