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
from collections import defaultdict


def visualize_data(data, k_means, label):
    """
    :param k_means: Input K mean points
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
    plt.scatter(k_means[:, 0], k_means[:, 1], c='blue', marker='x')
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("red=setosa, orange=versicolor, grey=virginica, blue=cluster mean")

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

    return distance


def assign_k_means(data, k_means, k):
    """
    Assign each datapoint to a mean point
    :param k: K number of mean points
    :param data: Input data points[x0, y0]
    :param k_means: Input K mean points[x0, y0]
    :return: Dictionary where each index is a list of assigned points
    """
    # default dictionary. Creates object if an object is accessed at index that doesnt exist
    point_dict = defaultdict(list)

    # distance matrix. Contains distance from k points to other points
    size = np.size(data, 0)
    distance_matrix = np.zeros((size, k))

    # for each mean point we calculate the distance to the other points
    for idx, point in enumerate(k_means):
        distance_matrix[:, idx] = calculate_distances(data, point)

    # for each point in our train set we check which mean is closest
    # then we create k-clusters of data by assigning the train points to a mean point
    for idx, point in enumerate(data):
        k_mean_index = int((np.where(distance_matrix[idx] == np.amin(distance_matrix[idx])))[0])
        point_dict[k_mean_index] = point_dict[k_mean_index] + [point]

    return point_dict


def update_k_means(data, k_means, k):
    """
    :param data: Training data
    :param k_means: Input K mean points
    :param k: K number of mean points
    :return: Updated K mean points after one iteration
    """
    point_dict = assign_k_means(data, k_means, k)

    # we sum over the column entries and divide by N to get the average of all assigned points
    for idk in range(k):
        k_means[idk] = np.sum(point_dict[idk], axis=0) / np.size(point_dict[idk], 0)

    return k_means, point_dict


def app_main():
    """ App main """
    iris = datasets.load_iris()
    data = iris.data[:, (2, 3)]  # take first 2 features (petal length/width)
    label = iris.target
    k = 3  # number of mean points
    itt = 10  # number of iterations

    # split into train test sets
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=False)

    # select initial means
    k_means = select_k_initial_means(x_train, k)

    # train the model
    for i in range(itt):
        k_means, k_point_dict = update_k_means(x_train, k_means, k)

    # run test data
    t_point_dict = assign_k_means(x_test, k_means, k)

    print(t_point_dict)

    # visualize the data
    visualize_data(x_train, k_means, y_train)
    # visualize_data(x_test, k_means, y_test)


if __name__ == "__main__":
    app_main()

# answers
# Vary the number of clusters and iterations. What do you observe?
# Higher number of iterations: k-mean becomes more accurate. At some point the result does not change.
# Number of clusters: 2 clusters makes the mean points very inaccurate. A point may be between 2 clusters.

# What is the impact of random initialization in k-means? How can you
# improve it?
# The solution may take longer to converge. It is possible to improve this by for example assigning a data point as the
# initial mean.

# What is disadvantage of using k-means algorithm?
# Have to choose K manually. Difficult to place initial positions of means.

# What are the similarities and differences between k-NN and k-means?
# Explain in detail.
# Similarities:
# - a way of grouping similar data
# - simple to implement
# Differences:
# - k-NN is supervised learning, k-means is unsupervised.
# - Easy to assess accuracy in k-NN, for k-means this is harder

