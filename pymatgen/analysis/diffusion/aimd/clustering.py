# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

"""
This module implements clustering algorithms to determine centroids, with
adaption for periodic boundary conditions. This can be used, for example, to
determine likely atomic positions from MD trajectories.
"""

import random
import warnings

import numpy as np
from pymatgen.util.coord import all_distances, pbc_diff

__author__ = "Shyue Ping Ong"
__copyright__ = "Copyright 2013, The Materials Virtual Lab"
__version__ = "0.1"
__maintainer__ = "Shyue Ping Ong"
__email__ = "ongsp@ucsd.edu"
__date__ = "3/18/15"


class Kmeans:
    """
    Simple kmeans clustering.
    """

    def __init__(self, max_iterations: int = 1000):
        """
        Args:
            max_iterations (int): Maximum number of iterations to run KMeans algo.
        """
        self.max_iterations = max_iterations

    def cluster(self, points, k, initial_centroids=None):
        """
        Args:
            points (ndarray): Data points as a mxn ndarray, where m is the
                number of features and n is the number of data points.
            k (int): Number of means.
            initial_centroids (np.array): Initial guess for the centroids. If
                None, a randomized array of points is used.

        Returns:
            centroids, labels, ss: centroids are the final centroids, labels
            provide the index for each point, and ss in the final sum squared
            distances.
        """
        centroids = np.array(random.sample(list(points), k)) if initial_centroids is None else initial_centroids

        # Initialize book keeping vars.
        iterations = 0
        old_centroids = None

        # Run the main k-means algorithm
        while not self.should_stop(old_centroids, centroids, iterations):
            # Save old centroids for convergence test. Book keeping.
            old_centroids = centroids
            iterations += 1
            # Assign labels to each datapoint based on centroids
            labels, ss = self.get_labels(points, centroids)

            # Assign centroids based on datapoint labels
            centroids = self.get_centroids(points, labels, k, centroids)

        labels, ss = self.get_labels(points, centroids)
        # We can get the labels too by calling getLabels(dataSet, centroids)
        return centroids, labels, ss

    @staticmethod
    def get_labels(points, centroids):
        """
        For each element in the dataset, chose the closest centroid.
        Make that centroid the element's label.

        Args:
            points: List of points
            centroids: List of centroids
        """
        dists = all_distances(points, centroids)
        min_dists = np.min(dists, axis=1)
        return np.where(dists == min_dists[:, None])[1], np.sum(min_dists ** 2)

    @staticmethod
    def get_centroids(points, labels, k, centroids):
        """
        Each centroid is the geometric mean of the points that
        have that centroid's label. Important: If a centroid is empty (no
        points have that centroid's label) you should randomly re-initialize it.

        Args:
            points: List of points
            labels: List of labels
            k: Number of means
            centroids: List of centroids
        """
        labels = np.array(labels)
        centroids = []
        for i in range(k):
            ind = np.where(labels == i)[0]
            if len(ind) > 0:
                centroids.append(np.average(points[ind, :], axis=0))
            else:
                centroids.append(get_random_centroid(points))
        return np.array(centroids)

    def should_stop(self, old_centroids, centroids, iterations):
        """
        Check for stopping conditions.

        Args:
            old_centroids: List of old centroids
            centroids: List of centroids
            iterations: Number of iterations thus far.
        """
        if iterations > self.max_iterations:
            warnings.warn(f"Max iterations {self.max_iterations} reached!")
            return True
        if old_centroids is None:
            return False
        return np.allclose(old_centroids, centroids)


class KmeansPBC(Kmeans):
    """
    A version of KMeans that work with PBC. Distance metrics have to change,
    as well as new centroid determination. The points supplied should be
    fractional coordinates.
    """

    def __init__(self, lattice, max_iterations=1000):
        """
        Args:
            lattice: Lattice
            max_iterations: Maximum number of iterations to run KMeans.
        """
        self.lattice = lattice
        self.max_iterations = max_iterations

    def get_labels(self, points, centroids):
        """
        For each element in the dataset, chose the closest centroid.
        Make that centroid the element's label.

        Args:
            points: List of points
            centroids: List of centroids
        """
        dists = self.lattice.get_all_distances(points, centroids)
        min_dists = np.min(dists, axis=1)
        return np.where(dists == min_dists[:, None])[1], np.sum(min_dists ** 2)

    def get_centroids(self, points, labels, k, centroids):
        """
        Each centroid is the geometric mean of the points that
        have that centroid's label. Important: If a centroid is empty (no
        points have that centroid's label) you should randomly re-initialize it.

        Args:
            points: List of points
            labels: List of labels
            k: Number of means
            centroids: List of centroids
        """
        m, n = points.shape
        labels = np.array(labels)
        new_centroids = []
        for i in range(k):
            ind = np.where(labels == i)[0]
            if len(ind) > 0:
                c = np.zeros(n)
                for j in ind:
                    dist, image = self.lattice.get_distance_and_image(centroids[i], points[j])
                    c += points[j] + image
                c /= len(ind)
                c = np.mod(c, 1)
            else:
                c = get_random_centroid(points)
                c = np.mod(c, 1)
            new_centroids.append(c)
        return np.array(new_centroids)

    def should_stop(self, old_centroids, centroids, iterations):
        """
        Check for stopping conditions.

        Args:
            old_centroids: List of old centroids
            centroids: List of centroids
            iterations: Number of iterations thus far.
        """
        if iterations > self.max_iterations:
            warnings.warn(f"Max iterations {self.max_iterations} reached!")
            return True
        if old_centroids is None:
            return False
        for c1, c2 in zip(old_centroids, centroids):
            if not np.allclose(pbc_diff(c1, c2), [0, 0, 0]):
                return False
        return True


def get_random_centroid(points):
    """
    Generate a random centroid based on points.

    Args:
        points: List of points.
    """
    m, n = points.shape
    maxd = np.max(points, axis=0)
    mind = np.min(points, axis=0)
    return np.array([random.uniform(mind[i], maxd[i]) for i in range(n)])


def get_random_centroids(points, k):
    """
    Generate k random centroids based on points.

    Args:
        points: List of points.
        k: Number of means.
    """
    centroids = []
    for i in range(k):
        centroids.append(get_random_centroid(points))
    return np.array(centroids)
