# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.


import unittest

import numpy as np
from scipy.cluster.vq import kmeans
import itertools
from pymatgen.core import Lattice
from pymatgen.analysis.diffusion.aimd.clustering import Kmeans, KmeansPBC
from pymatgen.util.coord import pbc_diff


class KmeansTest(unittest.TestCase):
    def test_cluster(self):
        data = np.random.uniform(size=(10, 5))
        data = list(data)
        d2 = np.random.uniform(size=(10, 5)) + ([5] * 5)
        data.extend(list(d2))
        d2 = np.random.uniform(size=(10, 5)) + ([-5] * 5)
        data.extend(list(d2))
        data = np.array(data)

        k = Kmeans()
        clusters = []
        for i in range(10):
            clusters.append(k.cluster(data, 3))
        c1, l1, ss = min(clusters, key=lambda d: d[2])
        c2, d = kmeans(data, 3)
        same = False
        for a in itertools.permutations(c2):
            if np.allclose(c1, a):
                same = True
                break
        self.assertTrue(same)


class KmeansPBCTest(unittest.TestCase):
    def test_cluster(self):
        lattice = Lattice.cubic(4)

        pts = []
        initial = [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.5, 0, 0]]
        for c in initial:
            for i in range(100):
                pts.append(np.array(c) + np.random.randn(3) * 0.01 + np.random.randint(3))
        pts = np.array(pts)
        k = KmeansPBC(lattice)
        centroids, labels, ss = k.cluster(pts, 4)
        for c1 in centroids:
            found = False
            for c2 in centroids:
                if np.allclose(pbc_diff(c1, c2), [0, 0, 0], atol=0.1):
                    found = True
                    break
            self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()
