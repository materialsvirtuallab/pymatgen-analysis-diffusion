# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.
from __future__ import annotations

import itertools
import unittest

import numpy as np
from scipy.cluster.vq import kmeans

from pymatgen.analysis.diffusion.aimd.clustering import Kmeans, KmeansPBC
from pymatgen.core import Lattice
from pymatgen.util.coord import pbc_diff


class KmeansTest(unittest.TestCase):
    def test_cluster(self) -> None:
        data = np.random.uniform(size=(10, 5))
        _data = list(data)
        d2 = np.random.uniform(size=(10, 5)) + ([5] * 5)
        _data.extend(list(d2))
        d2 = np.random.uniform(size=(10, 5)) + ([-5] * 5)
        _data.extend(list(d2))
        data = np.array(_data)

        k = Kmeans()
        clusters = []
        for _i in range(10):
            clusters.append(k.cluster(data, 3))
        c1, l1, ss = min(clusters, key=lambda d: d[2])
        c2, d = kmeans(data, 3)
        same = False
        for a in itertools.permutations(c2):
            if np.allclose(c1, a):
                same = True
                break
        assert same


class KmeansPBCTest(unittest.TestCase):
    def test_cluster(self) -> None:
        lattice = Lattice.cubic(4)

        _pts = []
        initial = [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.5, 0, 0]]
        for c in initial:
            for _i in range(100):
                _pts.append(np.array(c) + np.random.randn(3) * 0.01 + np.random.randint(3))
        pts = np.array(_pts)
        k = KmeansPBC(lattice)
        centroids, labels, ss = k.cluster(pts, 4)
        for c1 in centroids:
            found = False
            for c2 in centroids:
                if np.allclose(pbc_diff(c1, c2), [0, 0, 0], atol=0.1):
                    found = True
                    break
            assert found
