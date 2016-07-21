# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

from __future__ import division, unicode_literals, print_function
import numpy as np
from collections import Counter

__author__ = "Iek-Heng Chu"
__version__ = 1.0
__date__ = "05/15"

"""
 Algorithms for diffusion pathways analysis
"""

# TODO: ipython notebook example file, unittests


class ProbabilityDensityAnalysis(object):
    """
    Compute the time-averaged probability density distribution of selected species on a
    "uniform" (in terms of fractional coordinates) 3-D grid. Note that \int_{\Omega}d^3rP(r) = 1
    If you use this class, please consider citing the following paper:

    Zhu, Z.; Chu, I.-H.; Deng, Z. and Ong, S. P. "Role of Na+ Interstitials and Dopants
    in Enhancing the Na+ Conductivity of the Cubic Na3PS4 Superionic Conductor".
    Chem. Mater. (2015), 27, pp 8318–8325.
    """

    def __init__(self, structure, trajectories, interval=0.5,
                 species=("Li", "Na")):
        """
        Initialization.
        Args:
            structure (Structure): crystal structure
            trajectories (numpy array): ionic trajectories of the structure from MD simulations.
                It should be (1) stored as 3-D array [Ntimesteps, Nions, 3]
                where 3 refers to a,b,c components; (2) in fractional
                coordinates.
            interval(float): the interval between two nearest grid points (in Angstrom)
            species(list of str): list of species that are of interest
        """

        # initial settings
        trajectories = np.array(trajectories)

        # All fractional coordinates are between 0 and 1.
        trajectories -= np.floor(trajectories)
        assert np.all(trajectories >= 0) and np.all(trajectories <= 1)

        indices = [j for j, site in enumerate(structure) if site.specie.symbol in species]
        lattice = structure.lattice
        frac_interval = [interval / l for l in lattice.abc]
        nsteps = len(trajectories)

        # generate the 3-D grid
        ra = np.arange(0.0, 1.0, frac_interval[0])
        rb = np.arange(0.0, 1.0, frac_interval[1])
        rc = np.arange(0.0, 1.0, frac_interval[2])
        lens = [len(ra), len(rb), len(rc)]
        ngrid = lens[0] * lens[1] * lens[2]

        agrid = ra[:,None] * np.array([1, 0, 0])[None,:]
        bgrid = rb[:,None] * np.array([0, 1, 0])[None,:]
        cgrid = rc[:,None] * np.array([0, 0, 1])[None,:]

        grid = agrid[:, None, None] + bgrid[None, :, None] + cgrid[None, None, :]

        # calculate the time-averaged probability density function distribution Pr
        count = Counter()
        Pr = np.zeros(ngrid,dtype=np.double)

        for it in range(nsteps):
            fcoords = trajectories[it][indices,:]
            for fcoord in fcoords:
                # for each atom at time t, find the nearest grid point from
                # the 8 points that surround the atom
                corner_i = [int(c/d) for c, d in zip(fcoord, frac_interval)]
                next_i = np.zeros_like(corner_i, dtype=int)

                # consider PBC
                for i in range(3):
                    next_i[i] = corner_i[i] + 1 if corner_i[i] < lens[i]-1 else 0

                agrid = np.array([corner_i[0], next_i[0]])[:,None] * \
                        np.array([1, 0, 0])[None,:]
                bgrid = np.array([corner_i[1], next_i[1]])[:,None] * \
                        np.array([0, 1, 0])[None,:]
                cgrid = np.array([corner_i[2], next_i[2]])[:,None] * \
                        np.array([0, 0, 1])[None,:]

                grid_indices = agrid[:, None, None] + bgrid[None, :, None] + \
                               cgrid[None, None, :]
                grid_indices = grid_indices.reshape(8,3)

                mini_grid = [grid[indx[0],indx[1],indx[2]] for indx in grid_indices]
                dist_matrix = lattice.get_all_distances(mini_grid, fcoord)
                indx = np.where(dist_matrix == np.min(dist_matrix, axis=0)[None,:])[0][0]

                # 3-index label mapping to single index
                min_indx = grid_indices[indx][0] * len(rb) * len(rc) + \
                           grid_indices[indx][1] * len(rc) + grid_indices[indx][2]

                # make sure the index does not go out of bound.
                assert 0 <= min_indx < ngrid

                count.update([min_indx])

        for i, n in count.most_common(ngrid):
            Pr[i] = float(n)/nsteps/len(indices)/lattice.volume*ngrid

        Pr = Pr.reshape(lens[0], lens[1], lens[2])

        self.structure = structure
        self.trajectories = trajectories
        self.interval = interval
        self.lens = lens
        self.Pr = Pr

    @classmethod
    def from_diffusion_analyzer(cls, diffusion_analyzer, interval=0.5,
                 species=("Li", "Na")):
        """
        Create a ProbabilityDensityAnalysis from a diffusion_analyzer object.

        Args:
            diffusion_analyzer (DiffusionAnalyzer): A
                    pymatgen.analysis.diffusion_analyzer.DiffusionAnalyzer object
            interval(float): the interval between two nearest grid points (in Angstrom)
            species(list of str): list of species that are of interest
        """
        structure = diffusion_analyzer.structure
        trajectories = []

        for i, s in enumerate(diffusion_analyzer.get_drift_corrected_structures()):
            trajectories.append(s.frac_coords)

        trajectories = np.array(trajectories)

        return ProbabilityDensityAnalysis(structure, trajectories, interval=interval,
                 species=species)


    def to_chgcar(self, filename="CHGCAR.vasp"):
        """
        Save the probability density distribution in the format of CHGCAR,
        which can be visualized by VESTA.
        """

        count = 1
        VolinAu = self.structure.lattice.volume/(0.5291772083)**3
        symbols = self.structure.symbol_set
        natoms = [str(int(self.structure.composition[symbol])) for symbol in symbols]
        init_fcoords = np.array(self.structure.frac_coords)

        with open(filename,"w") as f:
            f.write(self.structure.formula+"\n")
            f.write(" 1.00 \n")

            for i in range(3):
                f.write(" {0} {1} {2} \n".format(*self.structure.lattice.matrix[i,:]))

            f.write(" " + " ".join(symbols) + "\n")
            f.write(" " + " ".join(natoms) + "\n")
            f.write("direct\n")
            for fcoord in init_fcoords:
                f.write(" {0:.8f}  {1:.8f}  {2:.8f} \n".format(*fcoord))

            f.write(" \n")
            f.write(" {0} {1} {2} \n".format(*self.lens))

            for i in range(self.lens[2]):
                for j in range(self.lens[1]):
                    for k in range(self.lens[0]):
                        f.write(" {0:.10e} ".format(self.Pr[k,j,i]*VolinAu))
                        if count % 5 == 0:
                            f.write("\n")
                        count += 1

        f.close()