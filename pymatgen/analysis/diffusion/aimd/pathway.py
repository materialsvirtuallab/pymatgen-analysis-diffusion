# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

"""
 Algorithms for diffusion pathway analysis
"""

from collections import Counter

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


class ProbabilityDensityAnalysis:
    r"""
    Compute the time-averaged probability density distribution of selected
    species on a "uniform" (in terms of fractional coordinates) 3-D grid.
    Note that \int_{\Omega}d^3rP(r) = 1

    If you use this class, please consider citing the following paper:

    Zhu, Z.; Chu, I.-H.; Deng, Z. and Ong, S. P. "Role of Na+ Interstitials and
    Dopants in Enhancing the Na+ Conductivity of the Cubic Na3PS4 Superionic
    Conductor". Chem. Mater. (2015), 27, pp 8318–8325.
    """

    def __init__(self, structure, trajectories, interval=0.5, species=("Li", "Na")):
        """
        Initialization.
        Args:
            structure (Structure): crystal structure
            trajectories (numpy array): ionic trajectories of the structure
                from MD simulations. It should be (1) stored as 3-D array [
                Ntimesteps, Nions, 3] where 3 refers to a,b,c components;
                (2) in fractional coordinates.
            interval(float): the interval between two nearest grid points
                (in Angstrom)
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

        agrid = ra[:, None] * np.array([1, 0, 0])[None, :]
        bgrid = rb[:, None] * np.array([0, 1, 0])[None, :]
        cgrid = rc[:, None] * np.array([0, 0, 1])[None, :]

        grid = agrid[:, None, None] + bgrid[None, :, None] + cgrid[None, None, :]

        # Calculate time-averaged probability density function distribution Pr
        count = Counter()
        Pr = np.zeros(ngrid, dtype=np.double)

        for it in range(nsteps):
            fcoords = trajectories[it][indices, :]
            for fcoord in fcoords:
                # for each atom at time t, find the nearest grid point from
                # the 8 points that surround the atom
                corner_i = [int(c / d) for c, d in zip(fcoord, frac_interval)]
                next_i = np.zeros_like(corner_i, dtype=int)

                # consider PBC
                for i in range(3):
                    next_i[i] = corner_i[i] + 1 if corner_i[i] < lens[i] - 1 else 0

                agrid = np.array([corner_i[0], next_i[0]])[:, None] * np.array([1, 0, 0])[None, :]
                bgrid = np.array([corner_i[1], next_i[1]])[:, None] * np.array([0, 1, 0])[None, :]
                cgrid = np.array([corner_i[2], next_i[2]])[:, None] * np.array([0, 0, 1])[None, :]

                grid_indices = agrid[:, None, None] + bgrid[None, :, None] + cgrid[None, None, :]
                grid_indices = grid_indices.reshape(8, 3)

                mini_grid = [grid[indx[0], indx[1], indx[2]] for indx in grid_indices]
                dist_matrix = lattice.get_all_distances(mini_grid, fcoord)
                indx = np.where(dist_matrix == np.min(dist_matrix, axis=0)[None, :])[0][0]

                # 3-index label mapping to single index
                min_indx = (
                    grid_indices[indx][0] * len(rb) * len(rc) + grid_indices[indx][1] * len(rc) + grid_indices[indx][2]
                )

                # make sure the index does not go out of bound.
                assert 0 <= min_indx < ngrid

                count.update([min_indx])

        for i, n in count.most_common(ngrid):
            Pr[i] = float(n) / nsteps / len(indices) / lattice.volume * ngrid

        Pr = Pr.reshape(lens[0], lens[1], lens[2])

        self.structure = structure
        self.trajectories = trajectories
        self.interval = interval
        self.lens = lens
        self.Pr = Pr
        self.species = species
        self.stable_sites = None

    @classmethod
    def from_diffusion_analyzer(cls, diffusion_analyzer, interval=0.5, species=("Li", "Na")):
        """
        Create a ProbabilityDensityAnalysis from a diffusion_analyzer object.

        Args:
            diffusion_analyzer (DiffusionAnalyzer): A
                pymatgen.analysis.diffusion_analyzer.DiffusionAnalyzer object
            interval(float): the interval between two nearest grid points (in
                Angstrom)
            species(list of str): list of species that are of interest
        """
        structure = diffusion_analyzer.structure
        trajectories = []

        for i, s in enumerate(diffusion_analyzer.get_drift_corrected_structures()):
            trajectories.append(s.frac_coords)

        trajectories = np.array(trajectories)

        return ProbabilityDensityAnalysis(structure, trajectories, interval=interval, species=species)

    def generate_stable_sites(self, p_ratio=0.25, d_cutoff=1.0):
        """
        Obtain a set of low-energy sites from probability density function with
        given probability threshold 'p_ratio'. The set of grid points with
        probability density higher than the threshold will further be clustered
        using hierachical clustering method, with no two clusters closer than the
        given distance cutoff. Note that the low-energy sites may converge more
        slowly in fast conductors (more shallow energy landscape) than in the slow
        conductors.

        Args:
            p_ratio (float): Probability threshold above which a grid point is
                considered as a low-energy site.
            d_cutoff (float): Distance cutoff used in hierachical clustering.

        Notes:
            The set of stable sites is stored in the `stable_sites` attribute
            as a Nx3 numpy array.
        """

        # Set of grid points with high probability density.
        grid_fcoords = []
        indices = np.where(self.Pr > self.Pr.max() * p_ratio)
        lattice = self.structure.lattice

        for (x, y, z) in zip(indices[0], indices[1], indices[2]):
            grid_fcoords.append([x / self.lens[0], y / self.lens[1], z / self.lens[2]])

        grid_fcoords = np.array(grid_fcoords)
        dist_matrix = np.array(lattice.get_all_distances(grid_fcoords, grid_fcoords))
        np.fill_diagonal(dist_matrix, 0)

        # Compressed distance matrix
        condensed_m = squareform((dist_matrix + dist_matrix.T) / 2.0)

        # Linkage function for hierachical clustering.
        z = linkage(condensed_m, method="single", metric="euclidean")
        cluster_indices = fcluster(z, t=d_cutoff, criterion="distance")

        # The low-energy sites must accommodate all the existing mobile ions.
        nions = len([e for e in self.structure if e.specie.symbol in self.species])
        nc = len(set(cluster_indices))

        if nc < nions:
            raise ValueError(
                f"The number of clusters ({nc}) is smaller than that of "
                f"mobile ions ({nions})! Please try to decrease either "
                "'p_ratio' or 'd_cut' values!"
            )

        # For each low-energy site (cluster centroid), its coordinates are obtained
        # by averaging over all the associated grid points within that cluster.
        stable_sites = []

        for i in set(cluster_indices):
            indices = np.where(cluster_indices == i)[0]

            if len(indices) == 1:
                stable_sites.append(grid_fcoords[indices[0]])
                continue

            # Consider periodic boundary condition
            members = grid_fcoords[indices] - grid_fcoords[indices[0]]
            members = np.where(members > 0.5, members - 1.0, members)
            members = np.where(members < -0.5, members + 1.0, members)
            members += grid_fcoords[indices[0]]

            stable_sites.append(np.mean(members, axis=0))

        self.stable_sites = np.array(stable_sites)

    def get_full_structure(self):
        """
        Generate the structure with the low-energy sites included. In the end, a
        pymatgen Structure object will be returned.
        """

        full_structure = self.structure.copy()
        for fcoord in self.stable_sites:
            full_structure.append("X", fcoord)

        return full_structure

    def to_chgcar(self, filename="CHGCAR.vasp"):
        """
        Save the probability density distribution in the format of CHGCAR,
        which can be visualized by VESTA.
        """

        count = 1
        VolinAu = self.structure.lattice.volume / 0.5291772083 ** 3
        symbols = self.structure.symbol_set
        natoms = [str(int(self.structure.composition[symbol])) for symbol in symbols]
        init_fcoords = np.array(self.structure.frac_coords)

        with open(filename, "w") as f:
            f.write(self.structure.formula + "\n")
            f.write(" 1.00 \n")

            for i in range(3):
                v = self.structure.lattice.matrix[i, :]
                f.write(f" {v[0]} {v[1]} {v[2]} \n")

            f.write(" " + " ".join(symbols) + "\n")
            f.write(" " + " ".join(natoms) + "\n")
            f.write("direct\n")
            for fcoord in init_fcoords:
                f.write(" {:.8f}  {:.8f}  {:.8f} \n".format(*fcoord))  # pylint: disable=C0209

            f.write(" \n")
            f.write(" {} {} {} \n".format(*self.lens))  # pylint: disable=C0209

            for i in range(self.lens[2]):
                for j in range(self.lens[1]):
                    for k in range(self.lens[0]):
                        f.write(f" {(self.Pr[k, j, i] * VolinAu):.10e} ")
                        if count % 5 == 0:
                            f.write("\n")
                        count += 1

        f.close()


class SiteOccupancyAnalyzer:
    """
    A class that analyzes the site occupancy of given species using MD trajectories.
    The occupancy of a site is determined based on the shortest distance scheme.

    .. attribute:: site_occ
        N x 1 numpy array that stores the occupancy of all sites associated with
        species. It has the same sequence as the given list of indices.

    .. attribute:: coords_ref
        N x 3 numpy array of fractional coordinates of all reference sites.

    .. attribute:: nsites
        Number of reference sites.

    .. attribute:: structure
        Initial structure.

    """

    def __init__(self, structure, coords_ref, trajectories, species=("Li", "Na")):
        """
        Args:
            structure (pmg_structure): Initial structure.
            coords_ref (N_s x 3 array): Fractional coordinates of N_s given sites
                at which the site occupancy will be computed.
            trajectories (Ntimesteps x Nions x 3 array): Ionic trajectories from MD
                simulation, where Ntimesteps is the number of time steps in MD
                simulation. Note that the coordinates are fractional.
            species(list of str): list of species that are of interest.
        """

        lattice = structure.lattice
        coords_ref = np.array(coords_ref)
        trajectories = np.array(trajectories)
        count = Counter()

        indices = [i for i, site in enumerate(structure) if site.specie.symbol in species]

        for it in range(len(trajectories)):
            dist_matrix = lattice.get_all_distances(coords_ref, trajectories[it, indices, :])
            labels = np.where(dist_matrix == np.min(dist_matrix, axis=0)[None, :])[0]
            count.update(labels)

        site_occ = np.zeros(len(coords_ref), dtype=np.double)

        for i, n in count.most_common(len(coords_ref)):
            site_occ[i] = n / float(len(trajectories))

        self.structure = structure
        self.coords_ref = coords_ref
        self.species = species
        self.indices = indices
        self.nsites = len(coords_ref)
        self.nsteps = len(trajectories)
        self.site_occ = site_occ

    def get_average_site_occupancy(self, indices):
        """
        Get the average site occupancy over a subset of reference sites.
        """
        return np.sum(self.site_occ[indices]) / len(indices)

    @classmethod
    def from_diffusion_analyzer(cls, coords_ref, diffusion_analyzer, species=("Li", "Na")):

        """
        Create a SiteOccupancyAnalyzer object using a diffusion_analyzer object.

        Args:
            coords_ref (nested list of floats): Fractional coordinates of a list
                of reference sites at which the site occupancy will be computed.
            diffusion_analyzer (DiffusionAnalyzer): A
                pymatgen.analysis.diffusion_analyzer.DiffusionAnalyzer object
            species(list of str): list of species that are of interest.
        """

        trajectories = []

        # Initial structure.
        structure = diffusion_analyzer.structure

        # Drifted corrected ionic trajectories
        for s in diffusion_analyzer.get_drift_corrected_structures():
            trajectories.append(s.frac_coords)

        return SiteOccupancyAnalyzer(structure, coords_ref, trajectories, species)
