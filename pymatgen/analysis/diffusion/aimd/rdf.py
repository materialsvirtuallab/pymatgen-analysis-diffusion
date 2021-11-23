# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

"""
RDF implementation.
"""

from collections import Counter
from math import ceil
from multiprocessing import cpu_count
from typing import List, Tuple, Union, Dict

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import norm
from joblib import delayed, Parallel
from pymatgen.core import Structure
from pymatgen.util.plotting import pretty_plot


class RadialDistributionFunction:
    """
    Calculate the average radial distribution function for a given set of
    structures.
    """

    def __init__(
        self,
        structures: List,
        indices: List,
        reference_indices: List,
        ngrid: int = 101,
        rmax: float = 10.0,
        cell_range: int = 1,
        sigma: float = 0.1,
    ):
        """
        Args:
            structures ([Structure]): List of structure
                objects with the same composition. Allow for ensemble averaging.
            ngrid (int): Number of radial grid points.
            rmax (float): Maximum of radial grid (the minimum is always zero)
                in Angstrom.
            cell_range (int): Range of translational vector elements associated
                with supercell. Default is 1, i.e. including the adjacent image
                cells along all three directions.
            sigma (float): Smearing of a Gaussian function.
            indices ([int]): A list of atom index of interest.
            reference_indices ([int]): set this option along with 'indices'
                parameter to compute radial distribution function.
        """

        if ngrid < 2:
            raise ValueError("ngrid should be greater than 1!")
        if sigma <= 0:
            raise ValueError("sigma should be a positive number!")

        if len(indices) < 1:
            raise ValueError("Given species are not in the structure!")

        lattices, rhos, fcoords_list, ref_fcoords_list = [], [], [], []

        dr = rmax / (ngrid - 1)
        interval = np.linspace(0.0, rmax, ngrid)
        rdf = np.zeros((ngrid), dtype=np.double)
        raw_rdf = np.zeros((ngrid), dtype=np.double)

        dns = Counter()  # type: ignore

        # Generate the translational vectors
        r = np.arange(-cell_range, cell_range + 1)
        arange = r[:, None] * np.array([1, 0, 0])[None, :]
        brange = r[:, None] * np.array([0, 1, 0])[None, :]
        crange = r[:, None] * np.array([0, 0, 1])[None, :]
        images = arange[:, None, None] + brange[None, :, None] + crange[None, None, :]
        images = images.reshape((len(r) ** 3, 3))

        # Find the zero image vector
        zd = np.sum(images ** 2, axis=1)
        indx0 = np.argmin(zd)

        for s in structures:
            latt = s.lattice
            lattices.append(latt)
            rhos.append(float(len(indices)) / latt.volume)
            all_fcoords = np.array(s.frac_coords)
            fcoords_list.append(all_fcoords[indices, :])
            ref_fcoords_list.append(all_fcoords[reference_indices, :])

        rho = sum(rhos) / len(rhos)  # The average density
        self.rhos = rhos
        self.rho = rho  # This is the average density

        for fcoords, ref_fcoords, latt in zip(fcoords_list, ref_fcoords_list, lattices):
            dcf = fcoords[:, None, None, :] + images[None, None, :, :] - ref_fcoords[None, :, None, :]
            dcc = latt.get_cartesian_coords(dcf)
            d2 = np.sum(dcc ** 2, axis=3)
            dists = [
                d2[u, v, j] ** 0.5
                for u in range(len(indices))
                for v in range(len(reference_indices))
                for j in range(len(r) ** 3)
                if indices[u] != reference_indices[v] or j != indx0
            ]
            r_indices = [int(dist / dr) for dist in filter(lambda e: e < rmax + 1e-8, dists)]
            dns.update(r_indices)

        for indx, dn in dns.most_common(ngrid):
            if indx > len(interval) - 1:
                continue

            # Volume of the thin shell
            ff = 4.0 / 3.0 * np.pi * (interval[indx + 1] ** 3 - interval[indx] ** 3)
            # print(norm.pdf(interval, interval[indx], sigma) * dn /
            #            float(len(reference_indices)) / ff / rho / len(
            #             fcoords_list) * dr)
            rdf[:] += (
                norm.pdf(interval, interval[indx], sigma)
                * dn
                / float(len(reference_indices))
                / ff
                / rho
                / len(fcoords_list)
                * dr
            )

            # additional dr factor renormalises overlapping gaussians.
            raw_rdf[indx] += dn / float(len(reference_indices)) / ff / rho / len(fcoords_list)

        self.structures = structures
        self.cell_range = cell_range
        self.rmax = rmax
        self.ngrid = ngrid
        self.species = {structures[0][i].species_string for i in indices}
        self.reference_species = {structures[0][i].species_string for i in reference_indices}
        self.indices = indices
        self.reference_indices = reference_indices
        self.dr = dr
        self.rdf = rdf
        self.raw_rdf = raw_rdf
        self.interval = interval

        # Finding peak based on smeared RDF
        self.peak_indices = find_peaks(rdf)[0]
        self.peak_r = [self.interval[i] for i in self.peak_indices]
        self.peak_rdf = [self.rdf[i] for i in self.peak_indices]

    @classmethod
    def from_species(
        cls,
        structures: List,
        ngrid: int = 101,
        rmax: float = 10.0,
        cell_range: int = 1,
        sigma: float = 0.1,
        species: Union[Tuple, List] = ("Li", "Na"),
        reference_species: Union[Tuple, List] = None,
    ):
        """
        Initialize using species.

        Args:
            structures (list of pmg_structure objects): List of structure
                objects with the same composition. Allow for ensemble averaging.
            ngrid (int): Number of radial grid points.
            rmax (float): Maximum of radial grid (the minimum is always zero).
            cell_range (int): Range of translational vector elements associated
                with supercell. Default is 1, i.e. including the adjacent image
                cells along all three directions.
            sigma (float): Smearing of a Gaussian function.
            species (list[string]): A list of specie symbols of interest.
            reference_species (list[string]): set this option along with
                'species' parameter to compute radial distribution function.
                eg: species=["H"], reference_species=["O"] to compute
                    O-H pair distribution in a water MD simulation.
        """
        indices = [j for j, site in enumerate(structures[0]) if site.specie.symbol in species]
        if reference_species:
            reference_indices = [j for j, site in enumerate(structures[0]) if site.specie.symbol in reference_species]

            if len(reference_indices) < 1:
                raise ValueError("Given reference species are not in the structure!")
        else:
            reference_indices = indices

        return cls(
            structures=structures,
            ngrid=ngrid,
            rmax=rmax,
            cell_range=cell_range,
            sigma=sigma,
            indices=indices,
            reference_indices=reference_indices,
        )

    @property
    def coordination_number(self):
        """
        returns running coordination number

        Returns:
            numpy array
        """
        # Note: The average density from all input structures is used here.
        intervals = np.append(self.interval, self.interval[-1] + self.dr)
        return np.cumsum(self.raw_rdf * self.rho * 4.0 / 3.0 * np.pi * (intervals[1:] ** 3 - intervals[:-1] ** 3))

    def get_rdf_plot(
        self,
        label: str = None,
        xlim: tuple = (0.0, 8.0),
        ylim: tuple = (-0.005, 3.0),
        loc_peak: bool = False,
    ):
        """
        Plot the average RDF function.

        Args:
            label (str): The legend label.
            xlim (list): Set the x limits of the current axes.
            ylim (list): Set the y limits of the current axes.
            loc_peak (bool): Label peaks if True.
        """

        if label is None:
            symbol_list = [e.symbol for e in self.structures[0].composition.keys()]
            symbol_list = [symbol for symbol in symbol_list if symbol in self.species]

            if len(symbol_list) == 1:
                label = symbol_list[0]
            else:
                label = "-".join(symbol_list)

        plt = pretty_plot(12, 8)
        plt.plot(self.interval, self.rdf, label=label, linewidth=4.0, zorder=1)

        if loc_peak:
            plt.scatter(
                self.peak_r,
                self.peak_rdf,
                marker="P",
                s=240,
                c="k",
                linewidths=0.1,
                alpha=0.7,
                zorder=2,
                label="Peaks",
            )

        plt.xlabel("$r$ ($\\rm\\AA$)")
        plt.ylabel("$g(r)$")
        plt.legend(loc="upper right", fontsize=36)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.tight_layout()

        return plt

    def export_rdf(self, filename: str):
        """
        Output RDF data to a csv file.

        Args:
            filename (str): Filename. Supported formats are csv and dat. If
                the extension is csv, a csv file is written. Otherwise,
                a dat format is assumed.
        """
        fmt = "csv" if filename.lower().endswith(".csv") else "dat"
        delimiter = ", " if fmt == "csv" else " "
        with open(filename, "wt") as f:
            if fmt == "dat":
                f.write("# ")
            f.write(delimiter.join(["r", "g(r)"]))
            f.write("\n")

            for r, gr in zip(self.interval, self.rdf):
                f.write(delimiter.join([str(v) for v in [r, gr]]))
                f.write("\n")


class RadialDistributionFunctionFast:
    """
    Fast radial distribution analysis.
    """

    def __init__(
        self,
        structures: Union[Structure, List[Structure]],
        rmin: float = 0.0,
        rmax: float = 10.0,
        ngrid: float = 101,
        sigma: float = 0.0,
        n_jobs=None,
    ):
        """
        This method calculates rdf on `np.linspace(rmin, rmax, ngrid)` points.

        Args:
            structures (list of pymatgen Structures): structures to compute RDF
            rmin (float): minimal radius
            rmax (float): maximal radius
            ngrid (int): number of grid points, defaults to 101
            sigma (float): smooth parameter
            n_jobs (int): number of CPUs in processing
        """
        if n_jobs is None:
            n_jobs = 1
        if n_jobs < 0:
            n_jobs = cpu_count()
        self.n_jobs = n_jobs
        if isinstance(structures, Structure):
            structures = [structures]
        self.structures = structures
        # Number of atoms in all structures should be the same
        assert len({len(i) for i in self.structures}) == 1
        elements = [[i.specie for i in j.sites] for j in self.structures]
        unique_elements_on_sites = [len(set(i)) == 1 for i in list(zip(*elements))]

        # For the same site index, all structures should have the same element there
        if not all(unique_elements_on_sites):
            raise RuntimeError("Elements are not the same at least for one site")

        self.rmin = rmin
        self.rmax = rmax
        self.ngrid = ngrid

        self.dr = (self.rmax - self.rmin) / (self.ngrid - 1)  # end points are on grid
        self.r = np.linspace(self.rmin, self.rmax, self.ngrid)  # type: ignore

        max_r = self.rmax + self.dr / 2.0  # add a small shell to improve robustness

        if self.n_jobs > 1:
            self.neighbor_lists = Parallel(n_jobs=self.n_jobs)(
                delayed(_get_neighbor_list)(s, max_r) for s in self.structures
            )
        else:
            self.neighbor_lists = [i.get_neighbor_list(max_r) for i in self.structures]
        # each neighbor list is a tuple of
        # center_indices, neighbor_indices, image_vectors, distances
        (
            self.center_indices,
            self.neighbor_indices,
            self.image_vectors,
            self.distances,
        ) = list(zip(*self.neighbor_lists))

        elements = np.array([str(i.specie) for i in structures[0]])  # type: ignore
        self.center_elements = [elements[i] for i in self.center_indices]
        self.neighbor_elements = [elements[i] for i in self.neighbor_indices]
        self.density = [{}] * len(self.structures)  # type: List[Dict]

        self.natoms = [i.composition.to_data_dict["unit_cell_composition"] for i in self.structures]

        for s_index, natoms in enumerate(self.natoms):
            for i, j in natoms.items():
                self.density[s_index][i] = j / self.structures[s_index].volume

        self.volumes = 4.0 * np.pi * self.r ** 2 * self.dr
        self.volumes[self.volumes < 1e-8] = 1e8  # avoid divide by zero
        self.n_structures = len(self.structures)
        self.sigma = ceil(sigma / self.dr)

    def _dist_to_counts(self, d):
        """
        Convert a distance array for counts in the bin

        Args:
            d: (1D np.array)
        Returns:
            1D array of counts in the bins centered on self.r
        """
        counts = np.zeros((self.ngrid,))
        indices = np.array(np.floor((d - self.rmin + 0.5 * self.dr) / self.dr), dtype=int)

        unique, val_counts = np.unique(indices, return_counts=True)
        counts[unique] = val_counts
        return counts

    def get_rdf(
        self,
        ref_species: Union[str, List[str]],
        species: Union[str, List[str]],
        is_average=True,
    ):
        """
        Args:
            ref_species (list of species or just single specie str): the reference species.
                The rdfs are calculated with these species at the center
            species (list of species or just single specie str): the species that we are interested in.
                The rdfs are calculated on these species.
            is_average (bool): whether to take the average over
                all structures

        Returns:
            (x, rdf) x is the radial points, and rdf is the rdf value.
        """
        if self.n_jobs > 1:
            all_rdfs = Parallel(n_jobs=self.n_jobs)(
                self.get_one_rdf(ref_species, species, i) for i in range(self.n_structures)
            )
            all_rdfs = [i[1] for i in all_rdfs]
        else:
            all_rdfs = [self.get_one_rdf(ref_species, species, i)[1] for i in range(self.n_structures)]
        if is_average:
            all_rdfs = np.mean(all_rdfs, axis=0)
        return self.r, all_rdfs

    def get_one_rdf(
        self,
        ref_species: Union[str, List[str]],
        species: Union[str, List[str]],
        index=0,
    ):
        """
        Get the RDF for one structure, indicated by the index of the structure
        in all structures

        Args:
            ref_species (list of species or just single specie str): the reference species.
                The rdfs are calculated with these species at the center
            species (list of species or just single specie str): the species that we are interested in.
                The rdfs are calculated on these species.
            index (int): structure index in the list

        Returns:
            (x, rdf) x is the radial points, and rdf is the rdf value.
        """
        if isinstance(ref_species, str):
            ref_species = [ref_species]

        if isinstance(species, str):
            species = [species]

        indices = (
            (np.isin(self.center_elements[index], ref_species))
            & (np.isin(self.neighbor_elements[index], species))
            & (self.distances[index] >= self.rmin - self.dr / 2.0)
            & (self.distances[index] <= self.rmax + self.dr / 2.0)
            & (self.distances[index] > 1e-8)
        )

        density = sum(self.density[index][i] for i in species)
        natoms = sum(self.natoms[index][i] for i in ref_species)
        distances = self.distances[index][indices]
        counts = self._dist_to_counts(distances)
        rdf_temp = counts / density / self.volumes / natoms
        if self.sigma > 1e-8:
            rdf_temp = gaussian_filter1d(rdf_temp, self.sigma)
        return self.r, rdf_temp

    def get_coordination_number(self, ref_species, species, is_average=True):
        """
        returns running coordination number

        Args:
            ref_species (list of species or just single specie str): the reference species.
                The rdfs are calculated with these species at the center
            species (list of species or just single specie str): the species that we are interested in.
                The rdfs are calculated on these species.
            is_average (bool): whether to take structural average

        Returns:
            numpy array
        """
        # Note: The average density from all input structures is used here.
        all_rdf = self.get_rdf(ref_species, species, is_average=False)[1]
        if isinstance(species, str):
            species = [species]
        density = [sum(i[j] for j in species) for i in self.density]
        cn = [np.cumsum(rdf * density[i] * 4.0 * np.pi * self.r ** 2 * self.dr) for i, rdf in enumerate(all_rdf)]
        if is_average:
            cn = np.mean(cn, axis=0)
        return self.r, cn


def _get_neighbor_list(structure, r) -> Tuple:
    """
    Thin wrapper to enable parallel calculations

    Args:
        structure (pymatgen Structure): pymatgen structure
        r (float): cutoff radius

    Returns:
        tuple of neighbor list
    """
    return structure.get_neighbor_list(r)
