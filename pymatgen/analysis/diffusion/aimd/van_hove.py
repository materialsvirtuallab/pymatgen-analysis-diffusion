# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

"""
Van Hove analysis for correlations.
"""

import itertools
from collections import Counter
from typing import List, Tuple, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pds
from scipy.stats import norm
from pymatgen.core import Structure
from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.typing import ArrayLike


from .rdf import RadialDistributionFunction

__author__ = "Iek-Heng Chu"
__version__ = "1.0"
__date__ = "Aug 9, 2017"


class VanHoveAnalysis:
    """
    Class for van Hove function analysis. In particular, self-part (Gs) and
    distinct-part (Gd) of the van Hove correlation function G(r,t)
    for given species and given structure are computed. If you use this class,
    please consider citing the following paper:

    Zhu, Z.; Chu, I.-H.; Deng, Z. and Ong, S. P. "Role of Na+ Interstitials
    and Dopants in Enhancing the Na+ Conductivity of the Cubic Na3PS4
    Superionic Conductor". Chem. Mater. (2015), 27, pp 8318–8325
    """

    def __init__(
        self,
        diffusion_analyzer: DiffusionAnalyzer,
        avg_nsteps: int = 50,
        ngrid: int = 101,
        rmax: float = 10.0,
        step_skip: int = 50,
        sigma: float = 0.1,
        cell_range: int = 1,
        species: Union[Tuple, List] = ("Li", "Na"),
        reference_species: Union[Tuple, List] = None,
        indices: List = None,
    ):
        """
        Initiation.

        Args:
            diffusion_analyzer (DiffusionAnalyzer): A
                pymatgen.analysis.diffusion.analyzer.DiffusionAnalyzer object
            avg_nsteps (int): Number of t0 used for statistical average
            ngrid (int): Number of radial grid points
            rmax (float): Maximum of radial grid (the minimum is always set zero)
            step_skip (int): # of time steps skipped during analysis. It defines
                the resolution of the reduced time grid
            sigma (float): Smearing of a Gaussian function
            cell_range (int): Range of translational vector elements associated
                with supercell. Default is 1, i.e. including the adjacent image
                cells along all three directions.
            species ([string]): a list of specie symbols of interest.
            reference_species ([string]): Set this option along with 'species'
                parameter to calculate the distinct-part of van Hove function.
                Note that the self-part of van Hove function is always computed
                only for those in "species" parameter.
            indices (list of int): If not None, only a subset of atomic indices
                will be selected for the analysis. If this is given, "species"
                parameter will be ignored.
        """

        # initial check
        if step_skip <= 0:
            raise ValueError("skip_step should be >=1!")

        n_ions, nsteps, ndim = diffusion_analyzer.disp.shape

        if nsteps <= avg_nsteps:
            raise ValueError("Number of timesteps is too small!")

        ntsteps = nsteps - avg_nsteps

        if ngrid - 1 <= 0:
            raise ValueError("Ntot should be greater than 1!")

        if sigma <= 0.0:
            raise ValueError("sigma should be > 0!")

        dr = rmax / (ngrid - 1)
        interval = np.linspace(0.0, rmax, ngrid)
        reduced_nt = int(ntsteps / float(step_skip)) + 1

        lattice = diffusion_analyzer.structure.lattice
        structure = diffusion_analyzer.structure

        if indices is None:
            indices = [j for j, site in enumerate(structure) if site.specie.symbol in species]

        ref_indices = indices
        if reference_species:
            ref_indices = [j for j, site in enumerate(structure) if site.specie.symbol in reference_species]

        rho = float(len(indices)) / lattice.volume

        # reduced time grid
        rtgrid = np.arange(0.0, reduced_nt)
        # van Hove functions
        gsrt = np.zeros((reduced_nt, ngrid), dtype=np.double)
        gdrt = np.zeros((reduced_nt, ngrid), dtype=np.double)

        tracking_ions = []  # type: ArrayLike
        ref_ions = []  # type: ArrayLike

        # auxiliary factor for 4*\pi*r^2
        aux_factor = 4.0 * np.pi * interval ** 2
        aux_factor[0] = np.pi * dr ** 2

        for i, ss in enumerate(diffusion_analyzer.get_drift_corrected_structures()):
            all_fcoords = np.array(ss.frac_coords)
            tracking_ions.append(all_fcoords[indices, :])
            ref_ions.append(all_fcoords[ref_indices, :])

        tracking_ions = np.array(tracking_ions)
        ref_ions = np.array(ref_ions)

        gaussians = norm.pdf(interval[:, None], interval[None, :], sigma) / float(avg_nsteps) / float(len(ref_indices))

        # calculate self part of van Hove function
        image = np.array([0, 0, 0])
        for it in range(reduced_nt):
            dns = Counter()  # type: ignore
            it0 = min(it * step_skip, ntsteps)
            for it1 in range(avg_nsteps):
                dists = [
                    lattice.get_distance_and_image(tracking_ions[it1][u], tracking_ions[it0 + it1][u], jimage=image)[0]
                    for u in range(len(indices))
                ]

                r_indices = [int(dist / dr) for dist in filter(lambda e: e < rmax, dists)]
                dns.update(r_indices)  # type: ignore

            for indx, dn in dns.most_common(ngrid):
                gsrt[it, :] += gaussians[indx, :] * dn

        # calculate distinct part of van Hove function of species
        r = np.arange(-cell_range, cell_range + 1)
        arange = r[:, None] * np.array([1, 0, 0])[None, :]
        brange = r[:, None] * np.array([0, 1, 0])[None, :]
        crange = r[:, None] * np.array([0, 0, 1])[None, :]
        images = arange[:, None, None] + brange[None, :, None] + crange[None, None, :]
        images = images.reshape((len(r) ** 3, 3))

        # find the zero image vector
        zd = np.sum(images ** 2, axis=1)
        indx0 = np.argmin(zd)

        for it in range(reduced_nt):
            dns = Counter()
            it0 = min(it * step_skip, ntsteps)

            for it1 in range(avg_nsteps):
                dcf = (
                    tracking_ions[it0 + it1, :, None, None, :]
                    + images[None, None, :, :]
                    - ref_ions[it1, None, :, None, :]
                )
                dcc = lattice.get_cartesian_coords(dcf)
                d2 = np.sum(dcc ** 2, axis=3)
                dists = [
                    d2[u, v, j] ** 0.5
                    for u in range(len(indices))
                    for v in range(len(ref_indices))
                    for j in range(len(r) ** 3)
                    if u != v or j != indx0
                ]

                r_indices = [int(dist / dr) for dist in filter(lambda e: e < rmax, dists)]
                dns.update(r_indices)

            for indx, dn in dns.most_common(ngrid):
                gdrt[it, :] += gaussians[indx, :] * dn / aux_factor[indx] / rho

        self.obj = diffusion_analyzer
        self.avg_nsteps = avg_nsteps
        self.step_skip = step_skip
        self.rtgrid = rtgrid
        self.interval = interval
        self.gsrt = gsrt
        self.gdrt = gdrt

        # time interval (in ps) in gsrt and gdrt.
        self.timeskip = self.obj.time_step * self.obj.step_skip * step_skip / 1000.0

    def get_3d_plot(self, figsize: Tuple = (12, 8), mode: str = "distinct"):
        """
        Plot 3D self-part or distinct-part of van Hove function, which is
        specified by the input argument 'type'.
        """

        assert mode in ["distinct", "self"]

        if mode == "distinct":
            grt = self.gdrt.copy()
            vmax = 4.0
            cb_ticks = [0, 1, 2, 3, 4]
            cb_label = "$G_d$($t$,$r$)"
        elif mode == "self":
            grt = self.gsrt.copy()
            vmax = 1.0
            cb_ticks = [0, 1]
            cb_label = r"4$\pi r^2G_s$($t$,$r$)"

        y = np.arange(np.shape(grt)[1]) * self.interval[-1] / float(len(self.interval) - 1)
        x = np.arange(np.shape(grt)[0]) * self.timeskip
        X, Y = np.meshgrid(x, y, indexing="ij")

        ticksize = int(figsize[0] * 2.5)

        plt.figure(figsize=figsize, facecolor="w")
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)

        labelsize = int(figsize[0] * 3)

        plt.pcolor(X, Y, grt, cmap="jet", vmin=grt.min(), vmax=vmax)
        plt.xlabel("Time (ps)", size=labelsize)
        plt.ylabel(r"$r$ ($\AA$)", size=labelsize)
        plt.axis([x.min(), x.max(), y.min(), y.max()])

        cbar = plt.colorbar(ticks=cb_ticks)
        cbar.set_label(label=cb_label, size=labelsize)
        cbar.ax.tick_params(labelsize=ticksize)
        plt.tight_layout()

        return plt

    def get_1d_plot(self, mode: str = "distinct", times: List = [0.0], colors: List = None):
        """
        Plot the van Hove function at given r or t.

        Args:
            mode (str): Specify which part of van Hove function to be plotted.
            times (list of float): Time moments (in ps) in which the van Hove
                            function will be plotted.
            colors (list strings/tuples): Additional color settings. If not set,
                            seaborn.color_plaette("Set1", 10) will be used.
        """
        if colors is None:
            import seaborn as sns

            colors = sns.color_palette("Set1", 10)

        assert mode in ["distinct", "self"]
        assert len(times) <= len(colors)

        if mode == "distinct":
            grt = self.gdrt.copy()
            ylabel = "$G_d$($t$,$r$)"
            ylim = [-0.005, 4.0]
        elif mode == "self":
            grt = self.gsrt.copy()
            ylabel = r"4$\pi r^2G_s$($t$,$r$)"
            ylim = [-0.005, 1.0]

        plt = pretty_plot(12, 8)

        for i, time in enumerate(times):
            index = int(np.round(time / self.timeskip))
            index = min(index, np.shape(grt)[0] - 1)
            new_time = index * self.timeskip
            label = str(new_time) + " ps"
            plt.plot(self.interval, grt[index], color=colors[i], label=label, linewidth=4.0)

        plt.xlabel(r"$r$ ($\AA$)")
        plt.ylabel(ylabel)
        plt.legend(loc="upper right", fontsize=36)
        plt.xlim(0.0, self.interval[-1] - 1.0)
        plt.ylim(ylim[0], ylim[1])
        plt.tight_layout()

        return plt


class EvolutionAnalyzer:
    """
    Analyze the evolution of structures during AIMD simulations.
    """

    def __init__(self, structures: List, rmax: float = 10, step: int = 1, time_step: int = 2):
        """
        Initialization the EvolutionAnalyzer from MD simulations. From the
        structures obtained from MD simulations, we can analyze the structure
        evolution with time by some quantitative characterization such as RDF
        and atomic distribution.

        If you use this class, please consider citing the following paper:
        Tang, H.; Deng, Z.; Lin, Z.; Wang, Z.; Chu, I-H; Chen, C.; Zhu, Z.;
        Zheng, C.; Ong, S. P. "Probing Solid–Solid Interfacial Reactions in
        All-Solid-State Sodium-Ion Batteries with First-Principles
        Calculations", Chem. Mater. (2018), 30(1), pp 163-173.

        Args:
            structures ([Structure]): The list of structures obtained from MD
                simulations.
            rmax (float): Maximum of radial grid (the minimum is always zero).
            step (int): indicate the interval of input structures, which is used
                to calculated correct time step.
            time_step(int): the time step in fs, POTIM in INCAR.
        """
        self.pairs = self.get_pairs(structures[0])
        self.structure = structures[0]
        self.structures = structures
        self.rmax = rmax
        self.step = step
        self.time_step = time_step

    @staticmethod
    def get_pairs(structure: Structure):
        """
        Get all element pairs in a structure.

        Args:
            structure (Structure): structure

        Returns:
            list of tuples
        """
        specie_list = [s.name for s in structure.types_of_specie]
        pairs = itertools.combinations_with_replacement(specie_list, 2)

        return list(pairs)

    @staticmethod
    def rdf(structure: Structure, pair: Tuple, ngrid: int = 101, rmax: float = 10):
        """
        Process rdf from a given structure and pair.

        Args:
            structure (Structure): input structure.
            pair (str tuple): e.g. ("Na", "Na").
            ngrid (int): Number of radial grid points.
            rmax (float): Maximum of radial grid (the minimum is always zero).

        Returns:
            rdf (np.array)
        """
        r = RadialDistributionFunction.from_species(
            [structure],
            ngrid=ngrid,
            rmax=rmax,
            sigma=0.1,
            species=(pair[0]),
            reference_species=(pair[1]),
        )

        return r.rdf

    @staticmethod
    def atom_dist(
        structure: Structure,
        specie: str,
        ngrid: int = 101,
        window: float = 1,
        direction: str = "c",
    ):
        """
        Get atomic distribution for a given specie.

        Args:
            structure (Structure): input structure
            specie (str): species string for an element
            ngrid (int): Number of radial grid points.
            window (float): number of atoms will be counted within the range
                (i-window, i+window), unit is angstrom.
            direction (str): Choose from "a", "b" and "c". Default is "c".

        Returns:
            density (np.array): atomic concentration along one direction.
        """
        if direction in ["a", "b", "c"]:
            l = structure.lattice.__getattribute__(direction)
            ind = ["a", "b", "c"].index(direction)
            assert window <= l, "Window range exceeds valid bounds!"
        else:
            raise ValueError("Choose from a, b and c!")

        atom_list = [site for site in structure.sites if site.species_string == specie]
        atom_total = structure.composition[specie]
        density = []

        for i in np.linspace(0, l - window, ngrid):
            atoms = []
            for j in [-1, 0, 1]:
                temp = [s for s in atom_list if i - window < s.coords[ind] % l + l * j < i + window]
                atoms.extend(temp)

            density.append(len(atoms) / atom_total)

        return np.array(density)

    def get_df(self, func: Callable, save_csv: str = None, **kwargs):
        """
        Get the data frame for a given pair. This step would be very slow if
        there are hundreds or more structures to parse.

        Args:
            func (FunctionType): structure to spectrum function. choose from
                rdf (to get radial distribution function, pair required) or
                get_atomic_distribution (to get atomic distribution, specie
                required). Extra parameters can be parsed using kwargs.
                e.g. To get rdf dataframe:
                    df = EvolutionAnalyzer.get_df(
                        func=EvolutionAnalyzer.rdf, pair=("Na", "Na"))
                e.g. To get atomic distribution:
                    df = EvolutionAnalyzer.get_df(
                        func=EvolutionAnalyzer.atom_dist, specie="Na")
            save_csv (str): save pandas DataFrame to csv.

        Returns:
            pandas.DataFrame object: index is the radial distance in Angstrom,
                and column is the time step in ps.
        """
        prop_table = []
        ngrid = kwargs.get("ngrid", 101)
        if func == self.rdf:
            kwargs["rmax"] = self.rmax

        for structure in self.structures:
            prop_table.append(func(structure, **kwargs))

        index = np.arange(len(self.structures)) * self.time_step * self.step / 1000
        columns = np.linspace(0, self.rmax, ngrid)
        df = pds.DataFrame(prop_table, index=index, columns=columns)

        if save_csv is not None:
            df.to_csv(save_csv)

        return df

    @staticmethod
    def get_min_dist(df: pds.DataFrame, tol: float = 1e-10):
        """
        Get the shortest pair distance from the given DataFrame.

        Args:
            df (DataFrame): index is the radial distance in Angstrom, and
                column is the time step in ps.
            tol (float): any float number less than tol is considered as zero.

        Returns:
            The shortest pair distance throughout the table.
        """
        # TODO: Add unittest
        for i, col in enumerate(df.columns):
            min_dist = df.min(axis="index")[i]
            if min_dist > tol:
                return float(col)
        raise RuntimeError("Getting min dist failed.")

    @staticmethod
    def plot_evolution_from_data(
        df: pds.DataFrame,
        x_label: str = None,
        cb_label: str = None,
        cmap=plt.cm.plasma,  # pylint: disable=E1101
    ):
        """
        Plot the evolution with time for a given DataFrame. It can be RDF,
        atomic distribution or other characterization data we might
        implement in the future.

        Args:

            df (pandas.DataFrame): input DataFrame object, index is the radial
                distance in Angstrom, and column is the time step in ps.
            x_label (str): x label
            cb_label (str): color bar label
            cmap (color map): the color map used in heat map.
                cmocean.cm.thermal is recommended
        Returns:
            matplotlib.axes._subplots.AxesSubplot object
        """
        import seaborn as sns

        sns.set_style("white")

        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["xtick.labelsize"] = 20
        plt.rcParams["ytick.labelsize"] = 20
        plt.rcParams["xtick.major.pad"] = 10

        fig, ax = plt.subplots(figsize=(12, 8), facecolor="w")
        ax = sns.heatmap(
            df,
            linewidths=0,
            cmap=cmap,
            annot=False,
            cbar=True,
            xticklabels=10,
            yticklabels=25,
            rasterized=True,
        )
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.collections[0].colorbar.set_label(cb_label, fontsize=30)

        plt.xticks(rotation="horizontal")

        plt.xlabel(x_label, fontsize=30)
        plt.ylabel("Time (ps)", fontsize=30)

        plt.yticks(rotation="horizontal")
        plt.tight_layout()

        return plt

    def plot_rdf_evolution(
        self,
        pair: Tuple,
        cmap=plt.cm.plasma,  # pylint: disable=E1101
        df: pds.DataFrame = None,
    ):
        """
        Plot the RDF evolution with time for a given pair.

        Args:
            pair (str tuple): e.g. ("Na", "Na")
            cmap (color map): the color map used in heat map.
                cmocean.cm.thermal is recommended
            df (DataFrame): external data, index is the radial distance in
                Angstrom, and column is the time step in ps.
        Returns:
            matplotlib.axes._subplots.AxesSubplot object
        """
        if df is None:
            df = self.get_df(func=EvolutionAnalyzer.rdf, pair=pair)
        x_label, cb_label = f"$r$ ({pair[0]}-{pair[1]}) ($\\rm\\AA$)", "$g(r)$"
        p = self.plot_evolution_from_data(df=df, x_label=x_label, cb_label=cb_label, cmap=cmap)

        return p

    def plot_atomic_evolution(
        self,
        specie: str,
        direction: str = "c",
        cmap=plt.cm.Blues,  # pylint: disable=E1101
        df: pds.DataFrame = None,
    ):
        """
        Plot the atomic distribution evolution with time for a given species.

        Args:
            specie (str): species string for an element.
            direction (str): Choose from "a", "b", "c". Default to "c".
            cmap (color map): the color map used in heat map.
            df (DataFrame): external data, index is the atomic distance in
                         Angstrom, and column is the time step in ps.
        Returns:
            matplotlib.axes._subplots.AxesSubplot object
        """
        if df is None:
            df = self.get_df(func=EvolutionAnalyzer.atom_dist, specie=specie, direction=direction)
        x_label, cb_label = (
            f"Atomic distribution along {direction}",
            "Probability",
        )
        p = self.plot_evolution_from_data(df=df, x_label=x_label, cb_label=cb_label, cmap=cmap)
        return p
