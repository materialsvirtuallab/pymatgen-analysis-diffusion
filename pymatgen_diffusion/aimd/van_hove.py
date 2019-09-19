# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

from collections import Counter
import numpy as np
import itertools
import pandas as pds
from pymatgen.analysis.diffusion_analyzer import DiffusionAnalyzer
from scipy import stats
from scipy.stats import norm
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from pymatgen.util.plotting import pretty_plot

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

    def __init__(self, diffusion_analyzer, avg_nsteps=50, ngrid=101, rmax=10.0,
                 step_skip=50, sigma=0.1, cell_range=1, species=("Li", "Na"),
                 reference_species=None, indices=None):
        """
        Initiation.

        Args:
            diffusion_analyzer (DiffusionAnalyzer): A
                pymatgen.analysis.diffusion_analyzer.DiffusionAnalyzer object
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
            indices = [j for j, site in enumerate(structure)
                       if site.specie.symbol in species]

        ref_indices = indices
        if reference_species:
            ref_indices = [j for j, site in enumerate(structure)
                           if site.specie.symbol in reference_species]

        rho = float(len(indices)) / lattice.volume

        # reduced time grid
        rtgrid = np.arange(0.0, reduced_nt)
        # van Hove functions
        gsrt = np.zeros((reduced_nt, ngrid), dtype=np.double)
        gdrt = np.zeros((reduced_nt, ngrid), dtype=np.double)

        tracking_ions = []
        ref_ions = []

        # auxiliary factor for 4*\pi*r^2
        aux_factor = 4.0 * np.pi * interval ** 2
        aux_factor[0] = np.pi * dr ** 2

        for i, ss in enumerate(
                diffusion_analyzer.get_drift_corrected_structures()):
            all_fcoords = np.array(ss.frac_coords)
            tracking_ions.append(all_fcoords[indices, :])
            ref_ions.append(all_fcoords[ref_indices, :])

        tracking_ions = np.array(tracking_ions)
        ref_ions = np.array(ref_ions)

        gaussians = norm.pdf(interval[:, None], interval[None, :],
                             sigma) / float(avg_nsteps) / float(
            len(ref_indices))

        # calculate self part of van Hove function
        image = np.array([0, 0, 0])
        for it in range(reduced_nt):
            dns = Counter()
            it0 = min(it * step_skip, ntsteps)
            for it1 in range(avg_nsteps):
                dists = [lattice.get_distance_and_image(tracking_ions[it1][u],
                                                        tracking_ions[
                                                            it0 + it1][u],
                                                        jimage=image)[0] for u
                         in range(len(indices))]
                dists = filter(lambda e: e < rmax, dists)

                r_indices = [int(dist / dr) for dist in dists]
                dns.update(r_indices)

            for indx, dn in dns.most_common(ngrid):
                gsrt[it, :] += gaussians[indx, :] * dn

        # calculate distinct part of van Hove function of species
        r = np.arange(-cell_range, cell_range + 1)
        arange = r[:, None] * np.array([1, 0, 0])[None, :]
        brange = r[:, None] * np.array([0, 1, 0])[None, :]
        crange = r[:, None] * np.array([0, 0, 1])[None, :]
        images = arange[:, None, None] + brange[None, :, None] + crange[None,
                                                                 None, :]
        images = images.reshape((len(r) ** 3, 3))

        # find the zero image vector
        zd = np.sum(images ** 2, axis=1)
        indx0 = np.argmin(zd)

        for it in range(reduced_nt):
            dns = Counter()
            it0 = min(it * step_skip, ntsteps)

            for it1 in range(avg_nsteps):
                dcf = (tracking_ions[it0 + it1, :, None, None, :] + images[None,
                                                                    None, :,
                                                                    :] -
                       ref_ions[it1, None, :, None, :])
                dcc = lattice.get_cartesian_coords(dcf)
                d2 = np.sum(dcc ** 2, axis=3)
                dists = [d2[u, v, j] ** 0.5 for u in range(len(indices))
                         for v in range(len(ref_indices))
                         for j in range(len(r) ** 3) if u != v or j != indx0]
                dists = filter(lambda e: e < rmax, dists)

                r_indices = [int(dist / dr) for dist in dists]
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

    def get_3d_plot(self, figsize=(12, 8), type="distinct"):
        """
        Plot 3D self-part or distinct-part of van Hove function, which is
        specified by the input argument 'type'.
        """

        assert type in ["distinct", "self"]

        if type == "distinct":
            grt = self.gdrt.copy()
            vmax = 4.0
            cb_ticks = [0, 1, 2, 3, 4]
            cb_label = "$G_d$($t$,$r$)"
        elif type == "self":
            grt = self.gsrt.copy()
            vmax = 1.0
            cb_ticks = [0, 1]
            cb_label = "4$\pi r^2G_s$($t$,$r$)"

        y = np.arange(np.shape(grt)[1]) * self.interval[-1] / float(
            len(self.interval) - 1)
        x = np.arange(
            np.shape(grt)[0]) * self.timeskip
        X, Y = np.meshgrid(x, y, indexing="ij")

        ticksize = int(figsize[0] * 2.5)

        plt.figure(figsize=figsize, facecolor="w")
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)

        labelsize = int(figsize[0] * 3)

        plt.pcolor(X, Y, grt, cmap="jet", vmin=grt.min(), vmax=vmax)
        plt.xlabel("Time (ps)", size=labelsize)
        plt.ylabel("$r$ ($\AA$)", size=labelsize)
        plt.axis([x.min(), x.max(), y.min(), y.max()])

        cbar = plt.colorbar(ticks=cb_ticks)
        cbar.set_label(label=cb_label, size=labelsize)
        cbar.ax.tick_params(labelsize=ticksize)
        plt.tight_layout()

        return plt

    def get_1d_plot(self, mode="distinct", times=[0.0], colors=None):
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
            ylabel = "4$\pi r^2G_s$($t$,$r$)"
            ylim = [-0.005, 1.0]

        plt = pretty_plot(12, 8)

        for i, time in enumerate(times):
            index = int(np.round(time / self.timeskip))
            index = min(index, np.shape(grt)[0] - 1)
            new_time = index * self.timeskip
            label = str(new_time) + " ps"
            plt.plot(self.interval, grt[index], color=colors[i], label=label,
                     linewidth=4.0)

        plt.xlabel("$r$ ($\AA$)")
        plt.ylabel(ylabel)
        plt.legend(loc='upper right', fontsize=36)
        plt.xlim(0.0, self.interval[-1] - 1.0)
        plt.ylim(ylim[0], ylim[1])
        plt.tight_layout()

        return plt


class RadialDistributionFunction:
    """
    Calculate the average radial distribution function for a given set of
    structures.
    """

    # todo: volume change correction for NpT RDF
    def __init__(self, structures, indices, reference_indices, ngrid=101,
                 rmax=10.0, cell_range=1, sigma=0.1):
        """
        Args:
            structures ([Structure]): List of structure
                objects with the same composition. Allow for ensemble averaging.
            ngrid (int): Number of radial grid points.
            rmax (float): Maximum of radial grid (the minimum is always zero).
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

        lattice = structures[0].lattice

        if len(indices) < 1:
            raise ValueError("Given species are not in the structure!")

        self.rho = float(len(indices)) / lattice.volume
        fcoords_list = []
        ref_fcoords_list = []

        for s in structures:
            all_fcoords = np.array(s.frac_coords)
            fcoords_list.append(all_fcoords[indices, :])
            ref_fcoords_list.append(all_fcoords[reference_indices, :])

        dr = rmax / (ngrid - 1)
        interval = np.linspace(0.0, rmax, ngrid)
        rdf = np.zeros((ngrid), dtype=np.double)
        raw_rdf = np.zeros((ngrid), dtype=np.double)

        dns = Counter()

        # generate the translational vectors
        r = np.arange(-cell_range, cell_range + 1)
        arange = r[:, None] * np.array([1, 0, 0])[None, :]
        brange = r[:, None] * np.array([0, 1, 0])[None, :]
        crange = r[:, None] * np.array([0, 0, 1])[None, :]
        images = arange[:, None, None] + \
                 brange[None, :, None] + crange[None, None, :]
        images = images.reshape((len(r) ** 3, 3))

        # find the zero image vector
        zd = np.sum(images ** 2, axis=1)
        indx0 = np.argmin(zd)

        for fcoords, ref_fcoords in zip(fcoords_list, ref_fcoords_list):
            dcf = fcoords[:, None, None, :] + \
                  images[None, None, :, :] - ref_fcoords[None, :, None, :]
            dcc = lattice.get_cartesian_coords(dcf)
            d2 = np.sum(dcc ** 2, axis=3)
            dists = [d2[u, v, j] ** 0.5 for u in range(len(indices))
                     for v in range(len(reference_indices))
                     for j in range(len(r) ** 3) if
                     indices[u] != reference_indices[v] or j != indx0]
            dists = filter(lambda e: e < rmax + 1e-8, dists)
            r_indices = [int(dist / dr) for dist in dists]
            dns.update(r_indices)

        for indx, dn in dns.most_common(ngrid):
            if indx > len(interval) - 1:
                continue

            ff = 4.0 / 3.0 * np.pi * (
                        interval[indx + 1] ** 3 - interval[indx] ** 3)

            rdf[:] += (stats.norm.pdf(interval, interval[indx], sigma) * dn /
                       float(len(reference_indices)) / ff / self.rho / len(
                        fcoords_list) * dr)
            # additional dr factor renormalises overlapping gaussians.
            raw_rdf[indx] += dn / float(
                len(reference_indices)) / ff / self.rho / len(
                fcoords_list)

        self.structures = structures
        self.cell_range = cell_range
        self.rmax = rmax
        self.ngrid = ngrid
        self.species = {structures[0][i].species_string for i in indices}
        self.reference_species = {structures[0][i].species_string for i in
                                  reference_indices}
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
    def from_species(cls, structures, ngrid=101, rmax=10.0, cell_range=1,
                     sigma=0.1, species=("Li", "Na"), reference_species=None):
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
        indices = [j for j, site in enumerate(structures[0])
                   if site.specie.symbol in species]
        if reference_species:
            reference_indices = [j for j, site in enumerate(structures[0])
                                 if site.specie.symbol in reference_species]

            if len(reference_indices) < 1:
                raise ValueError(
                    "Given reference species are not in the structure!")
        else:
            reference_indices = indices

        return cls(structures=structures, ngrid=ngrid, rmax=rmax,
                   cell_range=cell_range, sigma=sigma, indices=indices,
                   reference_indices=reference_indices)

    @property
    def coordination_number(self):
        """
        returns running coordination number

        Returns:
            numpy array
        """
        intervals = np.append(self.interval, self.interval[-1] + self.dr)
        return np.cumsum(self.raw_rdf * self.rho * 4.0 / 3.0 * np.pi *
                         (intervals[1:] ** 3 - intervals[:-1] ** 3))

    def get_rdf_plot(self, label=None, xlim=(0.0, 8.0), ylim=(-0.005, 3.0),
                     loc_peak=False):
        """
        Plot the average RDF function.

        Args:
            label (str): The legend label.
            xlim (list): Set the x limits of the current axes.
            ylim (list): Set the y limits of the current axes.
            loc_peak (bool): Label peaks if True.
        """

        if label is None:
            symbol_list = [e.symbol for e in
                           self.structures[0].composition.keys()]
            symbol_list = [symbol for symbol in symbol_list if
                           symbol in self.species]

            if len(symbol_list) == 1:
                label = symbol_list[0]
            else:
                label = "-".join(symbol_list)

        plt = pretty_plot(12, 8)
        plt.plot(self.interval, self.rdf, label=label, linewidth=4.0, zorder=1)

        if loc_peak:
            plt.scatter(self.peak_r, self.peak_rdf, marker="P", s=240, c='k',
                        linewidths=0.1, alpha=0.7, zorder=2, label="Peaks")

        plt.xlabel("$r$ ($\\rm\AA$)")
        plt.ylabel("$g(r)$")
        plt.legend(loc='upper right', fontsize=36)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.tight_layout()

        return plt

    def export_rdf(self, filename):
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
                f.write(delimiter.join(["%s" % v for v in [r, gr]]))
                f.write("\n")


class EvolutionAnalyzer:
    def __init__(self, structures, rmax=10, step=1, time_step=2):
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
    def get_pairs(structure):
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
    def rdf(structure, pair, ngrid=101, rmax=10):
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
            [structure], ngrid=ngrid, rmax=rmax, sigma=0.1, species=(pair[0]),
            reference_species=(pair[1]))

        return r.rdf

    @staticmethod
    def atom_dist(structure, specie, ngrid=101, window=1, direction="c"):
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

        atom_list = [site for site in structure.sites
                     if site.species_string == specie]
        atom_total = structure.composition[specie]
        density = []

        for i in np.linspace(0, l - window, ngrid):
            atoms = []
            for j in [-1, 0, 1]:
                temp = [
                    s for s in atom_list if
                    i - window < s.coords[ind] % l + l * j < i + window]
                atoms.extend(temp)

            density.append(len(atoms) / atom_total)

        return np.array(density)

    def get_df(self, func, save_csv=None, **kwargs):
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

        index = np.arange(
            len(self.structures)) * self.time_step * self.step / 1000
        columns = np.linspace(0, self.rmax, ngrid)
        df = pds.DataFrame(prop_table, index=index, columns=columns)

        if save_csv is not None:
            df.to_csv(save_csv)

        return df

    @staticmethod
    def get_min_dist(df, tol=1e-10):
        """
        Get the shortest pair distance from the given DataFrame.

        Args:
            df (DataFrame): index is the radial distance in Angstrom, and
                column is the time step in ps.
            tol (float): any float number less than tol is considered as zero.

        Returns:
            The shorted pair distance throughout the table.
        """
        # TODO: Add unittest
        for i, col in enumerate(df.columns):
            min_dist = df.min(axis="index")[i]
            if min_dist > tol:
                return float(col)

    @staticmethod
    def plot_evolution_from_data(df, x_label=None, cb_label=None,
                                 cmap=plt.cm.plasma):
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

        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams["font.family"] = 'sans-serif'
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20
        plt.rcParams['xtick.major.pad'] = 10

        fig, ax = plt.subplots(figsize=(12, 8), facecolor="w")
        ax = sns.heatmap(df, linewidths=0, cmap=cmap, annot=False, cbar=True,
                         xticklabels=10, yticklabels=25, rasterized=True)
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.collections[0].colorbar.set_label(cb_label, fontsize=30)

        plt.xticks(rotation="horizontal")

        plt.xlabel(x_label, fontsize=30)
        plt.ylabel('Time (ps)', fontsize=30)

        plt.yticks(rotation="horizontal")
        plt.tight_layout()

        return plt

    def plot_rdf_evolution(self, pair, cmap=plt.cm.plasma, df=None):
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
        x_label, cb_label = "$r$ ({}-{}) ($\\rm\AA$)".format(*pair), "$g(r)$"
        p = self.plot_evolution_from_data(df=df, x_label=x_label,
                                          cb_label=cb_label, cmap=cmap)

        return p

    def plot_atomic_evolution(self, specie, direction="c",
                              cmap=plt.cm.Blues, df=None):
        """
        Plot the atomic distribution evolution with time for a given species.

        Args:
            specie (str ): species string for an element.
            direction (str): Choose from "a", "b", "c". Default to "c".
            cmap (color map): the color map used in heat map.
            df (DataFrame): external data, index is the atomic distance in
                         Angstrom, and column is the time step in ps.
        Returns:
            matplotlib.axes._subplots.AxesSubplot object
        """
        if df is None:
            df = self.get_df(func=EvolutionAnalyzer.atom_dist, specie=specie,
                             direction=direction)
        x_label, cb_label = "Atomic distribution along {} ".format(
            direction), "Probability"
        p = self.plot_evolution_from_data(df=df, x_label=x_label,
                                          cb_label=cb_label, cmap=cmap)
        return p


if __name__ == "__main__":
    # import os
    # import json
    #
    # tests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    #                          "tests")
    # data_file = os.path.join(tests_dir, "cNa3PS4_pda.json")
    # data = json.load(open(data_file, "r"))
    # obj = DiffusionAnalyzer.from_dict(data)
    # structure_list = []
    # for i, s in enumerate(obj.get_drift_corrected_structures()):
    #     structure_list.append(s)
    #     if i == 9: break
    #
    # s = structure_list[0]
    # indices = [i for (i, site) in enumerate(s) if
    #            site.species_string in ["Na", "P", "S"]]
    # obj = RadialDistributionFunction(
    #     structures=structure_list, ngrid=101, rmax=10.0, cell_range=1,
    #     sigma=0.1, indices=indices, reference_indices=indices)

    pass
