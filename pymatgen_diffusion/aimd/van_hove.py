# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

from __future__ import division, unicode_literals, print_function
from collections import Counter
from scipy.stats import norm
import matplotlib.pyplot as plt
from pymatgen.util.plotting_utils import get_publication_quality_plot
from scipy import stats
import numpy as np

__author__ = "Iek-Heng Chu"
__version__ = 1.0
__date__ = "04/15"


class VanHoveAnalysis(object):
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
                 step_skip=50, sigma=0.1, cellrange=1, species=("Li", "Na"),
                 indices=None):
        """
        Initization.

        Args:
            diffusion_analyzer (DiffusionAnalyzer): A
                pymatgen.analysis.diffusion_analyzer.DiffusionAnalyzer object
            avg_nsteps (int): Number of t0 used for statistical average
            ngrid (int): Number of radial grid points
            rmax (float): Maximum of radial grid (the minimum is always set zero)
            step_skip (int): # of time steps skipped during analysis. It defines
                        the resolution of the reduced time grid
            sigma (float): Smearing of a Gaussian function
            cellrange (int): Range of translational vector elements associated with
                            supercell. Default is 1, i.e. including the adjecent
                            image cells along all three directions.
            species ([string]): a list of specie symbols of interest
            indices (list of int): If not None, only a subset of atomic indices
                            will be selected for the analysis.
        """

        # initial check
        if step_skip <= 0:
            raise ValueError("skip_step should be >=1!")

        nions, nsteps, ndim = diffusion_analyzer.disp.shape

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

        if indices is None:
            indices = [j for j, site in enumerate(diffusion_analyzer.structure)
                       if site.specie.symbol in species]

        rho = float(len(indices)) / lattice.volume

        # reduced time grid
        rtgrid = np.arange(0.0, reduced_nt)
        # van Hove functions
        gsrt = np.zeros((reduced_nt, ngrid), dtype=np.double)
        gdrt = np.zeros((reduced_nt, ngrid), dtype=np.double)

        tracking_ions = []

        # auxiliary factor for 4*\pi*r^2
        aux_factor = 4.0 * np.pi * interval ** 2
        aux_factor[0] = np.pi * dr ** 2

        for i, ss in enumerate(
                diffusion_analyzer.get_drift_corrected_structures()):
            all_fcoords = np.array(ss.frac_coords)
            tracking_ions.append(all_fcoords[indices, :])

        tracking_ions = np.array(tracking_ions)

        gaussians = norm.pdf(interval[:, None], interval[None, :], sigma) / \
                    float(avg_nsteps) / float(len(indices))

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
        r = np.arange(-cellrange, cellrange + 1)
        arange = r[:, None] * np.array([1, 0, 0])[None, :]
        brange = r[:, None] * np.array([0, 1, 0])[None, :]
        crange = r[:, None] * np.array([0, 0, 1])[None, :]
        images = arange[:, None, None] + brange[None, :, None] + \
                 crange[None, None, :]
        images = images.reshape((len(r) ** 3, 3))

        # find the zero image vector
        zd = np.sum(images ** 2, axis=1)
        indx0 = np.argmin(zd)

        for it in range(reduced_nt):
            dns = Counter()
            it0 = min(it * step_skip, ntsteps)

            for it1 in range(avg_nsteps):
                dcf = tracking_ions[it0 + it1, :, None, None, :] + \
                      images[None, None, :, :] - \
                      tracking_ions[it1, None, :, None, :]
                dcc = lattice.get_cartesian_coords(dcf)
                d2 = np.sum(dcc ** 2, axis=3)
                dists = [d2[u, v, j] ** 0.5 for u in range(len(indices)) for v
                         in range(len(indices)) \
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
        self.timeskip = self.obj.time_step * self.obj.step_skip * step_skip / \
                        1000.0

    def get_3d_plot(self, figsize=(12, 8), type="distinct"):
        """
        Plot 3D self-part or distinct-part of van Hove function, which is specified
        by the input argument 'type'.
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

    def get_1d_plot(self, type="distinct", times=[0.0], colors=["r", "g", "b"]):
        """
        Plot the van Hove function at given r or t.

        Args:
            type (str): Specify which part of van Hove function to be plotted.
            times (list of float): Time moments (in ps) in which the van Hove
                            function will be plotted.
            colors (list of str): Default list of colors for plotting.
        """

        assert type in ["distinct", "self"]
        assert len(times) <= len(colors)

        if type == "distinct":
            grt = self.gdrt.copy()
            ylabel = "$G_d$($t$,$r$)"
            ylim = [-0.005, 4.0]
        elif type == "self":
            grt = self.gsrt.copy()
            ylabel = "4$\pi r^2G_s$($t$,$r$)"
            ylim = [-0.005, 1.0]

        plt = get_publication_quality_plot(12, 8)

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


class RadialDistributionFunction(object):
    """
    Calculate the average radial distribution function for a given set of structures.
    """

    def __init__(self, structures, ngrid=101, rmax=10.0, cellrange=1, sigma=0.1,
                 species=("Li", "Na"), reference_species=None):
        """
        Args:
            structures (list of pmg_structure objects): List of structure
                objects with the same composition. Allow for ensemble averaging.
            ngrid (int): Number of radial grid points.
            rmax (float): Maximum of radial grid (the minimum is always set zero).
            cellrange (int): Range of translational vector elements associated
                with supercell. Default is 1, i.e. including the adjecent image
                cells along all three directions.
            sigma (float): Smearing of a Gaussian function.
            species ([string]): A list of specie symbols of interest.
            reference_species ([string]): set this option along with 'species'
                parameter to compute pair radial distribution function.
                eg: species=["H"], reference_species=["O"] to compute
                    O-H pair distribution in a water MD simulation.
        """

        if ngrid - 1 <= 0:
            raise ValueError("ngrid should be greater than 1!")

        if sigma <= 0.0:
            raise ValueError("sigma should be > 0!")

        lattice = structures[0].lattice
        indices = [j for j, site in enumerate(structures[0])
                   if site.specie.symbol in species]
        if len(indices) == 0:
            raise ValueError("Given species are not in the structure!")

        ref_indices = indices
        if reference_species:
            ref_indices = [j for j, site in enumerate(structures[0])
                           if site.specie.symbol in reference_species]

        self.rho = float(len(indices)) / lattice.volume
        fcoords_list = []
        ref_fcoords_list = []

        for s in structures:
            all_fcoords = np.array(s.frac_coords)
            fcoords_list.append(all_fcoords[indices, :])
            ref_fcoords_list.append(all_fcoords[ref_indices, :])

        dr = rmax / (ngrid - 1)
        interval = np.linspace(0.0, rmax, ngrid)
        rdf = np.zeros((ngrid), dtype=np.double)
        dns = Counter()

        # generate the translational vectors
        r = np.arange(-cellrange, cellrange + 1)
        arange = r[:, None] * np.array([1, 0, 0])[None, :]
        brange = r[:, None] * np.array([0, 1, 0])[None, :]
        crange = r[:, None] * np.array([0, 0, 1])[None, :]
        images = arange[:, None, None] + brange[None, :, None] + crange[None,
                                                                 None, :]
        images = images.reshape((len(r) ** 3, 3))

        # find the zero image vector
        zd = np.sum(images ** 2, axis=1)
        indx0 = np.argmin(zd)

        for fcoords, ref_fcoords in zip(fcoords_list, ref_fcoords_list):
            dcf = fcoords[:, None, None, :] + images[None, None, :,
                                              :] - ref_fcoords[None, :, None, :]
            dcc = lattice.get_cartesian_coords(dcf)
            d2 = np.sum(dcc ** 2, axis=3)
            dists = [d2[u, v, j] ** 0.5 for u in range(len(indices)) for v in
                     range(len(ref_indices))
                     for j in range(len(r) ** 3) if u != v or j != indx0]
            dists = filter(lambda e: e < rmax + 1e-8, dists)
            r_indices = [int(dist / dr) for dist in dists]
            dns.update(r_indices)

        for indx, dn in dns.most_common(ngrid):
            if indx > len(interval) - 1: continue

            if indx == 0:
                ff = np.pi * dr ** 2
            else:
                ff = 4.0 * np.pi * interval[indx] ** 2

            rdf[:] += stats.norm.pdf(interval, interval[indx], sigma) * dn \
                      / float(len(ref_indices)) / ff / self.rho / len(
                fcoords_list)

        self.structures = structures
        self.rdf = rdf
        self.interval = interval
        self.cellrange = cellrange
        self.rmax = rmax
        self.ngrid = ngrid
        self.species = species
        self.dr = dr

    @property
    def coordination_number(self):
        """
        returns running coordination number

        Returns:
            numpy array
        """
        return np.cumsum(self.rdf * self.rho * 4.0 * np.pi * self.interval ** 2 * self.dr)

    def get_rdf_plot(self, label=None, xlim=[0.0, 8.0], ylim=[-0.005, 3.0]):
        """
        Plot the average RDF function.
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

        plt = get_publication_quality_plot(12, 8)
        plt.plot(self.interval, self.rdf, color="r", label=label, linewidth=4.0)
        plt.xlabel("$r$ ($\AA$)")
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
