"""A module to perform diffusion analyses.

For example, calculating diffusivity from mean square displacements etc.). If you use this module, please consider
citing the following papers::

    Ong, S. P., Mo, Y., Richards, W. D., Miara, L., Lee, H. S., & Ceder, G.
    (2013). Phase stability, electrochemical stability and ionic conductivity
    of the Li10+-1MP2X12 (M = Ge, Si, Sn, Al or P, and X = O, S or Se) family
    of superionic conductors. Energy & Environmental Science, 6(1), 148.
    doi:10.1039/c2ee23355j

    Mo, Y., Ong, S. P., & Ceder, G. (2012). First Principles Study of the
    Li10GeP2S12 Lithium Super Ionic Conductor Material. Chemistry of Materials,
    24(1), 15-17. doi:10.1021/cm203303y
"""

from __future__ import annotations

import multiprocessing
import warnings
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from monty.json import MSONable
from scipy.optimize import curve_fit

from pymatgen.analysis.structure_matcher import OrderDisorderElementComparator, StructureMatcher
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.util.coord import pbc_diff

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from matplotlib.axes import Axes

    from pymatgen.util.typing import PathLike, SpeciesLike

__author__ = "Will Richards, Shyue Ping Ong"
__version__ = "0.2"
__maintainer__ = "Will Richards"
__email__ = "wrichard@mit.edu"
__status__ = "Beta"
__date__ = "5/2/13"


class DiffusionAnalyzer(MSONable):
    """
    Class for performing diffusion analysis.

    .. attribute: diffusivity

        Diffusivity in cm^2 / s

    .. attribute: chg_diffusivity

        Charge diffusivity in cm^2 / s

    .. attribute: conductivity

        Conductivity in mS / cm

    .. attribute: chg_conductivity

        Conductivity derived from Nernst-Einstein equation using charge
        diffusivity, in mS / cm

    .. attribute: diffusivity_components

        A vector with diffusivity in the a, b and c directions in cm^2 / s

    .. attribute: conductivity_components

        A vector with conductivity in the a, b and c directions in mS / cm

    .. attribute: diffusivity_std_dev

        Std dev in diffusivity in cm^2 / s. Note that this makes sense only
        for non-smoothed analyses.

    .. attribute: chg_diffusivity_std_dev

        Std dev in charge diffusivity in cm^2 / s. Note that this makes sense only
        for non-smoothed analyses.

    .. attribute: conductivity_std_dev

        Std dev in conductivity in mS / cm. Note that this makes sense only
        for non-smoothed analyses.

    .. attribute: diffusivity_components_std_dev

        A vector with std dev. in diffusivity in the a, b and c directions in
        cm^2 / cm. Note that this makes sense only for non-smoothed analyses.

    .. attribute: conductivity_components_std_dev

        A vector with std dev. in conductivity in the a, b and c directions
        in mS / cm. Note that this makes sense only for non-smoothed analyses.

    .. attribute: max_framework_displacement

        The maximum (drift adjusted) distance of any framework atom from its
        starting location in A.

    .. attribute: max_ion_displacements

        nions x 1 array of the maximum displacement of each individual ion.

    .. attribute: msd

        nsteps x 1 array of the mean square displacement of specie.

    .. attribute: mscd

        nsteps x 1 array of the mean square charge displacement of specie.

    .. attribute: msd_components

        nsteps x 3 array of the MSD in each lattice direction of specie.

    .. attribute: sq_disp_ions

        The square displacement of all ion (both specie and other ions) as a
        nions x nsteps array.

    .. attribute: dt

        Time coordinate array.

    .. attribute: haven_ratio
        Haven ratio defined as diffusivity / chg_diffusivity.
    """

    def __init__(
        self,
        structure: Structure,
        displacements: np.ndarray,
        specie: SpeciesLike,
        temperature: float,
        time_step: int,
        step_skip: int,
        smoothed: bool | str = "max",
        min_obs: int = 30,
        avg_nsteps: int = 1000,
        lattices: np.ndarray | None = None,
        c_ranges: Sequence | None = None,
        c_range_include_edge: bool = False,
        structures: Sequence[Structure] | None = None,
    ) -> None:
        """
        This constructor is meant to be used with pre-processed data.
        Other convenient constructors are provided as class methods (see
        from_vaspruns and from_files).

        Given a matrix of displacements (see arguments below for expected
        format), the diffusivity is given by:

            D = 1 / 2dt * <mean square displacement>

        where d is the dimensionality, t is the time. To obtain a reliable
        diffusion estimate, a least squares regression of the MSD against
        time to obtain the slope, which is then related to the diffusivity.

        For traditional analysis, use smoothed=False and weighted=False.

        Args:
            structure (Structure): Initial structure.
            displacements (array): Numpy array of with shape [site,
                time step, axis]
            specie (Element/Species): Species to calculate diffusivity for as a
                String. E.g., "Li".
            temperature (float): Temperature of the diffusion run in Kelvin.
            time_step (int): Time step between measurements.
            step_skip (int): Sampling frequency of the displacements (
                time_step is multiplied by this number to get the real time
                between measurements)
            smoothed (str): Whether to smooth the MSD, and what mode to smooth.
                Supported modes are:

                i. "max", which tries to use the maximum #
                   of data points for each time origin, subject to a
                   minimum # of observations given by min_obs, and then
                   weights the observations based on the variance
                   accordingly. This is the default.
                ii. "constant", in which each timestep is averaged over
                    the number of time_steps given by min_steps.
                iii. None / False / any other false-like quantity. No
                   smoothing.

            min_obs (int): Used with smoothed="max". Minimum number of
                observations to have before including in the MSD vs dt
                calculation. E.g. If a structure has 10 diffusing atoms,
                and min_obs = 30, the MSD vs dt will be
                calculated up to dt = total_run_time / 3, so that each
                diffusing atom is measured at least 3 uncorrelated times.
                Only applies in smoothed="max".
            avg_nsteps (int): Used with smoothed="constant". Determines the
                number of time steps to average over to get the msd for each
                timestep. Default of 1000 is usually pretty good.
            lattices (array): Numpy array of lattice matrix of every step. Used
                for NPT-AIMD. For NVT-AIMD, the lattice at each time step is
                set to the lattice in the "structure" argument.
            c_ranges (list): A list of fractional ranges of z-axis to define
                regions and calculate regional MSD and regional diffusivity.
                If the start and end positions of a diffusing specie between
                two time steps are all within the c_ranges, that displacement
                is collected to calculate MSD in that time step. units: Å.
                Default to None.
            c_range_include_edge (bool): Whether to include displacements start or end
                on the edge of the defined c_ranges into the calculation of regional
                MSD. Default to False.
            structures (list): A list of trajectory structures only used in the
                calculation of regional MSD and regional diffusivity. These structures
                should be the same as those used to construct the diffusion analyzer.
                Default to None.
        """
        self.structure = structure
        self.disp = displacements
        self.specie = specie
        self.temperature = temperature
        self.time_step = time_step
        self.step_skip = step_skip
        self.min_obs = min_obs
        self.smoothed = smoothed
        self.avg_nsteps = avg_nsteps
        self.lattices = lattices if lattices is not None else np.array([structure.lattice.matrix.tolist()])

        indices: list = []
        framework_indices = []
        for i, site in enumerate(structure):
            if site.specie.symbol == specie:
                indices.append(i)
            else:
                framework_indices.append(i)
        if self.disp.shape[1] < 2:
            self.diffusivity = 0.0
            self.conductivity = 0.0
            self.diffusivity_components = np.array([0.0, 0.0, 0.0])
            self.conductivity_components = np.array([0.0, 0.0, 0.0])
            self.max_framework_displacement = 0
        else:
            framework_disp = self.disp[framework_indices]
            drift = np.average(framework_disp, axis=0)[None, :, :]

            # drift corrected position
            dc = self.disp - drift

            nions, nsteps, dim = dc.shape

            self.indices = indices

            if not smoothed:
                timesteps = np.arange(0, nsteps)
            elif smoothed == "constant":
                if nsteps <= avg_nsteps:
                    raise ValueError("Not enough data to calculate diffusivity")
                timesteps = np.arange(0, nsteps - avg_nsteps)
            else:
                # limit the number of sampled timesteps to 200
                min_dt = int(1000 / (self.step_skip * self.time_step))
                max_dt = min(len(indices) * nsteps // self.min_obs, nsteps)
                if min_dt >= max_dt:
                    raise ValueError("Not enough data to calculate diffusivity")
                timesteps = np.arange(min_dt, max_dt, max(int((max_dt - min_dt) / 200), 1))

            dt = timesteps * self.time_step * self.step_skip

            # calculate the smoothed msd values
            msd = np.zeros_like(dt, dtype=np.double)
            sq_disp_ions = np.zeros((len(dc), len(dt)), dtype=np.double)
            msd_components = np.zeros((*dt.shape, 3))

            # calculate mean square charge displacement
            mscd = np.zeros_like(msd, dtype=np.double)

            # calculate regional msd and number of diffusing specie in those regions
            msd_c_range = np.zeros_like(dt, dtype=np.double)
            msd_c_range_components = np.zeros((*dt.shape, 3))
            indices_c_range = []
            for i, n in enumerate(timesteps):
                if not smoothed:
                    dx = dc[:, i : i + 1, :]
                    dcomponents = dc[:, i : i + 1, :]
                elif smoothed == "constant":
                    dx = dc[:, i : i + avg_nsteps, :] - dc[:, 0:avg_nsteps, :]
                    dcomponents = dc[:, i : i + avg_nsteps, :] - dc[:, 0:avg_nsteps, :]
                else:
                    dx = dc[:, n:, :] - dc[:, :-n, :]
                    dcomponents = dc[:, n:, :] - dc[:, :-n, :]

                # Get msd
                sq_disp = dx**2
                sq_disp_ions[:, i] = np.average(np.sum(sq_disp, axis=2), axis=1)
                msd[i] = np.average(sq_disp_ions[:, i][indices])
                msd_components[i] = np.average(dcomponents[indices] ** 2, axis=(0, 1))

                # Get regional msd
                if c_ranges and structures:
                    if not c_range_include_edge:
                        for index in indices:
                            if any(
                                lower < structures[i][index].c < upper and lower < structures[i + 1][index].c < upper
                                for (lower, upper) in c_ranges
                            ):
                                indices_c_range.append(index)
                    else:
                        for index in indices:
                            if any(
                                lower <= structures[i][index].c <= upper or lower <= structures[i + 1][index].c <= upper
                                for (lower, upper) in c_ranges
                            ):
                                indices_c_range.append(index)
                    msd_c_range[i] = np.average(sq_disp_ions[:, i][indices_c_range])
                    msd_c_range_components[i] = np.average(dcomponents[indices_c_range] ** 2, axis=(0, 1))

                # Get mscd
                sq_chg_disp = np.sum(dx[indices, :, :], axis=0) ** 2
                mscd[i] = np.average(np.sum(sq_chg_disp, axis=1), axis=0) / len(indices)

            conv_factor = get_conversion_factor(self.structure, self.specie, self.temperature)
            self.diffusivity, self.diffusivity_std_dev = get_diffusivity_from_msd(msd, dt, smoothed)
            (
                self.chg_diffusivity,
                self.chg_diffusivity_std_dev,
            ) = get_diffusivity_from_msd(mscd, dt, smoothed)
            diffusivity_components = np.zeros(3)
            diffusivity_components_std_dev = np.zeros(3)
            for j in range(3):
                diffusivity_components[j], diffusivity_components_std_dev[j] = (
                    np.array(get_diffusivity_from_msd(msd_components[:, j], dt, smoothed)) * 3
                )
            self.diffusivity_components = diffusivity_components
            self.diffusivity_components_std_dev = diffusivity_components_std_dev

            self.conductivity = self.diffusivity * conv_factor
            self.conductivity_std_dev = self.diffusivity_std_dev * conv_factor
            self.chg_conductivity = self.chg_diffusivity * conv_factor
            self.chg_conductivity_std_dev = self.chg_diffusivity_std_dev * conv_factor
            self.conductivity_components = self.diffusivity_components * conv_factor
            self.conductivity_components_std_dev = self.diffusivity_components_std_dev * conv_factor

            if c_ranges:
                (
                    self.diffusivity_c_range,
                    self.diffusivity_c_range_std_dev,
                ) = get_diffusivity_from_msd(msd_c_range, dt, smoothed)
                diffusivity_c_range_components = np.zeros(3)
                diffusivity_c_range_components_std_dev = np.zeros(3)
                for j in range(3):
                    (
                        diffusivity_c_range_components[j],
                        diffusivity_c_range_components_std_dev[j],
                    ) = np.array(get_diffusivity_from_msd(msd_c_range_components[:, j], dt, smoothed)) * 3

                self.diffusivity_c_range_components = diffusivity_c_range_components
                self.diffusivity_c_range_components_std_dev = diffusivity_c_range_components_std_dev

                n_specie_c_range = np.average([len(j) for j in indices_c_range])
                vol_c_range = np.sum([max(min(upper, 1), 0) - max(min(lower, 1), 0) for (upper, lower) in c_ranges])
                n_density_c_range = n_specie_c_range / vol_c_range
                conv_factor_c_range = conv_factor / len(indices) * n_density_c_range

                self.conductivity_c_range = self.diffusivity_c_range * conv_factor_c_range
                self.conductivity_c_range_std_dev = self.diffusivity_c_range_std_dev * conv_factor_c_range
                self.conductivity_c_range_components = self.diffusivity_c_range_components * conv_factor_c_range
                self.conductivity_c_range_components_std_dev = (
                    self.diffusivity_c_range_components_std_dev * conv_factor_c_range
                )

            # Drift and displacement information.
            self.drift = drift
            self.corrected_displacements = dc
            self.max_ion_displacements = np.max(np.sum(dc**2, axis=-1) ** 0.5, axis=1)
            self.max_framework_displacement = np.max(self.max_ion_displacements[framework_indices])
            self.msd = msd
            self.mscd = mscd
            self.haven_ratio = self.diffusivity / self.chg_diffusivity
            self.sq_disp_ions = sq_disp_ions
            self.msd_components = msd_components
            self.dt = dt
            self.indices = indices
            self.framework_indices = framework_indices

    def get_drift_corrected_structures(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Generator:
        """
        Returns an iterator for the drift-corrected structures. Use of
        iterator is to reduce memory usage as # of structures in MD can be
        huge. You don't often need all the structures all at once.

        Args:
            start(int): Applies a start to the iterator.
            stop (int): Applies a stop to the iterator.
            step (int): Applies a step to the iterator.
                Faster than applying it after generation, as it reduces the
                number of structures created.
        """
        coords = np.array(self.structure.cart_coords)
        species = self.structure.species_and_occu
        lattices = self.lattices
        nsites, nsteps, dim = self.corrected_displacements.shape

        for i in range(start or 0, stop or nsteps, step or 1):
            latt = lattices[0] if len(lattices) == 1 else lattices[i]
            yield Structure(
                latt,
                species,
                coords + self.corrected_displacements[:, i, :],
                coords_are_cartesian=True,
            )

    def get_summary_dict(self, include_msd_t: bool = False, include_mscd_t: bool = False) -> dict:
        """
        Provides a summary of diffusion information.

        Args:
            include_msd_t (bool): Whether to include mean square displace and
                time data with the data.
            include_mscd_t (bool): Whether to include mean square charge displace and
                time data with the data.

        Returns:
            (dict) of diffusion and conductivity data.
        """
        d = {
            "D": self.diffusivity,
            "D_sigma": self.diffusivity_std_dev,
            "D_charge": self.chg_diffusivity,
            "D_charge_sigma": self.chg_diffusivity_std_dev,
            "S": self.conductivity,
            "S_sigma": self.conductivity_std_dev,
            "S_charge": self.chg_conductivity,
            "D_components": self.diffusivity_components.tolist(),
            "S_components": self.conductivity_components.tolist(),
            "D_components_sigma": self.diffusivity_components_std_dev.tolist(),
            "S_components_sigma": self.conductivity_components_std_dev.tolist(),
            "specie": str(self.specie),
            "step_skip": self.step_skip,
            "time_step": self.time_step,
            "temperature": self.temperature,
            "max_framework_displacement": self.max_framework_displacement,
            "Haven_ratio": self.haven_ratio,
        }
        if include_msd_t:
            d["msd"] = self.msd.tolist()
            d["msd_components"] = self.msd_components.tolist()
            d["dt"] = self.dt.tolist()
        if include_mscd_t:
            d["mscd"] = self.mscd.tolist()
        return d

    def get_framework_rms_plot(self, granularity: int = 200, matching_s: Structure = None) -> Axes:
        """
        Get the plot of rms framework displacement vs time. Useful for checking
        for melting, especially if framework atoms can move via paddle-wheel
        or similar mechanism (which would show up in max framework displacement
        but doesn't constitute melting).

        Args:
            granularity (int): Number of structures to match
            matching_s (Structure): Optionally match to a disordered structure
                instead of the first structure in the analyzer. Required when
                a secondary mobile ion is present.

        Notes:
            The method doesn't apply to NPT-AIMD simulation analysis.
        """
        from pymatgen.util.plotting import pretty_plot

        if self.lattices is not None and len(self.lattices) > 1:
            warnings.warn("Note the method doesn't apply to NPT-AIMD simulation analysis!")

        ax = pretty_plot(12, 8)
        step = (self.corrected_displacements.shape[1] - 1) // (granularity - 1)
        f = (matching_s or self.structure).copy()
        f.remove_species([self.specie])
        sm = StructureMatcher(
            primitive_cell=False,
            stol=0.6,
            comparator=OrderDisorderElementComparator(),
            allow_subset=True,
        )
        _rms = []
        for s in self.get_drift_corrected_structures(step=step):
            s.remove_species([self.specie])
            d = sm.get_rms_dist(f, s)
            if d:
                _rms.append(d)
            else:
                _rms.append((1, 1))
        max_dt = (len(_rms) - 1) * step * self.step_skip * self.time_step
        if max_dt > 100000:
            plot_dt = np.linspace(0, max_dt / 1000, len(_rms))
            unit = "ps"
        else:
            plot_dt = np.linspace(0, max_dt, len(_rms))
            unit = "fs"
        rms = np.array(_rms)
        ax.plot(plot_dt, rms[:, 0], label="RMS")
        ax.plot(plot_dt, rms[:, 1], label="max")
        ax.legend(loc="best")
        ax.set_xlabel(f"Timestep ({unit})")
        ax.set_ylabel("normalized distance")
        return ax

    def get_msd_plot(self, mode: str = "specie") -> Axes:
        """
        Get the plot of the smoothed msd vs time graph. Useful for
        checking convergence. This can be written to an image file.

        Args:
            mode (str): Determines type of msd plot. By "species", "sites",
                or direction (default). If mode = "mscd", the smoothed mscd vs.
                time will be plotted.
        """
        from pymatgen.util.plotting import pretty_plot

        ax = pretty_plot(12, 8)
        plot_dt: np.ndarray
        if np.max(self.dt) > 100000:
            plot_dt = self.dt / 1000
            unit = "ps"
        else:
            plot_dt = self.dt
            unit = "fs"

        if mode == "species":
            for sp in sorted(self.structure.composition.keys()):
                indices = [i for i, site in enumerate(self.structure) if site.specie == sp]
                sd = np.average(self.sq_disp_ions[indices, :], axis=0)
                ax.plot(plot_dt, sd, label=str(sp))
            ax.legend(loc=2, prop={"size": 20})
        elif mode == "sites":
            for i, site in enumerate(self.structure):
                sd = self.sq_disp_ions[i, :]
                ax.plot(plot_dt, sd, label=f"{site.specie!s} - {i}")
            ax.legend(loc=2, prop={"size": 20})
        elif mode == "mscd":
            ax.plot(plot_dt, self.mscd, "r")
            ax.legend(["Overall"], loc=2, prop={"size": 20})
        else:
            # Handle default / invalid mode case
            ax.plot(plot_dt, self.msd, "k")
            ax.plot(plot_dt, self.msd_components[:, 0], "r")
            ax.plot(plot_dt, self.msd_components[:, 1], "g")
            ax.plot(plot_dt, self.msd_components[:, 2], "b")
            ax.legend(["Overall", "a", "b", "c"], loc=2, prop={"size": 20})

        ax.set_xlabel(f"Timestep ({unit})")
        if mode == "mscd":
            ax.set_ylabel("MSCD ($\\AA^2$)")
        else:
            ax.set_ylabel("MSD ($\\AA^2$)")
        return ax

    def plot_msd(self, mode: str = "default") -> None:
        """
        Plot the smoothed msd vs time graph. Useful for checking convergence.

        Args:
            mode (str): Can be "default" (the default, shows only the MSD for
                the diffusing specie, and its components), "ions" (individual
                square displacements of all ions), "species" (mean square
                displacement by specie), or "mscd" (overall mean square charge
                displacement for diffusing specie).
        """
        self.get_msd_plot(mode=mode)
        plt.show()

    def export_msdt(self, filename: str) -> None:
        """
        Writes MSD data to a csv file that can be easily plotted in other
        software.

        Args:
            filename (str): Filename. Supported formats are csv and dat. If
                the extension is csv, a csv file is written. Otherwise,
                a dat format is assumed.
        """
        fmt = "csv" if filename.lower().endswith(".csv") else "dat"
        delimiter = ", " if fmt == "csv" else " "
        with open(filename, "w") as f:
            if fmt == "dat":
                f.write("# ")
            f.write(delimiter.join(["t", "MSD", "MSD_a", "MSD_b", "MSD_c", "MSCD"]))
            f.write("\n")
            for dt, msd, msdc, mscd in zip(self.dt, self.msd, self.msd_components, self.mscd, strict=False):
                f.write(delimiter.join([str(v) for v in [dt, msd, *list(msdc), mscd]]))
                f.write("\n")

    @classmethod
    def from_structures(
        cls,
        structures: Sequence[Structure],
        specie: SpeciesLike,
        temperature: float,
        time_step: int,
        step_skip: int,
        initial_disp: np.ndarray = None,
        initial_structure: Structure = None,
        **kwargs,
    ) -> DiffusionAnalyzer:
        r"""
        Convenient constructor that takes in a list of Structure objects to
        perform diffusion analysis.

        Args:
            structures ([Structure]): list of Structure objects (must be
                ordered in sequence of run). E.g., you may have performed
                sequential VASP runs to obtain sufficient statistics.
            specie (Element/Species): Species to calculate diffusivity for as a
                String. E.g., "Li".
            temperature (float): Temperature of the diffusion run in Kelvin.
            time_step (int): Time step between measurements.
            step_skip (int): Sampling frequency of the displacements (
                time_step is multiplied by this number to get the real time
                between measurements)
            initial_disp (np.ndarray): Sometimes, you need to iteratively
                compute estimates of the diffusivity. This supplies an
                initial displacement that will be added on to the initial
                displacements. Note that this makes sense only when
                smoothed=False.
            initial_structure (Structure): Like initial_disp, this is used
                for iterative computations of estimates of the diffusivity. You
                typically need to supply both variables. This stipulates the
                initial structure from which the current set of displacements
                are computed.
            **kwargs: kwargs supported by the :class:`DiffusionAnalyzer`_.
                Examples include smoothed, min_obs, avg_nsteps.
        """
        _p, _lattices = [], []
        structure = structures[0]
        for s in structures:
            _p.append(np.array(s.frac_coords)[:, None])
            _lattices.append(s.lattice.matrix)
        if initial_structure is not None:
            _p.insert(0, np.array(initial_structure.frac_coords)[:, None])
            _lattices.insert(0, initial_structure.lattice.matrix)
        else:
            _p.insert(0, _p[0])
            _lattices.insert(0, _lattices[0])

        p = np.concatenate(_p, axis=1)
        dp = p[:, 1:] - p[:, :-1]
        dp = dp - np.round(dp)
        f_disp = np.cumsum(dp, axis=1)
        c_disp = []
        for i in f_disp:
            c_disp.append([np.dot(d, m) for d, m in zip(i, _lattices[1:], strict=False)])
        disp = np.array(c_disp)

        # If is NVT-AIMD, clear lattice data.
        lattices = np.array([_lattices[0]]) if np.array_equal(_lattices[0], _lattices[-1]) else np.array(_lattices)
        if initial_disp is not None:
            disp += initial_disp[:, None, :]

        return cls(
            structure,
            disp,
            specie,
            temperature,
            time_step,
            step_skip=step_skip,
            lattices=lattices,
            **kwargs,
        )

    @classmethod
    def from_vaspruns(
        cls,
        vaspruns: Sequence[Vasprun],
        specie: SpeciesLike,
        initial_disp: np.ndarray = None,
        initial_structure: Structure = None,
        **kwargs,
    ) -> DiffusionAnalyzer:
        r"""
        Convenient constructor that takes in a list of Vasprun objects to
        perform diffusion analysis.

        Args:
            vaspruns ([Vasprun]): List of Vaspruns (must be ordered  in
                sequence of MD simulation). E.g., you may have performed
                sequential VASP runs to obtain sufficient statistics.
            specie (Element/Species): Species to calculate diffusivity for as a
                String. E.g., "Li".
            initial_disp (np.ndarray): Sometimes, you need to iteratively
                compute estimates of the diffusivity. This supplies an
                initial displacement that will be added on to the initial
                displacements. Note that this makes sense only when
                smoothed=False.
            initial_structure (Structure): Like initial_disp, this is used
                for iterative computations of estimates of the diffusivity. You
                typically need to supply both variables. This stipulates the
                initial stricture from which the current set of displacements
                are computed.
            **kwargs: kwargs supported by the :class:`DiffusionAnalyzer`_.
                Examples include smoothed, min_obs, avg_nsteps.
        """

        def get_structures(vaspruns: Sequence[Vasprun]) -> Generator:
            step_skip = vaspruns[0].ionic_step_skip or 1
            final_structure = vaspruns[0].initial_structure
            temperature = vaspruns[0].parameters["TEEND"]
            time_step = vaspruns[0].parameters["POTIM"]
            for i, vr in enumerate(vaspruns):
                if i == 0:
                    yield step_skip, temperature, time_step
                # check that the runs are continuous
                fdist = pbc_diff(vr.initial_structure.frac_coords, final_structure.frac_coords)
                if np.any(fdist > 0.001):
                    raise ValueError("initial and final structures do not match.")
                final_structure = vr.final_structure

                assert (vr.ionic_step_skip or 1) == step_skip
                for s in vr.ionic_steps:
                    yield s["structure"]

        s = get_structures(vaspruns)
        step_skip, temperature, time_step = next(s)

        return cls.from_structures(
            structures=list(s),
            specie=specie,
            temperature=temperature,
            time_step=time_step,
            step_skip=step_skip,
            initial_disp=initial_disp,
            initial_structure=initial_structure,
            **kwargs,
        )

    @classmethod
    def from_files(
        cls,
        filepaths: Sequence[PathLike],
        specie: SpeciesLike,
        step_skip: int = 10,
        ncores: int | None = None,
        initial_disp: np.ndarray = None,
        initial_structure: Structure = None,
        **kwargs,
    ) -> DiffusionAnalyzer:
        r"""
        Convenient constructor that takes in a list of vasprun.xml paths to
        perform diffusion analysis.

        Args:
            filepaths ([str]): List of paths to vasprun.xml files of runs. (
                must be ordered in sequence of MD simulation). For example,
                you may have done sequential VASP runs and they are in run1,
                run2, run3, etc. You should then pass in
                ["run1/vasprun.xml", "run2/vasprun.xml", ...].
            specie (Element/Species): Species to calculate diffusivity for as a
                String. E.g., "Li".
            step_skip (int): Sampling frequency of the displacements (
                time_step is multiplied by this number to get the real time
                between measurements)
            ncores (int): Numbers of cores to use for multiprocessing. Can
                speed up vasprun parsing considerably. Defaults to None,
                which means serial. It should be noted that if you want to
                use multiprocessing, the number of ionic steps in all vasprun
                .xml files should be a multiple of the ionic_step_skip.
                Otherwise, inconsistent results may arise. Serial mode has no
                such restrictions.
            initial_disp (np.ndarray): Sometimes, you need to iteratively
                compute estimates of the diffusivity. This supplies an
                initial displacement that will be added on to the initial
                displacements. Note that this makes sense only when
                smoothed=False.
            initial_structure (Structure): Like initial_disp, this is used
                for iterative computations of estimates of the diffusivity. You
                typically need to supply both variables. This stipulates the
                initial structure from which the current set of displacements
                are computed.
            **kwargs: kwargs supported by the :class:`DiffusionAnalyzer`_.
                Examples include smoothed, min_obs, avg_nsteps.
        """
        if ncores is not None and len(filepaths) > 1:
            with multiprocessing.Pool(ncores) as p:
                vaspruns = p.imap(_get_vasprun, [(fp, step_skip) for fp in filepaths])
                return cls.from_vaspruns(
                    list(vaspruns),
                    specie=specie,
                    initial_disp=initial_disp,
                    initial_structure=initial_structure,
                    **kwargs,
                )

        def vr(filepaths: Sequence[PathLike]) -> Generator[Vasprun, None, None]:
            offset = 0
            for p in filepaths:
                v = Vasprun(p, ionic_step_offset=offset, ionic_step_skip=step_skip)
                yield v
                # Recompute offset.
                offset = (-(v.nionic_steps - offset)) % step_skip

        return cls.from_vaspruns(
            list(vr(filepaths)),
            specie=specie,
            initial_disp=initial_disp,
            initial_structure=initial_structure,
            **kwargs,
        )

    def as_dict(self) -> dict:
        """Returns: MSONable dict."""
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "structure": self.structure.as_dict(),
            "displacements": self.disp.tolist(),
            "specie": self.specie,
            "temperature": self.temperature,
            "time_step": self.time_step,
            "step_skip": self.step_skip,
            "min_obs": self.min_obs,
            "smoothed": self.smoothed,
            "avg_nsteps": self.avg_nsteps,
            "lattices": self.lattices.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> DiffusionAnalyzer:
        """
        Args:
            d (dict): Dict representation.

        Returns: DiffusionAnalyzer
        """
        structure = Structure.from_dict(d["structure"])
        return cls(
            structure,
            np.array(d["displacements"]),
            specie=d["specie"],
            temperature=d["temperature"],
            time_step=d["time_step"],
            step_skip=d["step_skip"],
            min_obs=d["min_obs"],
            smoothed=d.get("smoothed", "max"),
            avg_nsteps=d.get("avg_nsteps", 1000),
            lattices=np.array(d.get("lattices", [d["structure"]["lattice"]["matrix"]])),
        )


def get_conversion_factor(structure: Structure, species: SpeciesLike, temperature: float) -> float:
    """
    Conversion factor to convert between cm^2/s diffusivity measurements and
    mS/cm conductivity measurements based on number of atoms of diffusing
    species. Note that the charge is based on the oxidation state of the
    species (where available), or else the number of valence electrons
    (usually a good guess, esp for main group ions).

    Args:
        structure (Structure): Input structure.
        species (Element/Species): Diffusing species.
        temperature (float): Temperature of the diffusion run in Kelvin.

    Returns:
        Conversion factor.
        Conductivity (in mS/cm) = Conversion Factor * Diffusivity (in cm^2/s)
    """
    df_sp = get_el_sp(species)
    z = df_sp.oxi_state if hasattr(df_sp, "oxi_state") else df_sp.full_electronic_structure[-1][2]

    n = structure.composition[species]

    vol = structure.volume * 1e-24  # units cm^3
    return 1000 * n / (vol * const.N_A) * z**2 * (const.N_A * const.e) ** 2 / (const.R * temperature)


def _get_vasprun(args: tuple[str, int]) -> Vasprun:
    """Internal method to support multiprocessing."""
    return Vasprun(args[0], ionic_step_skip=args[1], parse_dos=False, parse_eigen=False)


def fit_arrhenius(
    temps: Sequence[float] | np.ndarray,
    diffusivities: Sequence[float] | np.ndarray,
    mode: Literal["linear", "exp"] = "linear",
    diffusivity_errors: Sequence[float] | np.ndarray | None = None,
) -> tuple[float, float, float | None]:
    """
    Returns Ea, c, standard error of Ea from the Arrhenius fit.
        D = c * exp(-Ea/kT).

    Args:
        temps ([float]): A sequence of temperatures. units: K
        diffusivities ([float]): A sequence of diffusivities (e.g.,
            from DiffusionAnalyzer.diffusivity). units: cm^2/s
        mode (str): The fitting mode. Supported modes are:
                i. "linear" (default), which fits ln(D) vs 1/T.
                ii. "exp", which fits D vs T.
            Hint: Use "exp" with diffusivity errors if the errors are
            not homoscadastic in the linear representation. Avoid using
            "exp" without errors.
        diffusivity_errors ([float]): A sequence of absolute errors in diffusivities. units: cm^2/s
    """
    if mode not in ["linear", "exp"]:
        raise ValueError("Mode must be 'linear' or 'exp'.")
    if mode == "linear":
        t_1 = 1 / np.array(temps)
        # Do a least squares regression of log(D) vs 1/T
        a = np.array([t_1, np.ones(len(temps))]).T
        w, res, _, _ = np.linalg.lstsq(a, np.log(diffusivities), rcond=None)
        w = np.array(w)
        n = len(temps)
        std_Ea = (res[0] / (n - 2) / (n * np.var(t_1))) ** 0.5 * const.k / const.e if n > 2 else None
        return -w[0] * const.k / const.e, np.exp(w[1]), std_Ea
    if mode == "exp":

        def arrhenius(t: np.ndarray, Ea: float, c: float) -> np.ndarray:
            return c * np.exp(-Ea / (const.k / const.e * t))

        guess = fit_arrhenius(temps, diffusivities, mode="linear")[0:2]  # Use linear fit to get initial guess

        popt, pcov = curve_fit(arrhenius, temps, diffusivities, guess, diffusivity_errors)
        return popt[0], popt[1], pcov[0, 0] ** 0.5
    return None


def get_diffusivity_from_msd(msd: np.ndarray, dt: np.ndarray, smoothed: bool | str = "max") -> tuple[float, float]:
    """
    Returns diffusivity and standard deviation of diffusivity.

        D = 1 / 2dt * <mean square displacement>

    where d is the dimensionality, t is the time. To obtain a reliable
    diffusion estimate, a least squares regression of the MSD against
    time to obtain the slope, which is then related to the diffusivity.

    For traditional analysis, use smoothed=False.

    Args:
        msd ([float]): A sequence of mean square displacements. units: Å^2
        dt ([float]): A sequence of time steps corresponding to MSD. units: fs
        smoothed (str): Whether to smooth the MSD, and what mode to smooth.
            Supported modes are::

            i. "max", which tries to use the maximum #
               of data points for each time origin, subject to a
               minimum # of observations given by min_obs, and then
               weights the observations based on the variance
               accordingly. This is the default.
            ii. "constant", in which each timestep is averaged over
                the number of time_steps given by min_steps.
            iii. None / False / any other false-like quantity. No
               smoothing.
    """

    def weighted_lstsq(a: np.ndarray, b: np.ndarray) -> tuple:
        if smoothed == "max":
            # For max smoothing, we need to weight by variance.
            w_root = (1 / dt) ** 0.5
            return np.linalg.lstsq(a * w_root[:, None], b * w_root, rcond=None)
        return np.linalg.lstsq(a, b, rcond=None)

    # Get self diffusivity
    a = np.ones((len(dt), 2))
    a[:, 0] = dt
    (m, c), res, rank, s = weighted_lstsq(a, msd)
    # m shouldn't be negative
    m = max(m, 1e-15)

    # factor of 10 is to convert from Å^2/fs to cm^2/s
    # factor of 6 is for dimensionality
    diffusivity = m / 60

    # Calculate the error in the diffusivity using the error in the
    # slope from the lst sq.
    # Variance in slope = n * Sum Squared Residuals / (n * Sxx - Sx
    # ** 2) / (n-2).
    n = len(dt)

    # Pre-compute the denominator since we will use it later.
    # We divide dt by 1000 to avoid overflow errors in some systems (
    # e.g., win). This is subsequently corrected where denom is used.
    denom = (n * np.sum((np.array(dt) / 1000) ** 2) - np.sum(np.array(dt) / 1000) ** 2) * (n - 2)
    diffusivity_std_dev = np.sqrt(n * res[0] / denom) / 60 / 1000
    return diffusivity, diffusivity_std_dev


def get_extrapolated_diffusivity(
    temps: Sequence[float], diffusivities: Sequence[float], new_temp: float, mode: Literal["linear", "exp"] = "linear"
) -> float:
    """
    Returns (Arrhenius) extrapolated diffusivity at new_temp.

    Args:
        temps ([float]): A sequence of temperatures. units: K
        diffusivities ([float]): A sequence of diffusivities (e.g.,
            from DiffusionAnalyzer.diffusivity). units: cm^2/s
        new_temp (float): desired temperature. units: K
        mode (str): The fitting mode. See fit_arrhenius for details.

    Returns:
        (float) Diffusivity at extrapolated temp in cm^2/s.
    """
    Ea, c, _ = fit_arrhenius(temps, diffusivities, mode)
    return c * np.exp(-Ea / (const.k / const.e * new_temp))


def get_extrapolated_conductivity(
    temps: Sequence[float],
    diffusivities: Sequence[float],
    new_temp: float,
    structure: Structure,
    species: SpeciesLike,
) -> float:
    """
    Returns extrapolated mS/cm conductivity.

    Args:
        temps ([float]): A sequence of temperatures. units: K
        diffusivities ([float]): A sequence of diffusivities (e.g.,
            from DiffusionAnalyzer.diffusivity). units: cm^2/s
        new_temp (float): desired temperature. units: K
        structure (structure): Structure used for the diffusivity calculation
        species (string/Species): conducting species

    Returns:
        (float) Conductivity at extrapolated temp in mS/cm.
    """
    return get_extrapolated_diffusivity(temps, diffusivities, new_temp) * get_conversion_factor(
        structure, species, new_temp
    )


def get_arrhenius_plot(
    temps: Sequence[float] | np.ndarray,
    diffusivities: Sequence[float] | np.ndarray,
    diffusivity_errors: Sequence[float] | np.ndarray | None = None,
    mode: Literal["linear", "exp"] = "linear",
    unit: Literal["eV", "meV"] = "meV",
    **kwargs,
) -> Axes:
    r"""
    Returns an Arrhenius plot.

    Args:
        temps ([float]): A sequence of temperatures.
        diffusivities ([float]): A sequence of diffusivities (e.g.,
            from DiffusionAnalyzer.diffusivity).
        diffusivity_errors ([float]): A sequence of errors for the
            diffusivities. If None, no error bar is plotted.
        mode (str): The fitting mode. See fit_arrhenius for details.
        unit (str): The unit for the activation energy. Supported units are
            "eV" and "meV".
        **kwargs:
            Any keyword args supported by matplotlib.pyplot.plot.

    Returns:
        A matplotlib.Axes object. Do ax.show() to show the plot.
    """
    Ea, c, _ = fit_arrhenius(temps, diffusivities, mode, diffusivity_errors)

    from pymatgen.util.plotting import pretty_plot

    ax = pretty_plot(12, 8)

    # log10 of the arrhenius fit
    t = np.linspace(min(temps), max(temps), 100) if mode == "exp" else np.array(temps)
    arr = c * np.exp(-Ea / (const.k / const.e * t))

    x = 1000 / np.array(temps) if mode == "linear" else np.array(temps)

    _ = ax.plot(x, diffusivities, "ko", x if mode == "linear" else t, arr, "k--", markersize=10, **kwargs)
    if diffusivity_errors is not None:
        n = len(diffusivity_errors)
        ax.errorbar(
            x[0:n],
            diffusivities[0:n],
            yerr=diffusivity_errors,
            fmt="ko",
            ecolor="k",
            capthick=2,
            linewidth=2,
        )
    ax.set_yscale("log") if mode == "linear" else None
    ax.text(
        0.6 if mode == "linear" else 0.1,
        0.85,
        f"E$_a$ = {(Ea * 1000):.0f} meV" if unit == "meV" else f"E$_a$ = {Ea:.2f} eV",
        fontsize=30,
        transform=ax.transAxes,
    )
    ax.set_ylabel("D (cm$^2$/s)")
    ax.set_xlabel("1000/T (K$^{-1}$)" if mode == "linear" else "T (K)")
    return ax
