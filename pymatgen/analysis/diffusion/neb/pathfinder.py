# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

"""Algorithms for NEB migration path analysis."""

from __future__ import annotations

import itertools
import logging
import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.linalg as la
import scipy.signal
import scipy.stats
from monty.json import MSONable
from scipy.interpolate import interp1d

from pymatgen.analysis.diffusion.utils.supercells import get_sc_fromstruct, get_start_end_structures
from pymatgen.core import PeriodicSite, Site, Structure
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

if TYPE_CHECKING:
    from pymatgen.symmetry.structure import SymmetrizedStructure

logger = logging.getLogger(__name__)

# TODO: (1) ipython notebook example files, unittests


class IDPPSolver:
    """
    A solver using image dependent pair potential (IDPP) algo to get an improved
    initial NEB path. For more details about this algo, please refer to
    Smidstrup et al., J. Chem. Phys. 140, 214106 (2014).
    """

    def __init__(self, structures):
        """
        Initialization.

        Args:
            structures (list of pmg_structure) : Initial guess of the NEB path
                (including initial and final end-point structures).
        """
        latt = structures[0].lattice
        natoms = structures[0].num_sites
        nimages = len(structures) - 2
        target_dists = []

        # Initial guess of the path (in Cartesian coordinates) used in the IDPP
        # algo.
        init_coords = []

        # Construct the set of target distance matrices via linear interpolation
        # between those of end-point structures.
        for i in range(1, nimages + 1):
            # Interpolated distance matrices
            dist = structures[0].distance_matrix + i / (nimages + 1) * (
                structures[-1].distance_matrix - structures[0].distance_matrix
            )

            target_dists.append(dist)

        target_dists = np.array(target_dists)

        # Set of translational vector matrices (anti-symmetric) for the images.
        translations = np.zeros((nimages, natoms, natoms, 3), dtype=np.float64)

        # A set of weight functions. It is set as 1/d^4 for each image. Here,
        # we take d as the average of the target distance matrix and the actual
        # distance matrix.
        weights = np.zeros_like(target_dists, dtype=np.float64)
        for ni in range(nimages):
            avg_dist = (target_dists[ni] + structures[ni + 1].distance_matrix) / 2.0
            weights[ni] = 1.0 / (avg_dist**4 + np.eye(natoms, dtype=np.float64) * 1e-8)

        for ni, i in itertools.product(range(nimages + 2), range(natoms)):
            frac_coords = structures[ni][i].frac_coords
            init_coords.append(latt.get_cartesian_coords(frac_coords))

            if ni not in [0, nimages + 1]:
                for j in range(i + 1, natoms):
                    img = latt.get_distance_and_image(frac_coords, structures[ni][j].frac_coords)[1]
                    translations[ni - 1, i, j] = latt.get_cartesian_coords(img)
                    translations[ni - 1, j, i] = -latt.get_cartesian_coords(img)

        self.init_coords = np.array(init_coords).reshape(nimages + 2, natoms, 3)  # pylint: disable=E1121
        self.translations = translations
        self.weights = weights
        self.structures = structures
        self.target_dists = target_dists
        self.nimages = nimages

    def run(
        self,
        maxiter=1000,
        tol=1e-5,
        gtol=1e-3,
        step_size=0.05,
        max_disp=0.05,
        spring_const=5.0,
        species=None,
    ):
        """
        Perform iterative minimization of the set of objective functions in an
        NEB-like manner. In each iteration, the total force matrix for each
        image is constructed, which comprises both the spring forces and true
        forces. For more details about the NEB approach, please see the
        references, e.g. Henkelman et al., J. Chem. Phys. 113, 9901 (2000).

        Args:
            maxiter (int): Maximum number of iterations in the minimization
                process.
            tol (float): Tolerance of the change of objective functions between
                consecutive steps.
            gtol (float): Tolerance of maximum force component (absolute value).
            step_size (float): Step size associated with the displacement of
                the atoms during the minimization process.
            max_disp (float): Maximum allowed atomic displacement in each
                iteration.
            spring_const (float): A virtual spring constant used in the NEB-like
                        relaxation process that yields so-called IDPP path.
            species (list of string): If provided, only those given species are
                allowed to move. The atomic positions of other species are
                obtained via regular linear interpolation approach.

        Returns:
            [Structure] Complete IDPP path (including end-point structures)
        """
        coords = self.init_coords.copy()
        old_funcs = np.zeros((self.nimages,), dtype=np.float64)
        idpp_structures = [self.structures[0]]

        if species is None:
            indices = list(range(len(self.structures[0])))
        else:
            species = [get_el_sp(sp) for sp in species]
            indices = [i for i, site in enumerate(self.structures[0]) if site.specie in species]

            if len(indices) == 0:
                raise ValueError("The given species are not in the system!")

        # Iterative minimization
        for _n in range(maxiter):
            # Get the sets of objective functions, true and total force
            # matrices.
            funcs, true_forces = self._get_funcs_and_forces(coords)
            tot_forces = self._get_total_forces(coords, true_forces, spring_const=spring_const)

            # Each atom is allowed to move up to max_disp
            disp_mat = step_size * tot_forces[:, indices, :]
            disp_mat = np.where(np.abs(disp_mat) > max_disp, np.sign(disp_mat) * max_disp, disp_mat)
            coords[1 : (self.nimages + 1), indices] += disp_mat

            max_force = np.abs(tot_forces[:, indices, :]).max()
            tot_res = np.sum(np.abs(old_funcs - funcs))

            if tot_res < tol and max_force < gtol:
                break

            old_funcs = funcs

        else:
            warnings.warn("Maximum iteration number is reached without convergence!", UserWarning)

        for ni in range(self.nimages):
            # generate the improved image structure
            new_sites = []

            for site, cart_coords in zip(self.structures[ni + 1], coords[ni + 1]):
                new_site = PeriodicSite(
                    site.species,
                    coords=cart_coords,
                    lattice=site.lattice,
                    coords_are_cartesian=True,
                    properties=site.properties,
                )
                new_sites.append(new_site)

            idpp_structures.append(Structure.from_sites(new_sites))

        # Also include end-point structure.
        idpp_structures.append(self.structures[-1])

        return idpp_structures

    @classmethod
    def from_endpoints(
        cls,
        endpoints,
        nimages: int = 5,
        sort_tol: float = 1.0,
        interpolate_lattices: bool = False,
    ):
        """
        A class method that starts with end-point structures instead. The
        initial guess for the IDPP algo is then constructed using linear
        interpolation.

        Args:
            endpoints (list of Structure objects): The two end-point structures.
            nimages (int): Number of images between the two end-points.
            sort_tol (float): Distance tolerance (in Angstrom) used to match the
                atomic indices between start and end structures. Need
                to increase the value in some cases.
            interpolate_lattices (bool): Whether to interpolate lattices between the start and end structures.
        """
        try:
            images = endpoints[0].interpolate(
                endpoints[1],
                nimages=nimages + 1,
                autosort_tol=sort_tol,
                interpolate_lattices=interpolate_lattices,
            )
        except Exception as e:
            if "Unable to reliably match structures " in str(e):
                warnings.warn(
                    "Auto sorting is turned off because it is unable" " to match the end-point structures!",
                    UserWarning,
                )
                images = endpoints[0].interpolate(
                    endpoints[1],
                    nimages=nimages + 1,
                    autosort_tol=0,
                    interpolate_lattices=interpolate_lattices,
                )
            else:
                raise e

        return IDPPSolver(images)

    def _get_funcs_and_forces(self, x):
        """
        Calculate the set of objective functions as well as their gradients,
        i.e. "effective true forces".
        """
        funcs = []
        funcs_prime = []
        trans = self.translations
        natoms = trans.shape[1]
        weights = self.weights
        target_dists = self.target_dists

        for ni in range(len(x) - 2):
            vec = [x[ni + 1, i] - x[ni + 1] - trans[ni, i] for i in range(natoms)]

            trial_dist = np.linalg.norm(vec, axis=2)
            aux = (trial_dist - target_dists[ni]) * weights[ni] / (trial_dist + np.eye(natoms, dtype=np.float64))

            # Objective function
            func = np.sum((trial_dist - target_dists[ni]) ** 2 * weights[ni])

            # "True force" derived from the objective function.
            grad = np.sum(aux[:, :, None] * vec, axis=1)

            funcs.append(func)
            funcs_prime.append(grad)

        return 0.5 * np.array(funcs), -2 * np.array(funcs_prime)

    @staticmethod
    def get_unit_vector(vec):
        """
        Calculate the unit vector of a vector.

        Args:
            vec: Vector.
        """
        return vec / np.sqrt(np.sum(vec**2))

    def _get_total_forces(self, x, true_forces, spring_const):
        """
        Calculate the total force on each image structure, which is equal to
        the spring force along the tangent + true force perpendicular to the
        tangent. Note that the spring force is the modified version in the
        literature (e.g. Henkelman et al., J. Chem. Phys. 113, 9901 (2000)).
        """
        total_forces = []
        natoms = np.shape(true_forces)[1]

        for ni in range(1, len(x) - 1):
            vec1 = (x[ni + 1] - x[ni]).flatten()
            vec2 = (x[ni] - x[ni - 1]).flatten()

            # Local tangent
            tangent = self.get_unit_vector(vec1) + self.get_unit_vector(vec2)
            tangent = self.get_unit_vector(tangent)

            # Spring force
            spring_force = spring_const * (np.linalg.norm(vec1) - np.linalg.norm(vec2)) * tangent

            # Total force
            flat_ft = true_forces[ni - 1].copy().flatten()
            total_force = true_forces[ni - 1] + (spring_force - np.dot(flat_ft, tangent) * tangent).reshape(natoms, 3)
            total_forces.append(total_force)

        return np.array(total_forces)


class MigrationHop(MSONable):
    """A convenience container representing a migration path."""

    def __init__(
        self,
        isite: Site,
        esite: Site,
        symm_structure: SymmetrizedStructure,
        host_symm_struct: SymmetrizedStructure = None,
        symprec: float = 0.001,
    ):
        """
        Args:
            isite: Initial site
            esite: End site
            symm_structure: SymmetrizedStructure
            host_symm_struct: SymmetrizedStructure of the host structure, used to for
                its spacegroup
            symprec: used to determine equivalence.
        """
        self.isite = isite
        self.esite = esite
        self.iindex = None
        self.eindex = None
        self.symm_structure = symm_structure
        self.host_symm_struct = host_symm_struct
        self.symprec = symprec
        self.msite = PeriodicSite(esite.specie, (isite.frac_coords + esite.frac_coords) / 2, esite.lattice)

        sg = (
            self.host_symm_struct.spacegroup  # type: ignore[union-attr]
            if host_symm_struct
            else self.symm_structure.spacegroup  # type: ignore[union-attr]
        )
        for i, sites in enumerate(self.symm_structure.equivalent_sites):
            if sg.are_symmetrically_equivalent([isite], [sites[0]]):
                self.iindex = i
            if sg.are_symmetrically_equivalent([esite], [sites[0]]):
                self.eindex = i

        # if no index was identified then loop over each site until something is found
        if self.iindex is None:
            for i, sites in enumerate(self.symm_structure.equivalent_sites):
                for itr_site in sites:
                    if sg.are_symmetrically_equivalent([isite], [itr_site]):
                        self.iindex = i
                        break
                else:
                    continue
                break
        if self.eindex is None:
            for i, sites in enumerate(self.symm_structure.equivalent_sites):
                for itr_site in sites:
                    if sg.are_symmetrically_equivalent([esite], [itr_site]):
                        self.eindex = i
                        break
                else:
                    continue
                break

        if self.iindex is None:
            raise RuntimeError(f"No symmetrically equivalent site was found for {isite}")
        if self.eindex is None:
            raise RuntimeError(f"No symmetrically equivalent site was found for {esite}")

    def __repr__(self):
        ifc = self.isite.frac_coords
        efc = self.esite.frac_coords
        wyk_symbols = self.symm_structure.wyckoff_symbols
        return (
            f"Path of {self.length:.4f} A from {self.isite.specie} "
            f"[{ifc[0]:.3f}, {ifc[1]:.3f}, {ifc[2]:.3f}] "
            f"(ind: {self.iindex}, Wyckoff: {wyk_symbols[self.iindex]}) "
            f"to {self.esite.specie} "
            f"[{efc[0]:.3f}, {efc[1]:.3f}, {efc[2]:.3f}] "
            f"(ind: {self.eindex}, Wyckoff: {wyk_symbols[self.eindex]})"
        )

    @property
    def length(self):
        """
        Returns:
            (float) Length of migration path.
        """
        return np.linalg.norm(self.isite.coords - self.esite.coords)

    def __hash__(self):
        return self.iindex + self.eindex

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if self.symm_structure != other.symm_structure:
            return False

        if abs(self.length - other.length) > 1e-3:
            return False

        return self.symm_structure.spacegroup.are_symmetrically_equivalent(
            (self.isite, self.msite, self.esite),
            (other.isite, other.msite, other.esite),
            self.symprec,
        )

    def get_structures(self, nimages=5, vac_mode=True, idpp=False, **idpp_kwargs):
        r"""
        Generate structures for NEB calculation.

        Args:
            nimages (int): Defaults to 5. Number of NEB images. Total number of
                structures returned in nimages+2.
            vac_mode (bool): Defaults to True. In vac_mode, a vacancy diffusion
                mechanism is assumed. The initial and end sites of the path
                are assumed to be the initial and ending positions of the
                vacancies. If vac_mode is False, an interstitial mechanism is
                assumed. The initial and ending positions are assumed to be
                the initial and ending positions of the interstitial, and all
                other sites of the same specie are removed. E.g., if NEBPaths
                were obtained using a Li4Fe4P4O16 structure, vac_mode=True would
                generate structures with formula Li3Fe4P4O16, while
                vac_mode=False would generate structures with formula
                LiFe4P4O16.
            idpp (bool): Defaults to False. If True, the generated structures
                will be run through the IDPPSolver to generate a better guess
                for the minimum energy path.
            **idpp_kwargs: Passthrough kwargs for the IDPPSolver.run.

        Returns:
            [Structure] Note that the first site of each structure is always
            the migrating ion. This makes it easier to perform subsequent
            analysis.
        """
        migrating_specie_sites, other_sites = self._split_migrating_and_other_sites(vac_mode)

        start_structure = Structure.from_sites([self.isite, *migrating_specie_sites, *other_sites])
        end_structure = Structure.from_sites([self.esite, *migrating_specie_sites, *other_sites])

        structures = start_structure.interpolate(end_structure, nimages=nimages + 1, pbc=False)

        if idpp:
            solver = IDPPSolver(structures)
            return solver.run(**idpp_kwargs)

        return structures

    def _split_migrating_and_other_sites(self, vac_mode):
        migrating_specie_sites = []
        other_sites = []
        for site in self.symm_structure.sites:
            if site.specie != self.isite.specie:
                other_sites.append(site)
            else:
                if self.isite.distance(site) <= 1e-8 or self.esite.distance(site) <= 1e-8:
                    migrating_specie_sites.append(site)
                    continue

                if vac_mode:
                    other_sites.append(site)
        return migrating_specie_sites, other_sites

    def get_sc_structures(
        self,
        vac_mode: bool,
        min_atoms: int = 80,
        max_atoms: int = 240,
        min_length: float = 10.0,
        tol: float = 1e-5,
    ) -> tuple[Structure, Structure, Structure]:
        """
        Construct supercells that represents the start and end positions for migration
        analysis.

        Args:
            vac_mode: If true simulate vacancy diffusion.
            max_atoms: Maximum number of atoms allowed in the supercell.
            min_atoms: Minimum number of atoms allowed in the supercell.
            min_length: Minimum length of the smallest supercell lattice vector.
            tol: toleranace for identifying isite/esite within base_struct

        Returns:
            Start, End, Base Structures.

            If not vacancy mode, the base structure is just the host lattice.
            If in vacancy mode, the base structure is the fully intercalated structure

        """
        migrating_specie_sites, other_sites = self._split_migrating_and_other_sites(vac_mode)
        if vac_mode:
            base_struct = Structure.from_sites(other_sites + migrating_specie_sites)
        else:
            base_struct = Structure.from_sites(other_sites)
        sc_mat = get_sc_fromstruct(
            base_struct=base_struct,
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_length=min_length,
        )
        start_struct, end_struct, base_sc = get_start_end_structures(
            self.isite,
            self.esite,
            base_struct,
            sc_mat,  # type: ignore
            vac_mode=vac_mode,
            tol=tol,
        )
        return start_struct, end_struct, base_sc

    def write_path(self, fname, **kwargs):
        r"""
        Write the path to a file for easy viewing.

        Args:
            fname (str): File name.
            **kwargs: Kwargs supported by NEBPath.get_structures.
        """
        sites = []
        for st in self.get_structures(**kwargs):
            sites.extend(st)
        st = Structure.from_sites(sites)
        st.to(filename=fname)


class DistinctPathFinder:
    """
    Determines symmetrically distinct paths between existing sites.
    The path info can then be used to set up either vacancy or interstitial
    diffusion (assuming site positions are known). Note that this works mainly
    for atomic mechanism, and does not work for correlated migration.
    """

    def __init__(
        self,
        structure,
        migrating_specie,
        max_path_length=None,
        symprec=0.1,
        perc_mode=">1d",
    ):
        """
        Args:
            structure: Input structure that contains all sites.
            migrating_specie (Specie-like): The specie that migrates. E.g.,
                "Li".
            max_path_length (float): Maximum length of NEB path in the unit
                of Angstrom. Defaults to None, which means you are setting the
                value to the min cutoff until finding 1D or >1D percolating paths.
            symprec (float): Symmetry precision to determine equivalence.
            perc_mode(str): The percolating type. Default to ">1d", because usually
                it is used to find possible NEB paths to form percolating networks.
                If you just want to check the min 1D percolation, set it to "1d".
        """
        self.structure = structure
        self.migrating_specie = get_el_sp(migrating_specie)
        self.symprec = symprec
        a = SpacegroupAnalyzer(self.structure, symprec=self.symprec)
        self.symm_structure = a.get_symmetrized_structure()

        junc = 0
        distance_list = []
        max_r = max_path_length or min(structure.lattice.abc)
        junc_cutoff = max_r
        for sites in self.symm_structure.equivalent_sites:
            if sites[0].specie == self.migrating_specie:
                site0 = sites[0]
                dists = []
                neighbors = self.symm_structure.get_neighbors(site0, r=max_r)
                for nn in sorted(neighbors, key=lambda nn: nn.nn_distance):
                    if nn.specie == self.migrating_specie:
                        dists.append(nn.nn_distance)
                if len(dists) > 2:
                    junc += 1
                distance_list.append(dists)
        # Avoid isolated atoms (# of neighbors < 2)
        if len(sorted(distance_list, key=len)[0]) < 2:
            path_cutoff = max_r
        # We define junction as atoms have at least three paths including
        # equivalent ones.
        elif junc == 0:
            path_cutoff = sorted(distance_list, key=lambda d: d[1])[-1][1]
        else:
            # distance_list are sorted as [[a0,a1,a2],[b0,b1,b2],[c0,c1,c2],...]
            # in which a0<a1<a2,b0<b1<b2,...
            # path_cutoff = max(a1,b1,c1,...), junc_cutoff=min(a2,b2,c2)
            path_cutoff = sorted(distance_list, key=lambda d: d[1])[-1][1]
            junc_distance_list = [d for d in distance_list if len(d) >= 3]
            junc_cutoff = sorted(junc_distance_list, key=lambda d: d[2])[0][2]

        if max_path_length is None:
            if perc_mode.lower() == "1d":
                self.max_path_length = path_cutoff
            else:
                self.max_path_length = max(junc_cutoff, path_cutoff)
        else:
            self.max_path_length = max_path_length

    def get_paths(self):
        """
        Returns:
            [MigrationHop] All distinct migration paths.
        """
        paths = set()
        for sites in self.symm_structure.equivalent_sites:
            if sites[0].specie == self.migrating_specie:
                site0 = sites[0]
                for nn in self.symm_structure.get_neighbors(site0, r=round(self.max_path_length, 3) + 0.01):
                    if nn.specie == self.migrating_specie:
                        path = MigrationHop(site0, nn, self.symm_structure)
                        paths.add(path)

        return sorted(paths, key=lambda p: p.length)

    def write_all_paths(self, fname, nimages=5, **kwargs):
        r"""
        Write a file containing all paths, using hydrogen as a placeholder for
        the images. H is chosen as it is the smallest atom. This is extremely
        useful for path visualization in a standard software like VESTA.

        Args:
            fname (str): Filename
            nimages (int): Number of images per path.
            **kwargs: Passthrough kwargs to path.get_structures.
        """
        sites = []
        for p in self.get_paths():
            structures = p.get_structures(nimages=nimages, species=[self.migrating_specie], **kwargs)
            sites.append(structures[0][0])
            sites.append(structures[-1][0])
            for s in structures[1:-1]:
                sites.append(PeriodicSite("H", s[0].frac_coords, s.lattice))
        sites.extend(structures[0].sites[1:])  # type: ignore
        Structure.from_sites(sites).to(filename=fname)


class NEBPathfinder:
    """
    General pathfinder for interpolating between two structures, where the
    interpolating path is calculated with the elastic band method with
    respect to the given static potential for sites whose indices are given
    in relax_sites, and is linear otherwise.

    If you use PathFinder algorithm for your research, please consider citing the
    following work::

        Ziqin Rong, Daniil Kitchaev, Pieremanuele Canepa, Wenxuan Huang, Gerbrand
        Ceder, The Journal of Chemical Physics 145 (7), 074112
    """

    def __init__(self, start_struct, end_struct, relax_sites, v, n_images=20, mid_struct=None):
        """
        Args:
            start_struct: Starting structure
            end_struct: End structure to interpolate
            relax_sites: List of site indices whose interpolation paths should
                be relaxed
            v: Static potential field to use for the elastic band relaxation
            n_images: Number of interpolation images to generate
            mid_struct: (optional) additional structure between the start and end
                structures to help.
        """
        self.__s1 = start_struct
        self.__s2 = end_struct
        self.__mid = mid_struct
        self.__relax_sites = relax_sites
        self.__v = v
        self.__n_images = n_images
        self.__images = None
        self.interpolate()

    def interpolate(self):
        """
        Finds a set of n_images from self.s1 to self.s2, where all sites except
        for the ones given in relax_sites, the interpolation is linear (as in
        pymatgen.core.structure.interpolate), and for the site indices given
        in relax_sites, the path is relaxed by the elastic band method within
        the static potential V.

        If a mid point is defined we will interpolate from s1--> mid -->s2
        The final number of images will still be n_images.
        """
        if self.__mid is not None:
            # to make arithmetic easier we will do the interpolation in two parts with n
            # images each then just take every other image at the end, this results in
            # exactly n images
            images_0 = self.__s1.interpolate(self.__mid, nimages=self.__n_images, interpolate_lattices=False)[:-1]
            images_1 = self.__mid.interpolate(self.__s2, nimages=self.__n_images, interpolate_lattices=False)
            images = images_0 + images_1
            images = images[::2]
        else:
            images = self.__s1.interpolate(self.__s2, nimages=self.__n_images, interpolate_lattices=False)
        for site_i in self.__relax_sites:
            start_f = images[0].sites[site_i].frac_coords
            end_f = images[-1].sites[site_i].frac_coords

            path = NEBPathfinder.string_relax(
                NEBPathfinder.__f2d(start_f, self.__v),
                NEBPathfinder.__f2d(end_f, self.__v),
                self.__v,
                n_images=(self.__n_images + 1),
                dr=[
                    self.__s1.lattice.a / self.__v.shape[0],
                    self.__s1.lattice.b / self.__v.shape[1],
                    self.__s1.lattice.c / self.__v.shape[2],
                ],
            )
            for image_i, image in enumerate(images):
                image.translate_sites(
                    site_i,
                    NEBPathfinder.__d2f(path[image_i], self.__v) - image.sites[site_i].frac_coords,
                    frac_coords=True,
                    to_unit_cell=True,
                )
        self.__images = images

    @property
    def images(self):
        """
        Returns a list of structures interpolating between the start and
        endpoint structures.
        """
        return self.__images

    def plot_images(self, outfile):
        """
        Generates a POSCAR with the calculated diffusion path with respect to the first
        endpoint.

        Args:
            outfile: Output file for the POSCAR.
        """
        sum_struct = self.__images[0].sites
        for image in self.__images:
            for site_i in self.__relax_sites:
                sum_struct.append(
                    PeriodicSite(
                        image.sites[site_i].specie,
                        image.sites[site_i].frac_coords,
                        self.__images[0].lattice,
                        to_unit_cell=True,
                        coords_are_cartesian=False,
                    )
                )
        sum_struct = Structure.from_sites(sum_struct, validate_proximity=False)
        p = Poscar(sum_struct)
        p.write_file(outfile)

    @staticmethod
    def string_relax(
        start,
        end,
        V,
        n_images=25,
        dr=None,
        h=3.0,
        k=0.17,
        min_iter=100,
        max_iter=10000,
        max_tol=5e-6,
    ):
        """
        Implements path relaxation via the elastic band method. In general, the
        method is to define a path by a set of points (images) connected with
        bands with some elasticity constant k. The images then relax along the
        forces found in the potential field V, counterbalanced by the elastic
        response of the elastic band. In general the endpoints of the band can
        be allowed to relax also to their local minima, but in this calculation
        they are kept fixed.

        Args:
            start: Starting point of the path calculation given in discrete
                coordinates with respect to the grid in V.
            end: Endpoints of the path calculation.
            V: potential field through which to calculate the path
            n_images: number of images used to define the path. In general
                anywhere from 20 to 40 seems to be good.
            dr: Conversion ratio from discrete coordinates to real coordinates
                for each of the three coordinate vectors
            h: Step size for the relaxation. h = 0.1 works reliably, but is
                slow. h=10 diverges with large gradients but for the types of
                gradients seen in CHGCARs, works pretty reliably
            k: Elastic constant for the band (in real units, not discrete)
            min_iter: Minimum number of iterations to perform. Defaults to 100.
            max_iter: Number of optimization steps the string will
                take before exiting (even if unconverged). Defaults to 10000.
            max_tol: Convergence threshold such that if the string moves by
                less than max_tol in a step, and at least min_iter steps have
                passed, the algorithm will terminate. Depends strongly on the
                size of the gradients in V, but 5e-6 works reasonably well for
                CHGCARs.
        """
        #
        # This code is based on the MATLAB example provided by
        # Prof. Eric Vanden-Eijnden of NYU
        # (http://www.cims.nyu.edu/~eve2/main.htm)
        #

        # logger.debug(f"Getting path from {start} to {end} (coords wrt V grid)")

        # Set parameters
        dr = np.array([1 / V.shape[0], 1 / V.shape[1], 1 / V.shape[2]]) if not dr else np.array(dr, dtype=float)
        keff = k * dr * n_images
        h0 = h

        # Initialize string
        g1 = np.linspace(0, 1, n_images)
        s0 = start
        s1 = end
        s = np.array([g * (s1 - s0) for g in g1]) + s0
        ds = s - np.roll(s, 1, axis=0)
        ds[0] = ds[0] - ds[0]
        ls = np.cumsum(la.norm(ds, axis=1))
        ls = ls / ls[-1]
        fi = interp1d(ls, s, axis=0)
        s = fi(g1)

        # Evaluate initial distances (for elastic equilibrium)
        ds0_plus = s - np.roll(s, 1, axis=0)
        ds0_minus = s - np.roll(s, -1, axis=0)
        ds0_plus[0] = ds0_plus[0] - ds0_plus[0]
        ds0_minus[-1] = ds0_minus[-1] - ds0_minus[-1]

        # Evaluate potential gradient outside the loop, as potential does not
        # change per step in this approximation.
        dV = np.gradient(V)

        # Evolve string
        for step in range(max_iter):
            # Gradually decay step size to prevent oscillations
            h = h0 * np.exp(-2 * (step - min_iter) / max_iter) if step > min_iter else h0
            # Calculate forces acting on string
            d = V.shape
            s0 = s.copy()  # store copy for endpoint fixing below (fixes GH 2732)
            edV = np.array(
                [
                    [
                        dV[0][int(pt[0]) % d[0]][int(pt[1]) % d[1]][int(pt[2]) % d[2]] / dr[0],
                        dV[1][int(pt[0]) % d[0]][int(pt[1]) % d[1]][int(pt[2]) % d[2]] / dr[0],
                        dV[2][int(pt[0]) % d[0]][int(pt[1]) % d[1]][int(pt[2]) % d[2]] / dr[0],
                    ]
                    for pt in s
                ]
            )
            # if(step % 100 == 0):
            #    logger.debug(edV)

            # Update according to force due to potential and string elasticity
            ds_plus = s - np.roll(s, 1, axis=0)
            ds_minus = s - np.roll(s, -1, axis=0)
            ds_plus[0] = ds_plus[0] - ds_plus[0]
            ds_minus[-1] = ds_minus[-1] - ds_minus[-1]
            Fpot = edV
            Fel = keff * (la.norm(ds_plus) - la.norm(ds0_plus)) * (ds_plus / la.norm(ds_plus))
            Fel += keff * (la.norm(ds_minus) - la.norm(ds0_minus)) * (ds_minus / la.norm(ds_minus))
            s -= h * (Fpot + Fel)

            # Fix endpoints
            s[0] = s0[0]
            s[-1] = s0[-1]

            # Re-parametrize string
            ds = s - np.roll(s, 1, axis=0)
            ds[0] = ds[0] - ds[0]
            ls = np.cumsum(la.norm(ds, axis=1))
            ls = ls / ls[-1]
            fi = interp1d(ls, s, axis=0)
            s = fi(g1)

            tol = la.norm((s - s0) * dr) / n_images / h

            if tol > 1e10:
                raise ValueError("Pathfinding failed, path diverged! Consider reducing h to avoid " "divergence.")

            if step > min_iter and tol < max_tol:
                logger.debug(f"Converged at {step=}")
                break

            if step % 100 == 0:
                logger.debug(f"{step=} - ds = {tol}")
        return s

    @staticmethod
    def __f2d(frac_coords, v):
        """
        Converts fractional coordinates to discrete coordinates with respect to
        the grid size of v.
        """
        # frac_coords = frac_coords % 1
        return (np.array(frac_coords) * np.array(v.shape)).astype(int)

    @staticmethod
    def __d2f(disc_coords, v):
        """
        Converts a point given in discrete coordinates with respect to the
        grid in v to fractional coordinates.
        """
        return np.array(
            [
                disc_coords[0] / v.shape[0],
                disc_coords[1] / v.shape[1],
                disc_coords[2] / v.shape[2],
            ]
        )


class StaticPotential:
    """
    Defines a general static potential for diffusion calculations. Implements
    grid-rescaling and smearing for the potential grid. Also provides a
    function to normalize the potential from 0 to 1 (recommended).
    """

    def __init__(self, struct, pot):
        """
        :param struct: atomic structure of the potential
        :param pot: volumentric data to be used as a potential
        """
        self.__v = pot
        self.__s = struct

    def get_v(self):
        """Returns the potential."""
        return self.__v

    def normalize(self):
        """Sets the potential range 0 to 1."""
        self.__v = self.__v - np.amin(self.__v)
        self.__v = self.__v / np.amax(self.__v)

    def rescale_field(self, new_dim):
        """
        Changes the discretization of the potential field by linear
        interpolation. This is necessary if the potential field
        obtained from DFT is strangely skewed, or is too fine or coarse. Obeys
        periodic boundary conditions at the edges of
        the cell. Alternatively useful for mixing potentials that originally
        are on different grids.

        :param new_dim: tuple giving the numpy shape of the new grid
        """
        v_dim = self.__v.shape
        padded_v = np.lib.pad(self.__v, ((0, 1), (0, 1), (0, 1)), mode="wrap")
        ogrid_list = np.array([list(c) for c in list(np.ndindex(v_dim[0] + 1, v_dim[1] + 1, v_dim[2] + 1))])
        v_ogrid = padded_v.reshape(((v_dim[0] + 1) * (v_dim[1] + 1) * (v_dim[2] + 1), -1))
        ngrid_a, ngrid_b, ngrid_c = np.mgrid[
            0 : v_dim[0] : v_dim[0] / new_dim[0],
            0 : v_dim[1] : v_dim[1] / new_dim[1],
            0 : v_dim[2] : v_dim[2] / new_dim[2],
        ]

        v_ngrid = scipy.interpolate.griddata(ogrid_list, v_ogrid, (ngrid_a, ngrid_b, ngrid_c), method="linear").reshape(
            (new_dim[0], new_dim[1], new_dim[2])
        )
        self.__v = v_ngrid

    def gaussian_smear(self, r):
        """
        Applies an isotropic Gaussian smear of width (standard deviation) r to
        the potential field. This is necessary to avoid finding paths through
        narrow minima or nodes that may exist in the field (although any
        potential or charge distribution generated from GGA should be
        relatively smooth anyway). The smearing obeys periodic
        boundary conditions at the edges of the cell.

        :param r - Smearing width in Cartesian coordinates, in the same units
            as the structure lattice vectors
        """
        # Since scaling factor in fractional coords is not isotropic, have to
        # have different radii in 3 directions
        a_lat = self.__s.lattice.a
        b_lat = self.__s.lattice.b
        c_lat = self.__s.lattice.c

        # Conversion factors for discretization of v
        v_dim = self.__v.shape
        r_frac = (r / a_lat, r / b_lat, r / c_lat)
        r_disc = (
            int(math.ceil(r_frac[0] * v_dim[0])),
            int(math.ceil(r_frac[1] * v_dim[1])),
            int(math.ceil(r_frac[2] * v_dim[2])),
        )

        # Apply smearing
        # Gaussian filter
        gauss_dist = np.zeros((r_disc[0] * 4 + 1, r_disc[1] * 4 + 1, r_disc[2] * 4 + 1))
        for g_a in np.arange(-2 * r_disc[0], 2 * r_disc[0] + 1, 1):
            for g_b in np.arange(-2 * r_disc[1], 2 * r_disc[1] + 1, 1):
                for g_c in np.arange(-2 * r_disc[2], 2 * r_disc[2] + 1, 1):
                    g = np.array([g_a / v_dim[0], g_b / v_dim[1], g_c / v_dim[2]]).T
                    gauss_dist[int(g_a + r_disc[0])][int(g_b + r_disc[1])][int(g_c + r_disc[2])] = (
                        la.norm(np.dot(self.__s.lattice.matrix, g)) / r
                    )
        gauss = scipy.stats.norm.pdf(gauss_dist)
        gauss = gauss / np.sum(gauss, dtype=float)
        padded_v = np.pad(
            self.__v,
            ((r_disc[0], r_disc[0]), (r_disc[1], r_disc[1]), (r_disc[2], r_disc[2])),
            mode="wrap",
        )
        smeared_v = scipy.signal.convolve(padded_v, gauss, mode="valid")
        self.__v = smeared_v


class ChgcarPotential(StaticPotential):
    """Implements a potential field based on the charge density output from VASP."""

    def __init__(self, chgcar, smear=False, normalize=True):
        """
        :param chgcar: Chgcar object based on a VASP run of the structure of
            interest (Chgcar.from_file("CHGCAR"))
        :param smear: Whether or not to apply a Gaussian smearing to the
            potential
        :param normalize: Whether or not to normalize the potential to range
            from 0 to 1
        """
        v = chgcar.data["total"]
        v = v / (v.shape[0] * v.shape[1] * v.shape[2])
        StaticPotential.__init__(self, chgcar.structure, v)
        if smear:
            self.gaussian_smear(2)
        if normalize:
            self.normalize()


class FreeVolumePotential(StaticPotential):
    """
    Implements a potential field based on geometric distances from atoms in the
    structure - basically, the potential
    is lower at points farther away from any atoms in the structure.
    """

    def __init__(self, struct, dim, smear=False, normalize=True):
        """
        :param struct: Unit cell on which to base the potential
        :param dim: Grid size for the potential
        :param smear: Whether or not to apply a Gaussian smearing to the
            potential
        :param normalize: Whether or not to normalize the potential to range
            from 0 to 1
        """
        self.__s = struct
        v = FreeVolumePotential.__add_gaussians(struct, dim)
        StaticPotential.__init__(self, struct, v)
        if smear:
            self.gaussian_smear(2)
        if normalize:
            self.normalize()

    @staticmethod
    def __add_gaussians(s, dim, r=1.5):
        gauss_dist = np.zeros(dim)
        for a_d in np.arange(0, dim[0], 1):
            for b_d in np.arange(0, dim[1], 1):
                for c_d in np.arange(0, dim[2], 1):
                    coords_f = np.array([a_d / dim[0], b_d / dim[1], c_d / dim[2]])
                    d_f = sorted(s.get_sites_in_sphere(coords_f, s.lattice.a), key=lambda x: x[1])[0][1]
                    # logger.debug(d_f)
                    gauss_dist[int(a_d)][int(b_d)][int(c_d)] = d_f / r
        return scipy.stats.norm.pdf(gauss_dist)


class MixedPotential(StaticPotential):
    """Implements a potential that is a weighted sum of some other potentials."""

    def __init__(self, potentials, coefficients, smear=False, normalize=True):
        """
        Args:
            potentials: List of objects extending the StaticPotential superclass
            coefficients: Mixing weights for the elements of the potentials list
            smear: Whether or not to apply a Gaussian smearing to the potential
            normalize: Whether or not to normalize the potential to range from
                0 to 1.
        """
        v = potentials[0].get_v() * coefficients[0]
        s = potentials[0].__s
        for i in range(1, len(potentials)):
            v += potentials[i].get_v() * coefficients[i]
        StaticPotential.__init__(self, s, v)
        if smear:
            self.gaussian_smear(2)
        if normalize:
            self.normalize()
