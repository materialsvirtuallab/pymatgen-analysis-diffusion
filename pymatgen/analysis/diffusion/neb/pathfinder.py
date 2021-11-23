# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

"""
Algorithms for NEB migration path analysis.
"""

import itertools
import logging
import warnings
from typing import Tuple

import numpy as np
from monty.json import MSONable
from pymatgen.core import Site
from pymatgen.core import PeriodicSite, Structure
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

__author__ = "Iek-Heng Chu"
__version__ = "1.0"
__date__ = "March 14, 2017"


# TODO: (1) ipython notebook example files, unittests
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.analysis.diffusion.utils.supercells import (
    get_sc_fromstruct,
    get_start_end_structures,
)

logger = logging.getLogger(__name__)


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
            weights[ni] = 1.0 / (avg_dist ** 4 + np.eye(natoms, dtype=np.float64) * 1e-8)

        for ni, i in itertools.product(range(nimages + 2), range(natoms)):
            frac_coords = structures[ni][i].frac_coords
            init_coords.append(latt.get_cartesian_coords(frac_coords))

            if ni not in [0, nimages + 1]:
                for j in range(i + 1, natoms):
                    img = latt.get_distance_and_image(frac_coords, structures[ni][j].frac_coords)[1]
                    translations[ni - 1, i, j] = latt.get_cartesian_coords(img)
                    translations[ni - 1, j, i] = -latt.get_cartesian_coords(img)

        self.init_coords = np.array(init_coords).reshape(nimages + 2, natoms, 3)
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
        for n in range(maxiter):
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
    def from_endpoints(cls, endpoints, nimages=5, sort_tol=1.0):
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
        """
        try:
            images = endpoints[0].interpolate(endpoints[1], nimages=nimages + 1, autosort_tol=sort_tol)
        except Exception as e:
            if "Unable to reliably match structures " in str(e):
                warnings.warn(
                    "Auto sorting is turned off because it is unable" " to match the end-point structures!",
                    UserWarning,
                )
                images = endpoints[0].interpolate(endpoints[1], nimages=nimages + 1, autosort_tol=0)
            else:
                raise e

        return IDPPSolver(images)

    def _get_funcs_and_forces(self, x):
        """
        Calculate the set of objective functions as well as their gradients,
        i.e. "effective true forces"
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
        return vec / np.sqrt(np.sum(vec ** 2))

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
    """
    A convenience container representing a migration path.
    """

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
            host_symm_struct: SymmetrizedStructure of the host structure, used to for its spacegroup
            symprec: used to determine equivalence
        """
        self.isite = isite
        self.esite = esite
        self.iindex = None
        self.eindex = None
        self.symm_structure = symm_structure
        self.symprec = symprec
        self.msite = PeriodicSite(esite.specie, (isite.frac_coords + esite.frac_coords) / 2, esite.lattice)
        if host_symm_struct:
            self.host_symm_structure = host_symm_struct
            sg = self.host_symm_structure.spacegroup
        else:
            sg = self.symm_structure.spacegroup
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
        return (
            f"Path of {self.length:.4f} A from {self.isite.specie} "
            f"[{ifc[0]:.3f}, {ifc[1]:.3f}, {ifc[2]:.3f}] "
            f"(ind: {self.iindex}, Wyckoff: {self.symm_structure.wyckoff_symbols[self.iindex]}) "
            f"to {self.esite.specie} "
            f"[{efc[0]:.3f}, {efc[1]:.3f}, {efc[2]:.3f}] "
            f"(ind: {self.eindex}, Wyckoff: {self.symm_structure.wyckoff_symbols[self.eindex]})"
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
            \*\*idpp_kwargs: Passthrough kwargs for the IDPPSolver.run.

        Returns:
            [Structure] Note that the first site of each structure is always
            the migrating ion. This makes it easier to perform subsequent
            analysis.
        """
        migrating_specie_sites, other_sites = self._split_migrating_and_other_sites(vac_mode)

        start_structure = Structure.from_sites([self.isite] + migrating_specie_sites + other_sites)
        end_structure = Structure.from_sites([self.esite] + migrating_specie_sites + other_sites)

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
    ) -> Tuple[Structure, Structure, Structure]:
        """
        Construct supercells that represents the start and end positions for migration analysis.

        Args:
            vac_mode: If true simulate vacancy diffusion.
            max_atoms: Maximum number of atoms allowed in the supercell.
            min_atoms: Minimum number of atoms allowed in the supercell.
            min_length: Minimum length of the smallest supercell lattice vector.

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
            self.isite, self.esite, base_struct, sc_mat, vac_mode=vac_mode  # type: ignore
        )
        return start_struct, end_struct, base_sc

    def write_path(self, fname, **kwargs):
        r"""
        Write the path to a file for easy viewing.

        Args:
            fname (str): File name.
            \*\*kwargs: Kwargs supported by NEBPath.get_structures.
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
            \*\*kwargs: Passthrough kwargs to path.get_structures.
        """
        sites = []
        for p in self.get_paths():
            structures = p.get_structures(nimages=nimages, species=[self.migrating_specie], **kwargs)
            sites.append(structures[0][0])
            sites.append(structures[-1][0])
            for s in structures[1:-1]:
                sites.append(PeriodicSite("H", s[0].frac_coords, s.lattice))
        sites.extend(structures[0].sites[1:])
        Structure.from_sites(sites).to(filename=fname)
