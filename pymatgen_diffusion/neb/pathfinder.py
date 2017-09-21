# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

from __future__ import division, unicode_literals, print_function

from pymatgen.core import Structure, PeriodicSite
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import warnings
import numpy as np
import itertools

__author__ = "Iek-Heng Chu"
__version__ = "1.0"
__date__ = "March 14, 2017"

"""
Algorithms for NEB migration path analysis.
"""


# TODO: (1) ipython notebook example files, unittests
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.transition_state import NEBAnalysis
import warnings

class IDPPSolver(object):
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
                structures[-1].distance_matrix - structures[0].distance_matrix)

            target_dists.append(dist)

        target_dists = np.array(target_dists)

        # Set of translational vector matrices (anti-symmetric) for the images.
        translations = np.zeros((nimages, natoms, natoms, 3), dtype=np.float64)

        # A set of weight functions. It is set as 1/d^4 for each image. Here,
        # we take d as the average of the target distance matrix and the actual
        # distance matrix.
        weights = np.zeros_like(target_dists, dtype=np.float64)
        for ni in range(nimages):
            avg_dist = (target_dists[ni]
                        + structures[ni + 1].distance_matrix) / 2.0
            weights[ni] = 1.0 / (avg_dist ** 4 +
                                 np.eye(natoms, dtype=np.float64) * 1e-8)

        for ni, i in itertools.product(range(nimages + 2), range(natoms)):
            frac_coords = structures[ni][i].frac_coords
            init_coords.append(latt.get_cartesian_coords(frac_coords))

            if ni not in [0, nimages + 1]:
                for j in range(i + 1, natoms):
                    img = latt.get_distance_and_image(
                        frac_coords, structures[ni][j].frac_coords)[1]
                    translations[ni - 1, i, j] = latt.get_cartesian_coords(img)
                    translations[ni - 1, j, i] = -latt.get_cartesian_coords(img)

        self.init_coords = np.array(init_coords).reshape(nimages + 2, natoms, 3)
        self.translations = translations
        self.weights = weights
        self.structures = structures
        self.target_dists = target_dists
        self.nimages = nimages

    def run(self, maxiter=1000, tol=1e-5, gtol=1e-3, step_size=0.05,
            max_disp=0.05, spring_const=5.0, species=None):
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
            indices = [i for i, site in enumerate(self.structures[0])
                       if site.specie in species]

            if len(indices) == 0:
                raise ValueError("The given species are not in the system!")

        # Iterative minimization
        for n in range(maxiter):
            # Get the sets of objective functions, true and total force
            # matrices.
            funcs, true_forces = self._get_funcs_and_forces(coords)
            tot_forces = self._get_total_forces(coords, true_forces,
                                                spring_const=spring_const)

            # Each atom is allowed to move up to max_disp
            disp_mat = step_size * tot_forces[:, indices, :]
            disp_mat = np.where(np.abs(disp_mat) > max_disp,
                                np.sign(disp_mat) * max_disp,
                                disp_mat)
            coords[1:(self.nimages + 1), indices] += disp_mat

            max_force = np.abs(tot_forces[:, indices, :]).max()
            tot_res = np.sum(np.abs(old_funcs - funcs))

            if tot_res < tol and max_force < gtol:
                break

            old_funcs = funcs

        else:
            warnings.warn(
                "Maximum iteration number is reached without convergence!",
                UserWarning)

        for ni in range(self.nimages):
            # generate the improved image structure
            new_sites = []

            for site, cart_coords in zip(self.structures[ni + 1], coords[ni + 1]):
                new_site = PeriodicSite(
                    site.species_and_occu, coords=cart_coords,
                    lattice=site.lattice, coords_are_cartesian=True,
                    properties=site.properties)
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
            images = endpoints[0].interpolate(endpoints[1], nimages=nimages + 1,
                                              autosort_tol=sort_tol)
        except Exception as e:
            if "Unable to reliably match structures " in str(e):
                warnings.warn("Auto sorting is turned off because it is unable"
                              " to match the end-point structures!",
                              UserWarning)
                images = endpoints[0].interpolate(endpoints[1],
                                                  nimages=nimages + 1,
                                                  autosort_tol=0)
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
            vec = [x[ni + 1, i] - x[ni + 1] - trans[ni, i]
                   for i in range(natoms)]

            trial_dist = np.linalg.norm(vec, axis=2)
            aux = (trial_dist - target_dists[ni]) * weights[ni] \
                  / (trial_dist + np.eye(natoms, dtype=np.float64))

            # Objective function
            func = np.sum((trial_dist - target_dists[ni]) ** 2 * weights[ni])

            # "True force" derived from the objective function.
            grad = np.sum(aux[:, :, None] * vec, axis=1)

            funcs.append(func)
            funcs_prime.append(grad)

        return 0.5 * np.array(funcs), -2 * np.array(funcs_prime)

    @staticmethod
    def get_unit_vector(vec):
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
            spring_force = spring_const * (np.linalg.norm(vec1) -
                                           np.linalg.norm(vec2)) * tangent

            # Total force
            flat_ft = true_forces[ni - 1].copy().flatten()
            total_force = true_forces[ni - 1] + (
                spring_force - np.dot(flat_ft, tangent) * tangent).reshape(
                natoms, 3)
            total_forces.append(total_force)

        return np.array(total_forces)


class MigrationPath(object):
    """
    A convenience container representing a migration path.
    """

    def __init__(self, isite, esite, symm_structure):
        """
        Args:
            isite: Initial site
            esite: End site
            symm_structure: SymmetrizedStructure
        """
        self.isite = isite
        self.esite = esite
        self.symm_structure = symm_structure
        self.msite = PeriodicSite(
            esite.specie,
            (isite.frac_coords + esite.frac_coords) / 2, esite.lattice)
        sg = self.symm_structure.spacegroup
        for i, sites in enumerate(self.symm_structure.equivalent_sites):
            if sg.are_symmetrically_equivalent([isite], [sites[0]]):
                self.iindex = i
            if sg.are_symmetrically_equivalent([esite], [sites[0]]):
                self.eindex = i

    def __repr__(self):
        return "Path of %.4f A from %s [%.3f, %.3f, %.3f] (ind: %d, Wyckoff: %s) to %s [%.3f, %.3f, %.3f] (ind: %d, Wyckoff: %s)" \
               % (self.length, self.isite.specie, self.isite.frac_coords[0], self.isite.frac_coords[1],
                  self.isite.frac_coords[2],
                  self.iindex, self.symm_structure.wyckoff_symbols[self.iindex],
                  self.esite.specie, self.esite.frac_coords[0], self.esite.frac_coords[1], self.esite.frac_coords[2],
                  self.eindex, self.symm_structure.wyckoff_symbols[self.eindex])

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

        return self.symm_structure.spacegroup.are_symmetrically_equivalent(
            (self.isite, self.msite, self.esite),
            (other.isite, other.msite, other.esite)
        )

    def get_structures(self, nimages=5, vac_mode=True, idpp=False,
                       **idpp_kwargs):
        """
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
        migrating_specie_sites = []
        other_sites = []
        isite = self.isite
        esite = self.esite

        for site in self.symm_structure.sites:
            if site.specie != isite.specie:
                other_sites.append(site)
            else:
                if vac_mode and (isite.distance(site) > 1e-8 and
                                         esite.distance(site) > 1e-8):
                    migrating_specie_sites.append(site)

        start_structure = Structure.from_sites(
            [self.isite] + migrating_specie_sites + other_sites)
        end_structure = Structure.from_sites(
            [self.esite] + migrating_specie_sites + other_sites)

        structures = start_structure.interpolate(end_structure,
                                                 nimages=nimages + 1,
                                                 pbc=False)

        if idpp:
            solver = IDPPSolver(structures)
            return solver.run(**idpp_kwargs)

        return structures

    def write_path(self, fname, **kwargs):
        """
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


class DistinctPathFinder(object):
    """
    Determines symmetrically distinct paths between existing sites.
    The path info can then be used to set up either vacancy or interstitial
    diffusion (assuming site positions are known). Note that this works mainly
    for atomic mechanism, and does not work for correlated migration.
    """

    def __init__(self, structure, migrating_specie, max_path_length=None,
                 symprec=0.1, perc_mode=">1d"):
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
                neighbors = self.symm_structure.get_neighbors(
                    site0, r=max_r)
                for nn, dist in sorted(neighbors, key=lambda n: n[-1]):
                    if nn.specie == self.migrating_specie:
                        dists.append(dist)
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
            [MigrationPath] All distinct migration paths.
        """
        paths = set()
        for sites in self.symm_structure.equivalent_sites:
            if sites[0].specie == self.migrating_specie:
                site0 = sites[0]
                for nn, dist in self.symm_structure.get_neighbors(
                        site0, r=round(self.max_path_length, 3) + 0.01):
                    if nn.specie == self.migrating_specie:
                        path = MigrationPath(site0, nn, self.symm_structure)
                        paths.add(path)

        return sorted(paths, key=lambda p: p.length)

    def write_all_paths(self, fname, nimages=5, **kwargs):
        """
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
            structures = p.get_structures(
                nimages=nimages, species=[self.migrating_specie], **kwargs)
            sites.append(structures[0][0])
            sites.append(structures[-1][0])
            for s in structures[1:-1]:
                sites.append(PeriodicSite("H", s[0].frac_coords, s.lattice))
        sites.extend(structures[0].sites[1:])
        Structure.from_sites(sites).to(filename=fname)

def combine_neb_plots(root_dirs, arranged_dirs=False, reverse_plot=False):
    """
    root_dirs (list): paths to the root directory with 00 ~ 0n subdirectories
    of the NEB calculation. E.g., 00, 01, 02...06 in which 00 and 06 are
    terminal relaxations and 01 ~ 05 are image relaxations.

    arranged_dirs: need to manually arrange 00 ~ 0n subdirectories of each
    root_dir to get the combined-barrier plot corresponding to the percolation
    path if the code gives a warning, which is due to close energies of
    terminal relaxations. Or only the barrier value is correct! E.g., if there
    are two root_dirs to combine, arrange in such a way that 06 of the first
    root_dir is 00 of the second root_dir.

    reverse_plot: reverse the plot or percolation direction.
    """
    x = StructureMatcher(ltol=0.2, stol=0.3)
    warn = False
    for fd_index in range(len(root_dirs)):
        if fd_index == 0:
            neb1 = NEBAnalysis.from_dir(root_dirs[fd_index])
            neb1_energies = list(neb1.energies)
            neb1_structures = neb1.structures
            neb1_forces = neb1.forces
            neb1_r = neb1.r
            continue

        neb2 = NEBAnalysis.from_dir(root_dirs[fd_index])
        neb2_energies = list(neb2.energies)

        neb1_start_s = neb1_structures[0]
        neb2_start_s, neb2_end_s = neb2.structures[0], neb2.structures[-1]

        if x.fit(neb1_start_s, neb2_start_s) == True \
                and x.fit(neb1_start_s, neb2_end_s) == True:
            warn = True
            if arranged_dirs:
                neb1_energies = neb1_energies[0:len(neb1_energies) - 1] \
                                + [(neb1_energies[-1] + neb2_energies[0]) / 2] \
                                + neb2_energies[
                                                                  1:]
                neb1_structures = neb1_structures + neb2.structures[1:]
                neb1_forces = list(neb1_forces) + list(neb2.forces)[1:]
                neb1_r = list(neb1_r) + [i + neb1_r[-1] for i in
                                         list(neb2.r)[1:]]

        if (x.fit(neb1_start_s, neb2_start_s) == True
            and x.fit(neb1_start_s, neb2_end_s) == False) \
                or (warn == True and arranged_dirs == False):
            neb1_energies = list(reversed(neb1_energies[1:])) + [
                (neb1_energies[0] + neb2_energies[0]) / 2] + neb2_energies[1:]
            neb1_structures = list(
                reversed((neb1_structures[1:]))) + neb2.structures
            neb1_forces = list(reversed(list(neb1_forces)[1:])) + list(
                neb2.forces)
            neb1_r = list(reversed(
                [i * -1 - neb1_r[-1] * -1 for i in list(neb1_r)[1:]])) + [
                         i + neb1_r[-1] for i in list(neb2.r)]

        elif x.fit(neb1_start_s, neb2_start_s) == False \
                and x.fit(neb1_start_s, neb2_end_s) == True:
            neb1_energies = (neb2_energies[0:len(neb2_energies) - 1]) + [
                (neb1_energies[0] + neb2_energies[-1]) / 2] + neb1_energies[1:]
            neb1_structures = (neb2.structures[
                               0:len(neb2_energies) - 1]) + neb1_structures
            neb1_forces = list(neb2.forces)[0:len(neb2_energies) - 1] + list(
                neb1_forces)
            neb1_r = list(reversed(
                [i * -1 - neb2.r[-1] * -1 for i in list(neb2.r)[1:]])) + [
                         i + neb2.r[-1] for i in list(neb1_r)]

        elif x.fit(neb1_start_s, neb2_start_s) == False \
                and x.fit(neb1_start_s, neb2_end_s) == False:
            raise ValueError("no matched structures for connection!")

    if warn:
        warnings.warn(
            "Need to arrange root_dirs or only the barrier value is correct!",
            Warning)

    if reverse_plot:
        na = NEBAnalysis(
            list(reversed([i * -1 - neb1_r[-1] * -1 for i in list(neb1_r)])),
            list(reversed(neb1_energies)),
            list(reversed(neb1_forces)), list(reversed(neb1_structures)))
    else:
        na = NEBAnalysis(neb1_r, neb1_energies, neb1_forces, neb1_structures)
    plt = na.get_plot()
    return plt
