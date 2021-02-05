# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.
"""
Functions to take entrys -> migrations graphs -> NEB inputs
"""

from pymatgen import Structure
import numpy as np
import logging

from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import find_optimal_cell_shape, get_deviation_from_optimal_cell_shape
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)
from pymatgen_diffusion.neb.pathfinder import MigrationPath
from pymatgen_diffusion.neb.full_path_mapper import FullPathMapper, ComputedEntryPath
from typing import Tuple, List, Union, Dict

__author__ = "Jimmy Shen"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Jimmy Shen"
__email__ = "jmmshn@lbl.gov"
__date__ = "April 11, 2019"

logger = logging.getLogger(__name__)


def get_cep_from_grouped_entries(
    base_entry: ComputedStructureEntry,
    inserted_entries: List[ComputedStructureEntry],
    migrating_species: str = "Li",
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5,
    initial_max_dist: float = 4.0,
    only_use_single: bool = False,
) -> ComputedEntryPath:
    """

    Args:
        base_entry (ComputedStructureEntry): Base entry for empty structure
        inserted_entries (list[ComputedStructureEntry]): List of structures with one or more working ions
        migrating_species (str, optional): Name of migrating_species. Defaults to 'Li'.
        ltol (float, optional): Fractional length tolerance. Defaults to 0.2.
        stol (float, optional): Site tolerance in Angstrom. Defaults to 0.3.
        angle_tol (float, optional): Angle tolerance in degrees. Defaults to 5.
        initial_max_dist (float, optional): initial distance threshold for connecting sites
        only_use_single: (bool, optional): only allow single cation insertions to be used to generate the CEP

    Returns:
        ComputedEntryPath: Object containing all of the symmetry equivalent hops in a system
    """

    if "aeccar" in base_entry.data.keys():
        # check this first because cep init will delete this
        get_chg_path = True
    else:
        get_chg_path = False

    single_inserted = []
    for entry in inserted_entries:
        # generate entries from single cation sites
        num_working = entry.composition.as_dict()[migrating_species]
        if only_use_single and num_working > 1:
            continue
        for itr, isite in enumerate(entry.structure.sites):
            if isite.species_string == migrating_species:
                new_struct = base_entry.structure.copy()
                new_struct.insert(
                    0, migrating_species, isite.frac_coords, properties=dict(magmom=0)
                )
                new_entry = ComputedStructureEntry(
                    structure=new_struct,
                    energy=(entry.energy - base_entry.energy) / num_working
                    + base_entry.energy,
                    entry_id=str(entry.entry_id) + "_" + str(itr),
                )
                single_inserted.append(new_entry)
    max_dist = initial_max_dist
    cep = None
    while max_dist < 10.0:
        if "aeccar" in base_entry.data.keys():
            cep = ComputedEntryPath(
                base_entry,
                single_cat_entries=single_inserted,
                migrating_specie=migrating_species,
                base_aeccar=base_entry.data["aeccar"],
                max_path_length=max_dist,
                ltol=ltol,
                stol=stol,
                angle_tol=angle_tol,
            )
            # since the aeccar is already stored we can removed it from the entry
        else:
            cep = ComputedEntryPath(
                base_entry,
                single_cat_entries=single_inserted,
                migrating_specie=migrating_species,
                base_aeccar=None,
                max_path_length=max_dist,
            )
        if len(cep.unique_hops) > 0:
            if "aeccar" in base_entry.data.keys():
                del cep.base_struct_entry.data["aeccar"]
            break
        else:
            max_dist += 0.2

    if cep is not None and get_chg_path:
        cep.populate_edges_with_chg_density_info()
    return cep


def get_sc_fromstruct(
    base_struct: Structure,
    min_atoms: int = 80,
    max_atoms: int = 240,
    min_length: float = 10.0,
) -> List[List[int]]:
    """
    Generate the best supercell from a unitcell.
    The CubicSupercellTransformation from PMG is much faster but don't iterate over as many
    supercell configurations so it's less able to find the best configuration in a give cell size.
    We try the PMG version first wit a cap on the number of atoms (max_atoms).
    If that fails to produce a supercell, we will use the ASE configuration
    that iterates overall more supercell configurations.

    Args:
        base_struct: structure of the unit cell
        max_atoms: Maximum number of atoms allowed in the supercell.
        min_atoms: Minimum number of atoms allowed in the supercell.
        min_length: Minimum length of the smallest supercell lattice vector.

    Returns:
        struc_sc: Supercell that is as close to cubic as possible
    """
    struct_sc = _get_sc_from_struct_pmg(base_struct, min_atoms, max_atoms, min_length)
    if struct_sc is None:
        logger.warning(
            "PMG Supercell generation failed, using the SC generation from "
            "ASE. This might take much longer."
        )
        struct_sc = _get_sc_from_struct_ase(base_struct, min_atoms, max_atoms)
    return struct_sc


def _get_sc_from_struct_pmg(
    base_struct: Structure,
    min_atoms: int = 80,
    max_atoms: int = 240,
    min_length: float = 10.0,
) -> List[List[int]]:
    """
    Generate the best supercell from a unitcell using the pymatgen CubicSupercellTransformation

    Args:
        base_struct: structure of the unit cell
        max_atoms: Maximum number of atoms allowed in the supercell.
        min_atoms: Minimum number of atoms allowed in the supercell.
        min_length: Minimum length of the smallest supercell lattice vector.

    Returns:
        3x3 matrix: supercell matrix

    """
    cst = CubicSupercellTransformation(
        min_atoms=min_atoms, max_atoms=max_atoms, min_length=min_length
    )

    try:
        cst.apply_transformation(base_struct)
    except BaseException:
        return None
    return cst.transformation_matrix


def _get_sc_from_struct_ase(
    base_struct, min_atoms=80, max_atoms=240
) -> List[List[int]]:
    """generate the best supercell from a unitcell using ASE's method, is slower but more exhaustive

    Args:
        base_struct (pymatgen.Structure): unit cell
        min_size (int, optional): Minimum number of atoms in the desired supercell. Defaults to 80.

    Returns:
        3x3 matrix: supercell matrix
    """

    num_cells_min = int(np.ceil(min_atoms / base_struct.num_sites))
    num_cells_max = int(np.ceil(max_atoms / base_struct.num_sites))
    res = []
    for icell in range(num_cells_min, num_cells_max):
        if icell % 2 != 0 or icell % 3 != 0:
            continue  # cells with many factors are more lifely to be square
        atoms = AseAtomsAdaptor().get_atoms(base_struct)
        logger.info(f"Getting cell shape {icell} x unit cell")
        sc_mat = find_optimal_cell_shape(atoms.cell, icell, "sc")

        deviation = get_deviation_from_optimal_cell_shape(np.dot(sc_mat, atoms.cell))

        dd = {"deviation": deviation, "P1": sc_mat}
        # base_struct_sc.to('poscar', f'/Users/lik/Desktop/enpoints_{base_struct_sc.num_sites}.vasp')
        res.append(dd)
        if deviation < 0.3:
            logger.debug("Found good cell")
            return sc_mat

    else:
        best_case = min(res, key=lambda x: x["deviation"])
    logger.warning(
        f"Could not find case with deviation from cubic was less than 0.3 using the ASE cubic supercell finder \
        \nThe best one had a deviation of {best_case['deviation']}"
    )
    return best_case["P1"]


def get_start_end_structs_from_hop(
    hop: MigrationPath, base_struct: Structure, sc_mat: List[List[Union[int, float]]]
) -> Tuple[Structure, Structure, Structure]:
    """
    Obtain the starting and terminating structures in a supercell for NEB calculations.

    Args:
        hop: object presenting the migration event
        base_struct: unit cell representation of the structure
        sc_mat: supercell transformation to create the simulation cell for the NEB calc

    Returns:
        initial structure, final structure, empty structure all in the supercell
    """
    base_sc = base_struct.copy() * sc_mat

    start_struct = base_struct.copy() * sc_mat
    end_struct = base_struct.copy() * sc_mat

    sc_mat_inv = np.linalg.inv(sc_mat)

    start_struct.insert(
        0,
        hop.esite.species_string,
        np.dot(hop.isite.frac_coords, sc_mat_inv),
        properties={"magmom": 0},
    )
    end_struct.insert(
        0,
        hop.esite.species_string,
        np.dot(hop.esite.frac_coords, sc_mat_inv),
        properties={"magmom": 0},
    )
    return start_struct, end_struct, base_sc


def get_start_end_structs(
    fpm: FullPathMapper,
    min_length: float = 10.0,
    store_base=True,
    min_atoms: int = 80,
    max_atoms: int = 240,
) -> Dict:
    """
    Iterate through the unique hops of a FullPathMapper object to generate the list of
    initial and final NEB calculation positions

    Args:
        fpm:  Migration graph
        min_length: smalled side legth
        store_base: if True return the base structure
        min_atoms: minimum number of atoms in the base cell
        max_atoms:  maximum number of atoms in the base cell (make sure this is not
        too strict)
    Returns:
        Dict: { unique hop label :
            {'start': start structure, 'end': end structure, 'base': empty structure}
        }
    """
    base = Structure.from_sites([isite for isite in fpm.symm_structure.copy()])
    base.remove_species([fpm.migrating_specie])
    sc_mat = get_sc_fromstruct(
        base, min_atoms=min_atoms, max_atoms=max_atoms, min_length=min_length
    )
    res = dict()
    for k, uniq_hop in fpm.unique_hops.items():
        start_struct, end_struct, base_struct = get_start_end_structs_from_hop(
            uniq_hop["hop"], base, sc_mat
        )
        if store_base:
            res[k] = {"start": start_struct, "end": end_struct, "base": base_struct}
        else:
            res[k] = {"start": start_struct, "end": end_struct}
    return res
