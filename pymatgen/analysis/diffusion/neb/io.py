# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.

"""
Generate input fiels for NEB calculations.
"""

import copy

from pymatgen.io.vasp.sets import MITRelaxSet, MITNEBSet
from pymatgen.core import Structure

__author__ = "Austen"


class MVLCINEBEndPointSet(MITRelaxSet):
    """
    Class for writing NEB end points relaxation inputs.
    """

    def __init__(self, structure, **kwargs):
        r"""
        Args:
            structure: Structure
            \*\*kwargs: Keyword args supported by VaspInputSets.
        """
        user_incar_settings = kwargs.get("user_incar_settings", {})
        defaults = {
            "ISIF": 2,
            "EDIFF": 5e-5,
            "EDIFFG": -0.02,
            "ISMEAR": 0,
            "ISYM": 0,
            "LCHARG": False,
            "LDAU": False,
            "NELMIN": 4,
        }

        if user_incar_settings != {}:
            defaults.update(user_incar_settings)
        kwargs["user_incar_settings"] = defaults

        super().__init__(structure, **kwargs)


class MVLCINEBSet(MITNEBSet):
    """
    MAVRL-tested settings for CI-NEB calculations. Note that these parameters
    requires the VTST modification of VASP from the Henkelman group. See
    http://theory.cm.utexas.edu/vtsttools/
    """

    def __init__(self, structures, **kwargs):
        r"""
        Args:
            structure: Structure
            \*\*kwargs: Keyword args supported by VaspInputSets.
        """
        user_incar_settings = kwargs.get("user_incar_settings", {})

        # CI-NEB settings
        defaults = {
            "EDIFF": 5e-5,
            "EDIFFG": -0.02,
            "IBRION": 3,
            "ICHAIN": 0,
            "IOPT": 1,
            "ISIF": 2,
            "ISMEAR": 0,
            "ISPIN": 2,
            "LCHARG": False,
            "LCLIMB": True,
            "LDAU": False,
            "LORBIT": 0,
            "NSW": 200,
            "POTIM": 0,
            "SPRING": -5,
        }
        if user_incar_settings != {}:
            defaults.update(user_incar_settings)

        kwargs["user_incar_settings"] = defaults

        super().__init__(structures, **kwargs)


def get_endpoints_from_index(structure, site_indices):
    """
    This class reads in one perfect structure and the two endpoint structures
    are generated using site_indices.

    Args:
        structure (Structure): A perfect structure.
        site_indices (list of int): a two-element list indicating site indices.

    Returns:
        endpoints (list of Structure): a two-element list of two endpoints
                                        Structure object.
    """

    if len(site_indices) != 2 or len(set(site_indices)) != 2:
        raise ValueError("Invalid indices!")
    if structure[site_indices[0]].specie != structure[site_indices[1]].specie:
        raise ValueError("The site indices must be " "associated with identical species!")

    s = structure.copy()
    sites = s.sites

    # Move hopping atoms to the beginning of species index.
    init_site = sites[site_indices[0]]
    final_site = sites[site_indices[1]]
    sites.remove(init_site)
    sites.remove(final_site)

    init_sites = copy.deepcopy(sites)
    final_sites = copy.deepcopy(sites)

    init_sites.insert(0, final_site)
    final_sites.insert(0, init_site)

    s_0 = Structure.from_sites(init_sites)
    s_1 = Structure.from_sites(final_sites)

    endpoints = [s_0, s_1]

    return endpoints


def get_endpoint_dist(ep_0, ep_1):
    """
    Calculate a list of site distances between two endpoints, assuming periodic
    boundary conditions.
    Args:
        ep_0 (Structure): the first endpoint structure.
        ep_1 (Structure): the second endpoint structure.
    Returns:
        dist (list): a list of distances between two structures.
    """
    ep_0.remove_oxidation_states()
    ep_1.remove_oxidation_states()
    assert ep_0.species == ep_1.species, "Formula mismatch!"
    assert ep_0.lattice.abc == ep_0.lattice.abc, "Lattice mismatch!"

    distances = []
    for site0, site1 in zip(ep_0, ep_1):
        fc = (site0.frac_coords, site1.frac_coords)
        d = ep_0.lattice.get_distance_and_image(fc[0], fc[1])[0]
        distances.append(d)

    return distances
