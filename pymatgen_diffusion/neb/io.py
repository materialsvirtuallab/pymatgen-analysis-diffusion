# coding: utf-8
# Copyright (c) Materials Virtual Lab.
# Distributed under the terms of the BSD License.


from __future__ import division, unicode_literals, print_function

from pymatgen.io.vasp.sets import MITRelaxSet, MITNEBSet
import os

__author__ = 'Austen'


class MVLCINEBEndPointSet(MITRelaxSet):
    """
    Class for writing NEB end points relaxation inputs.
    """

    def __init__(self, structure, **kwargs):
        user_incar_settings = kwargs.get("user_incar_settings", {})
        defaults = {
            "ISIF": 2,
            "EDIFFG": -0.02,
            "NELMIN": 4,
            "ISYM": 0,
            "EDIFF": 5e-5,
            "LDAU": False
        }

        if user_incar_settings != {}:
            defaults.update(user_incar_settings)
        kwargs["user_incar_settings"] = defaults

        super(MVLCINEBEndPointSet, self).__init__(structure, **kwargs)


class MVLCINEBSet(MITNEBSet):
    """
    MAVRL-tested settings for CI-NEB calculations. Note that these parameters
    requires the VTST modification of VASP from the Henkelman group. See
    http://theory.cm.utexas.edu/vtsttools/

    Args:
        nimages (int): Number of NEB images (excluding start and ending
            structures).
        user_incar_settings (dict): A dict specifying additional incar
            settings.
    """

    def __init__(self, structures, **kwargs):
        user_incar_settings = kwargs.get("user_incar_settings", {})

        # CI-NEB settings
        defaults = {
            "ISIF": 2,
            "EDIFF": 5e-5,
            "ISPIN": 2,
            "EDIFFG": -0.02,
            "NSW": 200,
            "IBRION": 3,
            "POTIM": 0,
            "ICHAIN": 0,
            "IOPT": 1,
            "LCLIMB": True,
            "LORBIT": 0,
            "ISMEAR": 0,
            "LDAU": False}
        if user_incar_settings != {}:
            defaults.update(user_incar_settings)

        kwargs["user_incar_settings"] = defaults

        super(MVLCINEBSet, self).__init__(structures, **kwargs)
