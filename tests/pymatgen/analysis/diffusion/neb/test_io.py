from __future__ import annotations

import os
import unittest
from typing import TYPE_CHECKING

from pymatgen.analysis.diffusion.neb.io import (
    MVLCINEBEndPointSet,
    MVLCINEBSet,
    get_endpoint_dist,
    get_endpoints_from_index,
)
from pymatgen.core import Structure

if TYPE_CHECKING:
    from pymatgen.util.typing import PathLike

__author__ = "hat003"

test_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)))


def get_path(path_str: PathLike, dirname: PathLike = "./") -> str:
    cwd = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(cwd, dirname, path_str)


class MVLCINEBEndPointSetTest(unittest.TestCase):
    endpoint = Structure.from_file(get_path("POSCAR0", dirname="io_files"))

    def test_incar(self) -> None:
        m = MVLCINEBEndPointSet(self.endpoint)
        incar_string = m.incar.get_str(sort_keys=True, pretty=True)
        incar_expect = """ALGO     =  Fast
EDIFF    =  5e-05
EDIFFG   =  -0.02
ENCUT    =  520.0
IBRION   =  2
ICHARG   =  1
ISIF     =  2
ISMEAR   =  0
ISPIN    =  2
ISYM     =  0
LCHARG   =  False
LDAU     =  False
LMAXMIX  =  4
LORBIT   =  11
LREAL    =  Auto
LWAVE    =  False
MAGMOM   =  35*0.6
NELM     =  200
NELMIN   =  4
NSW      =  99
PREC     =  Accurate
SIGMA    =  0.05"""
        assert incar_string == incar_expect

    def test_incar_user_setting(self) -> None:
        user_incar_settings = {
            "ALGO": "Normal",
            "EDIFFG": -0.05,
            "NELECT": 576,
            "NPAR": 4,
            "NSW": 100,
        }
        m = MVLCINEBEndPointSet(self.endpoint, user_incar_settings=user_incar_settings)
        incar_string = m.incar.get_str(sort_keys=True)
        incar_expect = """ALGO = Normal
EDIFF = 5e-05
EDIFFG = -0.05
ENCUT = 520
IBRION = 2
ICHARG = 1
ISIF = 2
ISMEAR = 0
ISPIN = 2
ISYM = 0
LCHARG = False
LDAU = False
LMAXMIX = 4
LORBIT = 11
LREAL = Auto
LWAVE = False
MAGMOM = 35*0.6
NELECT = 576
NELM = 200
NELMIN = 4
NPAR = 4
NSW = 100
PREC = Accurate
SIGMA = 0.05"""
        assert incar_string.strip() == incar_expect.strip()


class MVLCINEBSetTest(unittest.TestCase):
    structures = [Structure.from_file(get_path("POSCAR" + str(i), dirname="io_files")) for i in range(3)]

    def test_incar(self) -> None:
        m = MVLCINEBSet(self.structures)

        incar_string = m.incar.get_str(sort_keys=True)
        incar_expect = """ALGO = Fast
EDIFF = 5e-05
EDIFFG = -0.02
ENCUT = 520.0
IBRION = 3
ICHAIN = 0
ICHARG = 1
IMAGES = 1
IOPT = 1
ISIF = 2
ISMEAR = 0
ISPIN = 2
ISYM = 0
LCHARG = False
LCLIMB = True
LDAU = False
LMAXMIX = 4
LORBIT = 0
LREAL = Auto
LWAVE = False
MAGMOM = 35*0.6
NELM = 200
NELMIN = 6
NSW = 200
POTIM = 0
PREC = Accurate
SIGMA = 0.05
SPRING = -5"""
        assert incar_string.strip() == incar_expect.strip()

    def test_incar_user_setting(self) -> None:
        user_incar_settings = {"IOPT": 3, "EDIFFG": -0.05, "NPAR": 4, "ISIF": 3}
        m = MVLCINEBSet(self.structures, user_incar_settings=user_incar_settings)
        incar_string = m.incar.get_str(sort_keys=True, pretty=True)
        incar_expect = """ALGO     =  Fast
EDIFF    =  5e-05
EDIFFG   =  -0.05
ENCUT    =  520.0
IBRION   =  3
ICHAIN   =  0
ICHARG   =  1
IMAGES   =  1
IOPT     =  3
ISIF     =  3
ISMEAR   =  0
ISPIN    =  2
ISYM     =  0
LCHARG   =  False
LCLIMB   =  True
LDAU     =  False
LMAXMIX  =  4
LORBIT   =  0
LREAL    =  Auto
LWAVE    =  False
MAGMOM   =  35*0.6
NELM     =  200
NELMIN   =  6
NPAR     =  4
NSW      =  200
POTIM    =  0
PREC     =  Accurate
SIGMA    =  0.05
SPRING   =  -5"""
        assert incar_string.strip() == incar_expect.strip()


class UtilityTest(unittest.TestCase):
    """
    Unit test for outside methods in io.py
    """

    structure = Structure.from_file(get_path("POSCAR", dirname="io_files"))

    def test_get_endpoints_from_index(self) -> None:
        endpoints = get_endpoints_from_index(structure=self.structure, site_indices=[0, 1])
        ep_0 = endpoints[0].as_dict()
        ep_1 = endpoints[1].as_dict()
        ep_0_expect = Structure.from_file(get_path("POSCAR_ep0", dirname="io_files")).as_dict()
        ep_1_expect = Structure.from_file(get_path("POSCAR_ep1", dirname="io_files")).as_dict()

        assert ep_0 == ep_0_expect
        assert ep_1 == ep_1_expect

    def test_get_endpoint_dist(self) -> None:
        ep0 = Structure.from_file(get_path("POSCAR_ep0", dirname="io_files"))
        ep1 = Structure.from_file(get_path("POSCAR_ep1", dirname="io_files"))
        distances = get_endpoint_dist(ep0, ep1)

        self.assertAlmostEqual(max(distances), 6.3461081051543893, 7)
        assert min(distances) == 0.0
