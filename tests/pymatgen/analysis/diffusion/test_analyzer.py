# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.
from __future__ import annotations

import csv
import json
import os
import random

import matplotlib as mpl
import numpy as np
import pytest
import scipy.constants as const

from pymatgen.analysis.diffusion.analyzer import (
    DiffusionAnalyzer,
    fit_arrhenius,
    get_arrhenius_plot,
    get_conversion_factor,
)
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.util.testing import PymatgenTest

module_dir = os.path.dirname(os.path.abspath(__file__))


class FuncTest(PymatgenTest):
    def test_get_conversion_factor(self) -> None:
        s = PymatgenTest.get_structure("LiFePO4")
        # large tolerance because scipy constants changed between 0.16.1 and 0.17
        self.assertAlmostEqual(41370704.343540139, get_conversion_factor(s, "Li", 600), delta=20)

    def test_fit_arrhenius(self) -> None:
        Ea = 0.5
        k = const.k / const.e
        c = 12
        temps = np.array([300, 1000, 500])
        diffusivities = c * np.exp(-Ea / (k * temps))
        diffusivities *= np.array([1.00601834013, 1.00803236262, 0.98609720824])

        r = fit_arrhenius(temps, diffusivities)
        self.assertAlmostEqual(r[0], Ea)
        self.assertAlmostEqual(r[1], c)
        self.assertAlmostEqual(r[2], 0.000895566)

        r = fit_arrhenius(temps, diffusivities, mode="exp", diffusivity_errors=diffusivities * 0.01)
        self.assertAlmostEqual(r[0], Ea, 5)
        self.assertAlmostEqual(r[1], c, 2)
        self.assertAlmostEqual(r[2], 0.000904815)

        # when not enough values for error estimate
        r2 = fit_arrhenius([1, 2], [10, 10])
        self.assertAlmostEqual(r2[0], 0)
        self.assertAlmostEqual(r2[1], 10)
        assert r2[2] is None

        ax = get_arrhenius_plot(temps, diffusivities)
        assert isinstance(ax, mpl.axes.Axes)

        ax = get_arrhenius_plot(temps, diffusivities, mode="exp", diffusivity_errors=diffusivities * 0.01, unit="eV")
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_xlabel() == "T (K)"
        assert ax.get_ylabel() == "D (cm$^2$/s)"


class DiffusionAnalyzerTest(PymatgenTest):
    def test_init(self) -> None:
        # Diffusion vasprun.xmls are rather large. We are only going to use a
        # very small preprocessed run for testing. Note that the results are
        # unreliable for short runs.
        with open(os.path.join(module_dir, "DiffusionAnalyzer.json")) as f:
            dd = json.load(f)

            d = DiffusionAnalyzer.from_dict(dd)
            # large tolerance because scipy constants changed between 0.16.1 and 0.17
            self.assertAlmostEqual(d.conductivity, 74.165372613735684, 4)
            self.assertAlmostEqual(d.chg_conductivity, 232.8278799754324, 4)
            self.assertAlmostEqual(d.diffusivity, 1.16083658794e-06, 7)
            self.assertAlmostEqual(d.chg_diffusivity, 3.64565578208e-06, 7)
            self.assertAlmostEqual(d.conductivity_std_dev, 0.0097244677795984488, 7)
            self.assertAlmostEqual(d.diffusivity_std_dev, 9.1013023085561779e-09, 7)
            self.assertAlmostEqual(d.chg_diffusivity_std_dev, 7.20911399729e-10, 5)
            self.assertAlmostEqual(d.haven_ratio, 0.31854161048867402, 7)
            assert d.conductivity_components == pytest.approx([45.7903694, 26.1651956, 150.5406140])
            assert d.diffusivity_components == pytest.approx([7.49601236e-07, 4.90254273e-07, 2.24649255e-06], rel=0.2)
            assert d.conductivity_components_std_dev == pytest.approx([0.0063566, 0.0180854, 0.0217918], rel=0.1)
            assert d.diffusivity_components_std_dev == pytest.approx(
                [8.9465670e-09, 2.4931224e-08, 2.2636384e-08], abs=1e-3
            )
            assert d.mscd[0:4] == pytest.approx([0.69131064, 0.71794072, 0.74315283, 0.76703961], abs=1e-3)

            assert d.max_ion_displacements == pytest.approx(
                [
                    1.4620659693989553,
                    1.2787303484445025,
                    3.419618540097756,
                    2.340104469126246,
                    2.6080973517594233,
                    1.3928579365672844,
                    1.3561505956708932,
                    1.6699242923686253,
                    1.0352389639563648,
                    1.1662520093955808,
                    1.2322019205885841,
                    0.8094210554832534,
                    1.9917808504954169,
                    1.2684148391206396,
                    2.392633794162402,
                    2.566313049232671,
                    1.3175030435622759,
                    1.4628945430952793,
                    1.0984921286753002,
                    1.2864482076554093,
                    0.655567027815413,
                    0.5986961164605746,
                    0.5639091444309045,
                    0.6166004192954059,
                    0.5997911580422605,
                    0.4374606277579815,
                    1.1865683960470783,
                    0.9017064371676591,
                    0.6644840367853767,
                    1.0346375380664645,
                    0.6177630142863979,
                    0.7952002051914302,
                    0.7342686123054011,
                    0.7858047956905577,
                    0.5570732369065661,
                    1.0942937746885417,
                    0.6509372395308788,
                    1.0876687380413455,
                    0.7058162184725,
                    0.8298306317598585,
                    0.7813913747621343,
                    0.7337655232056153,
                    0.9057161616236746,
                    0.5979093093186919,
                    0.6830333586985015,
                    0.7926500894084628,
                    0.6765180009988608,
                    0.8555866032968998,
                    0.713087091642237,
                    0.7621007695790749,
                ],
            )

            assert d.sq_disp_ions.shape == (50, 206)
            assert d.lattices.shape == (1, 3, 3)
            assert d.mscd.shape == (206,)
            assert d.mscd.shape == d.msd.shape
            self.assertAlmostEqual(d.max_framework_displacement, 1.18656839605)

            ss = list(d.get_drift_corrected_structures(10, 1000, 20))
            assert len(ss) == 50
            n = random.randint(0, 49)
            n_orig = n * 20 + 10
            assert ss[n].cart_coords - d.structure.cart_coords + d.drift[:, n_orig, :] == pytest.approx(
                d.disp[:, n_orig, :],
            )

            d = DiffusionAnalyzer.from_dict(d.as_dict())
            assert isinstance(d, DiffusionAnalyzer)

            # Ensure summary dict is json serializable.
            json.dumps(d.get_summary_dict(include_msd_t=True))

            d = DiffusionAnalyzer(
                d.structure,
                d.disp,
                d.specie,
                d.temperature,
                d.time_step,
                d.step_skip,
                smoothed="max",
            )
            self.assertAlmostEqual(d.conductivity, 74.165372613735684, 4)
            self.assertAlmostEqual(d.diffusivity, 1.14606446822e-06, 7)
            self.assertAlmostEqual(d.haven_ratio, 0.318541610489, 6)
            self.assertAlmostEqual(d.chg_conductivity, 232.8278799754324, 4)
            self.assertAlmostEqual(d.chg_diffusivity, 3.64565578208e-06, 7)

            d = DiffusionAnalyzer(
                d.structure,
                d.disp,
                d.specie,
                d.temperature,
                d.time_step,
                d.step_skip,
                smoothed=False,
            )
            self.assertAlmostEqual(d.conductivity, 27.20479170406027, 4)
            self.assertAlmostEqual(d.diffusivity, 4.25976905436e-07, 7)
            self.assertAlmostEqual(d.chg_diffusivity, 1.6666666666666667e-17, 3)

            d = DiffusionAnalyzer(
                d.structure,
                d.disp,
                d.specie,
                d.temperature,
                d.time_step,
                d.step_skip,
                smoothed="constant",
                avg_nsteps=100,
            )

            self.assertAlmostEqual(d.conductivity, 47.404056230438741, 4)
            self.assertAlmostEqual(d.diffusivity, 7.4226016496716148e-07, 7)
            self.assertAlmostEqual(d.chg_conductivity, 1.06440821953e-09, 4)

            # Can't average over 2000 steps because this is a 1000-step run.
            self.assertRaises(  # noqa: PT027
                ValueError,
                DiffusionAnalyzer,
                d.structure,
                d.disp,
                d.specie,
                d.temperature,
                d.time_step,
                d.step_skip,
                smoothed="constant",
                avg_nsteps=2000,
            )

            d = DiffusionAnalyzer.from_structures(
                list(d.get_drift_corrected_structures()),
                d.specie,
                d.temperature,
                d.time_step,
                d.step_skip,
                smoothed=d.smoothed,
                avg_nsteps=100,
            )
            self.assertAlmostEqual(d.conductivity, 47.404056230438741, 4)
            self.assertAlmostEqual(d.diffusivity, 7.4226016496716148e-07, 7)

            d.export_msdt("test.csv")
            with open("test.csv") as f:
                _data = []
                for row in csv.reader(f):
                    if row:
                        _data.append(row)
            _data.pop(0)
            data = np.array(_data, dtype=np.float64)
            assert [row[1] for row in data] == pytest.approx(d.msd)
            assert [row[-1] for row in data] == pytest.approx(d.mscd)
            os.remove("test.csv")

    def test_init_npt(self) -> None:
        # Diffusion vasprun.xmls are rather large. We are only going to use a
        # very small preprocessed run for testing. Note that the results are
        # unreliable for short runs.
        with open(os.path.join(module_dir, "DiffusionAnalyzer_NPT.json")) as f:
            dd = json.load(f)
            d = DiffusionAnalyzer.from_dict(dd)
            # large tolerance because scipy constants changed between 0.16.1 and 0.17
            self.assertAlmostEqual(d.conductivity, 499.1504129387108, 4)
            self.assertAlmostEqual(d.chg_conductivity, 1219.5959181678043, 4)
            self.assertAlmostEqual(d.diffusivity, 8.40265434771e-06, 7)
            self.assertAlmostEqual(d.chg_diffusivity, 2.05305709033e-05, 6)
            self.assertAlmostEqual(d.conductivity_std_dev, 0.10368477696021029, 7)
            self.assertAlmostEqual(d.diffusivity_std_dev, 9.1013023085561779e-09, 7)
            self.assertAlmostEqual(d.chg_diffusivity_std_dev, 1.20834853646e-08, 6)
            self.assertAlmostEqual(d.haven_ratio, 0.409275240679, 7)
            assert d.conductivity_components == pytest.approx([455.178101, 602.252644, 440.0210014])
            assert d.diffusivity_components == pytest.approx([7.66242570e-06, 1.01382648e-05, 7.40727250e-06])
            assert d.conductivity_components_std_dev == pytest.approx([0.1196577, 0.0973347, 0.1525400])
            assert d.diffusivity_components_std_dev == pytest.approx([2.0143072e-09, 1.6385239e-09, 2.5678445e-09])

            assert d.max_ion_displacements == pytest.approx(
                [
                    1.13147881,
                    0.79899554,
                    1.04153733,
                    0.96061850,
                    0.83039864,
                    0.70246715,
                    0.61365911,
                    0.67965179,
                    1.91973907,
                    1.69127386,
                    1.60568746,
                    1.35587641,
                    1.03280378,
                    0.99202692,
                    2.03359655,
                    1.03760269,
                    1.40228350,
                    1.36315080,
                    1.27414979,
                    1.26742035,
                    0.88199589,
                    0.97700804,
                    1.11323184,
                    1.00139511,
                    2.94164403,
                    0.89438909,
                    1.41508334,
                    1.23660358,
                    0.39322939,
                    0.54264064,
                    1.25291806,
                    0.62869809,
                    0.40846708,
                    1.43415505,
                    0.88891241,
                    0.56259128,
                    0.81712740,
                    0.52700441,
                    0.51011733,
                    0.55557882,
                    0.49131002,
                    0.66740277,
                    0.57798671,
                    0.63521025,
                    0.50277142,
                    0.52878021,
                    0.67803443,
                    0.81161269,
                    0.46486345,
                    0.47132761,
                    0.74301293,
                    0.79285519,
                    0.48789600,
                    0.61776836,
                    0.60695847,
                    0.67767756,
                    0.70972268,
                    1.08232442,
                    0.87871177,
                    0.84674206,
                    0.45694693,
                    0.60417985,
                    0.61652272,
                    0.66444583,
                    0.52211986,
                    0.56544134,
                    0.43311443,
                    0.43027547,
                    1.10730439,
                    0.59829728,
                    0.52270635,
                    0.72327608,
                    1.02919775,
                    0.84423208,
                    0.61694764,
                    0.72795752,
                    0.72957755,
                    0.55491631,
                    0.68507454,
                    0.76745343,
                    0.96346584,
                    0.66672645,
                    1.06810107,
                    0.65705843,
                ],
            )

            assert d.sq_disp_ions.shape == (84, 217)
            assert d.lattices.shape == (1001, 3, 3)
            assert d.mscd.shape == (217,)
            assert d.mscd.shape == d.msd.shape

            self.assertAlmostEqual(d.max_framework_displacement, 1.43415505156)

            ss = list(d.get_drift_corrected_structures(10, 1000, 20))
            assert len(ss) == 50
            n = random.randint(0, 49)
            n_orig = n * 20 + 10
            assert ss[n].cart_coords - d.structure.cart_coords + d.drift[:, n_orig, :] == pytest.approx(
                d.disp[:, n_orig, :]
            )

            d = DiffusionAnalyzer.from_dict(d.as_dict())
            assert isinstance(d, DiffusionAnalyzer)

            # Ensure summary dict is json serializable.
            json.dumps(d.get_summary_dict(include_msd_t=True))

            d = DiffusionAnalyzer(
                d.structure,
                d.disp,
                d.specie,
                d.temperature,
                d.time_step,
                d.step_skip,
                smoothed="max",
            )
            self.assertAlmostEqual(d.conductivity, 499.1504129387108, 4)
            self.assertAlmostEqual(d.diffusivity, 8.40265434771e-06, 7)
            self.assertAlmostEqual(d.haven_ratio, 0.409275240679, 7)
            self.assertAlmostEqual(d.chg_diffusivity, 2.05305709033e-05, 7)

            d = DiffusionAnalyzer(
                d.structure,
                d.disp,
                d.specie,
                d.temperature,
                d.time_step,
                d.step_skip,
                smoothed=False,
            )
            self.assertAlmostEqual(d.conductivity, 406.5964019770787, 4)
            self.assertAlmostEqual(d.diffusivity, 6.8446082e-06, 7)
            self.assertAlmostEqual(d.chg_diffusivity, 1.03585877962e-05, 6)
            self.assertAlmostEqual(d.haven_ratio, 0.6607665413, 6)

            d = DiffusionAnalyzer(
                d.structure,
                d.disp,
                d.specie,
                d.temperature,
                d.time_step,
                d.step_skip,
                smoothed="constant",
                avg_nsteps=100,
            )

            self.assertAlmostEqual(d.conductivity, 425.77884571149525, 4)
            self.assertAlmostEqual(d.diffusivity, 7.167523809142514e-06, 7)
            self.assertAlmostEqual(d.chg_diffusivity, 9.33480892187e-06, 6)
            self.assertAlmostEqual(d.haven_ratio, 0.767827586952, 6)
            self.assertAlmostEqual(d.chg_conductivity, 554.5240271992852, 6)

            # Can't average over 2000 steps because this is a 1000-step run.
            self.assertRaises(  # noqa: PT027
                ValueError,
                DiffusionAnalyzer,
                d.structure,
                d.disp,
                d.specie,
                d.temperature,
                d.time_step,
                d.step_skip,
                smoothed="constant",
                avg_nsteps=2000,
            )

            d = DiffusionAnalyzer.from_structures(
                list(d.get_drift_corrected_structures()),
                d.specie,
                d.temperature,
                d.time_step,
                d.step_skip,
                smoothed=d.smoothed,
                avg_nsteps=100,
            )
            self.assertAlmostEqual(d.conductivity, 425.7788457114952, 4)
            self.assertAlmostEqual(d.diffusivity, 7.1675238091425148e-06, 7)
            self.assertAlmostEqual(d.haven_ratio, 0.767827586952, 7)
            self.assertAlmostEqual(d.chg_conductivity, 554.5240271992852, 6)

            d.export_msdt("test.csv")
            with open("test.csv") as f:
                _data = []
                for row in csv.reader(f):
                    if row:
                        _data.append(row)
            _data.pop(0)
            data = np.array(_data, dtype=np.float64)
            assert [row[1] for row in data] == pytest.approx(d.msd)
            assert [row[-1] for row in data] == pytest.approx(d.mscd)
            os.remove("test.csv")

    def test_from_structure_NPT(self) -> None:
        coords1 = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        coords2 = np.array([[0.0, 0.0, 0.0], [0.6, 0.6, 0.6]])
        coords3 = np.array([[0.0, 0.0, 0.0], [0.7, 0.7, 0.7]])
        lattice1 = Lattice.from_parameters(a=2.0, b=2.0, c=2.0, alpha=90, beta=90, gamma=90)
        lattice2 = Lattice.from_parameters(a=2.1, b=2.1, c=2.1, alpha=90, beta=90, gamma=90)
        lattice3 = Lattice.from_parameters(a=2.0, b=2.0, c=2.0, alpha=90, beta=90, gamma=90)
        s1 = Structure(coords=coords1, lattice=lattice1, species=["F", "Li"])
        s2 = Structure(coords=coords2, lattice=lattice2, species=["F", "Li"])
        s3 = Structure(coords=coords3, lattice=lattice3, species=["F", "Li"])
        structures = [s1, s2, s3]
        d = DiffusionAnalyzer.from_structures(
            structures,
            specie="Li",
            temperature=500.0,
            time_step=2,
            step_skip=1,
            smoothed=None,
        )
        assert d.disp[1] == pytest.approx(np.array([[0.0, 0.0, 0.0], [0.21, 0.21, 0.21], [0.40, 0.40, 0.40]]))

        ax = d.get_msd_plot()
        assert isinstance(ax, mpl.axes.Axes)
