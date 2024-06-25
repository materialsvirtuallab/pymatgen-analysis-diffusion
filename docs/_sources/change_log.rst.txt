Change Log
==========

v2024.6.25
----------
* Fix broken code due to pymatgen and matplotlib deprecated methods.

v2023.8.15
----------
* Complete migration of pymatgen.analysis.pathfinder over to pymatgen-analysis.diffusion.

v2022.4.22
----------
* Make MigrationHop MSONAble. (@acrutt)

v2021.4.29
----------
* Fixed msite calc order (@hmlli)
* Removed magmom (@jmmshn)

v2021.3.6
---------
* Fix for bad release due to error in MANIFEST.in.
* DiffusionAnalyzer has been migration from pymatgen to `pymatgen.analysis.diffusion.analyzer` for a more
  self-contained suite of tools for diffusion analysis.

v2021.3.5
---------
* pymatgen-diffusion is now released as a namespace package `pymatgen-analysis-diffusion` on PyPI. It should be
  imported via `pymatgen.analysis.diffusion` instead `pymatgen_diffusion`.

v2019.2.28
----------
* Py3k cleanup.

v2018.1.4
---------
* Compatibility with pymatgen v2018+.

v0.3.0
------
* NEB tools, including an implementation of an IDPP solver.
