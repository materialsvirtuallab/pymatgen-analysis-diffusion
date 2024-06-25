.. image:: https://github.com/materialsvirtuallab/pymatgen-analysis-diffusion/actions/workflows/testing.yml/badge.svg
      :alt: CI Status
      :target: https://github.com/materialsvirtuallab/pymatgen-diffusion/actions/workflows/testing.yml

.. image:: https://codecov.io/gh/materialsvirtuallab/pymatgen-analysis-diffusion/graph/badge.svg?token=4lH4UZcXye
      :target: https://codecov.io/gh/materialsvirtuallab/pymatgen-analysis-diffusion

pymatgen-analysis-diffusion
===========================

Formerly pymatgen-diffusion, this is an add-on to pymatgen for diffusion
analysis that is developed by the Materials Virtual Lab. Note that it relies on
pymatgen for structural manipulations, file io, and preliminary analyses. This is
and will always be, a scientific work in progress. Pls check back regularly for
more details.

Documentation available via `Github Pages <https://github.com/materialsvirtuallab/pymatgen-analysis-diffusion>`_.

Major Update (v2021.3.5)
========================

pymatgen-analysis-diffusion is now released as a namespace package `pymatgen-analysis-diffusion` on PyPI. It should be
imported via `pymatgen.analysis.diffusion` instead `pymatgen_diffusion`. To install this package via pip::

    pip install pymatgen-analysis-diffusion

Features (non-exhaustive!)
==========================

1. Van-Hove analysis
2. Probability density
3. Clustering (e.g., k-means with periodic boundary conditions).
4. Migration path finding and IDPP.

Citing
======

If you use pymatgen-diffusion in your research, please cite the following
work::

    Deng, Z.; Zhu, Z.; Chu, I.H.; Ong, S. P. Data-Driven First-Principles
    Methods for the Study and Design of Alkali Superionic Conductors,
    Chem. Mater., 2016, acs.chemmater.6b02648, doi:10.1021/acs.chemmater.6b02648.

You should also include the following citation for the pymatgen core package
given that it forms the basis for most of the analyses::

    Shyue Ping Ong, William Davidson Richards, Anubhav Jain, Geoffroy Hautier,
    Michael Kocher, Shreyas Cholia, Dan Gunter, Vincent Chevrier, Kristin A.
    Persson, Gerbrand Ceder. *Python Materials Genomics (pymatgen) : A Robust,
    Open-Source Python Library for Materials Analysis.* Computational
    Materials Science, 2013, 68, 314-319. doi:10.1016/j.commatsci.2012.10.028.

In addition, some of the analyses may also have relevant publications that
you should cite. Please consult the documentation of each module.

Contributing
============

We welcome contributions in all forms. If you'd like to contribute, please
fork this repository, make changes and send us a pull request!

Acknowledgments
===============

We gratefully acknowledge funding from the following agencies for the
development of this code:

1. US National Science Foundation’s Designing Materials to Revolutionize and
   Engineer our Future (DMREF) program under Grant No. 1436976 for the AIMD
   analysis package.
2. US Department of Energy, Oﬃce of Science, Basic Energy Sciences under
   Award No. DE-SC0012118 for the NEB analysis package.
