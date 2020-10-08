# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the Modified BSD License.

from setuptools import setup, find_packages

import os

SETUP_PTH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SETUP_PTH, "README.rst")) as f:
    desc = f.read()


setup(
    name="pymatgen-diffusion",
    packages=find_packages(),
    version="2020.10.8",
    install_requires=["pymatgen>=2019.10.3", "monty>=3.0.2"],
    extras_require={},
    package_data={},
    author="materials virtual lab",
    author_email="ongsp@eng.ucsd.edu",
    maintainer="materials virtual lab",
    url="https://github.com/materialsvirtuallab/pymatgen-diffusion/",
    license="BSD",
    description="Add-on to pymatgen for diffusion analysis.",
    long_description=desc,
    keywords=["VASP", "diffusion", "molecular dynamics", "MD"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ]
)