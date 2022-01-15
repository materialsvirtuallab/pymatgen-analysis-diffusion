# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the Modified BSD License.

from setuptools import setup, find_namespace_packages

import os

SETUP_PTH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SETUP_PTH, "README.rst")) as f:
    desc = f.read()


setup(
    name="pymatgen-analysis-diffusion",
    packages=find_namespace_packages(include=["pymatgen.analysis.*"]),
    version="2022.1.15",
    install_requires=["pymatgen>=2022.0.3", "joblib"],
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)