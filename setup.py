# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the Modified BSD License.

from setuptools import setup, find_packages


setup(
    name="pymatgen-diffusion",
    packages=find_packages(),
    version="0.1.0",
    install_requires=["pymatgen>=3.3.1", "monty>=0.7.1"],
    extras_require={},
    package_data={},
    author="materials virtual lab",
    author_email="ongsp@eng.ucsd.edu",
    maintainer="materials virtual lab",
    url="https://github.com/materialvirtuallab/pymatgen-diffusion/",
    license="MIT",
    description="Add-on to pymatgen for diffusion analysis.",
    long_description="Add-on to pymatgen for diffusion analysis.",
    keywords=["VASP", "gaussian", "diffusion", "molecular dynamics"],
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    scripts=[]
)
