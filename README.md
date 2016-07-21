# Pymatgen-diffusion

This is an add-on to pymatgen for diffusion analysis that is developed
by the Materials Virtual Lab. Note that it relies on pymatgen for structural 
manipulations, file io, and preliminary analyses. In particular, pymatgen's 
DiffusionAnalyzer is used heavily. The purpose of this add-on
is to provide other diffusion analyses, using trajectories extracted using the
DiffusionAnalyzer class. 

This is, and will always be, a scientific work in progress. Pls check back 
for more details.

# Features (non-exhaustive!)

1. Van-Hove analysis
2. Probability density
3. Clustering (e.g., k-means with periodic boundary conditions).

# Citing

If you use pymatgen-diffusion in your research, please cite the following
work:

    Zhi Deng, Zhuoying Zhu, Iek-Heng Chu, and Shyue Ping Ong. *Data-driven 
    First Principles Methods for the Study and Design of Alkali Superionic
    Conductors: A Case Study of Argyrodite Li6PS5Cl.* Submitted.

You should also include the following citation for the pymatgen core package
given that it forms the basis for most of the analyses.

    Shyue Ping Ong, William Davidson Richards, Anubhav Jain, Geoffroy Hautier,
    Michael Kocher, Shreyas Cholia, Dan Gunter, Vincent Chevrier, Kristin A.
    Persson, Gerbrand Ceder. *Python Materials Genomics (pymatgen) : A Robust,
    Open-Source Python Library for Materials Analysis.* Computational
    Materials Science, 2013, 68, 314-319. `doi:10.1016/j.commatsci.2012.10.028
    <http://dx.doi.org/10.1016/j.commatsci.2012.10.028>`_

In addition, some of pymatgen's functionality is based on scientific advances
/ principles developed by the computational materials scientists in our team.
Please refer to `pymatgen's documentation <http://pymatgen.org/>`_ on how to
cite them.

In addtion, some of the analyses may also have relevant publications that
you should cite. Please consult the documentation of each module.

# Contributing

We welcome contributions in all forms. If you'd like to contribute, please 
fork this repository, make changes and send us a pull request!

# Acknowledgements

This code is funded by the National Science Foundation’s Designing Materials
to Revolutionize and Engineer our Future (DMREF) program under Grant No. 
1436976.
