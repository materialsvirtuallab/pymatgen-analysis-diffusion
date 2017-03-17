Introduction
============

This is an add-on to pymatgen for diffusion analysis that is developed
by the Materials Virtual Lab. Note that it relies on pymatgen for structural
manipulations, file io, and preliminary analyses. In particular, pymatgen's
DiffusionAnalyzer is used heavily. The purpose of this add-on
is to provide other diffusion analyses, using trajectories extracted using the
DiffusionAnalyzer class.

This is, and will always be, a scientific work in progress. Pls check back
for more details.

Change Log
==========

:doc:`Change log </change_log>`

Features (non-exhaustive!)
==========================

1. Van-Hove analysis
2. Probability density
3. Clustering (e.g., k-means with periodic boundary conditions).
4. IDPP analysis.

API documentation
-----------------

For detailed documentation of all modules and classes, please refer to the
:doc:`API docs </modules>`.

.. toctree::

   modules

Citing
======

If you use this package, please consider citing the relevant publications
documented in each analysis.

Contributing
============

We welcome contributions in all forms. If you'd like to contribute, please
fork this repository, make changes and send us a pull request!

Acknowledgements
================

This code is funded by the National Science Foundation’s Designing Materials
to Revolutionize and Engineer our Future (DMREF) program under Grant No.
1436976.

License
=======

Pymatgen is released under the MIT License. The terms of the license are as
follows:

.. literalinclude:: ../LICENSE

Our Copyright Policy
====================

.. literalinclude:: ../COPYRIGHT


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _`pymatgen's Google Groups page`: https://groups.google.com/forum/?fromgroups#!forum/pymatgen/
.. _`PyPI` : http://pypi.python.org/pypi/pymatgen
.. _`Github page`: https://github.com/materialsproject/pymatgen/issues
.. _`custodian`: https://pypi.python.org/pypi/custodian
.. _`FireWorks`: https://pythonhosted.org/FireWorks/
.. _`Materials Virtual Lab`: http://www.materialsvirtuallab.org
