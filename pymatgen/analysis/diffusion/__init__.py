"""Package for diffusion analysis."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__author__ = "Materials Virtual Lab"
__email__ = "ongsp@eng.ucsd.edu"

try:
    __version__ = version("pymatgen")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass
