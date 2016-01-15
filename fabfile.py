"""
Deployment file to facilitate releases of custodian.
"""

from __future__ import division

__author__ = "Shyue Ping Ong"
__copyright__ = "Copyright 2012, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Shyue Ping Ong"
__email__ = "shyue@mit.edu"
__date__ = "Apr 29, 2012"

import glob

from fabric.api import local, lcd
from pymatgen_diffusion import __version__ as ver


def make_doc():
    with lcd("docs"):
        local("sphinx-apidoc -o . -f ../pymatgen_diffusion")
        #local("rm pymatgen_diffusion.tests.rst")
        for f in glob.glob("docs/*.rst"):
            if f.startswith('docs/pymatgen_diffusion') and f.endswith('rst'):
                newoutput = []
                suboutput = []
                subpackage = False
                with open(f, 'r') as fid:
                    for line in fid:
                        clean = line.strip()
                        if clean == "Subpackages":
                            subpackage = True
                        if not subpackage and not clean.endswith("tests"):
                            newoutput.append(line)
                        else:
                            if not clean.endswith("tests"):
                                suboutput.append(line)
                            if clean.startswith("pymatgen_diffusion") and not clean.endswith("tests"):
                                newoutput.extend(suboutput)
                                subpackage = False
                                suboutput = []

                with open(f, 'w') as fid:
                    fid.write("".join(newoutput))

        local("make html")


def publish():
    local("python setup.py release")


def test():
    local("nosetests")


def setver():
    local("sed s/version=.*,/version=\\\"{}\\\",/ setup.py > newsetup".format(ver))
    local("mv newsetup setup.py")


def release():
    setver()
    test()
    make_doc()
    publish()
