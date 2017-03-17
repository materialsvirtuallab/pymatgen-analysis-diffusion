# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

"""
Deployment file to facilitate releases of pymatgen.
Note that this file is meant to be run from the root directory of the pymatgen
repo.
"""

__author__ = "Shyue Ping Ong"
__email__ = "ongsp@ucsd.edu"
__date__ = "Sep 1, 2014"

import glob
import os
import json
import webbrowser
import requests
import re
import subprocess
from invoke import task

from monty.os import cd
from monty.tempfile import ScratchDir
from pymatgen_diffusion import __version__ as ver


@task
def make_doc(ctx):

    with cd("docs"):
        ctx.run("sphinx-apidoc -d 6 -o . -f ../pymatgen_diffusion")
        ctx.run("rm pymatgen_diffusion.*.tests.rst")
        ctx.run("cp ../CHANGES.rst change_log.rst")
        for f in glob.glob("*.rst"):
            if f.startswith('pymatgen_diffusion') and f.endswith('rst'):
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
        ctx.run("make html")
        #ctx.run("cp _static/* _build/html/_static")

        # Avoid ths use of jekyll so that _dir works as intended.
        ctx.run("touch _build/html/.nojekyll")


@task
def update_doc(ctx):
    with cd("docs/_build/html/"):
        ctx.run("git pull")
    make_doc(ctx)
    with cd("docs/_build/html/"):
        ctx.run("git add .")
        ctx.run("git commit -a -m \"Update dev docs\"")
        ctx.run("git push origin gh-pages")


@task
def publish(ctx):
    ctx.run("python setup.py release")


@task
def setver(ctx):
    ctx.run("sed s/version=.*,/version=\\\"{}\\\",/ setup.py > newsetup"
          .format(ver))
    ctx.run("mv newsetup setup.py")


@task
def release(ctx, notest=False):
    setver(ctx)
    if not notest:
        ctx.run("nosetests")
    publish(ctx)
    update_doc(ctx)

