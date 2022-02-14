#!/usr/bin/env python

"""
Deployment file to facilitate releases of pymatgen.analysis.diffusion.
"""


import datetime
import glob
import json
import os
import re

import requests
from invoke import task
from monty.os import cd

__author__ = "Shyue Ping Ong"
__date__ = "Apr 29, 2012"


NEW_VER = datetime.datetime.today().strftime("%Y.%-m.%-d")


@task
def make_doc(ctx):
    with cd("docs_rst"):
        ctx.run("cp ../CHANGES.rst change_log.rst")
        ctx.run("sphinx-apidoc --implicit-namespaces --separate -d 7 -o . -f ../pymatgen")
        ctx.run("rm pymatgen*.tests.rst")
        for f in glob.glob("*.rst"):
            if f.startswith("pymatgen") and f.endswith("rst"):
                newoutput = []
                suboutput = []
                subpackage = False
                with open(f) as fid:
                    for line in fid:
                        clean = line.strip()
                        if clean == "Subpackages":
                            subpackage = True
                        if not subpackage and not clean.endswith("tests"):
                            newoutput.append(line)
                        else:
                            if not clean.endswith("tests"):
                                suboutput.append(line)
                            if clean.startswith("pymatgen") and not clean.endswith("tests"):
                                newoutput.extend(suboutput)
                                subpackage = False
                                suboutput = []

                with open(f, "w") as fid:
                    fid.write("".join(newoutput))
        ctx.run("make html")

    with cd("docs"):
        ctx.run("cp -r html/* .")
        ctx.run("rm -r html")
        # Avoid ths use of jekyll so that _dir works as intended.
        ctx.run("touch .nojekyll")


@task
def set_ver(ctx):
    lines = []
    with open("pymatgen/analysis/diffusion/__init__.py") as f:
        for l in f:
            if "__version__" in l:
                lines.append('__version__ = "%s"' % NEW_VER)
            else:
                lines.append(l.rstrip())
    with open("pymatgen/analysis/diffusion/__init__.py", "wt") as f:
        f.write("\n".join(lines))

    lines = []
    with open("setup.py") as f:
        for l in f:
            lines.append(re.sub(r"version=([^,]+),", 'version="%s",' % NEW_VER, l.rstrip()))
    with open("setup.py", "wt") as f:
        f.write("\n".join(lines))
    ctx.run("black pymatgen")
    ctx.run("black setup.py")


@task
def update_doc(ctx):
    make_doc(ctx)
    with cd("docs"):
        ctx.run("git add .")
        ctx.run('git commit -a -m "Update dev docs"')
        ctx.run("git push")


@task
def publish(ctx):
    ctx.run("rm dist/*.*", warn=True)
    ctx.run("python setup.py sdist bdist_wheel")
    ctx.run("twine upload dist/*")


@task
def release_github(ctx):
    payload = {
        "tag_name": "v" + NEW_VER,
        "target_commitish": "master",
        "name": "v" + NEW_VER,
        "body": "v" + NEW_VER,
        "draft": False,
        "prerelease": False,
    }
    response = requests.post(
        "https://api.github.com/repos/materialsvirtuallab/pymatgen-diffusion/releases",
        data=json.dumps(payload),
        headers={"Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]},
    )
    print(response.text)


@task
def test(ctx):
    ctx.run("pytest pymatgen")


@task
def release(ctx):
    set_ver(ctx)
    # test(ctx)
    update_doc(ctx)
    publish(ctx)
    release_github(ctx)
