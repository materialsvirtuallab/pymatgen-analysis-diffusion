#!/usr/bin/env python

"""
Deployment file to facilitate releases of pymatgen.analysis.diffusion.
"""
from __future__ import annotations

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
def make_doc(ctx) -> None:
    with cd("docs_rst"):
        ctx.run("cp ../CHANGES.rst change_log.rst")
        ctx.run(
            "sphinx-apidoc --implicit-namespaces --separate -d 7 -o . -f ../pymatgen"
        )
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
                            if clean.startswith("pymatgen") and not clean.endswith(
                                "tests"
                            ):
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
def set_ver(ctx) -> None:

    lines = []
    with open("pyproject.toml") as f:
        for l in f:
            lines.append(
                re.sub(r"^version = ([^,]+)", f'version = "{NEW_VER}"', l.rstrip())
            )
    with open("pyproject.toml", "w") as f:
        f.write("\n".join(lines))
        f.write("\n")

    ctx.run("ruff format pyproject.toml")


@task
def update_doc(ctx) -> None:
    make_doc(ctx)
    with cd("docs"):
        ctx.run("git add .")
        ctx.run('git commit -a -m "Update dev docs"')
        ctx.run("git push")


@task
def publish(ctx) -> None:
    ctx.run("rm dist/*.*", warn=True)
    ctx.run("python -m build", warn=True)
    ctx.run("twine upload --skip-existing dist/*.whl", warn=True)
    ctx.run("twine upload --skip-existing dist/*.tar.gz", warn=True)


@task
def release_github(ctx) -> None:
    payload = {
        "tag_name": "v" + NEW_VER,
        "target_commitish": "master",
        "name": "v" + NEW_VER,
        "body": "v" + NEW_VER,
        "draft": False,
        "prerelease": False,
    }
    response = requests.post(
        "https://api.github.com/repos/materialsvirtuallab/pymatgen-analysis-diffusion/releases",
        data=json.dumps(payload),
        headers={"Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]},
    )
    print(response.text)


@task
def test(ctx) -> None:
    ctx.run("pytest pymatgen")


@task
def release(ctx) -> None:
    set_ver(ctx)
    # test(ctx)
    update_doc(ctx)
    publish(ctx)
    release_github(ctx)
