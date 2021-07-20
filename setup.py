#!/usr/bin/env python
from setuptools import setup, find_packages


def _parse_requirements():
    # TODO: ristretto is causing issues because the install fails when installing
    # from pipy. Need to sort this out when we have completed refactoring. For now
    # we'll just make sure setup.py can use ristretto as given in requirements.txt
    requirements = []

    for line in open("requirements.txt").readlines():
        if line.startswith("#"):
            continue
        elif "git+" in line and "ristretto" in line:
            line = f"ristretto @ {line}"
        requirements.append(line)
    return requirements


INSTALL_REQUIRES = _parse_requirements()

setup(
    name="cloudmetrics",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="Cloud pattern metrics toolkit",
    url="https://github.com/EUREC4A-UK/cloudmetrics",
    maintainer="Leif Denby, Martin Janssens",
    maintainer_email="l.c.denby[at]leeds.ac.uk",
    py_modules=["cloudmetrics"],
    packages=find_packages(include=["cloudmetrics"]),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    zip_safe=False,
)
