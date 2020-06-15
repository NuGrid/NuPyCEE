#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from setuptools import setup, find_packages
from codecs import open
from os import path, system
from re import compile as re_compile
import glob

__version__="3.0"

here = path.abspath(path.dirname(__file__))
def read(filename):
    kwds = {"encoding": "utf-8"} if sys.version_info[0] >= 3 else {}
    with open(filename, **kwds) as fp:
        contents = fp.read()
    return contents

# package_data and data_files behave similarly,
# but data_files requires pip and cannot be used with setuptools
data = list(glob.glob('NuPyCEE/data/**/', recursive=True))
for i,item in enumerate(data):
    data[i] = item[8:] + '*'

setup(
    name="NuPyCEE",
    packages=['NuPyCEE'],
    package_data={"NuPyCEE":data},
    version=__version__,
    author="Benoit Cote",
    #author_email="",  # <-- Direct complaints to this address.
    description="NuPyCEE codes",
    long_description=read(path.join(here, "README.md")),
    url="https://github.com/NuGrid/NuPyCEE",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    keywords="astronomy",
    install_requires=[
        "numpy",
        ],
)
