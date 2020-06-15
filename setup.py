#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import setuptools
from numpy.distutils.core import setup, Extension
from codecs import open
from os import path, system
from re import compile as re_compile

__version__="3.0"

here = path.abspath(path.dirname(__file__))
def read(filename):
    kwds = {"encoding": "utf-8"} if sys.version_info[0] >= 3 else {}
    with open(filename, **kwds) as fp:
        contents = fp.read()
    return contents

decay_mod = Extension(name="NuPyCEE.decay_module", sources=["NuPyCEE/decay_module.f95",])

# package_data and data_files behave similarly,
# but data_files requires pip and cannot be used with setuptools
setup(
    name="NuPyCEE",
    packages=['NuPyCEE'],
    package_data={"NuPyCEE":['burst.txt',
                             'decay_data/*',
                             'decay_data/fission/*',
                             'evol_tables/*',
                             'm_dm_evolution/*',
                             'yield_tables/*',
                             'yield_tables/other/*',
                             'yield_tables/iniabu/*',
                             'stellab_data/*',
                             'stellab_data/lmc_data/*',
                             'stellab_data/solar_normalization/*',
                             'stellab_data/carina_data/*',
                             'stellab_data/fornax_data/*',
                             'stellab_data/milky_way_data/*',
                             'stellab_data/sculptor_data/*'
                            ]},
    ext_modules=[decay_mod],
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
        "ipython",
        "matplotlib",
        "scipy",
        "jupyter",
        "pysph",
        "astropy",
        "nugridpy",
        "h5py"
        ],
)
