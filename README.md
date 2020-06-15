[![Build Status](https://travis-ci.org/NuGrid/NuPyCEE.svg?branch=master)](https://travis-ci.org/NuGrid/NuPyCEE) [![DOI](https://zenodo.org/badge/51356355.svg)](https://zenodo.org/badge/latestdoi/51356355)

NuPyCEE
=======

Public NuGrid Python Chemical Evolution Environment

This is a code repository containing the simple stellar population code SYGMA (Stellar Yields for Galactic Modeling Applications), the single-zone galaxy code OMEGA (One-zone Model for the Evolution of Galaxies), and the observational data plotting tool STELLAB (Stellar Abundances). 

**Requirement**: The codes are now in Python3 and use the "future" module.

**Online usage**: These tools can be used directly online via the public <a href="http://www.nugridstars.org/projects/wendi">WENDI interface</a>.

**Userguides**: See the <a href="https://github.com/NuGrid/NuPyCEE/tree/master/DOC"> Documentation </a> folder.

**Acknowledgments**: 

* SYGMA: Please refer to <a href="http://adsabs.harvard.edu/abs/2017arXiv171109172R">Ritter et al. (2017)</a>.

* OMEGA: Please refer to <a href="http://adsabs.harvard.edu/abs/2016arXiv160407824C">Côté et al. (2017)</a>.

* STELLAB: Please refer to the <a href="http://adsabs.harvard.edu/abs/2016ascl.soft10015R">NuPyCEE code library</a>.


### Installation Instructions
* Run `python setup.py install`
* You can import the individual models with:
	* `from NuPyCEE import omega`
	* `from NuPyCEE import sygma`
	* `from NuPyCEE import stellab`


### Installation of the Decay Module for Using Radioactive Isotopes

* In the NuPyCEE folder, type the following
	* `f2py -c decay_module.f95 -m decay_module`
	* **Note**: Use the f2py version that will be compatible with your Python version.
