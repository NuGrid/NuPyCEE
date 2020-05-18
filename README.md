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

* Create the directory where you want to download the codes.
* Go in that directory with a terminal and clone the GitHub repository:
	* `git clone https://github.com/NuGrid/NuPyCEE.git`
* From the same directory which contains the cloned NuPyCEE directory, you can import the codes in Python mode by typing:
	* `from NuPyCEE import omega`
	* `from NuPyCEE import sygma`
	* `from NuPyCEE import stellab`
* If you want to import the NuPyCEE codes from anywhere else within your work space, you have to update your Python path using a terminal:
	* `export PYTHONPATH="your_path_to_before_NuPyCEE:$PYTHONPATH"`
	* **Example**: `export PYTHONPATH="benoitcote/gce_code:$PYTHONPATH"`
	* **Important**: Do not forget `:$PYTHONPATH` at the end, otherwise the python path will be overwritten.
	* **Note**: All `export` commands should be put into your bash file. With MAC, it is the .bash_profile file in your home directory. Otherwise, you will need to define the paths each time you open a terminal.

### Installation of the Decay Module for Using Radioactive Isotopes

* In the NuPyCEE folder, type the following
	* `f2py -c decay_module.f95 -m decay_module`
	* **Note**: Use the f2py version that will be compatible with your Python version.
