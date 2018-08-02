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

* OMEGA: Please refer to <a href="h* ttp://adsabs.harvard.edu/abs/2016arXiv160407824C">Côté et al. (2017)</a>.

* STELLAB: Please refer to the <a href="http://adsabs.harvard.edu/abs/2016ascl.soft10015R">NuPyCEE code library</a>.


### Installation Instructions

* Create the directory where you want to download the codes.
* Go in that directory with a terminal and clone the GitHub repository.
	* `git clone https://github.com/NuGrid/NuPyCEE.git`
* Go into the NuPyCEE directory and install the codes.
	* `python setup.py develop`
	* **Note**: Use the Python version you will be working with.
* Set the path to access stellar yields and STELLAB data. This is the path where you are currently in (inside NuPyCEE).
	* `export SYGMADIR="your_path_to_NuPyCEE"`
	* **Example**: `export SYGMADIR="benoitcote/gce_code/NuPyCEE"`
* Update the python path to locate NuPyCEE. This is the path to the directory just before the NuPyCEE directory.
	* `export PYTHONPATH="your_path_to_before_NuPyCEE:$PYTHONPATH"`
	* **Example**: `export PYTHONPATH="benoitcote/gce_code:$PYTHONPATH"`
	* **Note**: Do not forget `:$PYTHONPATH` at the end, otherwise the python path will be overwritten.
* **Note**: All `export` commands should be put into your bash file. With MAC, it is the .bash_profile file in your home directory. Otherwise, you will need to define the paths each time you open a terminal.

* When in Python mode, you can import the code by typing `import omega`, `import sygma` and `import stellab`.