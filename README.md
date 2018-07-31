[![Build Status](https://travis-ci.org/NuGrid/NuPyCEE.svg?branch=master)](https://travis-ci.org/NuGrid/NuPyCEE) [![DOI](https://zenodo.org/badge/51356355.svg)](https://zenodo.org/badge/latestdoi/51356355)

NuPyCEE
=======
 
Public NuGrid Python Chemical Evolution Environment

This is a code repository containing the simple stellar population code SYGMA (Stellar Yields for Galactic Modeling Applications), the single-zone galaxy code OMEGA (One-zone Model for the Evolution of Galaxies), and the observational data plotting tool STELLAB (Stellar Abundances). 

**Requirement**: The codes are now in Python3 and use the "future" module.

**Online usage**: These tools can be used directly online via the public <a href="http://www.nugridstars.org/projects/wendi">WENDI interface</a>.

**Userguides**: All of our codes have <a href="http://nugrid.github.io/NuPyCEE/teaching.html">teaching iPython notebooks</a> and userguides that are available on <a href="http://www.nugridstars.org/projects/wendi">WENDI </a>. We also have <a href="http://nugrid.github.io/NuPyCEE/SPHINX/build/html/index.html">Sphinx documentation</a> to learn about the NuPyCEE functions and input parameters.

**Acknowledgments**: 

* SYGMA: Please refer to <a href="http://adsabs.harvard.edu/abs/2017arXiv171109172R">Ritter et al. (2017)</a>.

* OMEGA: Please refer to <a href="h* ttp://adsabs.harvard.edu/abs/2016arXiv160407824C">Côté et al. (2017)</a>.

* STELLAB: Please refer to the <a href="http://adsabs.harvard.edu/abs/2016ascl.soft10015R">NuPyCEE code library</a>.


Installation Instructions
=======
* Clone the NuPyCEE package.
* Go into the NuPyCEE directory.
* Run `python setup.py develop`
* Set the environment variable $SYGMADIR to point to this directory (used for accessing the yield tables).
	* `export SYGMADIR="your_path_to_go_inside_the_NuPyCEE_directory"`
* This should be enough if you used Anaconda Python. However, if using something else, you may have to set your `PYTHONPATH` environment variable to point to the directory containing the NuPyCEE folder.
	* `export PYTHONPATH="your_path_to_go_before_the_NuPyCEE_directory:$PYTHONPATH"`
* You can include the two `export` commands into your bash file so it automatically loads when you open a terminal.

* When in Python mode, you can import the code by typing
	* `from NuPyCEE import omega, sygma`