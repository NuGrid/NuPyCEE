# Installation Instructions
Everything is still very tentative, but here is what seems to work.

* Go into the NuPyCEE directory
* Run `python setup.py install` (note: if you plan to edit the NuPyCEE files, you can use `python setup.py develop`)
* Make sure to set the environment variable $SYGMADIR to point to this directory (used for accessing the yield tables)
* Now you can do something like `from NuPyCEE import omega, sygma`
