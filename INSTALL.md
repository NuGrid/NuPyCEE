# Installation Instructions
Everything is still very tentative, but here is what seems to work.

* Go into the NuPyCEE directory
* Run `python setup.py develop`
* Make sure to set the environment variable $SYGMADIR to point to this directory (used for accessing the yield tables)
* Now you can do something like `from NuPyCEE import omega, sygma`

Note: these were instructions with anaconda python.
If you are using something else, you may have to set your `PYTHONPATH` environment variable to point to the directory containing NuPyCEE.
