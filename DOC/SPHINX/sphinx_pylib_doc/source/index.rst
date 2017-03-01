NuGridpy Documentation
======================

Nugrid Pylib is a set of Python tools to access and work with (plot
etc) se hdf5 output from NuGrid (mppnp and ppn) and MESA as well as MESA
output in the 'LOGS' directory, which are the star.log file and the
logxxx.data files as well as.  These modules were written with an
interactive work mode in mind, in particular taking advantage of the
interactive ipython session that we usually start with
'ipython --q4thread --pylab' (you may not need the --q4thread if you
have a better installation of things than we do, it effects the way
show() is or is not implied working).

Once your session starts import the mesa or mppnp or ppn module
(depending on which type of data you are working with):

>>> import mesa as ms

and read the docstring:

>>> help(ms)

There are reasonable doc strings in the modules.  If you have made
tested and debugged improvements we are happy to know about them and
we may add them to the release available on the web page.  In
particular there are lots of plot methods (an imporved Kippenhahn
diagram to start with) that one can think of.  The tools provided here
are useful to us, but of course there are still many things that need
attention and improvement.  We have good list of scheduled improvements,
let me know if you want to help with these.

Contents
========

.. toctree::
   :maxdepth: 2

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

