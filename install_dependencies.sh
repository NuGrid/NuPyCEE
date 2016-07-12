###Required python modules for NUPYCEE.
#(You will need the pip package.)

#In the ideal case running this script is enough
#to install all requirements.


#The following will install required packages
#via pip

pip install ipython
pip install numpy
pip install matplotlib
#pip install mpl_toolkits
pip install scipy
pip install jupyter notebook

pip install pysph
pip install astropy
pip install nugridpy
pip install h5py


#The modules below should be already availale
#with the python on your system.
#(They are not available via pip and you might
#need to install them via your os package manager.)

# time
# copy
# math
# random
# os
# re
# imp
# cPickle
# getpass

#Possibly the module mpl_toolkits is not available.
#If 'import mpl_toolkits.mplot3d' does not work 
#you need to install it on your own.

#To start the interactive ipython session:

ipython --pylab --profile=numpy

