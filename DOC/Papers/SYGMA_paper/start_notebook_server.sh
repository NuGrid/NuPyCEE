#set the path to your NUPYCEE dir
cd ../../
export SYGMADIR=`pwd`
cd -
echo 'set SYGMADIR to '$SYGMADIR

#export path
export PYTHONPATH=$PYTHONPATH:$SYGMADIR

#Set the links properly to use the feature of yield table creation and SPHINX documentation
rm -f doc
ln -s $SYGMADIR/SPHINX/build/html doc
rm -f yield_tables
ln -s $SYGMADIR/yield_tables/ yield_tables

#starts notebook server
jupyter notebook

