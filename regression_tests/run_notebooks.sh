#here the notebooks will be added line by line to be executed with TRAVIS
#via runipy

#get all notebooks available via 


cp ../Teaching/*.ipynb .


## execute all notebooks via runipy
for f in *.ipynb
do
	echo "Processing $f"
	echo python code_to_add.py $f
	python code_to_add.py $f
	runipy "$f"
done


