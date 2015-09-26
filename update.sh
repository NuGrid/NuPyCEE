#sphinx doc, manually done with 'make html' in SPHINX dir
echo 'Updating SPHINX documentation...'
cd SPHINX
make html
cd ../
echo 'Updating SPHINX documentation done'
#for online access
echo 'compile DOC pdf'
cd DOC
./runlatex
echo 'compile DOC pdf done'
# finally submit
echo 'Submitting to bitbucket'
#inside the NUPYCEE dir
git commit  -am 'Sphinx update'
git push -u origin master
echo 'Sphinx update done'
echo 'Note: Update of web pages might take a moment'
