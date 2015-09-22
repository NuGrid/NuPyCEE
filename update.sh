echo 'Updating SPHINX documentation...'
cd SPHINX
update html
echo 'done'
echo 'Submitting to bitbucket'
#inside the NUPYCEE dir
git commit  -am 'Sphinx update'
git push -u origin master
echo 'Done
echo 'Note: Update of web pages might take a moment'
