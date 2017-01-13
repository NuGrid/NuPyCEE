import sys

filename=sys.argv[1]
def modify_code(filename):
	'''
	Add extra code to ipython notebooks
	'''

	#all content here (must be notebook format)
	f1=open('to_add_code.txt')
	lines=f1.readlines()

	f2=open(filename)
	lines2=f2.readlines()
	f2.close()
	lines3=[]

	done=False
	for k in range(len(lines2)):
		line=lines2[k]
		print 'line'
		if 'nbagg' in line:
			line=line.replace('nbagg','inline')
		if (('cells' in line) and (not done)):
			lines3=lines2[:k+1]
			for h in range(len(lines)):
				lines3.append(lines[h])
			done=True
			continue
		if done:
			lines3.append(line)
	f3=open(filename,'w')
	f3.write("\n".join(lines3))
	f3.close()

modify_code(filename)
