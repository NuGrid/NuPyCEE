
f1=open('wiersma_paper_data.txt')
lines=f1.readlines()
f1.close()

plotcount=0
wiersmadata=[]
table_header=[]
x=[]
y=[]
for k in range(len(lines)):
	if k==0:
		continue
	if 'H' in lines[k][0]:
		#print 'found H:'
		#print lines[k]
		if k == 1:
			table_header.append(lines[k].split())
			continue
		#plt.figure(fig[plotcount])
		#plotcount +=1
		#plt.plot(x,y,label='
	        table_header.append(lines[k].split())	
		wiersmadata.append([x,y])	
		x=[]
		y=[]
		continue
	line=lines[k].split()
	x.append(float(line[0]))
	y.append(float(line[1]))
#table_header.append(lines[k].split())
wiersmadata.append([x,y])
	
	

def getW(fig=2,specie='Fe',source='agb',Z=0.02):

	for k in range(len(table_header)):
		th=table_header[k]
		#print th[1][-1],str(fig)
		if (((th[1][-1] == str(fig)) and (th[2] == specie)) and ((source == th[3]) and (Z == float(th[4])))) :
			return wiersmadata[k]
