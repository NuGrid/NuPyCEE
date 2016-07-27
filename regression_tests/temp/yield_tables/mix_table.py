
f1=open('sn1a_ivo12_stable_z.txt')
lines1=f1.readlines()
f1.close()
f2=open('sn1a_ivo12_unstable_z.txt')
lines2=f2.readlines()
f2.close()

isolast=''
linesout=''
masslist=[]
for line in lines1:

	if ('H' in line[0]) or ('Isotopes' in line):
		continue
	iso=line.split('&')[1].strip()
	isoname=iso.split('-')[0]
	mass=iso.split('-')[1]
	if (isoname == isolast):
		isolist.append(iso)
		linelist.append(line)
		masslist.append(mass)
	else:

		if len(masslist)>0:
			
			#add file2 isotopes:

			for line2 in lines2:
				
		        	if ('H' in line2[0]) or ('Isotopes' in line2):
                			continue
				iso2=line2.split('&')[1].strip()
				isoname2=iso2.split('-')[0]
				mass2=iso2.split('-')[1]
				if isolast == isoname2:
					linelist.append(line2)
					masslist.append(mass2)

			#sort lists for a correct order
			sort_idx=sorted(range(len(masslist)),key=lambda x:masslist[x])
			
			for idx in sort_idx:
				linesout+=linelist[idx]
	
		#reset
		iso_list=[]
		masslist=[]
		linelist=[]		
		isolist.append(iso)
		linelist.append(line)
		masslist.append(mass)		
       
	if ( line==lines1[-1]):
		#sort lists for a correct order
		sort_idx=sorted(range(len(masslist)),key=lambda x:masslist[x])
		for idx in sort_idx:
			linesout+=linelist[idx]

		


	isolast=isoname

f=open('test','w')
f.write(linesout)
f.close()

