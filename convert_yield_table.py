import re
import read_yields as ry
import utils as u
#ry=

#f1=open('isotope_yield_table_portinari98_marigo01_gce.txt')

y1=ry.read_nugrid_yields('yield_tables/isotope_yield_table_portinari98_marigo01_gce.txt')

ids=y1.table_mz

#iniabu=
iniabu0p02=u.iniabu('yield_tables/iniabu/iniab2.0E-02GN93.ppn')
iniabu0p008=u.iniabu('yield_tables/iniabu/iniab8.0E-03GN93.ppn')
iniabu0p004=u.iniabu('yield_tables/iniabu/iniab4.0E-03GN93.ppn')
iniabu0p0004=u.iniabu('yield_tables/iniabu/iniab4.0E-04GN93.ppn')


for h in range(len(ids)):

	m=float(ids[h].split(',')[0].split('=')[1])
	z=float(ids[h].split(',')[1].split('=')[1][:-1])
	#print 'M=',m,'Z=',z
	if z==0.0127:
		iniabu=iniabu0p02
	elif z==0.008:
		iniabu=iniabu0p008
	elif z==0.004:
		iniabu=iniabu0p004
	elif z==0.0004:
		iniabu=iniabu0p0004
	else:
		print 'error ',m,z
	yields=y1.get(M=m,Z=z,quantity='Yields')
	isos=y1.get(M=m,Z=z,quantity='Isotopes')
	remn=y1.get(M=m,Z=z,quantity='Mfinal')
	mej=m-remn
	for k in range(len(isos)):
		b1=isos[k].split('-')[0].lower()
		b2=isos[k].split('-')[1]
		isos1=b1+(5-len(b1)-len(b2))*' '+b2
		ini_iso=iniabu.habu[isos1]			
		y_tot=ini_iso*mej  + yields[k]
		#print 'got y_tot: ',y_tot
		#YIELD MODIFICATION 0.5,2,0.5 for C,Mg,Fe
		if m>=9:
			print 'apply ele correction ',m
			if isos[k] in ['C-12','Fe-56']:
				y_tot = y_tot * 0.5
			elif isos[k] == 'Mg':
				y_tot = y_tot * 2.
		if y_tot<0:
			print 'Warning ',m,z,isos[k],y_tot,yields[k],ini_iso
			print 'set to zero'
			y_tot=0.
                y1.set( M=m, Z=z, specie=isos[k], value=y_tot)         


	'''
        for k in range(len(iniabu.habu)):		
                iso=iniabu.habu.keys()[k]
		iso_conv=re.split('(\d+)',iso)[0].strip().capitalize()+'-'+re.split('(\d+)',iso)[1]
		if iso_conv in ['B-10','B-11']:	
			continue
		y=y1.get(M=m,Z=z,quantity=iso_conv)
		ini_iso=iniabu.habu.value()[k]
                y_tot=ini_iso*mej  + yields[k]
		y1.set( M=m, Z=z, specie=iso_conv, value=y_tot)		
	'''
y1.write_table(filename='isotope_yield_table_portinari98_marigo01_gce_totalyields.txt')
print 'Created isotope_yield_table_portinari98_marigo01_gce_total_yields.txt'
