'''

        Superclass to extract yield data from tables
        and from mppnp simulations

        Christian Ritter 11/2013

        Two classes: One for reading and extracting of
        NuGrid table data, the other one for SN1a data.



'''


import matplotlib.pyplot as plt
import numpy as np
import os

color=['r','k','b','g']
marker_type=['o','p','s','D']
line_style=['--','-','-.',':']


#global notebookmode
notebookmode=False


class read_nugrid_parameter():

    def __init__(self,nugridtable):

        '''
                dir : specifing the filename of the table file

        '''
        table=nugridtable

        import os
        if '/' in table:
            self.label=table.split('/')[-1]
        else:
            self.label=table
        self.path=table
    	file1=open(nugridtable)
    	lines=file1.readlines()
    	file1.close()
        header1=[]
        table_header=[]
        yield_data=[]
        header_done=False
	ignore=False
	col_attrs_data=[]
	######read through all lines
        for line in lines:
            if 'H' in line[0]:
                if not 'Table' in line:
                    if header_done==False:
                        header1.append(line.strip())
                    else:
                        table_header[-1].append(line.strip())
                else:
		    ignore=False
		    #print line,'ignore',ignore
		    if ignore==True:
		        header_done=True
		        continue
		    
                    table_header.append([])
                    table_header[-1].append(line.strip())
                    yield_data.append([])
                    #lum_bands.append([])
                    #m_final.append([])
		    col_attrs_data.append([])
		    col_attrs_data[-1].append(line.strip())
                    header_done=True
		    continue
		if ignore==True:
		    continue
		if header_done==True:
			#col_attrs_data.append([])
			col_attrs_data[-1].append(float(line.split(':')[1]))
                continue
	    if ignore==True:
		continue
            if '&Age' in line:
                title_line=line.split('&')[1:]
                column_titles=[]
                for t in title_line:
                    yield_data[-1].append([])
                    column_titles.append(t.strip())
                #print column_titles
                continue
            #iso ,name and yields
	    iso_name=line.split('&')[1].strip()
	    #print line
	    #print line.split('&')
            yield_data[-1][0].append(float(line.split('&')[1].strip()))
            #if len(isotopes)>0:
            #        if not iso_name in isotopes:
	    #else:    	    
	    yield_data[-1][1].append(float(line.split('&')[2].strip()))
            # for additional data
            for t in range(2,len(yield_data[-1])):
	    	yield_data[-1][t].append(float(line.split('&')[t+1].strip()))
	#choose only isotoopes and right order
        ######reading finished
	#In [43]: tablesN.col_attrs
	#Out[43]: ['Isotopes', 'Yields', 'X0', 'Z', 'A']
        self.yield_data=yield_data
        #table header points to element in yield_data
        self.table_idx={}
        i=0
        self.col_attrs=[]
        self.table_mz=[]
        self.metallicities=[]
	#self.col_attrs=table_header
	#go through all MZ pairs
        for table1 in table_header:
	    #go through col_attrs
            for k in range(len(table1)):
                table1[k]=table1[k][2:]
                if 'Table' in table1[k]:
                    self.table_idx[table1[k].split(':')[1].strip()]=i
                    tablename=table1[k].split(':')[1].strip()
                    self.table_mz.append(tablename)
                    metal=tablename.split(',')[1].split('=')[1][:-1]
                    if float(metal) not in self.metallicities:
                        self.metallicities.append(float(metal))
                if table1 ==table_header[0]:
                    if 'Table' in table1[k]:
                        table1[k] = 'Table (M,Z):'
                    self.col_attrs.append(table1[k].split(':')[0].strip())
		    
            i+=1
        #define  header
        self.header_attrs={}
        #print 'header1: ',header1
        for h in header1:
            self.header_attrs[h.split(':')[0][1:].strip()]=h.split(':')[1].strip()
        self.data_cols=column_titles #previous data_attrs
        #self.kin_e=kin_e
        #self.lum_bands=lum_bands
        #self.m_final=m_final
	self.col_attrs_data=col_attrs_data


    def get(self,M=0,Z=-1,quantity=''):

        '''
                Allows to extract table data in 2 Modes:

                1) For extracting of table data for
                   star of mass M and metallicity Z.
                   Returns either table attributes,
                   given by yield.col_attrs
                   or table columns,
                   given by yield.data_cols.

                2) For extraction of a table attribute
                   from all available tables. Can be
                   directly used in the following way:

                   get(tableattribute)


                M: Stellar mass in Msun
                Z: Stellar metallicity (e.g. Z=0.02)
                quantity: table attribute or data column/data_cols

        '''

        all_tattrs=False
        if Z ==-1:
            if M ==0 and len(quantity)>0:
                quantity1=quantity
                all_tattrs=True
            elif (M in self.col_attrs) and quantity == '':
                quantity1=M
                all_tattrs=True
            else:
                print 'Error: Wrong input'
                return 0
            quantity=quantity1


        if (all_tattrs==False) and (not M ==0):
            inp='(M='+str(float(M))+',Z='+str(float(Z))+')'
            idx=self.table_idx[inp]

	if quantity in self.col_attrs:
		if all_tattrs==False:
			data=self.col_attrs_data[idx][self.col_attrs.index(quantity)]
			return data
		else:
			data=[]
			for k in range(len(self.table_idx)):
				data.append(self.col_attrs_data[k][self.col_attrs.index(quantity)])	
			return data

        if quantity=='masses':
            data_tables=self.table_mz
            masses=[]
            for table in data_tables:
                if str(float(Z)) in table:
                    masses.append(float(table.split(',')[0].split('=')[1]))


            return masses
        else:
            data=self.yield_data[idx]
	    idx_col=self.data_cols.index(quantity)
	    set1=data[idx_col]
	    return set1

    def add_parameter_write_table(self,table_header='',dcols=[],data=[[]],filename='isotope_yield_table_MESA_only_param_new.txt'):

	'''
		Allows to add more parameter to the parameter table.
		dcols=['Test1'], data=[[....]]
	'''

	import ascii_table as ascii1
	tables=self.table_mz
	yield_data=self.yield_data
	data_cols=self.data_cols
	col_attrs=self.col_attrs
	col_attrs_data1=self.col_attrs_data
	for k in range(len(tables)):
		if not tables[k]==table_header:
			continue
		mass=float(tables[k].split(',')[0].split('=')[1])
		metallicity=float(tables[k].split(',')[1].split('=')[1][:-1])

		#read out existing data
		col_attrs=col_attrs  #MZ pairs
		col_attrs_data=col_attrs_data1[k]

		#over col attrs, first is MZ pair which will be skipped, see special_header
		attr_lines=[]
		for h in range(1,len(col_attrs)):
			attr=col_attrs[h]
			idx=col_attrs.index(attr)
			# over MZ pairs
			attr_data=col_attrs_data[k][idx]
			line=attr+': '+'{:.3E}'.format(attr_data)
			attr_lines.append(line)

		#read in available columns
		data_new=yield_data[k]
		dcols_new=data_cols[:]
		#add more data...
		for h in range(len(dcols)):
			print 'h :',h
			data_new.append(data[h])
			dcols_new.append(dcols[h])
		dcols_new=[dcols_new[0]]+dcols_new[2:]+[dcols_new[1]]
		print 'dcols: ',dcols_new
		special_header='Table: (M='+str(mass)+',Z='+str(metallicity)+')'
		headers=[special_header]+attr_lines
		ascii1.writeGCE_table_parameter(filename=filename,headers=headers,data=data_new,dcols=dcols_new)



class read_nugrid_yields():

    def __init__(self,nugridtable,isotopes=[],excludemass=[]):

        '''
                dir : specifing the filename of the table file

        '''
        table=nugridtable

        import os
        if '/' in table:
            self.label=table.split('/')[-1]
        else:
            self.label=table
        self.path=table
        if notebookmode==True:
            os.system('sudo python cp.py '+nugridtable)
            file1=open('tmp/'+nugridtable)
            lines=file1.readlines()
            file1.close()
            os.system('sudo python delete.py '+nugridtable)
        else:
            file1=open(nugridtable)
            lines=file1.readlines()
            file1.close()
        header1=[]
        table_header=[]
        age=[]
        yield_data=[]
        #kin_e=[]
        #lum_bands=[]
        #m_final=[]
        header_done=False
	ignore=False
	col_attrs_data=[]
	######read through all lines
        for line in lines:
            if 'H' in line[0]:
                if not 'Table' in line:
                    if header_done==False:
                        header1.append(line.strip())
                    else:
                        table_header[-1].append(line.strip())
                else:
		    ignore=False
		    for kk in range(len(excludemass)):
		    	if float(excludemass[kk]) == float(line.split(',')[0].split('=')[1]):
				ignore=True
				#print 'ignore',float(line.split(',')[0].split('=')[1])
				break
		    #print line,'ignore',ignore
		    if ignore==True:
		        header_done=True
		        continue
		    
                    table_header.append([])
                    table_header[-1].append(line.strip())
                    yield_data.append([])
                    #lum_bands.append([])
                    #m_final.append([])
		    col_attrs_data.append([])
		    col_attrs_data[-1].append(line.strip())
                    header_done=True
		    continue
		if ignore==True:
		    continue
		if header_done==True:
			#col_attrs_data.append([])
			col_attrs_data[-1].append(float(line.split(':')[1]))
		#age is special col_attrs, used in chem_evol.py
                if 'Lifetime' in line:
                    age.append(float(line.split(':')[1]))
		'''
                if 'kinetic energy' in line:
                    kin_e.append(float(line.split(':')[1]))
                if 'band' in line:
                    lum_bands[-1].append(float(line.split(':')[1]))
                if 'Mfinal' in line:
                    m_final[-1].append(float(line.split(':')[1]))
		'''
                continue
	    if ignore==True:
		continue
            if '&Yields' in line:
                title_line=line.split('&')[1:]
                column_titles=[]
                for t in title_line:
                    yield_data[-1].append([])
                    column_titles.append(t.strip())
                #print column_titles
                continue
            #iso ,name and yields
	    iso_name=line.split('&')[1].strip()
	    #print line
	    #print line.split('&')
            yield_data[-1][0].append(line.split('&')[1].strip())
            #if len(isotopes)>0:
            #        if not iso_name in isotopes:
	    #else:    	    
	    yield_data[-1][1].append(float(line.split('&')[2].strip()))
            # for additional data
            for t in range(2,len(yield_data[-1])):
                if column_titles[t] == 'A' or column_titles[t] =='Z':
                    yield_data[-1][t].append(int(line.split('&')[t+1].strip()))

                else:
                    yield_data[-1][t].append(float(line.split('&')[t+1].strip()))
	#choose only isotoopes and right order
        ######reading finished
	#In [43]: tablesN.col_attrs
	#Out[43]: ['Isotopes', 'Yields', 'X0', 'Z', 'A']
	if len(isotopes)>0:
		#print 'correct for isotopes'
		data_new=[]
		for k in range(len(yield_data)):
			#print 'k'
			data_new.append([])
			#print 'len',len(yield_data[k])
			#print ([[]]*len(yield_data[k]))[0]
			for h in range(len(yield_data[k])):
				data_new[-1].append([])
			#print 'testaa',data_new[-1]
			data_all=yield_data[k]
			for iso_name in isotopes:
				if iso_name in data_all[0]:
					#print 'test',data_all[1][data_all[0].index(iso_name)]
					for hh in range(1,len(data_all)):
						data_new[-1][hh].append(data_all[hh][data_all[0].index(iso_name)])
					#data_new[-1][1].append(data_all[2][data_all[0].index(iso_name)])
					#data_new[-1][1].append(data_all[2][data_all[0].index(iso_name)])
				else:
                                        for hh in range(1,len(data_all)):
                                                data_new[-1][hh].append(0)
					#data_new[-1][1].append(0)
					#print 'GRID exclude',iso_name
				data_new[-1][0].append(iso_name)
		#print 'new list'
		#print data_new[0][0]
		#print data_new[0][1]
		yield_data=data_new
        self.yield_data=yield_data
        #table header points to element in yield_data
        self.table_idx={}
        i=0
        self.col_attrs=[]
        self.table_mz=[]
        self.metallicities=[]
	#self.col_attrs=table_header
	#go through all MZ pairs
        for table1 in table_header:
	    #go through col_attrs
            for k in range(len(table1)):
                table1[k]=table1[k][2:]
                if 'Table' in table1[k]:
                    self.table_idx[table1[k].split(':')[1].strip()]=i
                    tablename=table1[k].split(':')[1].strip()
                    self.table_mz.append(tablename)
                    metal=tablename.split(',')[1].split('=')[1][:-1]
                    if float(metal) not in self.metallicities:
                        self.metallicities.append(float(metal))
                if table1 ==table_header[0]:
                    if 'Table' in table1[k]:
                        table1[k] = 'Table (M,Z):'
                    self.col_attrs.append(table1[k].split(':')[0].strip())
		    
#col_attrs_data
                #table1.split(':')[1].strip()
            i+=1
        #define  header
        self.header_attrs={}
        #print 'header1: ',header1
        for h in header1:
            self.header_attrs[h.split(':')[0][1:].strip()]=h.split(':')[1].strip()
        self.data_cols=column_titles #previous data_attrs
        self.age=age
        #self.kin_e=kin_e
        #self.lum_bands=lum_bands
        #self.m_final=m_final
	self.col_attrs_data=col_attrs_data


    def set_col_attrs(self,M=0,Z=-1,quantity='',value=0):
	'''
		adds quantity with value to header of yield table with mass M and metallicity Z
		Note: creates for all tables the same quantity with value 0.

		if quantity is already available replace current value with new value

		quantites given by col_attrs
	'''

        inp='(M='+str(float(M))+',Z='+str(float(Z))+')'
        idx=self.table_idx[inp]
	if quantity in self.col_attrs:
		#quantity exists and will be overwritten
		idxq=self.col_attrs.index(quantity)
		self.col_attrs_data[idx][idxq]=value
	else:
		#create new entry
		self.col_attrs.append(quantity)
		#add for each table zero value
		for k in range(len(self.col_attrs_data)):
			if k==idx:
				newval=value
			else:
				newval=0.
			self.col_attrs_data[k].append(newval)	


    def set(self,M=0,Z=-1,specie='',value=0):

	'''
	    Replace the values in column 3 which
	    are usually the yields with value.
	    Use in combination with the write routine
	    to write out modification into new file.

	    M: initial mass to be modified
	    Z: initial Z to 
	    specie: quantity (e.g. yield) of specie will be modified

        '''

        inp='(M='+str(float(M))+',Z='+str(float(Z))+')'
        idx=self.table_idx[inp]
        data=self.yield_data[idx]
        idx_col=self.data_cols.index('Yields')
        set1=self.yield_data[idx][idx_col]
        specie_all= data[0]
        for k in range(len(set1)):
                    if specie == specie_all[k]:
                        #return set1[k]
			self.yield_data[idx][idx_col][k] = value


    def write_table(self,filename='isotope_yield_table_mod.txt',iolevel=0):

	'''
		Allows to write out table in NuGrid yield table format.
		Note that method has to be generalized for all tables
		and lines about NuGrid removed.

		fname: Table name

		needs ascii_table.py from NuGrid python tools

	'''

	import getpass
	user=getpass.getuser()
	import time
	date=time.strftime("%d %b %Y", time.localtime())
	
	
	tables=self.table_mz


	#write header attrs
	f=open(filename,'w')
	self.header_attrs
	
	out=''
	l='H Name: '+self.header_attrs['Name']+'\n'
	out = out +l
	l='H Data prepared by: '+user+'\n'	
	out=out +l
	l='H Data prepared date: '+date+'\n'
	out=out +l	
	l='H Isotopes: '+ self.header_attrs['Isotopes'] +'\n'
	out = out +l
	l='H Number of metallicities: '+self.header_attrs['Number of metallicities']+'\n'
	out = out +l
	l='H Units: ' + self.header_attrs['Units'] + '\n'
	out = out + l
	f.write(out)
	f.close()

	for k in range(len(tables)):
		if iolevel>0:
		        print 'Write table ',tables[k]
		mass=float(self.table_mz[k].split(',')[0].split('=')[1])
		metallicity=float(self.table_mz[k].split(',')[1].split('=')[1][:-1])
		data=self.yield_data[k]	
		#search data_cols
		idx_y=self.data_cols.index('Yields')
		yields=data[idx_y]
		idx_x0=self.data_cols.index('X0')
		mass_frac_ini=data[idx_x0]
		idx_specie=self.data_cols.index(self.data_cols[0])
		species=data[idx_specie]
		#over col attrs, first is MZ pair which will be skipped, see special_header
		attr_lines=[]
		for h in range(1,len(self.col_attrs)):
			attr=self.col_attrs[h]
			idx=self.col_attrs.index(attr)
			# over MZ pairs
			attr_data=self.col_attrs_data[k][idx]
			line=attr+': '+'{:.3E}'.format(attr_data)
			attr_lines.append(line)

		special_header='Table: (M='+str(mass)+',Z='+str(metallicity)+')'
	
		dcols=[self.data_cols[0],'Yields','X0']
		data=[species,list(yields),mass_frac_ini]

		headers=[special_header]+attr_lines
		write_single_table(filename=filename,headers=headers,data=data,dcols=dcols)
	print 'Yields table ',filename,' created.'

    def get(self,M=0.,Z=-1.,quantity='',specie=''):

        '''
                Allows to extract table data in 2 Modes:

                1) For extracting of table data for
                   star of mass M and metallicity Z.
                   Returns either table attributes,
                   given by yield.col_attrs
                   or table columns,
                   given by yield.data_cols.

                2) For extraction of a table attribute
                   from all available tables. Can be
                   directly used in the following way:

                   get(tableattribute)

	        Parameters
	        ----------

                M: float
			Stellar mass in Msun
			default: 0 
                Z: float 
			Stellar metallicity (e.g. 0.02)
		quantity: string
                	table attribute or data column/data_cols
                specie: string
			optional, return certain specie (e.g. 'H-1')


		table1.get(Z=0.02,quantity='masses')

		Examples
        	----------

		
        	>>> table1.get(M=2.0,Z=0.02,quantity='Yields')

        	>>> table1.get(Z=0.02,quantity='masses')
 



        '''
	#scale down to Z=0.00001
	#print 'get yields   ',Z
	if float(Z) == 0.00001:
		#scale abundance
		if quantity=='Yields':
			return self.get_scaled_Z(M=M,Z=Z,quantity=quantity,specie=specie)
		#Take all other parameter from Z=0.0001 case
		else:
			Z=0.0001

        all_tattrs=False
        if Z ==-1:
            if M ==0 and len(quantity)>0:
                quantity1=quantity
                all_tattrs=True
            elif (M in self.col_attrs) and quantity == '':
                quantity1=M
                all_tattrs=True
            else:
                print 'Error: Wrong input'
                return 0
            quantity=quantity1


        if (all_tattrs==False) and (not M ==0):
            inp='(M='+str(float(M))+',Z='+str(float(Z))+')'
            idx=self.table_idx[inp]
        #print 'len tableidx:',len(self.table_idx)
        #print 'len age',len(self.age)
	'''
        if quantity=='Lifetime':
            if all_tattrs==True:
                data=self.age
            else:
                data=self.age[idx]
            return data
        if quantity =='Total kinetic energy':
            if all_tattrs==True:
                data=self.kin_e
            else:
                data=self.kin_e[idx]
            return data
        if quantity == 'Lyman-Werner band':
            if all_tattrs==True:
                data=[list(i) for i in zip(*self.lum_bands)][0]
            else:
                data=self.lum_bands[idx][0]
            return data
        if quantity== 'Hydrogen-ionizing band':
            if all_tattrs==True:
                data=[list(i) for i in zip(*self.lum_bands)][1]
            else:
                data=self.lum_bands[idx][1]
            return data
        if quantity == 'High-energy band':
            if all_tattrs==True:
                data=[list(i) for i in zip(*self.lum_bands)][2]
            else:
                data=self.lum_bands[idx][2]
            return data
        if quantity == 'Mfinal':
            if all_tattrs==True:
                data=self.m_final
            else:
                data=self.m_final[idx][0]
            return data
        if quantity== 'Table (M,Z)':
            if all_tattrs==True:
                data=self.table_mz
            else:
                data=self.table_mz[idx]
            return data
	'''
	if quantity in self.col_attrs:
		if all_tattrs==False:
			data=self.col_attrs_data[idx][self.col_attrs.index(quantity)]
			return data
		else:
			data=[]
			for k in range(len(self.table_idx)):
				data.append(self.col_attrs_data[k][self.col_attrs.index(quantity)])	
			return data

        if quantity=='masses':
            data_tables=self.table_mz
            masses=[]
            for table in data_tables:
                if str(float(Z)) in table:
                    masses.append(float(table.split(',')[0].split('=')[1]))


            return masses
        else:
            data=self.yield_data[idx]
            if specie=='':
                idx_col=self.data_cols.index(quantity)
                set1=data[idx_col]
                return set1
            else:
                idx_col=self.data_cols.index('Yields')
                set1=data[idx_col]
                specie_all= data[0]
                for k in range(len(set1)):
                    if specie == specie_all[k]: #bug was here
                        return set1[k]

    def get_scaled_Z(self,table, table_yields,iniabu,iniabu_scale,M=0,Z=0,quantity='Yields',specie=''):

	'''
		Scaled down yields of isotopes 'He','C', 'O', 'Mg', 'Ca', 'Ti', 'Fe', 'Co','Zn','H','N'
	 	down to Z=1e-5 and Z=1e-6 (for Brian). The rest is set to zero.
	'''

	#print '####################################'
	#print 'Enter routine  get_scaled_Z'

	elem_prim=['He','C', 'O', 'Mg', 'Ca', 'Ti', 'Fe', 'Co','Zn','H']
	elem_sec=['N']

	##Scale down

	import re
	
	iniiso=[]
	iniabu_massfrac=[]
	for k in range(len(iniabu.habu)):
		iso=iniabu.habu.keys()[k]
		iniiso.append(re.split('(\d+)',iso)[0].strip().capitalize()+'-'+re.split('(\d+)',iso)[1])
		iniabu_massfrac.append(iniabu.habu.values()[k])
	iniiso_scale=[]
	iniabu_scale_massfrac=[]
	for k in range(len(iniabu_scale.habu)):
		iso=iniabu_scale.habu.keys()[k]
		iniiso_scale.append(re.split('(\d+)',iso)[0].strip().capitalize()+'-'+re.split('(\d+)',iso)[1])
		iniabu_scale_massfrac.append(iniabu_scale.habu.values()[k])


	grid_yields=[]
	grid_masses=[]
	isotope_names=[]
	origin_yields=[]
	for k in range(len(table.table_mz)):
		if 'Z=0.0001' in table.table_mz[k]:
			#print table.table_mz[k]
			mini=float(table.table_mz[k].split('=')[1].split(',')[0])
			grid_masses.append(mini)
			#this is production factor (see file name)
			prodfac=table.get(M=mini,Z=0.0001,quantity='Yields')
			isotopes=table.get(M=mini,Z=0.0001,quantity='Isotopes')
			#this is yields
			yields=table_yields.get(M=mini,Z=0.0001,quantity='Yields')
			mtot_eject=sum(yields)
			origin_yields.append([])
			#print 'tot eject',mtot_eject
			mout=[]
			sumnonh=0
			isotope_names.append([])
			for h in range(len(isotopes)):
				if not (isotopes[h].split('-')[0] in (elem_prim+elem_sec) ):
					#Isotopes/elements not considered/scaled are set to 0
					#mout.append(0)
					#isotope_names[-1].append(isotopes[h])
					continue
				isotope_names[-1].append(isotopes[h])
				idx=iniiso.index(isotopes[h])
				inix=iniabu_massfrac[idx]
				idx=iniiso_scale.index(isotopes[h])
				inix_scale=iniabu_scale_massfrac[idx]
				prodf=prodfac[isotopes.index(isotopes[h])]
				origin_yields[-1].append(yields[isotopes.index(isotopes[h])])
				if isotopes[h].split('-')[0] in elem_prim:
					#primary 
					mout1=(prodf-1.)*(inix_scale*mtot_eject) + (inix*mtot_eject)
					#check if amount destroyed was more than it was initial there
					if mout1<0:
						#print 'Problem with ',isotopes[h]
						#print 'Was more destroyed than evailable'
						#Then only what was there can be destroyed
						mout1=0
					#if isotopes[h] == 'C-13':
					#	print 'inix',inix
					#	print 'inixscale',inix_scale
					#	print 'prodf',prodf
					#	print (prodf)*(inix_scale*mtot_eject)
					#	print (inix*mtot_eject)	
				else:
					#secondary
					mout1=(prodf-1.)*(inix*mtot_eject) + (inix*mtot_eject)
				if (not isotopes[h]) == 'H-1' and (mout1>0):
					sumnonh+= (mout1 - (inix*mtot_eject))
				mout.append(mout1)
			#for mass conservation, assume total mass lost is same as in case of Z=0.0001
			idx_h=isotope_names[-1].index('H-1')		
			mout[idx_h]-=sumnonh
			for k in range(len(mout)):
				mout[k] = float('{:.3E}'.format(mout[k]))		
			grid_yields.append(mout)	



	####data

	idx=grid_masses.index(M)

        all_tattrs=False

	

        if specie=='':
	    return grid_yields[idx]
        else:
	    set1=data[idx]
	    names=isotope_names[idx]
	    for k in range(len(names)):
	        if specie in names[k]:
		    return set1[k]



class read_yield_sn1a_tables():

    def __init__(self,sn1a_table,isotopes=[]):


        '''
                Read SN1a tables.
                Fills up missing isotope yields
                with zeros.
                If different Zs are available
                do ...

        '''

        import re
        if notebookmode==True:
            os.system('sudo python cp.py '+sn1a_table)
            f1=open('tmp/'+sn1a_table)
            lines=f1.readlines()
            f1.close()
            os.system('sudo python delete.py '+sn1a_table)
        else:
            f1=open(sn1a_table)
            lines=f1.readlines()
            f1.close()
        iso=[]
        self.header=[]
        self.col_attrs=[]
        yields=[]
        metallicities=[]
	isotopes_avail=[]
        for line in lines:
            #for header
            if 'H' in line[0]:
                self.header.append(line)
                continue
            if ('Isotopes' in line) or ('Elements' in line):
                l=line.replace('\n','').split('&')[1:]
                self.col_attrs=l
                metallicities=l[1:]
                #print metallicities
                # metallicity dependent yields
                #if len(l)>2:
                #else:
                for k in l[1:]:
                    yields.append([])
                continue
            linesp=line.strip().split('&')[1:]
            iso.append(linesp[0].strip())
            #print iso
            for k in range(1,len(linesp)):
                yields[k-1].append(float(linesp[k]))

	#if isotope list emtpy take all isotopes
        if len(isotopes)==0:
		isotopes=iso
        yields1=[]
        #fill up the missing isotope yields with zero
        for z in range(len(yields)):
            yields1.append([])
            for iso1 in isotopes:
                #iso1=iso1.split('-')[1]+iso1.split('-')[0]
                #ison= iso1+((10-len(iso1))*' ')
                if iso1 in iso:
                    yields1[-1].append(yields[z][iso.index(iso1)])
                else:
                    yields1[-1].append(0.)
        self.yields=yields1
        self.metallicities=[]
        for m in metallicities:
            self.metallicities.append(float(m.split('=')[1]))
        #self.metallicities=metallicities
        #print yields1
	self.isotopes=iso

    def get(self,Z=0,quantity='Yields',specie=''):



        '''
                Allows to extract SN1a table data.
                If metallicity dependent yield tables
                were used, data is taken for the closest metallicity available
                to reach given Z

                quantity: if 'Yields' return yields
			  if 'Isotopes' return all isotopes available

        '''

	if quantity=='Yields':
        	idx = (np.abs(np.array(self.metallicities)-Z)).argmin()
        	yields=self.yields[idx]
        	return np.array(yields)
	elif quantity=='Isotopes':
		return self.isotopes


class read_yield_rawd_tables():

    def __init__(self,rawd_table,isotopes):


        '''
                Read RAWD tables.
                Fills up missing isotope yields
                with zeros.
                If different Zs are available
                do ...

        '''

        import re
        if notebookmode==True:
            os.system('sudo python cp.py '+rawd_table)
            f1=open('tmp/'+rawd_table)
            lines=f1.readlines()
            f1.close()
            os.system('sudo python delete.py '+rawd_table)
        else:
            f1=open(rawd_table)
            lines=f1.readlines()
            f1.close()
        iso=[]
        self.header=[]
        self.col_attrs=[]
        yields=[]
        metallicities=[]
        for line in lines:
            #for header
            if 'H' in line[0]:
                self.header.append(line)
                continue
            if ('Isotopes' in line) or ('Elements' in line):
                l=line.replace('\n','').split('&')[1:]
                self.col_attrs=l
                metallicities=l[1:]
                #print metallicities
                # metallicity dependent yields
                #if len(l)>2:
                #else:
                for k in l[1:]:
                    yields.append([])
                continue
            linesp=line.strip().split('&')[1:]
            iso.append(linesp[0].strip())
            #print iso
            for k in range(1,len(linesp)):
                yields[k-1].append(float(linesp[k]))

        yields1=[]
        #fill up the missing isotope yields with zero
        for z in range(len(yields)):
            yields1.append([])
            for iso1 in isotopes:
                #iso1=iso1.split('-')[1]+iso1.split('-')[0]
                #ison= iso1+((10-len(iso1))*' ')
                if iso1 in iso:
                    yields1[-1].append(yields[z][iso.index(iso1)])
                else:
                    yields1[-1].append(0.)
        self.yields=yields1
        self.metallicities=[]
        for m in metallicities:
            self.metallicities.append(float(m.split('=')[1]))
        #self.metallicities=metallicities
        #print yields1

    def get(self,Z=0,quantity='Yields',specie=''):



        '''
                Allows to extract rawd table data.
                If metallicity dependent yield tables
                were used, data is taken for the closest metallicity available
                to reach given Z

                quantity: yields only possible atm



        '''
        idx = (np.abs(np.array(self.metallicities)-Z)).argmin()
        yields=self.yields[idx]

        return np.array(yields)




'''
Adapted from NuGrid Utility class


'''

#import numpy as np
#import scipy as sc
#import ascii_table as att
#from scipy import optimize
#import matplotlib.pyplot as pl
#import os

class iniabu():
    '''
    This class in the utils package reads an abundance
    distribution file of the type iniab.dat. It then provides you
    with methods to change some abundances, modify, normalise and
    eventually write out the final distribution in a format that
    can be used as an initial abundance file for ppn. This class
    also contains a method to write initial abundance files for a
    MESA run, for a given MESA netowrk.
    '''
    # clean variables that we will use in this class

    filename = ''

    def __init__(self,filename):
        '''
        Init method will read file of type iniab.dat, as they are for
        example found in the frames/mppnp/USEPP directory.

        An instance of this class will have the following data arrays
        z      charge number
        a      mass number
        abu    abundance
        names  name of species
        habu   a hash array of abundances, referenced by species name
        hindex hash index returning index of species from name

        E.g. if x is an instance then x.names[4] gives you the
        name of species 4, and x.habu['c 12'] gives you the
        abundance of C12, and x.hindex['c 12'] returns
        4. Note, that you have to use the species names as
        they are provided in the iniabu.dat file.

        Example - generate modified input file ppn calculations:

        import utils
        p=utils.iniabu('iniab1.0E-02.ppn_asplund05')
        sp={}
        sp['h   1']=0.2
        sp['c  12']=0.5
        sp['o  16']=0.2
        p.set_and_normalize(sp)
        p.write('p_ini.dat','header for this example')

        p.write_mesa allows you to write this NuGrid initial abundance
        file into a MESA readable initial abundance file.
        '''
        f0=open(filename)
        sol=f0.readlines()
        f0.close

        # Now read in the whole file and create a hashed array:
        names=[]
        z=[]
        yps=np.zeros(len(sol))
        mass_number=np.zeros(len(sol))
        for i in range(len(sol)):
            z.append(int(sol[i][1:3]))
            names.extend([sol[i].split("         ")[0][4:]])
            yps[i]=float(sol[i].split("         ")[1])
            try:
                mass_number[i]=int(names[i][2:5])
            except ValueError:
                #print "WARNING:"
                #print "This initial abundance file uses an element name that does"
                #print "not contain the mass number in the 3rd to 5th position."
                #print "It is assumed that this is the proton and we will change"
                #print "the name to 'h   1' to be consistent with the notation used"
                #print "in iniab.dat files"
                names[i]='h   1'
            mass_number[i]=int(names[i][2:5])
        # now zip them together:
        hash_abu={}
        hash_index={}
        for a,b in zip(names,yps):
            hash_abu[a] = b

        for i in range(len(names)):
            hash_index[names[i]] = i

        self.z=z
        self.abu=yps
        self.a=mass_number
        self.names=names
        self.habu=hash_abu
        self.hindex=hash_index


    def iso_abundance(self,isos):
        '''
        This routine returns the abundance of a specific isotope. Isotope given as, e.g., 'Si-28' or as list ['Si-28','Si-29','Si-30']
        '''
        if type(isos) == list:
            dumb = []
            for it in range(len(isos)):
                dumb.append(isos[it].split('-'))
            ssratio = []
            isos = dumb
            for it in range(len(isos)):
                ssratio.append(self.habu[isos[it][0].ljust(2).lower() + str(int(isos[it][1])).rjust(3)])
        else:
            isos = isos.split('-')
            ssratio = self.habu[isos[0].ljust(2).lower() + str(int(isos[1])).rjust(3)]
        return ssratio


def read_iniabu(filename,isotopes):
    import read_yields as ry
    if notebookmode==True:
        os.system('sudo python cp.py '+'iniabu/'+filename)
        iniabu_class=ry.iniabu('tmp/'+filename)
        iniabu= np.array(iniabu_class.iso_abundance(isotopes))
        os.system('sudo python delete.py '+filename)
    else:
        iniabu_class=ry.iniabu(filename)
        iniabu= np.array(iniabu_class.iso_abundance(isotopes))
    return iniabu

def read_strip_param(filename):
	'''
		To read Elses simulatin files
	'''

	import read_yields as ry

	f1=open(filename)
	lines=f1.readlines()
	f1.close()
	info=['timebins','SFR','Mcool','Meject','Minfall','Mreinc','Mcoldgas','Mhotgas','Mejectedgas','Mstripej','Mstriphot','Mstripcold','Mstripstar']
	data=[]
	for k in range(len(lines)):
		#to skip header
		if k <14:
			continue
		#to read column header
		if k==14:
			cheader=lines[k].split()
			idx=[]
			for h in info:
				idx.append(cheader.index(h))
				data.append([])		
			continue
		#units line
		if k==15:
			continue
		line=lines[k].split()
		for i in range(len(idx)):
			data[i].append(float(line[idx[i]]))
	data_dict={}
	for k in range(len(data)):
		data_dict[info[k]]=data[k]		

	return data_dict


def write_single_table(filename,headers,data,dcols=['Isotopes','Yields','Z','A'],header_char='H',sldir='.',sep='&'):
	'''
	Method for writeing data in GCE format in Ascii files.
	Reads either elements or isotopes
	dcols[0] needs to contain either isotopes or elements

	Note the attribute name at position i in dcols will be associated
	with the column data at index i in data.
	Also the number of data columns(in data) must equal the number
	of data attributes (in dcols)
	Also all the lengths of that columns must all be the same.
	Input:
	filename: The file where this data will be written.
	Headers: A list of Header strings or if the file being written 
		 is of type trajectory, this is a List of strings
		 that contain header attributes and their associated 
		 values which are seperated by a '='. 
	dcols: A list of data attributes
	data:  A list of lists (or of numpy arrays).
	header_char  the character that indicates a header lines
	sldir: Where this fill will be written.
	sep: What seperatesa the data column attributes
	trajectory: Boolean of if we are writeing a trajectory type file
	'''

	import re
	import utils as u

	#check if input are elements or isotopes
	if not '-' in data[0][0]:
		iso_inp=False
		dcols=dcols+['Z']
	else:
		iso_inp=True
		dcols=dcols+['Z','A']
	#Attach Z and A
	if iso_inp==True:
		data.append([])
		data.append([])
		u.convert_specie_naming_from_h5_to_ppn(data[0])
		Z=u.znum_int
		A=u.amass_int
		for i in range(len(data[0])):
			zz=str(int(Z[i]))
			aa=str(int(A[i]))
			data[1][i]='{:.3E}'.format(data[1][i])+' '
			data[-2].append(zz)
			data[-1].append(aa)


	else:
		#in order to get Z , create fake isotope from element
		fake_iso=[]
		for k in range(len(data[0])):
			fake_iso.append(data[0][k]+'-99')
		#print fake_iso
		data.append([])
		u.convert_specie_naming_from_h5_to_ppn(fake_iso)
		Z=u.znum_int
		for i in range(len(data[0])):
			zz=str(int(Z[i]))
			data[1][i]='{:.3E}'.format(data[1][i])+' '
			data[-1].append(zz)


	if sldir.endswith(os.sep):
		filename = str(sldir)+str(filename)
	else:
		filename = str(sldir)+os.sep+str(filename)
	tmp=[] #temp variable
	lines=[]#list of the data lines
	lengthList=[]# list of the longest element (data or column name)
		     # in each column
	#CR to surpress too much output	     
	#if os.path.exists(filename):
		#print 'This method will add table to existing file '+ filename
	
	if len(data)!=len(dcols):
		print 'The number of data columns does not equal the number of Data attributes'
		print 'returning none'
		return None
	for i in xrange(len(headers)):
		tmp.append(header_char+' '+headers[i]+'\n')
	headers=tmp
	tmp=''
	
	for i in xrange(len(data)): #Line length stuff
		length=len(dcols[i])+1
		for j in xrange(len(data[i])):
			tmp2=data[i][j]
			if isinstance(data[i][j],float):
				tmp2='{:.3E}'.format(data[i][j])+' '
				data[i][j] = tmp2
			if len(str(tmp2))>length:
				length=len(str(tmp2))
		lengthList.append(length)
	
	tmp=''
	tmp1=''
	for i in xrange(len(dcols)):
		tmp1=dcols[i]
		if len(dcols[i]) < lengthList[i]:
			j=lengthList[i]-len(dcols[i])
			for k in xrange(j):
				tmp1+=' '
		tmp+=sep+tmp1
	tmp+='\n'
	dcols=tmp
	tmp=''
	for i in xrange(len(data[0])):
		for j in xrange(len(data)):
			if type(data[j][i]) == str:
				#match = re.match(r"([a-z]+)([0-9]+)",data[j][i], re.I)
				#items = match.groups()
				tmp1=data[j][i]#items[0].capitalize()+'-'+items[1]
				if len(str(data[j][i])) < lengthList[j]:
					l=lengthList[j]-len(tmp1)
					for k in xrange(l):
						tmp1+=' '
				extra=''	
			#else:
			#        tmp1=data[j][i]
			#        if len(data[j][i]) < lengthList[j]:
			#                l=lengthList[j]-len(data[j][i]))
			#                for k in xrange(l):
			#                        tmp1+=' '


			tmp+=sep+tmp1
		lines.append(tmp+'\n')
		tmp=''
		
	f=open(filename,'a')
	for i in xrange(len(headers)):
		f.write(headers[i])
	f.write(dcols)
	for i in xrange(len(lines)):
		f.write(lines[i])
	
	f.close()
	return None


def write_tables(data,data_cols,Zs,Ms,isos,col_attrs,col_attrs_data,units='Msun, year',table_name='Yield table',filename='isotope_yield_table_mod.txt',iolevel=0):

	'''
		Allows to write out table in NuGrid yield table format.
		Note that method has to be generalized for all tables
		and lines about NuGrid removed.

		fname: Table name

		needs ascii_table.py from NuGrid python tools

	'''

	import getpass
	user=getpass.getuser()
	import time
	date=time.strftime("%d %b %Y", time.localtime())
	
	
	#write header attrs
	f=open(filename,'w')
	
	out=''
	l='H Name: '+table_name+'\n'
	out = out +l
	l='H Data prepared by: '+user+'\n'	
	out=out +l
	l='H Data prepared date: '+date+'\n'
	out=out +l	
	isos_str=isos[0]
	for k in range(1,len(isos)):
		isos_str = isos_str +', '+isos[k]
	l='H Isotopes: '+ isos_str +'\n'
	out = out +l
	l='H Number of metallicities: '+str(len(Zs))+'\n'
	out = out +l
	l='H Units: ' + units+ '\n'
	out = out + l
	f.write(out)
	f.close()

	#MZ pairs
	#assume same isotopes for each star
	A_isos=[]
	Z_isos=[]
	for k in range(len(isos)):
		A_isos.append(int(isos[k].split('-')[1]))
		Z_iso = get_z_from_el(isos[k].split('-')[0])
		Z_isos.append(int(Z_iso))

	for i in range(len(Zs)):
		table_headers=[]
		for M in Ms[i]:
			inp='Table: (M='+str(float(M))+',Z='+str(float(Zs[i]))+')'
			table_headers.append(inp)	

		for k in range(len(table_headers)):
			if iolevel>0:
				print 'Write table ',table_headers[k]
			attr_lines=[]
			for h in range(len(col_attrs)):
				line=col_attrs[h]+': '+'{:.3E}'.format(col_attrs_data[i][k][h])
				attr_lines.append(line)

			special_header=table_headers[k]
		
			data1=[]
			dcols=[]
			dcols.append('Isotopes')
			data1.append(isos)
			for h in range(len(data_cols)):
				dcols.append(data_cols[h])
				data1.append(data[i][k][h][:])
			#dcols=[self.data_cols[0],'Yields','X0']
			#data=[species,list(yields),mass_frac_ini]
			#data1.append(Z_isos)
			#dcols.append('Z')
			#data1.append(A_isos)
			#dcols.append('A')

			headers=[special_header]+attr_lines
			write_single_table(filename=filename,headers=headers,data=data1,dcols=dcols)
	print 'Yields table ',filename,' created.'

def get_z_from_el(element):
    '''
    Very simple function that gives the atomic number AS A STRING when given the element symbol.
    Uses predefined a dictionnary.
    Parameter :
    element : string
    For the other way, see get_el_from_z
    '''
    dict_name={'Ru': '44', 'Re': '75', 'Ra': '88', 'Rb': '37', 'Rn': '86', 'Rh': '45', 'Be': '4', 'Ba': '56', 'Bi': '83', 'Br': '35', 'H': '1', 'P': '15', 'Os': '76', 'Hg': '80', 'Ge': '32', 'Gd': '64', 'Ga': '31', 'Pr': '59', 'Pt': '78', 'C': '6', 'Pb': '82', 'Pa': '91', 'Pd': '46', 'Cd': '48', 'Po': '84', 'Pm': '61', 'Ho': '67', 'Hf': '72', 'K': '19', 'He': '2', 'Mg': '12', 'Mo': '42', 'Mn': '25', 'O': '8', 'S': '16', 'W': '74', 'Zn': '30', 'Eu': '63', 'Zr': '40', 'Er': '68', 'Ni': '28', 'Na': '11', 'Nb': '41', 'Nd': '60', 'Ne': '10', 'Fr': '87', 'Fe': '26', 'B': '5', 'F': '9', 'Sr': '38', 'N': '7', 'Kr': '36', 'Si': '14', 'Sn': '50', 'Sm': '62', 'V': '23', 'Sc': '21', 'Sb': '51', 'Se': '34', 'Co': '27', 'Cl': '17', 'Ca': '20', 'Ce': '58', 'Xe': '54', 'Lu': '71', 'Cs': '55', 'Cr': '24', 'Cu': '29', 'La': '57', 'Li': '3', 'Tl': '81', 'Tm': '69', 'Th': '90', 'Ti': '22', 'Te': '52', 'Tb': '65', 'Tc': '43', 'Ta': '73', 'Yb': '70', 'Dy': '66', 'I': '53', 'U': '92', 'Y': '39', 'Ac': '89', 'Ag': '47', 'Ir': '77', 'Al': '13', 'As': '33', 'Ar': '18', 'Au': '79', 'At': '85', 'In': '49'}
    return int(dict_name[element])


