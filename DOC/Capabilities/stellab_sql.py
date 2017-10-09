
# coding: utf-8

# # The SQL database for Stellab

# Towards a stellab widget interface, similar to SYGMA and OMEGA widgets.

# SQL code to create stellab database. The table structure is shown below.

# Tables:
# * refs: Table with literature references:
#     * refid: unique reference id (PK)
#     * fauthor: first author
#     * year
#     * nasaads: link to nasa ads page
# 
# * galaxies
#     * gali: unique ref id
#     * name: galaxy name
# 
# * solarnorms
#     * normid: unique reference id (PK)
#     * H, He... abundance entries
#     * refids
# * abu_table_reg
#     * abu_reg_id uniqure reference
#     * refid to look up literature
#     * table name to look up abundance table
#     * normid to look up normalization used
# 
# * multiple abundance tables, all with
#     * abuid: unique reference id
#     * element ratio1, element ratio2...
#     * error1, errror2 ...
# 
# Rules:
# * Internal connection only via (private keys) PKs.
# * On user level no interaction with PKs.
# * As such table entries (except PKs) can be easily modified without allowing any disconnections between tables.
# 
# Tips:
# * Use not null
# * fixed lenght type arrays are faster

# In[1]:

#We choose sqlite3 because it does not require to install mysql or run a server.
import sqlite3
import pandas as pd
import stellab as st


# SQLite is a C library that provides a lightweight disk-based database that doesn’t require a separate server process and allows accessing the database using a nonstandard variant of the SQL query language. 

class stellab_sql():


    	def __init__(self,db_name='stellab.db',load_db=False):

	    if not load_db:
	       self.create_database(database_name=db_name)
		    
	def create_database(self,database_name):

	    import os 
	    try:
	    	os.remove(database_name)
	    except:
		pass
	    #create table
	    db=sqlite3.connect(database_name)
	    self.db=db
	    curser=self.db.cursor()
	    
	    #create table which holds all paper references
	    curser.execute('''CREATE TABLE IF NOT EXISTS refs (refid integer primary key autoincrement, fauthor text, year year, nasads text)''')
	    #create table which holds all galaxy names
	    curser.execute('''CREATE TABLE IF NOT EXISTS galaxies (galid integer primary key autoincrement, gal_name text)''')
	    #create table which holds data for solar normalizations.
	    curser.execute('''CREATE TABLE IF NOT EXISTS solarnorms (normid integer primary key autoincrement, H float, He float, refid integer)''')

	    #table which holds names and info about all existing abundance tables. It has a foreign key referencing galaxies table
	    curser.execute('''CREATE TABLE IF NOT EXISTS abu_table_reg (abu_reg_id integer primary key autoincrement, refid integer, normid integer,abutable text,galid integer, FOREIGN KEY(galid) REFERENCES galaxies(galid))''')

	    self.db.commit()
	    print 'database created.'

	# In[5]:

	def get_column_names(self,table):
	    tmp=self.db.execute("PRAGMA table_info(%s)" % table).fetchall()
	    return [entry[1] for entry in tmp]


	# ***
	# ### Routines for adding data.

	# In[61]:

	def check_solar_normalization(self,name,year):
	    '''
	    Checks for available entry in solar normalization table. For internal use only.
	    '''        
	    #check if paper for normalization data exists already.
	    results = self.db.execute('''SELECT refid FROM refs WHERE (fauthor = '%s' and year = '%s' )''' % (name,year)).fetchall()  #% ('Anders','1989'))
	    refid = results
	    if len(refid)==1:
		refid = refid[0][0]
		results = self.db.execute('''SELECT normid FROM solarnorms WHERE (refid = %s )''' %(refid))
		normid = results.fetchall()
		if len(normid)==1:
		    print 'found corresponding normalization data.'
		    return normid[0][0]
		else:
		    #add normalization data
		    print 'normalization data is missing. Add the data first by using add_normalization().'
		    return -1
	    else:
		print 'Solar normalization data is not in database. Add the data first by using add_normalization().'
		return -1


	# In[62]:

	def check_galaxy(self,galaxy_name):
	    '''
	    Checks for available entry in galaxy table. For internal use only.

	    '''
	    results = self.db.execute('''SELECT galid FROM galaxies WHERE (gal_name = '%s') ''' % galaxy_name).fetchall()
	    #if galaxy does not exist
	    if len(results)==0:
		galid = -1
	    else:
		galid = results[0][0]
	    return galid    


	# In[156]:

	def check_refs(self,ref_paper):
	    '''
	    Checks for available entry in refs table. For internal use only.
	    '''
	    name=ref_paper[0]
	    year=ref_paper[1]
	    results = self.db.execute('''SELECT refid FROM refs WHERE (fauthor = '%s' and year = '%s' )''' % (name,year)).fetchall()  #% ('Anders','1989'))
	    refid = results
	    if len(refid)==0: 
		refid = -1
	    else:
		refid=refid[0][0]
	    return refid    


	# In[162]:

	def check_abu_table_reg(self,ref_paper):
	    '''
	    Checks for available entry in reg_table_reg table. For internal use only.
	    '''    
	    name=ref_paper[0]
	    year=ref_paper[1]   
	    results= self.db.execute('''SELECT abu_reg_id FROM abu_table_reg a INNER JOIN refs r ON a.refid = r.refid 
		WHERE (fauthor = '%s' and year = '%s')''' % (name,year) ).fetchall()
	    abu_reg_id = results
	    if len(abu_reg_id)==0: 
		abu_reg_id = -1
	    else:
		abu_reg_id=abu_reg_id[0][0]
	    return abu_reg_id


	# In[8]:

	def add_solar_normalization(self,norm_paper,norm_label,norm_data):
	    '''
	    Add data to normalization table solarnorms and the corresponding paper information to table refs, if necessary.
	    E.g.
	    norm_paper=['Venn',2012,'http://adsabs.harvard.edu/abs/2012ApJ...751..102V']
	    norm_label=['H','He']
	    norm_data = [-2.81,0.34,]

	    '''
	    #check if paper for normalization data exists in table refs.
	    name=norm_paper[0]
	    year=norm_paper[1]
	    nasads=norm_paper[2]
	    results = self.db.execute('''SELECT refid FROM refs WHERE (fauthor = '%s' and year = '%s' )''' % (name,year)).fetchall()  #% ('Anders','1989'))
	    refid = results
	    #if paper is not available in  refs, add it
	    if len(refid)==0:
		print 'add paper for solar normalization to table refs.'
		self.add_paper_ref(name,year,nasads)
		results = self.db.execute('''SELECT refid FROM refs WHERE (fauthor = '%s' and year = '%s' )''' % (name,year)).fetchall()   #% ('Anders','1989'))
		#get PK refid for paper entry
		refid = results               
	    refid = refid[0][0]
	    
	    #check for normalization data in table solarnorms, if not available, add it.
	    results = self.db.execute('''SELECT normid FROM solarnorms WHERE (refid = %s )''' %(refid)).fetchall()
	    normid = results
	    if len(normid)==1:
		print 'normalization data is already available.'
	    else:         
		#check if columns of solarnorms table include all entries of norm_label. If not, add new columns.
		columns= self.get_column_names('solarnorms') 
		for k in range(len(norm_label)):
		    if not norm_label[k] in columns:
			#print norm_label[k],'missing in columns, add new column'
			self.db.execute('''ALTER TABLE solarnorms ADD %s float''' %norm_label[k])                  
		#add normalization data
		str_tmp = 'refid, '
		for k in range(len(norm_label)):
			str_tmp+= (norm_label[k] + ',')
		str_tmp = str_tmp[:-1]
		sql_prepr = tuple([str_tmp])
		str_tmp = str(refid)+', '
		for k in range(len(norm_data)):
			str_tmp+= str(norm_data[k]) + ','
		str_tmp = str_tmp[:-1]
		sql_prepr = sql_prepr + tuple([str_tmp])
		self.db.execute('''INSERT INTO solarnorms (%s) VALUES (%s)''' % sql_prepr)
		self.db.commit()


	def add_paper_ref(self,name,year,nasads):
	    '''
	    Add a new paper entry to the table refs, if it does not exist yet. Return PK ref_id of table refs.
	    e.g.
	    name='Anders'
	    year=1989
	    nasads='http://ukads.nottingham.ac.uk/abs/1993A%26A...271..587G'
	    '''   
	    results = self.db.execute('''SELECT refid FROM refs WHERE (fauthor = '%s' and year = '%s' )''' % 
				 (name,year)).fetchall()
	    if len(results)==0:
		    print 'add paper related to abundance to table refs.'
		    self.db.execute('''INSERT INTO refs (fauthor, year,nasads) VALUES ('%s','%s','%s') ''' % (name,year,nasads))
		    self.db.commit()
		    results=self.db.execute('''SELECT refid FROM refs WHERE (fauthor='%s' and year='%s') ''' % (name,year)).fetchall()
		    ref_id=results[0][0]
	    else:
		print 'abu ref paper exists already. do nothing'
		ref_id = results[0][0]
	    return ref_id


	# In[10]:

	def add_galaxy(self,galaxy_name):
	    results = self.db.execute('''SELECT galid FROM galaxies WHERE (gal_name = '%s') ''' % galaxy_name).fetchall()
	    #galaxy does not exist
	    if len(results)==0:
		print 'galaxy name ',galaxy_name
		self.db.execute('''INSERT INTO galaxies (gal_name) VALUES ('%s') ''' % (galaxy_name))
		self.db.commit()
		results=self.db.execute('''SELECT galid FROM galaxies WHERE (gal_name= '%s') ''' % (galaxy_name)).fetchall()
		galid = results[0][0]
	    else:
		galid = results[0][0]
	    return galid


	# In[11]:

	def add_abundance_data(self,abu_paper,abu_norm,abu_label,abu_data):
	    '''
	    Adding new abundance data to database. Creates new table. Requires corresponding solar normalization to be available
	    in database.
	    e.g.
	    abundance_paper=['Venn',2012,'http://adsabs.harvard.edu/abs/2012ApJ...751..102V','Milky Way']
	    abundance_norm=['Anders','1989']
	    abundance_label=['[Fe/H]','err']
	    abundance_data = [-2.81,0.34]    
	    '''
	    
	    ######check if solar normalization paper and data already exists in database
	    #get PK norm_id for normalization table, if not available return -1
	    norm_id = self.check_solar_normalization(abu_norm[0],abu_norm[1])
	    if norm_id == -1: return
	    
	    ######check if abundance paper into database, if it does not exist, add it.
	    #get PK ref_id for refs table entry
	    ref_id = self.add_paper_ref(abu_paper[0],abu_paper[1],abu_paper[2])

	    ######check galaxy name, if it does not exist add it.
	    gal_name = abu_paper[3]
	    #get PK galid for galaxy table entry
	    galid=self.add_galaxy(gal_name)
	    
	    ###### add abundance data    
	    table_name = ''
	    #check if table already exists in registry table for abundance tables
	    results = self.db.execute('''SELECT ar.abu_reg_id FROM abu_table_reg ar WHERE (normid = '%s' and refid = '%s' and galid = '%s')''' % (norm_id,ref_id,galid)).fetchall()
	    if len(results)==0:
		
		#get the latest key entry
		abu_reg_ids = self.db.execute('''SELECT abu_reg_id FROM abu_table_reg''').fetchall()
		if len(abu_reg_ids)==0:
		    abu_reg_id = 0
		else:
		    abu_reg_id = abu_reg_ids[-1][0]
		#create new key entry
		abu_reg_id = abu_reg_id + 1
		
		#name table according to numbering of abu_reg_id
		tablename = 'abu_table_%s' % abu_reg_id
		print 'create table ',tablename
		#add new table
				     
		#create table
		str_tmp=''
		for k in range(len(abu_label)):
		    str_tmp+=' , '+abu_label[k] +' float '
				     
		sql_prepr = tuple([tablename])+tuple([str_tmp])
				     
		#create database
		#print 'create abu table entry'
		self.db.execute('''CREATE TABLE IF NOT EXISTS %s (abuid integer primary key autoincrement %s )''' %
				    sql_prepr)
				     
		#### add abundance data: 1 entry line        
		str_tmp1=''
		for k in range(len(abu_label)):
			str_tmp1+=abu_label[k]+','
		str_tmp1 = str_tmp1[:-1]
		str_tmp2=''
		for k in range(len(abu_data)):
			str_tmp2+=str(abu_data[k])+','
		str_tmp2 = str_tmp2[:-1]        
		
		#create tuple for SQL input
		sql_prepr = tuple([tablename]) + tuple([str_tmp1]) + tuple([str_tmp2])
		#print sql_prepr           
		self.db.execute('''INSERT INTO %s (%s) VALUES (%s)''' %sql_prepr)
		
		#insert into table registry last, after abundance table was created successfully.                     
		self.db.execute('''INSERT INTO abu_table_reg (refid, normid,abutable,galid) VALUES ('%s','%s','%s','%s') ''' %
				     (ref_id,norm_id,tablename,galid))
		#print 'Table ',sql_prepr,' , ',' created!'
	    else:
		print 'abundance table with same paper reference and solar normalization exists already! Do nothing.'


	# ***
	# ### Routines for retrieving data.

	# In[12]:

	# style of pandas sheet to allow nasads link to be clickable
	def make_clickable(self,val):
	    '''
	    internal function for pandas display
	    '''
	    if 'http' in str(val):
		return '<a href="{}">{}</a>'.format(val,val)
	    else:
		return val

	def remove_columns(self,columns,data,rcolumns):
	    '''
	    Function to remove columns with names in rcolumns. For internal use only.
	    '''

	    columns_update = [column for column in columns if not column in rcolumns]
	    data_update=[] 
	    for k in range(len(data)):
		data_tmp=list(data[k])
	    	data_update.append([data_tmp[i] for i in range(len(columns)) if not columns[i] in rcolumns])

	    return columns_update,data_update #[tuple(data_update)]

	# In[13]:

	def get_solar_normalizations(self,norm_paper=[],data_x_y=False,extra_info=False):
	    '''
	    Access either specific solar normalization data, specified through norm_paper or all solar normalization
	    data when norm_paper=[].
	    e.g. norm_paper=['Anders',1989]
	    '''
	    
	    all_data= self.db.execute('''SELECT sn.*,r.fauthor,r.year,r.nasads FROM solarnorms sn INNER JOIN refs as r ON sn.refid = r.refid''').fetchall()
	    columns = self.get_column_names('solarnorms')
	    columns=columns + ['fauthor','year','nasaads']
	 
	    if not len(norm_paper) == 0:
		idx=-1
		for k in range(len(all_data)):
		    print all_data[k]
		    if norm_paper[0] in all_data[k] and norm_paper[1] in all_data[k]:
			idx=k
			break
		if idx == -1: return 'normalization table not found.'
		data = [all_data[idx]]
	    else:
		data = all_data

	    if not extra_info:
		rcolumns=['normid','refid']
		columns,data = self.remove_columns(columns,data,rcolumns)

	    if data_x_y:
		return columns,data        
	    else:    
		return pd.DataFrame(data=data,columns=columns).style.format(self.make_clickable)


	# In[14]:

	def get_paper_refs(self,extra_info=False):
	    '''
	    Access all the paper references available in the database.
	    '''
	    data= self.db.execute('''SELECT * FROM refs''').fetchall()
	    columns = self.get_column_names('refs')

	    if not extra_info:
		rcolumns=['refid']
		columns,data = self.remove_columns(columns,data,rcolumns)

	    return pd.DataFrame(data=data,columns=columns).style.format(self.make_clickable)


	# In[15]:

	def get_galaxies(self,data_x_y=False):
	    data = self.db.execute('''SELECT gal_name FROM galaxies''').fetchall()
	    columns = ['Galaxies']    
	    if data_x_y:
		return columns,data
	    
	    else:
		return pd.DataFrame(data=data,columns=columns) 


	# In[16]:

	def get_overview_abundance_tables(self,data_x_y=False,extra_info=False):
	    '''
	    Overview over all available abundance tables. 
	    '''
	    data= self.db.execute('''SELECT a.*,r.fauthor,r.year,r.nasads FROM abu_table_reg a INNER JOIN refs r ON a.refid = r.refid''').fetchall()
	    columns = self.get_column_names('abu_table_reg')
	    columns=columns + ['fauthor','year','nasaads']

	    if not extra_info:
		rcolumns=['normid','refid','abu_reg_id','galid','abutable']
		columns,data = self.remove_columns(columns,data,rcolumns)

	    if data_x_y:
		return columns,data
	    
	    else:
		return pd.DataFrame(data=data,columns=columns).style.format(self.make_clickable)


	# In[17]:

	def get_abundance_data(self,abu_paper,data_x_y=False,extra_info=False):
	    '''
	    Access abundance data from specific paper abu_paper
	    e.g. abu_paper=['Venn',2012,'Milky Way']   
	    '''
	    fname = abu_paper[0]
	    year = abu_paper[1]
	    galaxy_name= abu_paper[2]
	    
	    #check for availability of galaxy_name
	    galid=self.check_galaxy(galaxy_name)
	    if galid==-1: return 'galaxy name not available.'
	    
	    #check for availability, instead of sql query
	    columns,all_data = self.get_overview_abundance_tables(data_x_y=True,extra_info=True)
	    
	    idx=-1
	    for k in range(len(all_data)):
		if ((fname in all_data[k] and year in all_data[k]) and (galid in all_data[k])):
		    idx=k
		    break
	    if idx==-1: return 'abundance table not found in database.'
	    data= all_data[idx]
	    #print columns
	    idx=columns.index('abutable')
	    tablename = data[idx]
	    #get the abundance data from specific table tablename
	    data= self.db.execute('''SELECT * FROM %s ''' % tablename).fetchall()
	    print data
	    columns=self.get_column_names(tablename)

	    if not extra_info:
		rcolumns=['abuid']
		columns,data = self.remove_columns(columns,data,rcolumns)



	    if data_x_y:
		return columns,data
	    else:
		return pd.DataFrame(data=data,columns=columns)   


	# ***
	# ### Routines for updating existing data

	# In[213]:

	def update_paper_refs(self,refs_paper,update):
	    '''
	    Update table entry selectd by refid of paper reference table and providing arguments
	    fauthor, year, nasads
	    refs_paper=['Anders',1989]
	    update={'nasads':'http://ukads.nottingham.ac.uk/abs/1993A%26A...271..587G','year':92999}

	    '''
	    
	    refid = self.check_refs(refs_paper)
	    if refid == -1: return 'Paper reference not available'
	    
	    all_columns=self.get_column_names('refs')
	    value_declaration = ''
	    for k,column in enumerate(update):
		if not column in all_columns:
		    return 'Column argument ',column,'is not available'
		value_declaration += str(column)+'=' + "'" + str(update[column])+"',"
	    value_declaration = value_declaration[:-1]
	    self.db.execute('''UPDATE refs SET %s WHERE (refid = %s)''' % (value_declaration,refid))
	    self.db.commit()
	    print 'Paper reference updated.'


	# In[212]:

	def update_solar_normalizations(self,norm_paper,update):
	    '''
	    Update table entry of solar normalization table selectd via norm_paper and providing arguments
	    available as columns in the table.
	    norm_paper=['Anders',1989]
	    update={'H':0.7,'He':0.3}

	    '''
	    
	    normid = self.check_solar_normalization(norm_paper[0],norm_paper[1])
	    if normid == -1: return 'Table entry not available.'
	    
	    all_columns_solarnorms=self.get_column_names('solarnorms')
	    all_columns_refs=self.get_column_names('refs')
	    
	    
	    #check if update requires updating table refs
	    for k,column in enumerate(update):
		if column in all_columns_refs:
		    print 'Column value ',column,' belongs to paper reference table. Update value via update_paper_refs()'
		    return
	    solarnorms=update                
		
	    value_declaration = ''
	    for k,column in enumerate(solarnorms):
		if not column in all_columns_solarnorms:
		    print 'Column argument ',column,'is not available'
		    return 
		value_declaration += str(column)+'=' + "'" + str(solarnorms[column])+"',"
	    value_declaration = value_declaration[:-1]
	    self.db.execute('''UPDATE solarnorms SET %s WHERE (normid = %s)''' % (value_declaration,normid))
	    self.db.commit()
	    print 'Solar normalization entries updated.'


	# I refraid from implementing an update capability for the abundance table because these cannot be checked by hand. Instead I recommend deleting the whole table.

	# In[214]:

	def update_abundance_data(self,abu_paper,abu_norm,abu_label,abu_data):
	    '''
	    '''
	    
	    #is abu table available?
	    abu_reg_id=self.check_abu_table_reg(abu_paper)
	    if abu_reg_id==-1: return 'abundance table not available.'
	    
	    norm_id = self.check_solar_normalization(abu_norm[0],abu_norm[1])
	    if norm_id == -1: return
	    
	    #get ref_id
	    ref_id = self.add_paper_ref(abu_paper[0],abu_paper[1],abu_paper[2])    
	    
	    #get galid
	    galid=self.add_galaxy(abu_paper[3])

	    
	    #get table name
	    results=self.db.execute('''SELECT abutable FROM abu_table_reg WHERE (abu_reg_id = %s)''' % (abu_reg_id)).fetchall()
	    tablename = results[0][0]
	    
	    
	    #delete whole table
	    try:
		self.db.execute('''DROP table %s''' % (tablename)) 
	    except:
		pass
	    
	    #create table
	    str_tmp=''
	    for k in range(len(abu_label)):
		str_tmp+=' , '+abu_label[k] +' float '
				     
	    sql_prepr = tuple([tablename])+tuple([str_tmp])
				     
	    #create table
	    self.db.execute('''CREATE TABLE IF NOT EXISTS %s (abuid integer primary key autoincrement %s )''' %
				    sql_prepr)
				     
	    #### add abundance data: 1 entry line        
	    str_tmp1=''
	    for k in range(len(abu_label)):
		str_tmp1+=abu_label[k]+','
	    str_tmp1 = str_tmp1[:-1]
	    str_tmp2=''
	    for k in range(len(abu_data)):
		    str_tmp2+=str(abu_data[k])+','
	    str_tmp2 = str_tmp2[:-1]        
		
	    #create tuple for SQL input
	    sql_prepr = tuple([tablename]) + tuple([str_tmp1]) + tuple([str_tmp2])
	    #print sql_prepr           
	    self.db.execute('''INSERT INTO %s (%s) VALUES (%s)''' %sql_prepr)
		
	    #insert into table registry last, after abundance table was created successfully.                     
	    self.db.execute('''INSERT INTO abu_table_reg (refid, normid,abutable,galid) VALUES ('%s','%s','%s','%s') ''' %
				     (ref_id,norm_id,tablename,galid))    
	    

	    self.db.commit()
	    print 'Abundance data table updated.'


	# In[215]:

	def update_galaxy(self,galaxy_name,update):
	    '''
	    Update table entry of solar normalization table selectd via norm_paper and providing arguments
	    available as columns in the table.
	    galname='Milky Way'
	    update={'Galaxies':'MW'}
	    '''
	    galid=self.check_galaxy(galaxy_name)
	    if galid==-1: return 'galaxy name not available.'
	    new_galname=update['Galaxies']

	    self.db.execute('''UPDATE galaxies SET gal_name='%s' WHERE (gal_name = '%s')''' % (new_galname,galaxy_name))
	    self.db.commit()
	    print 'Galaxy table entry updated.'
	    


