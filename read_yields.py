from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

'''

    Superclass to extract yield data from tables
    and from mppnp simulations

    Christian Ritter 11/2013

    Two classes: One for reading and extracting of
    NuGrid table data, the other one for SN1a data.

    = = = = = = = = = = = = = = = = 

    New version: Benoit Cote, May 2020

    - Restructuration and cleaning
    - read_nugrid_parameter class kept unchanged

'''


# Import Python packages
import numpy as np
import copy
import os


##############################################
#                                            #
#            CLASS Read Yields               #
#                                            #
##############################################
class read_yields( object ):


    '''

    General yields table class that allows to do basic operations
    on the tables (e.g., get yields, set yields, ...)

    '''


    ##############################################
    #                Constructor                 #
    ##############################################
    def __init__(self, table_path=None, table_type=None, \
                 isotopes=None):

        # Make arguments self parameters
        self.table_path = table_path
        self.table_type = table_type
        self.isotopes = isotopes

        # Thresholds used for tolerances
        self.X0_tol = 0.01


    ##############################################
    #                    Get                     #
    ##############################################
    def get(self, M=None, Z=None, quantity=None, isotopes=[]):

        '''

        Return a quantity associated with a model in the yields table

        Arguments
        =========

            M: Initial mass of the model
            Z: Initial metallicity of the model
            quantity: "Lifetime", "Mfinal", "Yields", "C-12", ...
            isotopes: If empty, use self.isotopes for Yields and Xo
                      If provide Yields and X0 will follow that list

        '''

        # Check whether the model exists
        model_label = self._model_label(M,Z)
        if not model_label in self.models:
            print("Error - Model "+model_label+" not found in"+self.table_path)
            return None

        # Return quantity if that is only one value
        if quantity in self.key_one_item:
            return self.table[model_label][quantity]

        # Return the full yields
        elif quantity == "Yields":
            return self.__get_field_list(model_label, isotopes=isotopes, field="Yields")

        # Return the full yields
        elif quantity == "X0":
            return self.__get_field_list(model_label, isotopes=isotopes, field="X0")

        # Return a specific isotope
        elif quantity in self.isotopes:
            return self.table[model_label]["Yields"][quantity]

        # Return error
        else:
            print("Error - Quantity "+quantity+" not found in "+self.table_path)


    ##############################################
    #               Get Field List               #
    ##############################################
    def __get_field_list(self, model_label, isotopes=[], field="Yields"):

        '''

        Return an array of isotopic yields, in sync with the 
        self.isotopes array

        Arguments
        =========

            model_label: Model (M,Z) at which the yields are requested
            isotopes: If empty, use self.isotopes for Yields and Xo
                      If provide Yields and X0 will follow that list
            field: "Yields" or "X0"

        '''

        # Select the list of isotopes
        if len(isotopes) == 0:
            iso_list = self.isotopes
        else:
            iso_list = isotopes
        nb_isotopes = len(iso_list)

        # Declare the yields list
        yields_temp = np.zeros(nb_isotopes)

        # Collect the yields of every isotopes
        for i_iso in range(nb_isotopes):
            iso = iso_list[i_iso]
            if iso in self.isotopes:
                yields_temp[i_iso] = self.table[model_label][field][iso]
            else:
                yields_temp[i_iso] = 1.0e-30

        # Return the yields
        return yields_temp


    ##############################################
    #                    Set                     #
    ##############################################
    def set(self, M=None, Z=None, specie=None, value=None):

        '''

        Overwrite the yieldsof a specific specie (e.g., C-12)
        of a given stellar model

        Arguments
        =========

            M: Initial mass
            Z: Initial metallicity
            specie: Isotope to be overwriten
            value: Overwrite value

        '''

        # Check whether the model exists
        model_label = self._model_label(M,Z)
        if not model_label in self.models:
            print("Error - Model "+model_label+" not found in"+self.table_path)

        # Check whether the isotopes is available
        elif not specie in self.isotopes:
            print("Error - Isotope "+specie+" not found.")

        # Replace the isotope value
        else:
            self.table[model_label]["Yields"][specie] = copy.deepcopy(value)


    ##############################################
    #                 Model Label                #
    ##############################################
    def _model_label(self, M, Z):

        '''

        Create and return the label of a model

        Arguments
        =========

            M: Mass of the model
            Z: Initial metallicity of the model

        '''

        # Mass- and metallicity-dependent model
        if not M == None and not Z == None:
            return '(M='+str(float(M))+',Z='+str(float(Z))+')'

        # Metallicity-dependent model
        elif M == None:
            return '(M=None,Z='+str(float(Z))+')'


    ##############################################
    #             Split NuPyCEE Line             #
    ##############################################
    def _split_NuPyCEE_line(self, line):

        '''

        Split a yields line from a yields table, and remove "&"


        '''

        # Remove white spaces
        split = line.split()

        # Remove the & characters
        for i_col in range(len(split)):
            split[i_col] = split[i_col].split("&")[-1]

        # Return the column labels
        return split


    ##############################################
    #               Get File Lines               #
    ##############################################
    def _get_file_lines(self, path):

        '''

        Return all lines from a file


        '''

        # Open file and read all lines
        ff = open(path)
        lines = ff.readlines()

        # Close file and return lines
        ff.close()
        return lines


    ##############################################
    #            Initialize Parameters           #
    ##############################################
    def _initialize_parameters(self):

        '''

        Declare general parameters for reading yields tables


        ''' 

        # Define whether the list of isotopes is provided
        iso_provided = True
        if self.isotopes == None:
            iso_provided = False

        # Define the properties of the table
        table = dict()
        table["Header"] = []

        # Define the list of models
        models = []

        # Read all lines in the yields table
        lines = self._get_file_lines(self.table_path)

        # Return parameters
        return iso_provided, table, models, lines


    ##############################################
    #            Create Yields Entry             #
    ##############################################
    def _create_yields_entry(self, model_label, iso_provided):

        '''

        Create a new dictionary entry for the yields of a new model

        Argument
        ========

            model_label: name of the new model found in the yields table file
            iso_provided: True if the list of isotopes is pre-defined
       
        ''' 

        # Add the yields entry
        self.table[model_label] = dict()
        self.table[model_label]["Yields"] = dict()

        # Declare isotopes if needed
        if iso_provided:
            for iso in self.isotopes:
                self.table[model_label]["Yields"][iso] = 0.0


    ##############################################
    #            Add Yields to Dict              #
    ##############################################
    def _add_yields_to_dict(self, model_label, iso, value, iso_provided):

        '''

        Add yields of a specific isotopes to the self.table dictionary

        Argument
        ========

            table: yields table dictionary
            model_label: name of the new model found in the yields table file
            iso_provided: True if the list of isotopes is pre-defined
       
        ''' 

        # If the list of isotopes is pre-defined ..
        if iso_provided:

            # If the isotope in the yields file is wanted ..
            if iso in self.isotopes:

                # Add the yields to the dictionary
                self.table[model_label]["Yields"][iso] = float(value)

        # Add the yields to the dictionary
        else:
            self.table[model_label]["Yields"][iso] = float(value)


    ##############################################
    #              Create M Z Lists              #
    ##############################################
    def _create_M_Z_lists(self, define_M=False, define_Z=False):

        '''

        From the table dictionary, extract the list of masses
        and metallicities available in the yields table file


        '''

        # Define the lists
        if define_M:
            self.M_list = []
        if define_Z:
            self.Z_list = []

        # For each model ..
        for model in self.models:

            # Extract initial mass of the model
            if define_M:
                M_temp = float(model.split("=")[1].split(",")[0])
                if M_temp not in self.M_list:
                    self.M_list.append(M_temp)

            # Extract initial metallicity of the model
            if define_Z:
                Z_temp = float(model.split("=")[2].split(")")[0])
                if Z_temp not in self.Z_list:
                    self.Z_list.append(Z_temp)

        # Get the number of entries
        if define_M:
            self.nb_M = len(self.M_list)
        if define_Z:
            self.nb_Z = len(self.Z_list)


    ##############################################
    #                 Run Tests                  #
    ##############################################
    def _run_tests(self):

        '''

        Series of tests to see whether the yields table
        is suitable for GCE calculations


        '''

        # Check whether all absolute masses are provided
        self.__check_absolute_masses()

        # Check whether X0 is provided for net yields
        self.__check_net_yields()

        # TODO check mass and isotope consistency
        # .. self.isotopes + what is in Yields and X0


    ##############################################
    #              Check Net Yields              #
    ##############################################
    def __check_net_yields(self):

        '''

        Check whether the initial composition of the models
        are provided, and whether they add up to 1.0.


        '''

        # Set the potential use of net yields
        self.net_yields_available = True

        # For each model ..
        for model in self.models:

            # Check if X0 was in the table
            if not "X0" in self.table[model].keys():
                self.net_yields_available = False

            # Check if the initial composition add to 1.0
            else: 
                X0_sum = sum(self.table[model]["X0"].values())
                ratio_tol = np.minimum(X0_sum,1.0) / np.maximum(X0_sum,1.0)
                if ratio_tol < self.X0_tol:
                    self.net_yields_available = False

        # Set initial composition to None if cannot use net yields
        if not self.net_yields_available:
            for model in self.models:
                self.table[model]["X0"] = None
                self.table[model]["Net_yields"] = None

        # Calculate net yields if we can
        else:
            for model in self.models:
                self.table[model]["Net_yields"] = dict()
                for iso in self.isotopes:
                    self.table[model]["Net_yields"][iso] = self.table[model]["Yields"][iso] - \
                     self.table[model]["X0"][iso] * self.table[model]["M_ejected"]


    ##############################################
    #            Check Absolute Masses           #
    ##############################################
    def __check_absolute_masses(self):

        '''

        Make sure the final remnant masses and the total
        ejected masses are provided and consistent


        '''

        # For each model ..
        for model in self.models:

            # Fill total ejected mass
            M_ejected = sum(self.table[model]["Yields"].values())
            self.table[model]["M_ejected"] = M_ejected

            # Get the initial and final stellar mass
            if not self.table_type == "Z_dependent":
                M_initial = float(model.split(",")[0].split("=")[1])
                self.table[model]["M_initial"] = M_initial
                self.table[model]["M_final"] = M_initial - M_ejected



##############################################
#                                            #
#          CLASS Read Yields M Z             #
#                                            #
##############################################
class read_yields_M_Z( read_yields ):


    '''

    Yields table class for mass- and metallicity-dependent yields
    Inherites from read_yields class


    '''


    ##############################################
    #                Constructor                 #
    ##############################################
    def __init__(self, table_path, isotopes=None):


        # Define the type of yields table
        table_type = "M_Z_dependent"
        self.key_one_item = ["Lifetime", "M_final"]

        # Initialize the common parameters
        read_yields.__init__(self, table_path=table_path, \
                table_type=table_type, isotopes=isotopes)

        # Read the yields table
        self.__read_M_Z_table()

        # Create list of masses and metallicities
        self._create_M_Z_lists(define_M=True, define_Z=True)

        # Run test functions to make sure everything is ok
        self._run_tests()


    ##############################################
    #               Read M Z Table               #
    ##############################################
    def __read_M_Z_table(self):

        '''

        Read the mass- and metallicity-dependent yields table
        and store all the relevant properties


        '''

        # Initialize parameters
        iso_provided, self.table, self.models, lines = self._initialize_parameters()
        iso_list_read = []

        # For each line ..
        for line in lines:

            # Collect header
            if (line[0] == "H") and \
                not "H Table:" in line and \
                not "H Lifetime:" in line and \
                not "H Mfinal:" in line:
                    self.table["Header"].append(line)

            # If this is a new model ..
            elif "H Table:" in line:

                # Add the model to the list
                model_label = ""
                for char in line.split(":")[-1]:
                    if not char == " ":
                        model_label += char
                model_label = model_label[:-1]
                self.models.append(model_label)

                # Create new entry for the table dictionary
                self._create_yields_entry(model_label, iso_provided)

            # Collect lifetime
            elif "H Lifetime:" in line:
                self.table[model_label]["Lifetime"] = float(line.split(":")[-1])

            # Collect the final mass
            elif "H Mfinal:" in line:
                self.table[model_label]["M_final"] = float(line.split(":")[-1])

            # Collect the column labels
            elif "&Isotopes" in line:
                columns = self._split_NuPyCEE_line(line)

                # Add entry for the initial composition
                if "X0" in columns:
                    i_X0 = columns.index("X0")
                    self.table[model_label]["X0"] = dict()
                    if iso_provided:
                        for iso in self.isotopes:
                            self.table[model_label]["X0"][iso] = 0.0

            # If yields are to be read ..
            elif "&" in line:

                # Split the line
                split = self._split_NuPyCEE_line(line)

                # Add yields
                self._add_yields_to_dict(model_label, split[0], split[1], iso_provided)

                # Add isotopes to the list if not predefined
                if not iso_provided and not split[0] in iso_list_read:
                    iso_list_read.append(split[0])

                # Add initial composition
                if "X0" in columns:
                    self.table[model_label]["X0"][split[0]] = float(split[i_X0])

        # Collect the list of isotopes if needed
        if not iso_provided:
            self.isotopes = copy.deepcopy(iso_list_read)
        self.nb_isotopes = len(self.isotopes)



##############################################
#                                            #
#           CLASS Read Yields Z              #
#                                            #
##############################################
class read_yields_Z( read_yields ):


    '''

    Yields table class for metallicity-dependent yields
    Inherites from read_yields class


    '''


    ##############################################
    #                Constructor                 #
    ##############################################
    def __init__(self, table_path, isotopes=None):


        # Define the type of yields table
        table_type = "Z_dependent"
        self.key_one_item = []

        # Initialize the common parameters
        read_yields.__init__(self, table_path=table_path, \
                table_type=table_type, isotopes=isotopes)

        # Read the yields table
        self.__read_Z_table()

        # Create list of metallicities
        self._create_M_Z_lists(define_M=False, define_Z=True)

        # Run test functions to make sure everything is ok
        self._run_tests()


    ##############################################
    #                Read Z Table                #
    ##############################################
    def __read_Z_table(self):

        '''

        Read the metallicity-dependent yields table
        and store all the relevant properties


        '''

        # Initialize parameters
        iso_provided, self.table, self.models, lines = self._initialize_parameters()
        iso_list_read = []

        # For each line in the yields file ..
        for line in lines:

            # Collect header
            if line[0] == "H":
                self.table["Header"].append(line)

            # If this is the line where metallicities are defined ..
            elif "&Isotopes" in line:

                # Split the line
                columns = self._split_NuPyCEE_line(line)

                # For each metallicity available ..
                for i_col in range(1, len(columns)):

                    # Create the label of the model
                    model_label = "(M=None,"+columns[i_col]+")"
                    self.models.append(model_label)

                    # Create new entry for the table dictionary
                    self._create_yields_entry(model_label, iso_provided)

            # If yields are to be read ..
            elif "&" in line:

                # Split the line
                split = self._split_NuPyCEE_line(line)

                # For each metallicity (each model) ..
                for i_m in range(len(self.models)):

                    # Set the column index in the yields file
                    i_col = i_m + 1

                    # Add yields
                    self._add_yields_to_dict(\
                        self.models[i_m], split[0], split[i_col], iso_provided)

                    # Add isotopes to the list if not predefined
                    if not iso_provided and not split[0] in iso_list_read:
                        iso_list_read.append(split[0])

        # Collect the list of isotopes if needed
        if not iso_provided:
            self.isotopes = copy.deepcopy(iso_list_read)
        self.nb_isotopes = len(self.isotopes)



##############################################
#                                            #
#               CLASS Iniabu                 #
#                                            #
##############################################
class iniabu():

    '''

    Reads NuPyCEE initial abundance files

    '''

    ##############################################
    #                Constructor                 #
    ##############################################
    def __init__(self, iniabu_path):


        # Assign self parameters
        self.iniabu_path = iniabu_path

        # Read the initial abundance file
        self.__read_iniabu()


    ##############################################
    #                Read Iniabu                 #
    ##############################################
    def __read_iniabu(self):

        '''

        Read the initial abundance file and store abundances and isotopes


        '''

        # Read all lines in the initial abundance file
        ff = open(self.iniabu_path)
        lines = ff.readlines()
        ff.close

        # Define the abundance dictionary
        self.abundances = dict()

        # For each line ..
        for line in lines:

            # Split the line into isotope and abundance
            iso, abundance = self.__split_iniabu_line(line)

            # Add the abundance to the dictionary
            self.abundances[iso] = abundance


    ##############################################
    #                Read Iniabu                 #
    ##############################################
    def __split_iniabu_line(self, line):

        '''

        Split a iniabu-file line into isotope and abundance


        '''

        # Split the line
        split = line.split()

        # H-1
        if "PROT" in line:
            return "H-1", float(split[-1])

        # If isotope is not H-1 ..
        else:

            # Get the isotope name without "-"
            if len(split) == 3:
                iso_temp = split[1]
            else:
                iso_temp = split[1] + split[2]

            # Reformat the isotope
            iso = ""
            for i_str in range(len(iso_temp)):
                if iso_temp[i_str].isdigit() and not "-" in iso:
                    iso += "-"
                iso += iso_temp[i_str]
            iso = iso.capitalize()

            # Return isotope and the abundance
            return iso, float(split[-1])


    ##############################################
    #               Iso Abundance                #
    ##############################################
    def iso_abundance(self, isotopes):

        '''

        Return the list of abundance of the iniabu file

        Argument
        ========

            isotopes: List of isotope requested

        '''

        # Declare the list of abundances to be returned
        abu = []

        # For each isotope in the requested list
        for i_iso in range(len(isotopes)):

            # Add abundance
            if isotopes[i_iso] in self.abundances.keys():
                abu.append(self.abundances[isotopes[i_iso]])
            else:
                abu.append(1.0e-30)

        # Return the abundances
        return abu



# Below
# # # # # # # # # # # # # # # # # # # # # #
#                                         # 
#           NOT    UPDATED    YET         #
#                                         # 
# # # # # # # # # # # # # # # # # # # # # #



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
                if not 'Table:' in line:
                    if header_done==False:
                        header1.append(line.strip())
                    else:
                        table_header[-1].append(line.strip())
                else:
                    ignore=False
                    #print (line,'ignore',ignore)
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
                #print (column_titles)
                continue
            #iso ,name and yields
            iso_name=line.split('&')[1].strip()
            #print (line)
            #print (line.split('&'))
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
        #print ('header1: ',header1)
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
                print ('Error: Wrong input')
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
                        print ('h :',h)
                        data_new.append(data[h])
                        dcols_new.append(dcols[h])
                dcols_new=[dcols_new[0]]+dcols_new[2:]+[dcols_new[1]]
                print ('dcols: ',dcols_new)
                special_header='Table: (M='+str(mass)+',Z='+str(metallicity)+')'
                headers=[special_header]+attr_lines
                ascii1.writeGCE_table_parameter(filename=filename,headers=headers,data=data_new,dcols=dcols_new)


