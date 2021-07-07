from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

'''

    Class that reads in mass- and metallicity-dependent yields tables
    and combine them in order to cover all mass ranges, from the lowest-
    mass AGB to the most massive star.

'''

# Import Python packages
import numpy as np
import copy
import os

# Define where is the working directory
nupy_path = os.path.dirname(os.path.realpath(__file__))

# Import NuPyCEE codes
import NuPyCEE.read_yields as ry
from NuPyCEE.utils import interpolation

class yields_combiner():

    '''
    Input parameters (yields_combiner.py)
    ================

    zero_log : float
        Value that replaces zeros in the yields table when
        the codes need to interpolate yields in log-log space

        Default value : 1.0e-30

    '''


    ##############################################
    ##               Constructor                ##
    ##############################################
    def __init__(self, zero_log=1.0e-30):

        # Convert input parameters into self parameters
        self.zero_log = zero_log


    ##############################################
    #               Combine Tables               #
    ##############################################
    def combine_tables(self, path_list, isotopes, log_q=True, log_Z=True, overwrite=[]):

        '''
        Combine different yields table and create a unique
        yields table that has a uniform grid of masses and
        metallicities.

        Parameters
        ==========
            path_list: Pathes to all yields tables to be combined
                       The order does not matter
            isotopes:  List of isotopes to be in the combined table
            log_q:     Whether quantity should be interpolated in log space
            log_Z:     Whether metallicities should be interpolated in log space
            overwrite: List of table paths that will overwrite the combined table.
                       See function __overwrite_models(..) for more information.

        '''

        # Set the list of fields that includes isotopes abundances
        self.__set_iso_fields()

        # Read all yields table that need to be combined
        nb_ytables, ytable_list = self.__read_all_yields(path_list, isotopes)

        # Sort the tables from low-mass to high-mass
        ytable_list = self.__sort_ytables(nb_ytables, ytable_list)
        if ytable_list == None:
            print("Problem -- overlaping input yields")
            return None

        # Extract the list of metallicities covered by all tables
        Z_list = self.__combine_Z(nb_ytables, ytable_list)

        # Overwrite specific stellar models if necessary
        if len(overwrite) > 0:
            ytable_list = self.__overwrite_models(\
                nb_ytables, ytable_list, overwrite, isotopes)

        # Remove net yields fields if this option cannot be
        # accomodated with all individual yields tables.
        # This needs to be after __overwrite_models().
        ytable_list = self.__adjust_net_yields_option(nb_ytables, ytable_list)

        # Uniform the metallicity grid of each individual yields table
        ytable_list = self.__uniform_ytables(\
                        nb_ytables, ytable_list, Z_list, log_q, log_Z)

        # Combine the yields table objects into one
        ytable_comb = self.__combine_yields_dictionaries(\
                        nb_ytables, ytable_list, Z_list)

        # Calculate net yields if possible (needed for interpolated models)
        ytable_comb.check_net_yields()

        # Return the combined yields table object
        return ytable_comb



    ##############################################
    #              Uniform Ytables               #
    ##############################################
    def __uniform_ytables(self, nb_ytables, ytable_list, Z_list, log_q, log_Z):

        '''
        Take all yields table objects, and add interpolated metallicities
        in order to be uniform with the combined yields table.

        Parameters
        ==========
            nb_ytables:  Number of yields tables to be ultimately combined
            ytable_list: List of read_yields objects
            Z_list:      List of metallicities covered by all yields table
            log_q:       Whether quantity should be interpolated in log space
            log_Z:       Whether metallicities should be interpolated in log space

        '''

        # For each table, and for each covered metallicity ..
        for i_t in range(nb_ytables):
            for Z in Z_list:

                # If this metallicity is not originally covered by the table ..
                if Z not in ytable_list[i_t].Z_list:

                    # Create a new entry for this metallicity
                    ytable_list[i_t] = \
                        self.__add_Z_to_ytable(Z, ytable_list[i_t], log_q, log_Z)

            # Update the list of metallicities in the table
            ytable_list[i_t].Z_list = copy.deepcopy(Z_list)
            ytable_list[i_t].nb_Z = len(Z_list)

        # Return the uniform yields tables
        return ytable_list



    ##############################################
    #          Adjust Net Yields Option          #
    ##############################################
    def __adjust_net_yields_option(self, nb_ytables, ytable_list):

        '''
        Scan all yields table objects that will compose the combined
        table, and remove the net yields option for all object if at
        least one of them cannot work with net yields. This is because
        the combined yields will be used in NuPyCEE as if it was a
        single yields table.

        Parameters
        ==========
            nb_ytables:  Number of yields tables to be ultimately combined
            ytable_list: List of read_yields objects

        '''

        # Keep track of whether the net yields option can be used
        can_use_net_yields = True

        # For each yields table object ..
        for ytable in ytable_list:

            # Check if net yields can be used
            if not ytable.net_yields_available:
                can_use_net_yields = False
                break

        # Remove the net yields option if needed
        if not can_use_net_yields:
            self.__set_iso_fields(net_yields_available=False)
            for i_t in range(nb_ytables):
                ytable_list[i_t].net_yields_available = False
                for model in ytable_list[i_t].models:
                    del ytable_list[i_t].table[model]["X0"]
                    del ytable_list[i_t].table[model]["Net_yields"]

        # Return the yields table objects
        return ytable_list



    ##############################################
    #              Add Z to Ytable               #
    ##############################################
    def __add_Z_to_ytable(self, Z_int, ytable, log_q, log_Z):

        '''
        Add stellar yields to a table by interpolating quantities
        between metallicities.

        Parameters
        ==========
            Z_int:  Metallicity at which quantities will be interpolated
            ytable: Yields table object to which new yields will be added
            log_q:  Whether quantity should be interpolated in log space
            log_Z:  Whether metallicities should be interpolated in log space

        '''

        # Get the list of quantities needed for each stellar model
        quantity_list = list(ytable.table[ytable.models[0]].keys())

        # For each stellar masses in the yields table ..
        for M in ytable.M_list:

            # Create a new entry in the table
            model_label = ytable._model_label(M, Z_int)
            ytable.table[model_label] = dict()
            ytable.models.append(model_label)

            # For each quantity ..
            for quantity in quantity_list:

                # Interpolate the quantity (if not net yields, they are 
                # calculated later on to avoid log10(0) problems)
                if quantity == "Net_yields":
                    quantity_int = None
                else:
                    quantity_int = self.__interpolate_specific_M_vs_Z(\
                        M, Z_int, ytable, quantity, log_q, log_Z)

                # Add the interpolation to the table
                ytable.table[model_label][quantity] = quantity_int

            # Re-convert isotopic numpy arrays into a dictionaries
            ytable = self.__recover_iso_dictionaries(ytable, model_label)

        # Return the table with additional entries
        return ytable


    ##############################################
    #         Recover Iso Dictionaries           #
    ##############################################
    def __recover_iso_dictionaries(self, ytable, model_label):

        '''
        Convert "Yields" and "X0" from numpy arrays to dictionaries
        using the isotope as the key of each entry.

        Parameters
        ==========
            ytable:      Yields table object
            model_label: Dictionary key to the targeted stellar model

        '''

        # For each isotopic abundances ..
        for key in self.iso_fields:

            # Declare the dictionary that will overwrite the numpy array
            ab = dict()

            # For each entry (isotope) ..
            for i_iso in range(ytable.nb_isotopes):

                # Create an isotope entry in the dictionary
                ab[ytable.isotopes[i_iso]] = copy.deepcopy(ytable.table[model_label][key][i_iso])

            # Overwrite the isotopic abundances
            ytable.table[model_label][key] = copy.deepcopy(ab)

        # Returned the yields table object
        return ytable



    ##############################################
    #              Set Iso Fields                #
    ##############################################
    def __set_iso_fields(self, net_yields_available=True):

        '''
        Set the dictionary keys that include isotopic abundances
        in the yields table object.

        Parameters
        ==========
            net_yields_available: True is net yields can be used

        '''

        # If net yields are available ..
        if net_yields_available:
            self.iso_fields = ["Yields", "X0"]

        # If net yields are not available ..
        else:
            self.iso_fields = ["Yields"]



    ##############################################
    #         Combine Yields Dictionaries        #
    ##############################################
    def __combine_yields_dictionaries(self, nb_ytables, ytable_list, Z_list):

        '''
        Combines different yields table (read_yields) objects together,
        and update the various properties (e.g. list of mass) in order
        to reflect the full list of combined stellar models.

        Parameters
        ==========
            nb_ytables:  Number of yields tables to be ultimately combined
            ytable_list: List of read_yields objects
            Z_list:      List of metallicities covered by all yields table

        '''

        # Make the first table object the combined table
        ytable_comb = ytable_comb = ytable_list[0]

        # For all other tables ..        
        for i_t in range(1, nb_ytables):

            # Add the yields to the combined table
            ytable_comb.table.update(ytable_list[i_t].table)

            # Add masses to the combined table
            ytable_comb.M_list = ytable_comb.M_list + ytable_list[i_t].M_list

            # Add the list of models
            ytable_comb.models = ytable_comb.models + ytable_list[i_t].models

        # Add the metallicities (without duplicate)
        ytable_comb.Z_list = copy.deepcopy(Z_list)

        # Calculate the initial, final, and total ejected mass
        ytable_comb.check_absolute_masses()

        # Return the combined yields
        return ytable_comb



    ##############################################
    #             Overwrite Models               #
    ##############################################
    def __overwrite_models(self, nb_ytables, ytable_list, overwrite, isotopes):

        '''
        Overwrite models in the combined yields table. For example,
        if overwrite=[path1,path2] and it points to table1 with 
        M=20,25 and table2 with M=25, then it will first overwrite
        the 20 and 25 Msun model in the combined table using table1,
        and then overwrite (again) the 25 Msun using table2. The 
        results will be a combined table where the 20 Msun comes
        from table1 and the 25 Msun model comes from table2. Overwrite
        should be in order of increasing priority.

        Parameters
        ==========
            nb_ytables:  Number of yields tables to be ultimately combined
            ytable_list: List of read_yields objects
            overwrite:   List of table paths that will overwrite the combined table.
            isotopes:    List of isotopes to be in the combined table

        '''

        # Copy the list of all models of the combined table
        all_models = []
        for ytable in ytable_list:
            all_models = all_models + ytable.models

        # Read all yields table aimed to overwrite
        ov_table_list = []
        for path in overwrite:
            ov_table_list.append(ry.read_yields_M_Z(\
                os.path.join(nupy_path,path), isotopes=isotopes))

        # For each table that will overwrite ..
        for i_t in range(len(ov_table_list)):
            ov_table = ov_table_list[i_t]

            # Check if all models exist in the combined table
            models_exist = True
            for model in ov_table.models:
                if model not in all_models:
                    models_exist = False

            # If the overwrite process can be accomodated ..
            if models_exist:

                # Overwrite models one by one
                for model in ov_table.models:
                    for i_main in range(nb_ytables):
                        if model in ytable_list[i_main].models:
                            ytable_list[i_main].table[model] = copy.deepcopy(ov_table.table[model])

                # Signal that the net yields option need to be turned off if needed
                # See __adjust_net_yields_option()
                if not ov_table.net_yields_available:
                    ytable_list[0].net_yields_available = False

            # Do not overwrite if it cannot be fully accomodated
            else:
                print("Warning - The following table did not overwrite, due to"+\
                    " a mismatch with the list of models in the main yields tables.")
                print(overwrite[i_t])

        # Return the overwritten yields tables
        return ytable_list



    ##############################################
    #               Read All Yields              #
    ##############################################
    def __read_all_yields(self, path_list, isotopes):

        '''
        Read all yields tables that need to be combined, and
        return the corresponding read_yields object.

        Parameters
        ==========
            path_list: Pathes to all yields tables to be combined
                       The order does not matter
            isotopes:  List of isotopes in the combined yields table

        '''

        # Declare the yields table objects
        ytable_list = []
        nb_ytables = len(path_list)

        # Read all tables
        for path in path_list:
             ytable_list.append(ry.read_yields_M_Z(os.path.join(nupy_path,path), isotopes=isotopes))

        # Return the objects
        return nb_ytables, ytable_list



    ##############################################
    #                 Sort YTables               #
    ##############################################
    def __sort_ytables(self, nb_ytables, ytable_list):

        '''
        Look at the list of masses in each yields table and return 
        the list of tables sorted by increasing masses. For example
        if ytables_list = [massive, agb], then the function will return
        [agb, massive], so that agb masses appears before massive masses
        in the combined yields table.

        Parameters
        ==========
            nb_ytables:  Number of yields tables to be ultimately combined
            ytable_list: List of read_yields objects

        '''

        # Declare the mass range of each yields table
        M_min = []; M_max = []; M_range = []

        # For each yields table ..
        for ytable in ytable_list:

            # Collect the min and max masses
            M_min.append(np.min(ytable.M_list))
            M_max.append(np.max(ytable.M_list))

            # Return warning if there are mass overlaps with other tables
            for mr in M_range:
                if  (mr[0] <= M_min[-1] and M_min[-1] <= mr[1]) or \
                    (mr[0] <= M_max[-1] and M_max[-1] <= mr[1]):
                    return None

            # Keep in memory the current mass range if there is no overlap
            M_range.append([M_min[-1],M_max[-1]])

        # Sort the list of yields tables (in increasing mass)
        ytables_sorted = []
        for i_t in np.argsort(M_min):
            ytables_sorted.append(ytable_list[i_t])

        # Return the sorted list
        return ytables_sorted



    ##############################################
    #                  Combine Z                 #
    ##############################################
    def __combine_Z(self, nb_ytables, ytable_list):

        '''
        Combine the metallicities covered by all yields tables,
        while excluding duplicates.

        Parameters
        ==========
            nb_ytables:  Number of yields tables to be ultimately combined
            ytable_list: List of read_yields objects

        '''

        # Declare the list containing the metallicities of each yields table
        Z_lists = []

        # For each yields table ..
        for ytable in ytable_list: 

            # Collect its list of metallicities
            Z_lists.append(ytable.Z_list)

        # Combine and return metallicities 
        return self.combine_no_duplicate(Z_lists)



    ##############################################
    #             Combine No Duplicate           #
    ##############################################
    def combine_no_duplicate(self, lists):

        '''
        Combine items from multiple lists, while excluding duplicates.

        Parameters
        ==========
            lists: list of lists for which their contents will be combined

        '''

        # Declare the combined list of values (no duplicate)
        list_no_dup = []

        # For each entry in each list ..
        for l in lists:
            for entry in l:

                # Add the entry if not already in the combined list
                if entry not in list_no_dup:
                    list_no_dup.append(entry)

        # Return the combined list (in decreasing order)
        return sorted(list_no_dup)[::-1]



    ##############################################
    #         Interpolate Specific M vs Z        #
    ##############################################
    def __interpolate_specific_M_vs_Z(self, M, Z_int, ytable, quantity, \
                                      log_q, log_Z):

        '''
        Interpolate the quantity (e.g., yields) of a specific stellar
        mass M as a function of metallicity Z, for a single yields table.

        Parameters
        ==========
            M:        Stellar mass
            Z_int:    Metallicity at which the yields need to be interpolated
            ytable:   Yields table object (read_yields.py)
            quantity: Yields table dictionary fields, eg. "Yields", "X0"
            log_q:    Whether quantity should be interpolated in log space
            log_Z:    Whether metallicities should be interpolated in log space

        '''

        # Copy list of metallicities
        Z_arr = copy.deepcopy(ytable.Z_list)

        # Do not interpolate is Z_int is out of bound
        Z_min, Z_max = np.min(ytable.Z_list), np.max(ytable.Z_list)
        if Z_int <= Z_min: return ytable.get(M=M, Z=Z_min, quantity=quantity)
        elif Z_int >= Z_max: return ytable.get(M=M, Z=Z_max, quantity=quantity)

        # Extract the quantity at all Z, for the specific M
        quantity_arr = self.__extract_specific_M_vs_Z(M, ytable, quantity)

        # Convert values in log10 if needed
        if log_Z:
            Z_arr = np.log10(Z_arr)
            Z_int = np.log10(Z_int)
        if log_q:
            quantity_arr = self.__log_quantity(quantity_arr, ytable, quantity)

        # Interpolate the quantity
        quantity_int = self.__interpolate_general(\
                            Z_arr, quantity_arr, Z_int, quantity)

        # Reconvert the quantity into linear scale
        if log_q:
            quantity_int = 10**quantity_int

        # Return the interpolated quantity
        return quantity_int



    ##############################################
    #          Extract Specific M vs Z           #
    ##############################################
    def __extract_specific_M_vs_Z(self, M, ytable, quantity):

        '''
        Creates a numpy array containing the quantity (e.g., yields) as
        a function of metallicity, for a specific stellar mass M.

        Parameters
        ==========
            M:        Stellar mass
            ytable:   Yields table object
            quantity: Quantity (dictionary field), e.g. "Yields", "X0"

        '''

        # Declare the output array
        if quantity in self.iso_fields:
            quantity_arr = np.zeros((ytable.nb_Z, ytable.nb_isotopes))
        else:
            quantity_arr = np.zeros(ytable.nb_Z)

        # Extract the quantity for each metallicity available in the table
        for i_Z in range(ytable.nb_Z):
            quantity_arr[i_Z] = ytable.get(M=M, Z=ytable.Z_list[i_Z], quantity=quantity)

        # Return the array
        return quantity_arr



    ##############################################
    #                 Log Quantity               #
    ##############################################
    def __log_quantity(self, quantity_arr, ytable, quantity):

        '''
        Takes an input quantities, replaces 0 by self.zero_log, and
        convert the values into log10 values.

        Parameters
        ==========
            quantity_arr: Input array that needs to be in log10
            ytable:       Yields table object
            quantity:     Quantity (dictionary field), e.g. "Yields", "X0"

        '''

        # If the input array are isotopic abundances ..
        if quantity in self.iso_fields:

            # Look for zeros and convert then
            for i_Z in range(ytable.nb_Z):
                for i_iso in range(ytable.nb_isotopes):
                    if quantity_arr[i_Z][i_iso] == 0.0:
                        quantity_arr[i_Z][i_iso] = self.zero_log

        # If the input array is a 1D array ..
        else:

            # Look for zeros and convert then
            for i_Z in range(ytable.nb_Z):
                if quantity_arr[i_Z] == 0.0:
                    quantity_arr[i_Z] = self.zero_log

        # Return the log10 version of the input quantity_arr
        return np.log10(quantity_arr)



    ##############################################
    #             Interpolate General            #
    ##############################################
    def __interpolate_general(self, x_arr, y_arr, xx, quantity):

        '''
        Interpolate quantities according to the interpolation routine
        in chem_evol.py.

        Parameters
        ==========
            x_arr:    Coordinate array
            y_arr:    1-D or 2-D numpy array for interpolation
            xx:       Value for which y_arr must be interpolated
            quantity: Yields table dictionary fields, eg. "Yields", "X0"

        '''

        # Prepare the empty list of interpolation coefficients
        if quantity in self.iso_fields: 
            interp_list = [[None]*len(y) for y in y_arr]
        else:
            interp_list = [None]*len(y_arr)

        # Find lower-bound interpolation index
        # .. if x_arr is in increasing order
        if x_arr[0] < x_arr[-1]:
            indx = 0
            while x_arr[indx+1] < xx:
                indx += 1

        # Find lower-bound interpolation index
        # .. if x_arr is in decreasing order
        if x_arr[0] > x_arr[-1]:
            indx = 0
            while x_arr[indx+1] > xx:
                indx += 1

        # Return the interpolated quantities
        return interpolation(x_arr, y_arr, xx, indx, interp_list)










        
