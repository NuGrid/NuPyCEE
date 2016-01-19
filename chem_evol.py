'''

Chemical Evolution - chem_evol.py

Functionality
=============

This is the superclass inherited by the SYGMA and the OMEGA modules.  It provides
common functions for initialization and for the evolution of one single timestep.


Made by
=======

MAY2015: B. Cote

The core of this superclass is a reorganization of the functions previously found in
earlier versions of SYGMA:

v0.1 NOV2013: C. Fryer, C. Ritter

v0.2 JAN2014: C. Ritter

v0.3 APR2014: C. Ritter, J. F. Navarro, F. Herwig, C. Fryer, E. Starkenburg, 
              M. Pignatari, S. Jones, K. Venn, P. A. Denissenkov & the
              NuGrid collaboration

v0.4 FEB2015: C. Ritter, B. Cote


Usage
=====

See sygma.py and omega.py

'''

# Standard packages
import numpy as np
import time as t_module
import copy
import math
import random
import os
import re
from pylab import polyfit
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.interpolate import UnivariateSpline
from mpl_toolkits.mplot3d import Axes3D
from imp import *
import matplotlib.pyplot as plt

# Variable enabling to work in notebooks
global notebookmode
notebookmode=True

# Set the working space to the current directory
global global_path
try:
    if os.environ['SYGMADIR']:
        global_path = os.environ['SYGMADIR']
except KeyError:
    global_path=os.getcwd()
global_path=global_path+'/'

# Import the class that reads the input yield tables
import read_yields as ry


class chem_evol(object):


    '''
    Input parameters (chem_evol.py)
    ================

    special_timesteps : integer
        Number of special timesteps.  This option (already activated by default)
        is activated when special_timesteps > 0.  It uses a logarithm timestep
        scheme which increases the duration of timesteps throughout the simulation.
        Default value : 30

    dt : float
        Duration of the first timestep [yr] if special_timesteps is activated.
        Duration of each timestep if special_timesteps is desactivated.
        Default value : 1.0e6

    tend : float
        Total duration of the simulation [yr].
        Default value : 13.0e9

    imf_bdys : list
        Upper and lower mass limits of the initial mass function (IMF) [Mo].
        Default value : [0.1,100]

    imf_yields_range : list
        Initial mass of stars that contribute to stellar ejecta [Mo].
        Default value : [1,30]

    imf_type : string
	Choices : 'salpeter', 'chabrier', 'kroupa', 'alphaimf'
        'alphaimf' creates a custom IMF with a single power-law covering imf_bdys.
        Default value : 'kroupa'

    alphaimf : float
        Aplha index of the custom IMF, dN/dM = Constant * M^-alphaimf
        Default value : 2.35

    imf_bdys_pop3 : list
        Upper and lower mass limits of the IMF of PopIII stars [Mo]. 
        Default value : [0.1,100]

    imf_yields_range_pop3 : list
        Initial mass of stars that contribute to PopIII stellar ejecta [Mo].
        PopIII stars ejecta taken from Heger et al. (2010)
        Default value : [10,30]

    iniZ : float
        Initial metallicity of the gas in mass fraction (e.g. Solar Z = 0.02).
        Choices : 0.0, 0.0001, 0.001, 0.006, 0.01, 0.02
                 (-1.0 to use non-default yield tables)
        Default value : 0.0

    Z_trans : float
        Variable used when interpolating stellar yields as a function of Z.
        Transition Z below which PopIII yields are used, and above which default
        yields are used.
        Default value : -1 (not active)

    mgal : float
        Initial mass of gas in the simulation [Mo].
        Default value : 1.6e11

    sn1a_on : boolean
	True or False to include or exclude the contribution of SNe Ia.
        Default value : True

    sn1a_rate : string
        SN Ia delay-time distribution function used to calculate the SN Ia rate.
	Choices : 'power_law' - custom power law, set parameter with beta_pow (similar to Maoz & Mannucci 2012)
                  'gauss' - gaussian DTD, set parameter with gauss_dtd
                  'exp' - exponential DTD, set parameter with exp_dtd
		  'maoz' - specific power law from Maoz & Mannucci (2012)
        Default value : 'power_law'

    beta_pow : float
        Slope of the power law for custom SN Ia rate, R = Constant * t^-beta_pow.
        Default value : -1.0

    gauss_dtd : list
        Contains parameter for the gaussian DTD: first the characteristic time [yrs] (gaussian center) 
        and then the width of the distribution [yrs]. 

    exp_dtd : float
        Characteristic delay time [yrs] for the e-folding DTD.

    nb_1a_per_m : float
        Number of SNe Ia per stellar mass formed in a simple stellar population.
        Default value : 1.0e-03

    transitionmass : float
        Initial mass which marks the transition from AGB to massive stars [Mo].
        Default value : 8.0

    exclude_masses : list
        Contains initial masses to be excluded from the simulation; default exclude 32Msun and 60Msun of solar Z because
        they are usable only in Z=0.02 SSP simulations


    table : string
        Path pointing toward the stellar yield tables for massive and AGB stars.
        Default value : 'yield_tables/isotope_yield_table.txt' (NuGrid)

    sn1a_table : string
        Path pointing toward the stellar yield table for SNe Ia.
        Default value : 'yield_tables/sn1a_t86.txt' (Tielemann et al. 1986)

    iniabu_table : string
        Path pointing toward the table of initial abuncances in mass fraction.
        Default value : 'yield_tables/iniabu/iniab2.0E-02GN93.ppn'

    yield_interp : if 'None' : no yield interpolation, no interpolation of total ejecta
		   if 'lin' - Simple linear yield interpolation.
                   if 'wiersma' - Interpolation method from Wiersma+ (2009)

    iolevel : int
        Specifies the amount of output for testing purposes (up to 3).
        Default value : 0


    Run example
    ===========

    See sygma.py and omega.py

    '''


    ##############################################
    ##               Constructor                ##
    ##############################################
    def __init__(self, imf_type='kroupa', alphaimf=2.35, imf_bdys=[0.1,100], \
             sn1a_rate='power_law', iniZ=0.0, dt=1e6, special_timesteps=30, \
             tend=13e9, mgal=1.6e11, transitionmass=8, iolevel=0, \
             ini_alpha=True, table='yield_tables/isotope_yield_table.txt', \
             hardsetZ=-1, sn1a_on=True, sn1a_table='yield_tables/sn1a_t86.txt',\
             iniabu_table='', \
             extra_source_on=False, \
             extra_source_table='yield_tables/mhdjet_NTT_delayed.txt', \
             pop3_table='yield_tables/popIII_heger10.txt', \
             imf_bdys_pop3=[0.1,100], imf_yields_range_pop3=[10,30], \
             starbursts=[], beta_pow=-1.0,gauss_dtd=[3.3e9,6.6e8],\
             exp_dtd=2e9,nb_1a_per_m=1.0e-3,direct_norm_1a=-1,Z_trans=-1, \
             f_arfo=1, imf_yields_range=[1,30],exclude_masses=[32,60],\
             netyields_on=False,wiersmamod=False,yield_interp='lin',\
             input_yields=False,t_merge=-1.0,\
             popIII_on=True,ism_ini=np.array([]),\
             ytables_in=np.array([]), zm_lifetime_grid_nugrid_in=np.array([]),\
             isotopes_in=np.array([]), ytables_pop3_in=np.array([]),\
             zm_lifetime_grid_pop3_in=np.array([]), ytables_1a_in=np.array([]),\
             dt_in=np.array([]),dt_split_info=np.array([]),\
             ej_massive=np.array([]), ej_agb=np.array([]),\
             ej_sn1a=np.array([]), ej_massive_coef=np.array([]),\
             ej_agb_coef=np.array([]), ej_sn1a_coef=np.array([]),\
             dt_ssp=np.array([])):

        # Initialize the history class which keeps the simulation in memory
	self.history = self.__history()

        # If we need to assume the current baryonic ratio ...
        if mgal < 0.0:

            # Use a temporary mgal value for chem_evol __init__ function
            mgal = 1.0e06
            self.bar_ratio = True

        # If we use the input mgal parameter ...
        else:
            self.bar_ratio = False

        # Attribute the input parameters to the current object
        self.history.mgal = mgal
        self.history.tend = tend
        self.history.dt = dt
	self.history.sn1a_rate = sn1a_rate
        self.history.imf_bdys = imf_bdys
        self.history.transitionmass = transitionmass
        self.mgal = mgal
        self.transitionmass = transitionmass
        self.iniZ = iniZ
	self.imf_bdys=imf_bdys
	self.imf_bdys_pop3=imf_bdys_pop3
	self.imf_yields_range_pop3=imf_yields_range_pop3
	self.extra_source_on = extra_source_on
	self.table = table
        self.iniabu_table = iniabu_table
        self.sn1a_table = sn1a_table
        self.extra_source_table = extra_source_table
        self.pop3_table = pop3_table
	self.hardsetZ = hardsetZ
	self.starbursts = starbursts
	self.imf_type = imf_type
	self.alphaimf = alphaimf
	self.sn1a_on = sn1a_on
        self.special_timesteps = special_timesteps
        self.iolevel = iolevel
        self.nb_1a_per_m = nb_1a_per_m
	self.direct_norm_1a=direct_norm_1a
        self.Z_trans = Z_trans
        if sn1a_rate == 'maoz':
            self.beta_pow = -1.0
        else:
            self.beta_pow = beta_pow
        self.gauss_dtd = gauss_dtd
	self.exp_dtd=exp_dtd
        self.normalized = False # To avoid normalizing SN Ia rate more than once
        self.f_arfo = f_arfo
        self.imf_yields_range = imf_yields_range
        self.exclude_masses=exclude_masses
        self.netyields_on=netyields_on
	self.wiersmamod=wiersmamod
	self.yield_interp=yield_interp
        self.t_merge = t_merge
        self.ism_ini = ism_ini
        self.dt_in = dt_in

        # Normalization constants for the Kroupa IMF
        if imf_type == 'kroupa':
            self.p0 = 1.0
            self.p1 = 0.08**(-0.3 + 1.3)
            self.p2 = 0.5**(-1.3 + 2.3)
            self.p3 = 1**(-2.3 +2.3)

        # Parameter that determines if not enough gas is available for star formation
        self.not_enough_gas_count = 0
        self.not_enough_gas = False

        # Check for incompatible inputs - Error messages
        self.__check_inputs()
	
        # If the yield tables has already been read previously ...
        if input_yields:

            # Assign the input yields and lifetimes
            self.ytables = ytables_in
            self.zm_lifetime_grid_nugrid = zm_lifetime_grid_nugrid_in
            self.history.isotopes = isotopes_in
            self.nb_isotopes = len(self.history.isotopes)
            self.ytables_pop3 = ytables_pop3_in
            self.zm_lifetime_grid_pop3 = zm_lifetime_grid_pop3_in
            self.ytables_1a = ytables_1a_in
            self.default_yields = True
            self.extra_source_on = False
            self.ytables_extra = 0

        # If the yield tables need to be read from the files ...
        else:

            # Initialisation of the yield tables
            self.__set_yield_tables()

        # Initialisation of the composition of the gas reservoir
        ymgal = self._get_iniabu()
        self.len_ymgal = len(ymgal)

        # Initialisation of the timesteps 
        timesteps = self.__get_timesteps()
        self.history.timesteps = timesteps
        self.nb_timesteps = len(timesteps)

        # Initialisation of the storing arrays
        mdot, ymgal, ymgal_massive, ymgal_agb, ymgal_1a, mdot_massive, mdot_agb, \
        mdot_1a, sn1a_numbers, sn2_numbers, imf_mass_ranges, \
        imf_mass_ranges_contribution, imf_mass_ranges_mtot = \
        self._get_storing_arrays(ymgal)

        # Initialisation of the composition of the gas reservoir
        if len(self.ism_ini) > 0:
            for i_ini in range(0,self.len_ymgal):
                ymgal[0][i_ini] = self.ism_ini[i_ini]

        # Output information
        if iolevel >= 1:
             print 'Number of timesteps: ', '{:.1E}'.format(len(timesteps))

        # Add the initialized arrays to the history class
        self.history.gas_mass.append(sum(ymgal[0]))
        self.history.ism_iso_yield.append(ymgal[0])
        self.history.ism_iso_yield_agb.append(ymgal_agb[0])
        self.history.ism_iso_yield_1a.append(ymgal_1a[0])
        self.history.ism_iso_yield_massive.append(ymgal_massive[0])
        self.history.sn1a_numbers.append(0)
        self.history.sn2_numbers.append(0)
        self.history.m_locked = []
        self.history.m_locked_agb = []
        self.history.m_locked_massive = []

        # Keep track of the mass-loss rate of massive stars
        self.massive_ej_rate = []
        for k in range(self.nb_timesteps + 1):
            self.massive_ej_rate.append(0.0)

        # Attribute arrays and variables to the current object
        self.mdot = mdot
        self.ymgal = ymgal
        self.ymgal_massive = ymgal_massive
        self.ymgal_agb = ymgal_agb
        self.ymgal_1a = ymgal_1a
        self.mdot_massive = mdot_massive
        self.mdot_agb = mdot_agb
        self.mdot_1a = mdot_1a
        self.sn1a_numbers = sn1a_numbers
        self.sn2_numbers = sn2_numbers
        self.imf_mass_ranges = imf_mass_ranges
        self.imf_mass_ranges_contribution = imf_mass_ranges_contribution
        self.imf_mass_ranges_mtot = imf_mass_ranges_mtot

        # Set the initial time and metallicity 
        zmetal = self._getmetallicity(0)
        self.history.metallicity.append(zmetal)
        self.t = 0
        self.history.age.append(self.t)
        self.zmetal = zmetal

        # Get coefficients for the fraction of white dwarfs fit (2nd poly)
        self.__get_coef_wd_fit()

        # Output information
        if iolevel > 0:
            print '### Start with initial metallicity of ','{:.4E}'.format(zmetal)
            print '###############################'


    ##############################################
    #                 Check Inputs               #
    ##############################################
    def __check_inputs(self):

        '''
        This function checks for incompatible input entries and stops
        the simulation if needed.

        '''
        
        import sys
        self.need_to_quit = False
        # Total duration of the simulation
        if self.history.tend > 1.5e10:
            print 'Error - tend must be less than or equal to 1.5e10 years.'
            self.need_to_quit = True

        # Timestep
        if self.history.dt > self.history.tend:
            print 'Error - dt must be smaller or equal to tend.'
            self.need_to_quit = True

        # Transition mass between AGB and massive stars 
        #if #(self.transitionmass <= 7)or(self.transitionmass > 12):
           # print 'Error - transitionmass must be between 7 and 12 Mo.'
           # self.need_to_quit = True
        if not self.transitionmass ==8:
            print 'Warning: Non-default transitionmass chosen. Use in agreement '\
            'with yield input!'

        # IMF
        if not self.imf_type in ['salpeter','chabrier','kroupa','input', \
            'alphaimf','chabrieralpha','fpp']:
            print 'Error - Selected imf_type is not available.'
            self.need_to_quit = True

        # IMF yields range
        #if self.imf_yields_range[0] < 1:
        #    print 'Error - imf_yields_range lower boundary must be >= 1.'
            #self.need_to_quit = True
        
        #if (self.imf_yields_range[0] >= self.imf_bdys[1]) or \
        #   (self.imf_yields_range[0] <= self.imf_bdys[0]) or \
         #  (self.imf_yields_range[1] >= self.imf_bdys[1]):
        if ((self.imf_yields_range[0] >  self.imf_bdys[1]) or \
            (self.imf_yields_range[1] < self.imf_bdys[0])):
            print 'Error - part of imf_yields_range must be within imf_bdys.'
            self.need_to_quit = True
        if (self.transitionmass<self.imf_yields_range[0])\
             or (self.transitionmass>self.imf_yields_range[1]):
            print 'Error - Transitionmass outside imf yield range'
            self.need_to_quit = True


        # SN Ia delay-time distribution function
        if not self.history.sn1a_rate in \
            ['exp','gauss','maoz','power_law']:
            print 'Error - Selected sn1a_rate is not available.'
            self.need_to_quit = True

        # Initial metallicity for the gas
        #if not self.iniZ in [0.0, 0.0001, 0.001, 0.006, 0.01, 0.02]:
        #    print 'Error - Selected iniZ is not available.'
        #    self.need_to_quit = True

        # If popIII stars are used ...
        if self.iniZ == 0.0:

            # IMF and yield boundary ranges
            if not self.imf_yields_range_pop3 == [10,30]:
                print 'Error - Selected imf_yields_range_pop3 not included yet.'
                self.need_to_quit = True
            if (self.imf_yields_range_pop3[0] >= self.imf_bdys_pop3[1]) or \
               (self.imf_yields_range_pop3[1] <= self.imf_bdys_pop3[0]):
                print 'Error - imf_yields_range_pop3 must be within \
                          imf_bdys_pop3.'
                self.need_to_quit = True
              
            if self.netyields_on == True:
                print 'Error - net yields setting not usable with PopIII at the moment.'
                self.need_to_quit = True

    ##############################################
    #                  Get Iniabu                #
    ##############################################
    def _get_iniabu(self):

        '''
        This function returns the initial gas reservoir, ymgal, containing
        the mass of all the isotopes considered by the stellar yields.  

        '''

        # Zero metallicity gas reservoir
        if self.iniZ == 0:

            # If an input iniabu table is provided ...
            if len(self.iniabu_table) > 0:
                iniabu=ry.iniabu(global_path + self.iniabu_table)
                if self.iolevel >0:
                    print 'Use initial abundance of ', self.iniabu_table
                ymgal_gi = np.array(iniabu.iso_abundance(self.history.isotopes)) * \
                       self.mgal

            else:

                # Get the primordial composition of Walker et al. (1991)
	        iniabu_table = 'yield_tables/bb_walker91.txt'
                ytables_bb = ry.read_yield_sn1a_tables( \
                    global_path+iniabu_table, self.history.isotopes)

                # Assign the composition to the gas reservoir
                ymgal_gi = ytables_bb.get(quantity='Yields') * self.mgal

                # Output information
                if self.iolevel > 0:
                    print 'Use initial abundance of ', iniabu_table

        # Already enriched gas reservoir
        else:

            # If an input iniabu table is provided ...
            if len(self.iniabu_table) > 0:
                iniabu=ry.iniabu(global_path + self.iniabu_table)
                if self.iolevel > 0:
                    print 'Use initial abundance of ', self.iniabu_table

            # If NuGrid's yields are used ...
            #else self.default_yields:
            else:

                # Define all the Z and abundance input files considered by NuGrid
                ini_Z = [0.01, 0.001, 0.0001, 0.02, 0.006, 0.00001, 0.000001]
                ini_list = ['iniab1.0E-02GN93.ppn', 'iniab1.0E-03GN93_alpha.ppn', \
                            'iniab1.0E-04GN93_alpha.ppn', 'iniab2.0E-02GN93.ppn', \
                            'iniab6.0E-03GN93_alpha.ppn', \
                            'iniab1.0E-05GN93_alpha_scaled.ppn', \
                            'iniab1.0E-06GN93_alpha_scaled.ppn']

                # Pick the composition associated to the input iniZ
                for metal in ini_Z:
                    if metal == float(self.iniZ):
                        iniabu = ry.iniabu(global_path + \
                        'yield_tables/iniabu/' + ini_list[ini_Z.index(metal)])
                        if self.iolevel>0:
                            print 'Use initial abundance of ', \
                            ini_list[ini_Z.index(metal)]
                        break

            # Input file for the initial composition ...
            #else:
            #    iniabu=ry.iniabu(global_path + iniabu_table)
            #    print 'Use initial abundance of ', iniabu_table

            # Assign the composition to the gas reservoir
            ymgal_gi = np.array(iniabu.iso_abundance(self.history.isotopes)) * \
                       self.mgal

        # Make sure the total mass of gas does not exceed mgal
        if sum(ymgal_gi) > self.mgal:
            ymgal_gi[0] = ymgal_gi[0] - (sum(ymgal_gi) - self.mgal)

        # Return the gas reservoir
        return ymgal_gi


    ##############################################
    #                Get Timesteps               #
    ##############################################
    def __get_timesteps(self):

        '''
        This function calculates and returns the duration of every timestep.  

        '''

        # Declaration of the array containing the timesteps
        timesteps_gt = []

        # If the timesteps are given as an input ...
        if len(self.dt_in) > 0:

            # Copy the timesteps
            timesteps_gt = self.dt_in

        # If the timesteps need to be calculated ...
        else:

            # If all the timesteps have the same duration ...
            if self.special_timesteps <= 0:

                # Make sure the last timestep is equal to tend
                counter = 0
                step = 1
                laststep = False
                t = 0
                t0 = 0
                while(True):
                    counter+=step
                    if (self.history.tend/self.history.dt)==0:
                        if (self.history.dt*counter)>self.history.tend:
                            break
                    else:
                        if laststep==True:
                            break
                        if (self.history.dt*counter+step)>self.history.tend:
                            counter=(self.history.tend/self.history.dt)
                            laststep=True
                    t=counter
                    timesteps_gt.append(int(t-t0)*self.history.dt)
                    t0=t

            # If the special timestep option is chosen ...
            if self.special_timesteps > 0:

                # Use a logarithm scheme
                times1 = np.logspace(np.log10(self.history.dt), \
                         np.log10(self.history.tend), self.special_timesteps)
                times1 = [0] + list(times1)
                timesteps_gt = np.array(times1[1:]) - np.array(times1[:-1])

        # If a timestep needs to be added to be synchronized with
        # the external program managing merger trees ...
        if self.t_merge > 0.0:

            # Declare the new timestep array
            timesteps_new = []

            # Find the interval where the step needs to be added
            i_temp = 0
            t_temp = timesteps_gt[0]
            while t_temp < self.t_merge:
                timesteps_new.append(timesteps_gt[i_temp])
                i_temp += 1
                t_temp += timesteps_gt[i_temp]

            # Add the extra timestep
            dt_up_temp = t_temp - self.t_merge
            dt_low_temp = timesteps_gt[i_temp] - dt_up_temp
            timesteps_new.append(dt_low_temp)
            timesteps_new.append(dt_up_temp)

            # Keep the t_merger index in memory
            self.i_t_merger = i_temp

            # Add the rest of the timesteps
            # Skip the current one that just has been split
            for i_dtnew in range(i_temp+1,len(timesteps_gt)):
                timesteps_new.append(timesteps_gt[i_dtnew])

            # Replace the timesteps array to be returned
            timesteps_gt = timesteps_new

        # Return the duration of all timesteps
        return timesteps_gt


    ##############################################
    #             Get Storing Arrays             #
    ##############################################
    def _get_storing_arrays(self, ymgal):

        '''
        This function declares and returns all the arrays containing information
        about the evolution of the stellar ejecta, the gas reservoir, the star
        formation rate, and the number of core-collapse SNe, SNe Ia, and white
        dwarfs.

        Argument
        ========

          ymgal : Initial gas reservoir. This function extends it to all timesteps

        '''

        # Number of timesteps and isotopes
        nb_dt_gsa = self.nb_timesteps
        nb_iso_gsa = len(self.history.isotopes)

        # Stellar ejecta
        mdot = []
        for k in range(nb_dt_gsa):
            mdot.append(nb_iso_gsa * [0])

        # Gas reservoir
        ymgal = [list(ymgal)]
        for k in range(nb_dt_gsa):
            ymgal.append(nb_iso_gsa * [0])

        # Massive stars, AGB stars, and SNe Ia ejecta
        ymgal_massive = []
        ymgal_agb = []
        ymgal_1a = []
        for k in range(nb_dt_gsa + 1):
            ymgal_massive.append(nb_iso_gsa * [0])
            ymgal_agb.append(nb_iso_gsa * [0])
            ymgal_1a.append(nb_iso_gsa * [0])
        mdot_massive = copy.deepcopy(mdot)
        mdot_agb     = copy.deepcopy(mdot)
        mdot_1a      = copy.deepcopy(mdot)

        # Number of SNe Ia and core-collapse SNe
        sn1a_numbers = [0] * nb_dt_gsa
        sn2_numbers  = [0] * nb_dt_gsa
        self.wd_sn1a_range  = [0] * nb_dt_gsa
        self.wd_sn1a_range1 = [0] * nb_dt_gsa

        # Star formation
        self.number_stars_born = [0] * (nb_dt_gsa + 1)

        # Related to the IMF
        self.history.imf_mass_ranges = [[]] * (nb_dt_gsa + 1)
        imf_mass_ranges = []
        imf_mass_ranges_contribution = [[]] * (nb_dt_gsa + 1)
        imf_mass_ranges_mtot = [[]] * (nb_dt_gsa + 1)

        # Return all the arrays
        return mdot, ymgal, ymgal_massive, ymgal_agb, ymgal_1a, mdot_massive, \
               mdot_agb, mdot_1a, sn1a_numbers, sn2_numbers, imf_mass_ranges, \
               imf_mass_ranges_contribution, imf_mass_ranges_mtot


    ##############################################
    #               Get Coef WD Fit              #
    ##############################################
    def __get_coef_wd_fit(self):

        '''
        This function calculates the coefficients for the fraction of white
        dwarfs fit in the form of f_wd = a*lg(t)**2 + b*lg(t) + c.  Only 
        progenitor stars for SNe Ia are considered. 

        '''

        # Get the number of masses per metallicity in the grid
        nb_m_per_z = len(self.ytables.table_mz) / \
                     len(self.ytables.metallicities)

        # Extract the masses of each metallicity
        m_per_z = []
        for i_gcwf in range(0,nb_m_per_z):
            m_temp = re.findall("\d+.\d+", \
                     self.ytables.table_mz[i_gcwf])
            m_per_z.append(float(m_temp[0]))

        # Create the complete M-axis in a 1-D array
        m_complete = []
        for i_gcwf in range(0,len(self.ytables.metallicities)):
            m_complete += m_per_z

        # Only consider stars between 3 and 8 Mo
        lg_m_fit = []
        lg_t_fit = []
        for i_gcwf in range(0,len(m_complete)):
            if m_complete[i_gcwf] >= 3.0 and m_complete[i_gcwf] <= 8.0:
                lg_m_fit.append(np.log10(m_complete[i_gcwf]))
                lg_t_fit.append(np.log10(self.ytables.age[i_gcwf]))

        # Create fit lgt = a*lgM**2 + b*lgM + c
        a_fit, b_fit, c_fit = polyfit(lg_m_fit, lg_t_fit, 2)

        # Array of lifetimes
        t_f_wd = []
        m_f_wd = []
        t_max_f_wd = 10**(a_fit*0.47712**2 + b_fit*0.47712 + c_fit)
        t_min_f_wd = 10**(a_fit*0.90309**2 + b_fit*0.90309 + c_fit)
        self.t_3_0 = t_max_f_wd
        self.t_8_0 = t_min_f_wd
        nb_m = 15
        dm_wd = (8.0 - 3.0) / nb_m
        m_temp = 3.0
        for i_gcwf in range(0,nb_m):
            m_f_wd.append(m_temp)
            t_f_wd.append(10**(a_fit*np.log10(m_temp)**2 + \
                               b_fit*np.log10(m_temp) + c_fit))
            m_temp += dm_wd

        # Calculate the total number of progenitor stars
        n_tot_prog_inv = 1 / self._imf(3.0,8.0,1)

        # For each lifetime ...
        f_wd = []
        for i_gcwf in range(0,len(t_f_wd)):

            # Calculate the fraction of white dwarfs
            f_wd.append(self._imf(m_f_wd[i_gcwf],8.0,1)*n_tot_prog_inv)

        # Calculate the coefficients for the fit f_wd vs t
        self.a_wd, self.b_wd, self.c_wd, self.d_wd = \
            polyfit(t_f_wd, f_wd, 3)


    ##############################################
    #                  Evol Stars                #
    ##############################################
    def _evol_stars(self, i):

        '''
        This function executes a part of a single timestep with the simulation
        managed by either OMEGA or SYGMA.  It converts gas into stars, calculates
        the stellar ejecta of the new simple stellar population (if any), and adds
        its contribution to the total ejecta coming from all stellar populations.

        Argument
        ========

          i : Index of the current timestep

        '''

        # Update the time of the simulation.  Here, i is in fact the end point
        # of the current timestep which extends from i-1 to i.
        self.t += self.history.timesteps[i-1]

        # Output information
        if self.iolevel >= 1:
            tenth = self.nb_timesteps / (10.)
            if i%tenth == 0:
                print 'time and metallicity and total mass:'
                print '{:.3E}'.format(self.t),'{:.4E}'.format(self.zmetal), \
                      '{:.4E}'.format(sum(self.ymgal[i-1]))
                print 'time and metallicity and total mass:'
                print '{:.3E}'.format(self.t),'{:.4E}'.format(self.zmetal), \
                      '{:.4E}'.format(sum(self.ymgal[i-1]))

        # Initialisation of the mass locked into stars
        self.m_locked = 0
        self.m_locked_agb = 0
        self.m_locked_massive = 0

        # If stars are forming during the current timestep ...
        if self.sfrin > 0:

            # Limit the SFR if there is not enough gas
            if self.sfrin > 1.0:
                print 'Warning -- Not enough gas to sustain the SFH.', i
                self.sfrin = 1.0
                self.not_enough_gas = True
                self.not_enough_gas_count += 1

            # Output information
            if self.iolevel >= 1:
                print '################## Star formation at ', \
                      '{:.3E}'.format(self.t), \
                      '(Z='+'{:.4E}'.format(self.zmetal)+') of ',self.sfrin

            # Lock gas into stars
            f_lock = 1.0e0 - self.sfrin
            for k in range(self.len_ymgal):
                self.ymgal[i][k] = f_lock * self.ymgal[i-1][k]
                self.ymgal_agb[i][k] = f_lock * self.ymgal_agb[i-1][k]
                self.ymgal_1a[i][k] = f_lock * self.ymgal_1a[i-1][k]
                self.ymgal_massive[i][k] = f_lock * self.ymgal_massive[i-1][k]
                self.m_locked += self.sfrin * self.ymgal[i-1][k]

            # Output information
            if self.iolevel >= 1:
                print 'Mass locked away:','{:.3E}'.format(self.m_locked), \
                      ', new ISM mass:','{:.3E}'.format(sum(self.ymgal[i]))

            # Calculate stellar ejecta and the number of SNe
            self.__sfrmdot(i)

        # If no star is forming during the current timestep ...
        else:

            # Use the previous gas reservoir for the current timestep
            self.ymgal[i] = self.ymgal[i-1]
            self.ymgal_agb[i] = self.ymgal_agb[i-1]
            self.ymgal_1a[i] = self.ymgal_1a[i-1]
            self.ymgal_massive[i] = self.ymgal_massive[i-1]

            # Output information
            if self.iolevel > 2:
                print 'Finished getting mdot',self.gettime()
		
        # Add stellar ejecta to the gas reservoir
        self.ymgal[i] = np.array(self.ymgal[i]) + np.array(self.mdot[i-1])
        self.ymgal_agb[i] = np.array(self.ymgal_agb[i])+np.array(self.mdot_agb[i-1])
        self.ymgal_1a[i] = np.array(self.ymgal_1a[i]) + np.array(self.mdot_1a[i-1])
        self.ymgal_massive[i] = np.array(self.ymgal_massive[i]) + \
                                np.array(self.mdot_massive[i-1])

        # Convert the mass ejected by massive stars into rate
        if self.history.timesteps[i-1] == 0.0:
            self.massive_ej_rate[i-1] = 0.0
        else:
            self.massive_ej_rate[i-1] = sum(self.mdot_massive[i-1]) / \
                self.history.timesteps[i-1]


    ##############################################
    #                Update History              #
    ##############################################
    def _update_history(self, i):

        '''
        This function adds the state of current timestep into the history class.

        Argument
        ========

          i : Index of the current timestep

        Note
        ====

          This function is decoupled from evol_stars() because OMEGA modifies
          the quantities between evol_stars() and the update of the history class.

        '''

        # Keep the current in memory
        self.history.metallicity.append(self.zmetal)
        self.history.age.append(self.t)
        self.history.gas_mass.append(sum(self.ymgal[i]))
        self.history.ism_iso_yield.append(self.ymgal[i])
        self.history.ism_iso_yield_agb.append(self.ymgal_agb[i])
        self.history.ism_iso_yield_1a.append(self.ymgal_1a[i])
        self.history.ism_iso_yield_massive.append(self.ymgal_massive[i])
        self.history.sn1a_numbers.append(self.sn1a_numbers[i-1])
        self.history.sn2_numbers.append(self.sn2_numbers[i-1])
        self.history.m_locked.append(self.m_locked)	
        self.history.m_locked_agb.append(self.m_locked_agb)
        self.history.m_locked_massive.append(self.m_locked_massive)


    ##############################################
    #             Update History Final           #
    ##############################################
    def _update_history_final(self):

        '''
        This function adds the total stellar ejecta to the history class as well
        as converting isotopes into chemical elements.

        '''

        # Fill the last bits of the history class
        self.history.mdot = self.mdot
        self.history.imf_mass_ranges_contribution=self.imf_mass_ranges_contribution
        self.history.imf_mass_ranges_mtot = self.imf_mass_ranges_mtot

        # Convert isotopes into elements
        for h in range(len(self.history.ism_iso_yield)):
            conv = self._iso_abu_to_elem(self.history.ism_iso_yield[h])
            self.history.ism_elem_yield.append(conv[1])
            conv = self._iso_abu_to_elem(self.history.ism_iso_yield_agb[h])
            self.history.ism_elem_yield_agb.append(conv[1])
            conv = self._iso_abu_to_elem(self.history.ism_iso_yield_1a[h])
            self.history.ism_elem_yield_1a.append(conv[1])
            conv = self._iso_abu_to_elem(self.history.ism_iso_yield_massive[h])
            self.history.ism_elem_yield_massive.append(conv[1])

            # List of all the elements
            if h == 0:
               self.history.elements = conv[0]


    ##############################################
    #              Set Yield Tables              #
    ##############################################
    def __set_yield_tables(self): 

	'''
        This function sets the variables associated with the yield tables
        and the stellar lifetimes used to calculate the evolution of stars.

	'''

        # Set if the yields are the default ones or not
        if not int(self.iniZ) == int(-1):
            default_yields = True
            self.default_yields = True
        else:
            default_yields = False
            self.default_yields = False

        # Read stellar yields
        ytables = ry.read_nugrid_yields(global_path + self.table,excludemass=self.exclude_masses)
        self.ytables = ytables

        # Interpolate stellar lifetimes
        self.zm_lifetime_grid_nugrid = self.__interpolate_lifetimes_grid(ytables)

        # Get the isotopes considered by NuGrid
        mtest=float(ytables.table_mz[0].split(',')[0].split('=')[1])
        ztest=float(ytables.table_mz[0].split(',')[1].split('=')[1][:-1])
        isotopes = ytables.get(mtest,ztest,'Isotopes')
        self.history.isotopes = isotopes
        self.nb_isotopes = len(self.history.isotopes)

        # Read PopIII stars yields - Heger et al. (2010)
        self.ytables_pop3 = ry.read_nugrid_yields( \
            global_path+self.pop3_table,isotopes,excludemass=self.exclude_masses)

        # Check compatibility between PopIII stars and NuGrid yields
        isotopes_pop3 = self.ytables_pop3.get(10,0.0,'Isotopes')
        if not isotopes_pop3 == isotopes:
            print 'Warning - Network not set correctly.'

        # Interpolate PopIII stellar lifetimes
        self.zm_lifetime_grid_pop3 = \
            self.__interpolate_lifetimes_pop3(self.ytables_pop3)

        # Read SN Ia yields
        import sys
        sys.stdout.flush()
        self.ytables_1a = ry.read_yield_sn1a_tables( \
            global_path + self.sn1a_table, isotopes)

        # Should be modified to include extra source for the yields
        self.extra_source_on = False
        self.ytables_extra = 0
        if self.extra_source_on == True:
            self.ytables_extra = ry.read_yield_sn1a_tables( \
                global_path + extra_source_table, isotopes)

        # Output information
        if self.iolevel >= 1:
            print 'Warning - Use isotopes with care.'
            print isotopes


    ##############################################
    #                  SFR Mdot                  #
    ##############################################
    def __sfrmdot(self, i):

        '''
        This function calculates the stellar yields per isotope from the
        stars recently formed. These yields are distributed in the mdot
        array to progressively release the ejecta with time (in the upcoming
        timesteps), using stellar lifetimes.

        Argument
        ========

          i : Index of the current timestep

        '''

        # Output information
        if self.iolevel >= 2:
            print 'Entering sfrmdot routine'
            print 'Getting yields',self.gettime()

        # Interpolate yields in the mass-metallicity plane
        mstars, yields_all, func_total_ejecta, yields_extra = \
            self.__inter_mm_plane(i)

        # Get the mass fraction of stars that contribute to the stellar ejecta
        # relative to the total stellar mass within imf_bdys.
        mfac = self.__get_mfac()

        # Get the mass boundaries on which each star yields will be applied.
        # Ex: 12 Mo model yields might be used for all stars from 8 to 12.5 Mo.
        mass_bdys,mstars = self.__get_mass_bdys(mstars)
      
        # Apply the IMF on the mass boundaries
        mass_bdys, mstars, yields_all, massfac, mftot = \
            self.__apply_imf_mass_bdys( mass_bdys, mstars, yields_all )

        # Output information
        if self.iolevel >= 1:
            print 'Stars under consideration', \
                  '(take into account user-selected imf ends):'
	    for kk in range(len(mass_bdys)-1):
	    	print mass_bdys[kk],'|',mstars[kk],'|',mass_bdys[kk+1]
	    print 'lens: ',len(mass_bdys),len(mstars)
            #print mstars
            #print 'with (corrected) mass bdys:####'
            #print mass_bdys
            #print 'Scaling factor for gas mass (mfac)', mfac

        # Calculate the mass locked away from the old ejecta of massive and AGB stars
        massfac = self.__mlock_agb_massive(mstars,mfac,mftot,massfac,mass_bdys,i)

        # Save the contribution of each mass range interval
        self.imf_mass_ranges_contribution[i] = \
            [len(self.ymgal[0])*[0]] * (len(mass_bdys)-1)
        imf_mass_ranges=[]
        for k in range(len(mass_bdys)-1):
            imf_mass_ranges.append([mass_bdys[k], mass_bdys[k+1]])
        self.history.imf_mass_ranges[i] = imf_mass_ranges
        self.imf_mass_ranges = imf_mass_ranges

        # Calculate ejecta from stars recently formed and add it to the mdot arrays
        self.__calculate_ejecta(mstars, yields_all, yields_extra, \
            mass_bdys,massfac, i, func_total_ejecta)

        # Add the contribution of SNe Ia, if any ...
        if (self.sn1a_on == True) and (self.zmetal > 0 or self.hardsetZ > 0):
            if not (self.imf_bdys[0] > 8 or self.imf_bdys[1] < 3):
                self.__sn1a_contribution(i)


    ##############################################
    #               Inter MM Plane               #
    ##############################################
    def __inter_mm_plane(self, i):

        '''
        This function returns the appropriate interpolated yields for this timestep.

        Argument
        ========

          i : Index of the current timestep

        '''

        # If Zmetal is above the lowest non-zero Z available in the yields ...
        if self.zmetal >= self.ytables.metallicities[-1]:

            # Get the interpolated stellar yields for the current metallicity.
            mstars, yields_all, func_total_ejecta = self.__interpolate_yields( \
                self.zmetal, self.ymgal[i-1], self.ytables)

            # Copy the stellar lifetimes
            self.zm_lifetime_grid_current = self.zm_lifetime_grid_nugrid

        # If Z_trans < Zmetal < the lowest available Z
        elif self.zmetal > self.Z_trans:

            # Use the yields with the lowest metallicity
            mstars, yields_all, func_total_ejecta = self.__interpolate_yields( \
                self.ytables.metallicities[-1], self.ymgal[i-1], self.ytables)

            # Copy the stellar lifetimes
            self.zm_lifetime_grid_current = self.zm_lifetime_grid_nugrid

        # If Zmetal < Z_trans
        else:

            # Output information
            if self.iolevel > 0:
                print 'Taking POPIII yields from yield_tables/popIII_heger10.txt'
                print 'Currently only PopIII massive stars between 10Msun and', \
                      '100Msun with their yield contributions included'
                print ' lower IMF minimum (imf_bdys) fixed to 10Msun for now'

            # Use PopIII stars yields
            mstars, yields_all, func_total_ejecta = \
                self.__interpolate_yields_pop3(0.0,self.ymgal[i-1], self.ytables)

            # Copy PopIII stars lifetimes
            self.zm_lifetime_grid_current = self.zm_lifetime_grid_pop3

        # In case of extra source ...
        # In case an extra yield source (e.g. r process) is considered
        # Use new interpolation method here as well?
        yields_extra = []
        if self.default_yields == False:
            if self.extra_source_on:
                tables_Z = self.ytables_extra.metallicities
                for tz in tables_Z:
                    if self.zmetal > tz:
                        yields_extra = \
                        self.ytables_extra.get(Z=tz, quantity='Yields')
                        break
                    if self.zmetal <= tables_Z[-1]:
                        yields_extra = \
                        self.ytables_extra.get(Z=tables_Z[-1],quantity='Yields')
                        break

        # Output information
        if self.iolevel > 2:
            print 'yields retrieved', self.gettime()
	if self.iolevel > 1:
            for k in range(len(mstars)):
                print 'test yield input in sfrmdot'
                print 'test Mstars:',mstars[k]
                for h in range(len(yields_all[k])):
                    if yields_all[k][h] < 0:
                        print 'yields', yields_all[k][h],'of isotope',isotope[h]

        # Return the results of the interpolation
        return mstars, yields_all, func_total_ejecta, yields_extra


    ##############################################
    #                  Get Mfac                  #
    ##############################################
    def __get_mfac(self):

        '''
        This function calculates and returns the mass fraction of stars
        contributing to the stellar ejecta (yield tables) relative to
        the total mass of a stellar population weighted by the IMF having
        lower and upper mass limits set by imf_bdys.

        Example : If only stars in imf_yields_range contribute to the ejecta, 
                  the rest is just matter locked away --> mfac is scaling factor.

        '''

        # Select the appropriate IMF mass boundary and yield range
        if (self.zmetal > self.Z_trans) or (self.hardsetZ > 0.0): 
            bd_gm = self.imf_bdys
            yr_gm = self.imf_yields_range
        else:
            bd_gm = self.imf_bdys_pop3
            yr_gm = self.imf_yields_range_pop3

        # If the IMF mass boundary includes the yield range ...
        if bd_gm[1] >= yr_gm[1] and bd_gm[0] <= yr_gm[0]:
            mfac = (self._imf(yr_gm[0], yr_gm[1], 2)) / \
                   (self._imf(bd_gm[0], bd_gm[1], 2))

        # If the IMF mass boundary overlaps the yield range but its lower mass
        # limit is not low enough to cover the entire yield range ...
        if ( bd_gm[1] >= yr_gm[1] and (bd_gm[0]>yr_gm[0] and bd_gm[0]<yr_gm[1])): 
            mfac = (self._imf(bd_gm[0], yr_gm[1], 2)) / \
                   (self._imf(bd_gm[0], bd_gm[1], 2))

        # If the IMF mass boundary overlaps the yield range but its upper mass
        # limit is not high enough to cover the entire yield range ...
        if (bd_gm[0] <= yr_gm[0] and (bd_gm[1]<yr_gm[1] and bd_gm[1]>yr_gm[0])):
            mfac = (self._imf(yr_gm[0], bd_gm[1], 2)) / \
                   (self._imf(bd_gm[0], bd_gm[1], 2))

        # If the yield range includes the IMF mass boundary ...
        if bd_gm[1] < yr_gm[1] and bd_gm[0] > yr_gm[0]:
            mfac = 1.0

        # Return the mass fraction
        return mfac


    ##############################################
    #               Get Mass Bdys                #
    ##############################################
    def __get_mass_bdys(self, mstars):

        '''
        This function returns the array mass_bdys containing the initial stellar
        masses where the code switches from the yields of a certain stellar model
        to another.  The yields of each stellar model available in the input tables
        are applied to all the stars within a certain mass boundary.  The
        array mass_bdys represents those mass boundaries.

        This function also adds the transition mass in mass_bdys.  This must be
        done in order to force the code to have a fixed transition between AGB stars
        and massive stars.

        '''
        # Select the apropriate yield mass range
	if (self.zmetal > self.Z_trans) or (self.hardsetZ > 0):
            yr_gmb = self.imf_yields_range
        else:
            yr_gmb = self.imf_yields_range_pop3
        # Lowest stellar mass contributing to the ejecta
        mass_bdys = [yr_gmb[0]]

        m_stars=[]
        # For every stellar initial mass already in the old mass array ...
        for w in range(len(mstars)):
            #print w,mass_bdys
            # Add the transition mass if we are at the right mass
            if (mstars[w-1] < self.transitionmass) and \
                   (mstars[w] > self.transitionmass):
                mass_bdys.append(self.transitionmass)
                if (w == len(mstars) - 1):
                    mass_bdys.append(yr_gmb[1]) 
		m_stars.append(mstars[w-1])
                continue

            # Calculate the mass where the code switches to another star yield. 
            if (w>0) and (w < len(mstars) - 1):
                #inside imf interval
                newbdy=(mstars[w-1] + mstars[w]) / 2.e0
                if (newbdy > yr_gmb[0]) and (newbdy<yr_gmb[1]):
                    mass_bdys.append(newbdy)
                    if not mstars[w-1] in m_stars:
                        m_stars.append(mstars[w-1])
		if (newbdy > yr_gmb[0]) and (newbdy>=yr_gmb[1]):
                    if not mstars[w-1] in m_stars:
                        m_stars.append(mstars[w-1])		    
		    mass_bdys.append(yr_gmb[1])
		    break
                #print newbdy,yr_gmb[0]
            if mstars[w] == yr_gmb[0]:
                    if not mstars[w] in m_stars:
                        m_stars.append(mstars[w])
	    '''
      	    if mstars[w] == yr_gmb[1]:
		    newbdy=(mstars[w-1] + mstars[w]) / 2.e0
                    if not mstars[w] in m_stars:
                        m_stars.append(mstars[w])
      		    #mass_bdys.append(newbdy)
		    mass_bdys.append(yr_gmb[1])
		    break
	    '''
            # Highest stellar masses contributing to the ejecta 
            if ((w == len(mstars) - 1) or mstars[w] == yr_gmb[1]):
                newbdy=(mstars[w-1] + mstars[w]) / 2.e0
                oldnewbdy=(mstars[w-2] + mstars[w-1]) / 2.e0
                if newbdy < yr_gmb[1]:
		    if not newbdy in mass_bdys:
                    	mass_bdys.append(newbdy)
		    if not mstars[w-1] in m_stars:
                    	m_stars.append(mstars[w-1])
		    if not mstars[w] in m_stars: 
		    #if not mstars[w] == yr_gmb[1]:
                    	m_stars.append(mstars[w])
                elif oldnewbdy<yr_gmb[1]:
		    if not mstars[w-1] in m_stars:
                    	m_stars.append(mstars[w-1])
                if not newbdy == yr_gmb[1]:
		    if not yr_gmb[1] in mass_bdys:
                    	mass_bdys.append(yr_gmb[1])
		#if mstars[w] == yr_gmb[1]:
        	break
	     

        # Output information
        if self.iolevel >= 1:
            print '__get_mass_bdys: mass_bdys: ',mass_bdys
            print '__get_mass_bdys: m_stars',m_stars

        # Return the array containing the boundary masses for applying yields
        return mass_bdys,m_stars


    ##############################################
    #            Apply IMF Mass Bdys             #
    ##############################################
    def __apply_imf_mass_bdys(self, mass_bdys, mstars, yields_all):

        '''
        This function applies the IMF to the yields mass boundaries to know how
        many stars and how much mass is in each bin.  That information will be
        used to scale the specific yields used in each mass bin.

        Arguments
        =========

          mass_bdys : Stellar mass boundaries where yields are applied.
          mstars : Initial mass of stellar models available in the input yields.
          yields_all : Stellar yields available from the input.

        '''

        # Copy of the IMF boundary according to the current gas metallicity
        if (self.zmetal > self.Z_trans) or (self.hardsetZ > 0.0):
            imf_bd = self.imf_bdys
        else:
            imf_bd = self.imf_bdys_pop3

        # Declaration of local variables
        mftot = 0.0      # Total mass of the yield range limited by the IMF
        massfac = []     # Number of stars in each mass bin (weighted by the IMF)
        mstars1 = []     # Copy of mstars but limited by the IMF
        yields_all1 = [] # Copy of yields_all but limited by the IMF
        mass_bdys1 = []  # Copy of mass_bdys but limited by the IMF

        # For each stellar mass bin ...
        for w in range(len(mstars)):

            # If the mass bin is outside of the IMF mass range ...
            if (mass_bdys[1+w] <= imf_bd[0]) or (mass_bdys[w] >= imf_bd[1]):

                # Skip this bin
                continue

            # If the lower mass limit of the IMF is inside the mass bin ...
            if imf_bd[0] >= mass_bdys[w] and imf_bd[0] <= mass_bdys[w+1]:

                # Use the lower mass limit of the IMF for the bin lower limit
                massfac.append(self._imf(imf_bd[0], mass_bdys[w+1], 1))
                mftot += (self._imf(imf_bd[0], mass_bdys[w+1], 2))
                mass_bdys1.append(imf_bd[0])

            # If the upper mass limit of the IMF is inside the mass bin ...
            elif mass_bdys[w] <= imf_bd[1] and mass_bdys[w+1] >= imf_bd[1]:

                # Use the upper mass limit of the IMF for the bin upper limit
                massfac.append( self._imf(mass_bdys[w], imf_bd[1], 1) )
                mftot += (self._imf(mass_bdys[w],imf_bd[1], 2))
                mass_bdys1.append(mass_bdys[w])
                mass_bdys1.append(imf_bd[1])

            # If the whole mass bin is within the IMF mass boundary ...
            else:

                # Keep the provided mass boundary for this bin
                massfac.append(self._imf(mass_bdys[w], mass_bdys[w+1], 1))
                mftot += (self._imf(mass_bdys[w], mass_bdys[w+1], 2))
                mass_bdys1.append(mass_bdys[w])
                if (w == len(mstars)-1) and imf_bd[1] > mass_bdys[w+1]:
                    mass_bdys1.append(mass_bdys[w+1])

            # Add the mass of the stellar model providing yields if inside the IMF
            mstars1.append(mstars[w])
            yields_all1.append(yields_all[w])
        
        # Update the arrays now corrected for the IMF
        mass_bdys = mass_bdys1
        mstars = mstars1
        yields_all = yields_all1

        # Keep the mass boundaries in memory
        self.history.mass_bdys = mass_bdys

        # Return the IMF corrected arrays
        return mass_bdys, mstars, yields_all, massfac, mftot


    ##############################################
    #              Mlock AGB Massive             #
    ##############################################
    def __mlock_agb_massive(self, mstars, mfac, mftot, massfac, mass_bdys, i):

        '''
        This function calculates the mass of massive and AGB ejecta locked
        away by the star formation.

        Arguments
        =========

          mstars : Initial mass of stellar models available in the input yields.
          mfac : Mass fraction of stars that contribute to the stellar ejecta
                 relative to the total stellar mass within imf_bdys.
          mftot : Total mass of the yield range limited by the IMF
          massfac :  Number of stars in each mass bin (weighted by the IMF)
          mass_bdys : Stellar mass boundaries where yields are applied.
          i : Index of the current timestep

        '''

        # Testing variable
        #mtest = 0.0

        # For each mass bin ...
        for w in range(len(mstars)):

            # Calculate the mass of massive or AGB ejecta locked away
            # !! Still assumption all stars in interval same mass !!
            massfac[w] = self.m_locked * mfac / mftot * massfac[w]  
	    mtot_massfac = self.m_locked * mfac / mftot * \
                                    self._imf(mass_bdys[w], mass_bdys[w+1], 2)
	    self.imf_mass_ranges_mtot[i].append(mtot_massfac)

            #mtest=mtest+massfac[w]*self._imf(mass_bdys[w],mass_bdys[w+1],2)
	    #mtest=mtest+mtot_massfac

            # Add locked mass to the appropriate ejecta
            if mstars[w] > self.transitionmass:
                self.m_locked_massive += mtot_massfac 
            else:
                self.m_locked_agb += mtot_massfac

        # Output information
        if self.iolevel >= 1:
            print 'Total mass of the gas in stars:'
            #print '{:.3E}'.format(mtest)
            print 'AGB: ','{:.3E}'.format(self.m_locked_agb)
            print 'Massive: ','{:.3E}'.format(self.m_locked_massive)

        # Return the modified massfac array
        return massfac


    ##############################################
    #             Calculate Ejecta               #
    ##############################################
    def __calculate_ejecta(self, mstars, yields_all, yields_extra, mass_bdys, \
                           massfac, i, func_total_ejecta):

        '''
        This function calculates the ejecta coming from the stars that are forming
        during the current timestep.  The ejecta is then distributed over time, in
        the future upcoming timesteps, according to the stellar lifetimes.

        Arguments
        =========

          mstars : Initial mass of stellar models available in the input yields.
          yields_all : Stellar yields available from the input.
          yields_extra : Stellar yields from extra source.
          mass_bdys : Stellar mass boundaries where yields are applied.
          massfac :  Number of stars in each mass bin (weighted by the IMF)
          i : Index of the current timestep

        '''

        # Variables used for testing
        count_numbers_c = 0
        count_numbers_f = 0

        # For every star having yields, from massive down to low-mass stars ...
        for w in range(len(mstars))[::-1]:

            # Copy the desired yields and mass boundary
            yields = yields_all[w]
            maxm = mass_bdys[w+1]
            minm = mass_bdys[w]
	    
            # Get stellar lifetimes associated to mass bin lower and upper limits
            lifetimemin = self.__find_lifetimes(round(self.zmetal,6), maxm)
            lifetimemax = self.__find_lifetimes(round(self.zmetal,6), minm)

            # Output information
            if self.iolevel >= 2:
                lifetime_gridmass = \
                    self.__find_lifetimes(round(self.zmetal,6), mstars[w])
                print '###############################################'
                print 'Mass range: Mass and lifetime for ', mstars[w], 'Msun:', \
                      '(table age: ','{:.3E}'.format(lifetime_gridmass),'):'
                print maxm,'{:.3E}'.format(lifetimemin)
                print minm,'{:.3E}'.format(lifetimemax)

            # Move to the next bin if the lifetime is longer than the simulation
            if sum(self.history.timesteps) < lifetimemin:
                if self.iolevel >= 2:
                    print 'Ejection of ', mstars[w], ' mass range at ', \
                          '{:.3E}'.format(lifetimemin),'excluded due to timelimit'
                continue

            # Variable used for testing
            count_numbers_c += massfac[w]

            # Get imf (number) coefficient in mass interval
            # This is the normalisation constant of the IMF with Mtot = m_locked
            p_number = massfac[w] / (self._imf(minm, maxm, 1))

            # Keep the contribution of this mass bin in memory
            for k in range(len(self.imf_mass_ranges_contribution[i])):
                r = self.imf_mass_ranges[k]
                if r[0] >= minm and r[1] <= maxm:
                    self.imf_mass_ranges_contribution[i][k] += \
                        (np.array(yields) * massfac[w])

            # Distribute the yields in current and future timesteps
            self.__distribute_yields(i, count_numbers_c, count_numbers_f, yields, \
            yields_extra, maxm, minm, lifetimemin, lifetimemax, p_number, \
            mstars, w, func_total_ejecta)


    ##############################################
    #            Distribute Yields               #
    ##############################################
    def __distribute_yields(self, i, count_numbers_c, count_numbers_f, yields, \
          yields_extra, maxm, minm, lifetimemin, lifetimemax, p_number, mstars, \
          w, func_total_ejecta):

        '''
        This function distributes the ejecta of a specific stellar mass bin over
        the current and the next following timesteps. In practical terms, this
        function adds the contribution of the specific stellar mass bin to the mdot
        arrays.

        Arguments
        =========

          i : Index of the current timestep.
          count_numbers_c : Variable for testing.
          count_numbers_f : Variable for testing.
          yields : Stellar yields for the considered mass bin.
          yields_extra : Stellar yields from extra source.
          maxm : Maximum stellar mass in the considered bin.
          minm : Minimum stellar mass in the considered bin.
          lifetimemin : Minimum stellar lifetime in the considered bin.
          lifetimemax : Maximum stellar lifetime in the considered bin.
          p_number : IMF (number) coefficient in the mass interval.
          mstars : Initial mass of stellar models available in the input yields.
          w : Index of the stellar model providing the yields

        '''

        # Initialisation of the local variables
        count_numbers_t = 0  # Star count
        tt = 0               # Time since the formation of the considered stars
        text = False         # Print info.. does not look useful
        firstmin = True      # True if the first ejecta is still to come 
        min_old = 0
        minm1 = 0.0
        maxm1 = 0.0

        # For every timestep, beginning with the current timestep ...
        for j in range(i-1, self.nb_timesteps):

            # Lifetime limits given by the timestep j (will be modified below)
            lifetimemin1 = tt 
            tt += self.history.timesteps[j]
            lifetimemax1 = tt

            # Set time and mass boundaries for this timestep
            firstmin, maxm1, minm1, continue_bol, break_bol, min_old = \
                self.__set_t_m_bdys(tt, lifetimemin, lifetimemin1, firstmin, \
                  lifetimemax, lifetimemax1, maxm, maxm1, minm, minm1, j, min_old)

            # Verify if last function demanded a continue or a break
            if continue_bol:
                continue
            elif break_bol:
                break

	    self.history.t_m_bdys.append([minm1, maxm1])

            # Calculate the number of stars contributing in this timestep j
            number_stars = p_number * (self._imf(minm1, maxm1, 1))

            # Cumulate the number of stars for testing
            count_numbers_f += number_stars
            count_numbers_t += number_stars

            # Output information
            if text==False :
                text = True
                if self.iolevel >= 1:
                    print mstars[w], 'Wind (if massive +SN2): of Z=', \
                    '{:.6E}'.format(self.zmetal),' at time ', \
                    '{:.3E}'.format(sum(self.history.timesteps[:j+1])),\
		    'with lifetime: ','{:.3E}'.format(self.__find_lifetimes(round(self.zmetal,6), mstars[w]))
            if self.iolevel > 1:
                if sum(yields) < 0:				
                    print 'Yields are negative', sum(yields)

            # Add the yields to mdot arrays
            break_bol = self.__add_yields_mdot( minm1, maxm1, yields, \
                yields_extra, i, j, w, number_stars, mstars, tt, lifetimemax, \
                    p_number, func_total_ejecta)

            # Look if the last function demanded a break
            if break_bol:
                break


    ##############################################
    #               Set t m Bdys                 #
    ##############################################
    def __set_t_m_bdys(self, tt, lifetimemin, lifetimemin1, firstmin, \
                lifetimemax, lifetimemax1, maxm, maxm1, minm, minm1, j, min_old):

        '''
        Part of __distribute_yields() function

        '''

        # Values to be returned
        continue_bol = False
        break_bol = False

        # If no ejecta occurs before the end of the timestep j ...
        # Skip this timestep
        if tt < lifetimemin:
            continue_bol = True
            return firstmin, maxm1, minm1, continue_bol, break_bol, min_old

        # If the ejecta is added for the first time ...
        # Adjust the time and mass boundary for this timestep
        elif firstmin:
            lifetimemin1 = lifetimemin
            maxm1 = maxm

        # If this is the last timestep to receive the ejecta ...
        # Adjust the time and mass boundary for this timestep
        if tt >= lifetimemax:
            lifetimemax1 = lifetimemax
            minm1 = minm

        # Stop if the considered lifetime is longer than the simulation
        if (lifetimemax > sum(self.history.timesteps) and \
                                            j == self.nb_timesteps-1):  
            if self.iolevel > 0:
                print 'Ejection of mass at ','{:.3E}'.format(lifetimemax), \
                      'excluded due to timelimit'
            break_bol = True
            return firstmin, maxm1, minm1, continue_bol, break_bol, min_old
 
        # In case of timestep being inside the liftime/mass interval,
        # old minm1 becomes new maxm1 
        if not firstmin:
            maxm1 = min_old

        # Set firstmin to False since first ejecta is considered at this point
        firstmin = False
 
        # If the lifetime of the considered star lasts beyond this timestep j...
        # Find minm1 from lifetime max1
        if tt < lifetimemax:
            minm1 = self.__find_lifetimes(round(self.zmetal,6), \
                                    mass=[minm,maxm], lifetime=lifetimemax1)
            min_old = minm1

        # Output information
        if self.iolevel >= 3:
            print '(timestep:) ','{:.3E}'.format(lifetimemin1), maxm1, \
                  'to','{:.3E}'.format(lifetimemax1),minm1

        return firstmin, maxm1, minm1, continue_bol, break_bol, min_old


    ##############################################
    #              Add yields Mdot               #
    ##############################################
    def __add_yields_mdot(self, minm1, maxm1, yields, yields_extra, i, j, w,\
            number_stars, mstars, tt, lifetimemax, p_number, func_total_ejecta):

        '''
        This function adds the yields of the stars formed during timestep i
        in a future upcoming timestep j.  The stars here are only a fraction
        of the whole stellar population that just formed.

        Arguments
        =========

          minm1 : Minimum stellar mass having ejecta in this timestep j.
          maxm1 : Minimum stellar mass having ejecta in this timestep j.
          yields : Stellar yields for the considered mass bin.
          yields_extra : Stellar yields from extra source.
          i : Index of the current timestep.
          j : Index of the future timestep where ejecta is added.
          w : Index of the stellar model providing the yields.
          number_stars : Number of stars having ejecta in timestep j.
          mstars : Initial mass of stellar models available in the input yields.
          tt : Time between timestep j and timestep i.
          lifetimemax : Maximum lifetime .
          p_number : IMF (number) coefficient in the mass interval.

        '''

        # Scale the total yields (see func_total_eject)
	if not (self.yield_interp == 'None'):
        	scalefactor = (func_total_ejecta(minm1) + \
                              func_total_ejecta(maxm1)) / 2.0 / sum(yields)
	else:
		scalefactor = 1

        # Output information
        if self.iolevel > 1:
            print 'Scalefactor:', scalefactor

        # For every isotope ...
        for k in range(len(self.ymgal[i])):

            # Add the total ejecta
            if k >= 76 and mstars[w] > self.transitionmass:
                self.mdot[j][k] = self.mdot[j][k] + \
                    self.f_arfo * yields[k] * scalefactor * number_stars
            else:
                self.mdot[j][k] = self.mdot[j][k] + \
                    yields[k] * scalefactor * number_stars

            # For massive stars ...
            if mstars[w] > self.transitionmass:

                # In the case of an extra source in massive stars ...
                if self.extra_source_on:
                    self.mdot_massive[j][k] = self.mdot_massive[j][k] + \
                        yields_extra[k] * scalefactor * number_stars
                    self.mdot[j][k] = self.mdot[j][k] + \
                        yields_extra[k] * scalefactor * number_stars
                    
                # Add the contribution of massive stars
                if k >= 76:
                    self.mdot_massive[j][k] = self.mdot_massive[j][k] + \
                        self.f_arfo * yields[k] * scalefactor * number_stars
                else:
                    self.mdot_massive[j][k] = self.mdot_massive[j][k] + \
                        yields[k] * scalefactor * number_stars

            # Add contribution of AGB stars
            else:
                self.mdot_agb[j][k] = self.mdot_agb[j][k] + \
                    yields[k] * scalefactor * number_stars

        # Count the number of core-collapse SNe
        if mstars[w] > self.transitionmass:
            self.sn2_numbers[j] += number_stars
        if ((minm1 >= 3) and (maxm1 <= 8)) or ((minm1 < 3) and (maxm1 > 8)):
            self.wd_sn1a_range1[j] += number_stars
        elif minm1 < 3 and maxm1 > 3:
            self.wd_sn1a_range1[j] += p_number * (self._imf(3, maxm1, 1))
        elif minm1 < 8 and maxm1 > 8:	
            self.wd_sn1a_range1[j] += p_number * (self._imf(minm1, 8, 1,))

        # Sum the total number of stars born in the current timestep i (not j)
        self.number_stars_born[i] += number_stars
		
        # Exit loop if no more ejecta to be distributed 
        if tt >= lifetimemax:
            break_bol = True
        else:
            break_bol = False

        # Return whether the parent function needs to break or not
        return break_bol


    ##############################################
    #             SN Ia Contribution             #
    ##############################################
    def __sn1a_contribution(self, i):

        '''
        This function calculates the contribution of SNe Ia in the stellar ejecta,
        and adds it to the mdot array.
   
        Argument
        ========

          i : Index of the current timestep.

        '''

        # Set the IMF normalization constant for a 1 Mo stellar population
        # Normalization constant is only used if inte = 0 in the IMF call
        self._imf(0, 0, -1, 0)

        # Get SN Ia yields
        tables_Z = self.ytables_1a.metallicities
        for tz in tables_Z:
            if self.zmetal > tz:
                yields1a = self.ytables_1a.get(Z=tz, quantity='Yields')
                break
            if self.zmetal <= tables_Z[-1]:
                yields1a = self.ytables_1a.get(Z=tables_Z[-1], quantity='Yields')
                break

        # If the selected SN Ia rate depends on the number of white dwarfs ...
        if self.history.sn1a_rate == 'exp' or \
           self.history.sn1a_rate == 'gauss' or \
           self.history.sn1a_rate == 'maoz' or \
           self.history.sn1a_rate == 'power_law':

            # Get the lifetimes of the considered stars
            spline_lifetime, spline_min_time = self.__get_spline_info()

        # Normalize the SN Ia rate if not already done
        if self.history.sn1a_rate == 'exp' and not self.normalized:
            self.__normalize_efolding(spline_min_time)
        elif self.history.sn1a_rate == 'gauss' and not self.normalized:
            self.__normalize_gauss(spline_min_time)
        elif (self.history.sn1a_rate == 'maoz' or \
            self.history.sn1a_rate == 'power_law') and not self.normalized:
            self.__normalize_maoz(spline_min_time)

        # Initialisation of the cumulated time and number of SNe Ia
        sn1a_output = 0
        sn1a_sum = 0
        tt = 0

        # For every upcoming timestep j, starting with the current one ...
        for j in range(i-1, self.nb_timesteps):

            # Set the upper and lower time boundary of the timestep j
            timemin = tt
            tt += self.history.timesteps[j]
            timemax = tt

            # Calculate the number of SNe Ia if with Vogelsberger SN Ia rate
            if self.history.sn1a_rate=='vogelsberger':
            	n1a = self.__vogelsberger13(timemin, timemax)

            # No SN Ia if the minimum current stellar lifetime is too long
            if spline_min_time > timemax:
                n1a = 0

            # If SNe Ia occur during this timestep j ...		
            else:

                # Set the lower time limit for the integration
                if timemin < spline_min_time:
                    timemin = spline_min_time 

                # For an exponential SN Ia rate ...
                if self.history.sn1a_rate == 'exp':
       
                    # Calculate the number of SNe Ia and white dwarfs (per Mo)
                    n1a, wd_number = self.__efolding(timemin, \
                        timemax, spline_lifetime)

                # For a power law SN Ia rate ...
                elif self.history.sn1a_rate == 'maoz' or \
                     self.history.sn1a_rate == 'power_law':

                    # Calculate the number of SNe Ia and white dwarfs (per Mo)
                    n1a, wd_number = self.__maoz12_powerlaw(timemin, \
                        timemax, spline_lifetime)

                # For a gaussian SN Ia rate ...
                elif self.history.sn1a_rate == 'gauss':

                    # Calculate the number of SNe Ia and white dwarfs (per Mo)
                    n1a, wd_number = self.__gauss(timemin, \
                        timemax, spline_lifetime)

                # Cumulate the number of white dwarfs in the SN Ia mass range
                self.wd_sn1a_range[j] += (wd_number * self.m_locked)	

            # Convert number of SNe Ia per Mo into real number of SNe Ia
            n1a = n1a * self.m_locked

            # Cumulate the number of SNe Ia
            self.sn1a_numbers[j] += n1a
            sn1a_sum += n1a

            # Output information
            if sn1a_output == 0 :
                if self.iolevel >= 2:
                    print 'SN1a (pop) start to contribute at time ', \
                          '{:.3E}'.format((timemax))
                sn1a_output = 1

            # Add the contribution of SNe Ia to the timestep j
            self.mdot[j] = np.array(self.mdot[j]) +  np.array(n1a * yields1a)
            self.mdot_1a[j] = np.array(self.mdot_1a[j]) + np.array(n1a*yields1a)


    ##############################################
    #              Vogelsberger 13               #
    ##############################################
    def __vogelsberger13(self, timemin,timemax):

        '''
        This function returns the number of SNe Ia occuring within a given time
        interval using the Vogelsberger et al. (2013) delay-time distribution
        function.
   
        Arguments
        =========

          timemin : Lower boundary of the time interval.
          timemax : Upper boundary of the time interval.

        '''

        # Define the minimum age for a stellar population to host SNe Ia
        fac = 4.0e7

        # If stars are too young ...
        if timemax < fac:

            # No SN Ia
            n1a = 0

        # If the age fac is in between the given time interval ...
        elif timemin <= fac:

            # Limit the lower time boundary to fac 
            timemin = fac
            n1a = quad(self.__vb, timemin, timemax, args=(fac))[0]

        # If SNe Ia occur during the whole given time interval ...
        else:

            # Use the full time range
            n1a = quad(self.__vb, timemin, timemax, args=(fac))[0]

        # Exit if the IMF boundary do not cover 3 - 8 Mo (SN Ia progenitors)
        if not ( (self.imf_bdys[0] < 3) and (self.imf_bdys[1] > 8)):
            print '!!!!!IMPORTANT!!!!'
            print 'With Vogelsberger SNIa implementation selected mass', \
                  'range not possible.'
            import sys
            sys.exit('Choose mass range which either fully includes' + \
                     'range from 3 to 8Msun or fully excludes it'+ \
                     'or use other SNIa implementations')

        # Return the number of SNe Ia per Mo
        return n1a


    ##############################################
    #            Vogelsberger 13 - DTD           #
    ##############################################
    def __vb(self, tt, fac1):

        '''
        This function returns the rate of SNe Ia using the delay-time distribution
        of Vogelsberger et al. (2013) at a given time

        Arguments
        =========

          tt : Age of the stellar population
          fac1 : Minimum age for the stellar population to host SNe Ia

        '''

        # Return the rate of SN
        fac2 = 1.12
        return 1.3e-3 * (tt / fac1)**(-fac2) * (fac2 - 1.0) / fac1


    ##############################################
    #                  Spline 1                  #
    ##############################################
    def __spline1(self, t_s):

        '''
        This function returns the lower mass boundary of the SN Ia progenitors
        from a given stellar population age.
   
        Arguments
        =========

          t_s : Age of the considered stellar population

        '''

        # Set the very minimum mass for SN Ia progenitors
        minm_prog1a = 3.0

        # Limit the minimum mass to the lower mass limit of the IMF if needed
        if self.imf_bdys[0] > minm_prog1a:
            minm_prog1a = self.imf_bdys[0]

        # Return the minimum mass
        return float(max(minm_prog1a, 10**self.spline_lifetime(np.log10(t_s))))


    ##############################################
    #                  WD Number                 #
    ##############################################
    def __wd_number(self, m, t):

        '''
        This function returns the number of white dwarfs, at a given time, which 
        had stars of a given initial mass as progenitors.  The number is 
        normalized to a stellar population having a total mass of 1 Mo. 
   
        Arguments
        =========

          m : Initial stellar mass of the white dwarf progenitors
          t : Age of the considered stellar population

        '''
   
        # Calculate the stellar mass associated to the lifetime t
        mlim = float(10**self.spline_lifetime(np.log10(t)))

        # Set the maximum mass for SN Ia progenitor
        maxm_prog1a = 8.0

        # Limit the maximum progenitor mass to the IMF upper limit, if needed
        if 8.0 > self.imf_bdys[1]:
            maxm_prog1a = self.imf_bdys[1]

        # Return the number of white dwarfs, if any
        if mlim > maxm_prog1a:
            return 0
        else:
            mmin=0
            mmax=0
            inte=0
            return  float(self._imf(mmin,mmax,inte,m))


    ##############################################
    #                 Maoz SN Rate               #
    ##############################################
    def __maoz_sn_rate(self, m, t):

        '''
        This function returns the rate of SNe Ia, at a given stellar population
        age, coming from stars having a given initial mass.  It uses the delay-
        time distribution of Maoz & Mannucci (2012).
   
        Arguments
        =========

          m : Initial stellar mass of the white dwarf progenitors
          t : Age of the considered stellar population

        '''

        # Factors 4.0e-13 and 1.0e9 need to stay there !
        # Even if the rate is re-normalized.
        return  self.__wd_number(m,t) * 4.0e-13 * (t/1.0e9)**self.beta_pow


    ##############################################
    #               Maoz SN Rate Int             #
    ##############################################
    def __maoz_sn_rate_int(self, t):

        '''
        This function returns the rate of SNe Ia, at a given stellar population
        age, coming from all the possible progenitors.  It uses the delay-time
        distribution of Maoz & Mannucci (2012).
   
        Arguments
        =========

          t : Age of the considered stellar population

        '''

        # Return the SN Ia rate integrated over all possible progenitors
        return quad(self.__maoz_sn_rate, self.__spline1(t), 8, args=t)[0]


    ##############################################
    #               Maoz12 PowerLaw              #
    ##############################################
    def __maoz12_powerlaw(self, timemin, timemax, spline_lifetime):

        '''
        This function returns the total number of SNe Ia (per Mo formed) and 
        white dwarfs for a given time interval.  It uses the delay-time
        distribution of Maoz & Mannucci (2012).
   
        Arguments
        =========

          timemin : Lower limit of the time (age) interval
          timemax : Upper limit of the time (age) interval

        '''

        # Avoid the zero in the integration
        if timemin == 0:
            timemin = 1

        # Maximum mass for SN Ia progenitor
        maxm_prog1a = 8.0

        # Get stellar masses associated with lifetimes of timemax and timemin
        spline1_timemax = float(self.__spline1(timemax))
        spline1_timemin = float(self.__spline1(timemin))

        # Calculate the number of SNe Ia per Mo of star formed
        #n1a = self.A_maoz * quad(self.__maoz_sn_rate_int, timemin, timemax)[0]

        # Initialisation of the number of SNe Ia (IMPORTANT)
        n1a = 0.0

        # If SNe Ia occur during this time interval ...
        if timemax > self.t_8_0 and timemin < 13.0e9:

            # If the fraction of white dwarfs needs to be integrated ...
            if timemin < self.t_3_0:

                # Get the upper and lower time limits for this integral part
                t_temp_up = min(self.t_3_0,timemax)
                t_temp_low = max(self.t_8_0,timemin)

                # Calculate a part of the integration
                temp_up = self.a_wd * t_temp_up**(self.beta_pow+4.0) / \
                             (self.beta_pow+4.0) + \
                          self.b_wd * t_temp_up**(self.beta_pow+3.0) / \
                             (self.beta_pow+3.0) + \
                          self.c_wd * t_temp_up**(self.beta_pow+2.0) / \
                             (self.beta_pow+2.0)
                temp_low = self.a_wd * t_temp_low**(self.beta_pow+4.0) / \
                              (self.beta_pow+4.0) + \
                           self.b_wd * t_temp_low**(self.beta_pow+3.0) / \
                              (self.beta_pow+3.0) + \
                           self.c_wd * t_temp_low**(self.beta_pow+2.0) / \
                              (self.beta_pow+2.0)

                # Natural logarithm if beta_pow == -1.0
                if self.beta_pow == -1.0:
                    temp_up += self.d_wd*np.log(t_temp_up)
                    temp_low += self.d_wd*np.log(t_temp_low)

                # Normal integration if beta_pow != -1.0
                else:
                    temp_up += self.d_wd * t_temp_up**(self.beta_pow+1.0) / \
                                  (self.beta_pow+1.0)
                    temp_low += self.d_wd * t_temp_low**(self.beta_pow+1.0) / \
                                   (self.beta_pow+1.0)

                # Add the number of SNe Ia (with the wrong units)
                n1a = (temp_up - temp_low)

            # If the integration continues beyond the point where all 
            # progenitor white dwarfs are present (this should not be an elif)
            if timemax > self.t_3_0:

                # Get the upper and lower time limits for this integral part
                t_temp_up = min(13.0e9,timemax)
                t_temp_low = max(self.t_3_0,timemin)

                # Natural logarithm if beta_pow == -1.0
                if self.beta_pow == -1.0:
                    temp_int = np.log(t_temp_up) - np.log(t_temp_low)

                # Normal integration if beta_pow != -1.0
                else:
                    temp_int = (t_temp_up**(self.beta_pow+1.0) - \
                         t_temp_low*(self.beta_pow+1.0)) / (self.beta_pow+1.0)

                # Add the number of SNe Ia (with the wrong units)
                n1a += temp_int

            # Add the right units 
            n1a = n1a * self.A_maoz * 4.0e-13 / 10**(9*self.beta_pow)

        # Calculate the number of white dwarfs
        #number_wd = quad(self.__wd_number, spline1_timemax, maxm_prog1a, \
        #    args=timemax)[0] - quad(self.__wd_number, spline1_timemin, \
        #        maxm_prog1a, args=timemin)[0]
        number_wd = 1.0 # Temporary .. should be modified if nb_wd is needed..

        # Return the number of SNe Ia (per Mo formed) and white dwarfs
        return n1a, number_wd


    ##############################################
    #                 Exp SN Rate               #
    ##############################################
    def __exp_sn_rate(self, m,t):

        '''
        This function returns the rate of SNe Ia, at a given stellar population
        age, coming from stars having a given initial mass. It uses the exponential
        delay-time distribution of Wiersma et al. (2009).
   
        Arguments
        =========

          m : Initial stellar mass of the white dwarf progenitors
          t : Age of the considered stellar population

        '''

        # E-folding timescale of the exponential law
        tau=self.exp_dtd #Wiersma default: 2e9
        mmin=0
        mmax=0
        inte=0

        # Return the SN Ia rate at time t coming from stars of mass m
        return self.__wd_number(m,t) * np.exp(-t/tau) / tau


    ##############################################
    #              Wiersma09 E-Folding           #
    ##############################################
    def __efolding(self, timemin, timemax, spline_lifetime):

        '''
        This function returns the total number of SNe Ia (per Mo formed) and 
        white dwarfs for a given time interval.  It uses the exponential delay-
        time distribution of Wiersma et al. (2009).
   
        Arguments
        =========

          timemin : Lower limit of the time (age) interval
          timemax : Upper limit of the time (age) interval

        '''

        # Avoid the zero in the integration (exp function)
        if timemin == 0:
            timemin = 1

        # Set the maximum mass of the progenitors of SNe Ia
        maxm_prog1a = 8.0
        if 8 > self.imf_bdys[1]:
            maxm_prog1a = self.imf_bdys[1]

        # Calculate the number of SNe Ia per Mo of star formed
        n1a = self.A_exp * dblquad(self.__exp_sn_rate, timemin, timemax, \
            lambda x: self.__spline1(x), lambda x: maxm_prog1a)[0]

        # Calculate the number of white dwarfs per Mo of star formed
        number_wd = quad(self.__wd_number, self.__spline1(timemax), maxm_prog1a, \
            args=timemax)[0] - quad(self.__wd_number, self.__spline1(timemin), \
                maxm_prog1a, args=timemin)[0]
	    
        # Return the number of SNe Ia and white dwarfs
        return n1a, number_wd


    ##############################################
    #             Normalize WEfolding            #
    ##############################################
    def __normalize_efolding(self, spline_min_time):

        '''
        This function normalizes the SN Ia rate of a gaussian.
   
        Argument
        ========

          spline_min_time : Minimum stellar lifetime.

        '''

        # Set the maximum mass of progenitors of SNe Ia
        maxm_prog1a = 8.0
        if maxm_prog1a > self.imf_bdys[1]:
            maxm_prog1a = self.imf_bdys[1]

        # Maximum time of integration
        ageofuniverse = 1.3e10

        # Calculate the normalisation constant
        self.A_exp = self.nb_1a_per_m / dblquad(self.__exp_sn_rate, \
            spline_min_time, ageofuniverse, \
            lambda x:self.__spline1(x), lambda x:maxm_prog1a)[0]

	if self.direct_norm_1a >0:
		self.A_exp=self.direct_norm_1a

        # Avoid renormalizing during the next timesteps
        self.normalized = True


    ##############################################
    #                Gauss SN Rate               #
    ##############################################
    def __gauss_sn_rate(self, m, t):

        '''
        This function returns the rate of SNe Ia, at a given stellar population
        age, coming from stars having a given initial mass.  It uses a gaussian
        delay-time distribution similar to Wiersma09.
   
        Arguments
        =========

          m : Initial stellar mass of the white dwarf progenitors
          t : Age of the considered stellar population

        '''

        # Gaussian characteristic delay timescale, and its sigma value
        tau = self.gauss_dtd[0]   #Wiersma09 defaults:1.0e9 
        sigma = self.gauss_dtd[1] #Wiersma09 defaults: 0.66e9

        # Return the SN Ia rate at time t coming from stars of mass m
        return self.__wd_number(m,t) * 1.0 / np.sqrt(2 * np.pi * sigma**2) * \
            np.exp(-(t - tau)**2 / (2 * sigma**2))


    ##############################################
    #                Wiersma09 Gauss             #
    ##############################################
    def __gauss(self, timemin, timemax, spline_lifetime):

        '''
        This function returns the total number of SNe Ia (per Mo formed) and 
        white dwarfs for a given time interval.  It uses the gaussian delay-
        time distribution of Wiersma et al. (2009).
   
        Arguments
        =========

          timemin : Lower limit of the time (age) interval
          timemax : Upper limit of the time (age) interval

        '''

        # Set the maximum mass of the progenitors of SNe Ia
        maxm_prog1a = 8.0
        if 8 > self.imf_bdys[1]:
            maxm_prog1a=self.imf_bdys[1]
	
        # Calculate the number of SNe Ia per Mo of star formed
        n1a = self.A_gauss * dblquad(self.__gauss_sn_rate, timemin, timemax, \
            lambda x:self.__spline1(x), lambda x:maxm_prog1a)[0]

        # Calculate the number of white dwarfs per Mo of star formed
        number_wd = quad(self.__wd_number, self.__spline1(timemax), maxm_prog1a, \
            args=timemax)[0] - quad(self.__wd_number, self.__spline1(timemin), \
                maxm_prog1a, args=timemin)[0]

        # Return the number of SNe Ia and white dwarfs
        return n1a, number_wd


    ##############################################
    #               Normalize WGauss             #
    ##############################################
    def __normalize_gauss(self, spline_min_time):

        '''
        This function normalizes the SN Ia rate of a gaussian (similar to Wiersma09).
   
        Argument
        ========

          spline_min_time : Minimum stellar lifetime.

        '''

        # Set the maximum mass of progenitors of SNe Ia
        maxm_prog1a = 8.0
        if maxm_prog1a > self.imf_bdys[1]:
            maxm_prog1a = self.imf_bdys[1]

        # Maximum time of integration
        ageofuniverse = 1.3e10

        # Calculate the normalisation constant
        self.A_gauss = self.nb_1a_per_m / dblquad(self.__gauss_sn_rate, \
            spline_min_time, ageofuniverse, \
            lambda x:self.__spline1(x), lambda x:maxm_prog1a)[0]

        # Avoid renormalizing during the next timesteps
        self.normalized = True


    ##############################################
    #                Get Spline Info             #
    ##############################################
    def __get_spline_info(self):

        '''
        This function returns the stellar lifetimes used in sn1a_contribution()
   
        '''

        # Get the [metallicity, log mass, log lifetime] arrays
        # Values for spline fit must be increasing
        zm_lifetime_grid = self.zm_lifetime_grid_current
        idx_z = (np.abs(zm_lifetime_grid[0] - self.zmetal)).argmin()
        grid_lifetimes = list(zm_lifetime_grid[2][idx_z])[::-1]
        grid_masses = list(zm_lifetime_grid[1])[::-1]

        # Information for spline functions 
        spline_degree1 = 2
        smoothing1 = 0
        boundary = [None, None]

        # Define the spline
        self.spline_lifetime = UnivariateSpline(grid_lifetimes, \
            np.log10(grid_masses), bbox=boundary, k=spline_degree1, s=smoothing1)
        spline_lifetime = self.spline_lifetime
        spline_min_time = 10**grid_lifetimes[0]

	#tarray=range(1e6,1e9)
	#marray=[]
	#for k in range(len(tarray)):
	#	m=self.spline_lifetime(np.log10(tarray[k]))
	#	marray.append(m)
	#plt.plot(tarray,marray)
	#plt.title('Test')

        # Return all the lifetimes (log) and the miminum lifetime
        return spline_lifetime, spline_min_time


    ##############################################
    #                Normalize Maoz              #
    ##############################################
    def __normalize_maoz(self, spline_min_time):

        '''
        This function normalizes the SN Ia rate of Maoz or any power law.
   
        Argument
        ========

          spline_min_time : Minimum stellar lifetime.

        '''

        # Set the maximum mass of progenitors of SNe Ia
#        maxm_prog1a = 8.0
#        if maxm_prog1a > self.imf_bdys[1]:
#            maxm_prog1a = self.imf_bdys[1]

        # Maximum time of integration
#        ageofuniverse = 1.3e10

        # Calculate the normalisation constant
#        self.A_maoz = self.nb_1a_per_m / quad(self.__maoz_sn_rate_int, \
#            spline_min_time, ageofuniverse)[0]
#        print self.A_maoz

        # Calculate the first part of the integral
        temp_8_0 = self.a_wd*self.t_8_0**(self.beta_pow+4.0)/(self.beta_pow+4.0)+\
                   self.b_wd*self.t_8_0**(self.beta_pow+3.0)/(self.beta_pow+3.0)+\
                   self.c_wd*self.t_8_0**(self.beta_pow+2.0)/(self.beta_pow+2.0)
        temp_3_0 = self.a_wd*self.t_3_0**(self.beta_pow+4.0)/(self.beta_pow+4.0)+\
                   self.b_wd*self.t_3_0**(self.beta_pow+3.0)/(self.beta_pow+3.0)+\
                   self.c_wd*self.t_3_0**(self.beta_pow+2.0)/(self.beta_pow+2.0)

        # Natural logarithm if beta_pow == -1.0
        if self.beta_pow == -1.0:
            temp_8_0 += self.d_wd*np.log(self.t_8_0)
            temp_3_0 += self.d_wd*np.log(self.t_3_0)
            temp_13gys = np.log(13.0e9) - np.log(self.t_3_0)

        # Normal integration if beta_pow != -1.0
        else:
            temp_8_0 += self.d_wd*self.t_8_0**(self.beta_pow+1.0)/(self.beta_pow+1.0)
            temp_3_0 += self.d_wd*self.t_3_0**(self.beta_pow+1.0)/(self.beta_pow+1.0)
            temp_13gys = (13.0e9**(self.beta_pow+1.0) - \
                         self.t_3_0**(self.beta_pow+1.0)) / (self.beta_pow+1.0)

        # Calculate the normalization constant
        self.A_maoz = self.nb_1a_per_m * 10**(9*self.beta_pow) / 4.0e-13 / \
                   (temp_3_0 - temp_8_0 + temp_13gys)

        # Avoid renormalizing during the next timesteps
        self.normalized = True


    ##############################################
    #                     IMF                    #
    ##############################################
    def _imf(self, mmin, mmax, inte, mass=0):

	'''
        This function returns, using the IMF, the number or the mass of all
        the stars within a certain initial stellar mass interval.

        Arguments
        =========

          mmin : Lower mass limit of the interval.
          mmax : Upper mass limit of the interval.
          inte : 1 - Return the number of stars.
                 2 - Return the stellar mass.
                 0 - Return the number of stars having a mass 'mass'
                -1 - Return the IMF proportional constant when normalized to 1 Mo.
          mass : Mass of a star (if inte == 0).

        '''

        # Return zero if there is an error in the mass boundary
        if mmin>mmax:
            if self.iolevel > 1:
                print 'Warning in _imf function'
                print 'mmin:',mmin
                print 'mmax',mmax
                print 'mmin>mmax'
                print 'Assume mmin == mmax'
            return 0

        # Salpeter IMF or any power law
        if self.imf_type == 'salpeter' or self.imf_type == 'alphaimf':

            # Choose the right option
            if inte == 0:
                return self.imfnorm * self.__g1_power_law(mass)
            if inte == 1: 
                return quad(self.__g1_power_law, mmin, mmax)[0]
            if inte == 2:
                return quad(self.__g2_power_law, mmin, mmax)[0]
            if inte == -1:
                self.imfnorm = 1.0 / quad(self.__g2_power_law, \
                    self.imf_bdys[0], self.imf_bdys[1])[0]

	# Custom IMF with the file imf_input.py
	if self.imf_type=='input':

            # Load the file
            ci = load_source('custom_imf', global_path + '/imf_input.py')
	    self.ci = ci
            # Choose the right option
            if inte == 0:
                return self.imfnorm * self.__g1_costum(mass)
            if inte == 1:
                return quad(self.__g1_costum, mmin, mmax)[0]
            if inte == 2:
                return quad(self.__g2_costum, mmin, mmax)[0]
            if inte == -1:
                self.imfnorm =1.0 / quad(self.__g2_costum, \
                    self.imf_bdys[0], self.imf_bdys[1])[0]
		
	# Chabrier IMF
	elif self.imf_type=='chabrier':

            # Choose the right option
            if inte == 0:
                return self.imfnorm * self.__g1_chabrier(mass)
            if inte == 1:
                return quad(self.__g1_chabrier, mmin, mmax)[0]
            if inte == 2:
                return quad(self.__g2_chabrier, mmin, mmax)[0]
            if inte == -1:
                self.imfnorm = 1.0 / quad(self.__g2_chabrier, \
                    self.imf_bdys[0], self.imf_bdys[1])[0]

	# Chabrier - Alpha custom - IMF
	elif self.imf_type=='chabrieralpha':

            # Choose the right option
            if inte == 0:
                return self.imfnorm * self.__g1_chabrier_alphaimf(mass)
            if inte == 1:
                return quad(self.__g1_chabrier_alphaimf, mmin, mmax)[0]
            if inte == 2:
                return quad(self.__g2_chabrier_alphaimf, mmin, mmax)[0]
            if inte == -1:
                self.imfnorm = 1.0 / quad(self.__g2_chabrier_alphaimf, \
                    self.imf_bdys[0], self.imf_bdys[1])[0]

	# Kroupa IMF
	elif self.imf_type=='kroupa':

            # Choose the right option
            if inte == 0:
                return self.imfnorm * self.__g1_kroupa(mass)
            if inte == 1:
                return quad(self.__g1_kroupa, mmin, mmax)[0]	
            if inte == 2:
                return quad(self.__g2_kroupa, mmin, mmax)[0]
            if inte == -1:
                self.imfnorm = 1.0 / quad(self.__g2_kroupa, \
                    self.imf_bdys[0], self.imf_bdys[1])[0]

	# Ferrini, Pardi & Penco (1990)
	elif self.imf_type=='fpp':

            # Choose the right option
            if inte == 0:
                return self.imfnorm * self.__g1_fpp(mass)
            if inte == 1:
                return quad(self.__g1_fpp, mmin, mmax)[0]
            if inte == 2:
                return quad(self.__g2_fpp, mmin, mmax)[0]
            if inte == -1:
                self.imfnorm = 1.0 / quad(self.__g2_fpp, \
                    self.imf_bdys[0], self.imf_bdys[1])[0]


    ##############################################
    #                G1 Power Law                #
    ##############################################
    def __g1_power_law(self, mass):

	'''
        This function returns the number of stars having a certain stellar mass
        with a Salpeter IMF or a similar power law.

        Arguments
        =========

          mass : Stellar mass.

        '''

        # Select the right alpha index
        if self.imf_type == 'salpeter':
            return mass**(-2.35)
        elif self.imf_type == 'alphaimf':
            return mass**(-self.alphaimf)
        else:
            return 0


    ##############################################
    #                G2 Power Law                #
    ##############################################
    def __g2_power_law(self, mass):

	'''
        This function returns the total mass of stars having a certain initial
        mass with a Salpeter IMF or a similar power law.

        Arguments
        =========

          mass : Stellar mass.

        '''

        # Select the right alpha index
        if self.imf_type == 'salpeter':
            return mass * mass**(-2.35)
        elif self.imf_type == 'alphaimf':
            return mass * mass**(-self.alphaimf)
        else:
            return 0


    ##############################################
    #                  G1 Costum                 #
    ##############################################
    def __g1_costum(self, mass):

	'''
        This function returns the number of stars having a certain stellar mass
        with a custom IMF.

        Arguments
        =========

          mass : Stellar mass.
          ci : File containing the custom IMF.

        '''

        # Return the number of stars
        return self.ci.custom_imf(mass)


    ##############################################
    #                  G2 Costum                 #
    ##############################################
    def __g2_costum(self, mass):

	'''
        This function returns the total mass of stars having a certain stellar
        mass with a custom IMF.

        Arguments
        =========

          mass : Stellar mass.
          ci : File containing the custom IMF.

        '''

        # Return the total mass of stars
        return mass * self.ci.custom_imf(mass)


    ##############################################
    #                 G1 Chabrier                #
    ##############################################
    def __g1_chabrier(self, mass):

	'''
        This function returns the number of stars having a certain stellar mass
        with a Chabrier IMF.

        Arguments
        =========

          mass : Stellar mass.

        '''

        # Select the right mass regime
        if mass <= 1:
            return 0.158 * (1.0 / mass) * \
                np.exp(-np.log10(mass/0.079)**2 / (2.0 * 0.69**2))
        else:
            return 0.0443 * mass**(-2.3)


    ##############################################
    #                 G2 Chabrier                #
    ##############################################
    def __g2_chabrier(self, mass):

	'''
        This function returns the total mass of stars having a certain stellar
        mass with a Chabrier IMF.

        Arguments
        =========

          mass : Stellar mass.

        '''

        # Select the right mass regime
        if mass <= 1:
            return 0.158 * np.exp( -np.log10(mass/0.079)**2 / (2.0 * 0.69**2))
        else:
            return 0.0443 * mass * mass**(-2.3)


    ##############################################
    #            G1 Chabrier AlphaIMF            #
    ##############################################
    def __g1_chabrier_alphaimf(self, mass):

	'''
        This function returns the number of stars having a certain stellar mass
        with a Chabrier IMF.

        Arguments
        =========

          mass : Stellar mass.

        '''

        # Select the right mass regime
        if mass <= 1:
            return 0.158 * (1.0 / mass) * \
                np.exp(-np.log10(mass/0.079)**2 / (2.0 * 0.69**2))
        else:
            return 0.0443 * mass**(-self.alphaimf)


    ##############################################
    #            G2 Chabrier AlphaIMF            #
    ##############################################
    def __g2_chabrier_alphaimf(self, mass):

	'''
        This function returns the total mass of stars having a certain stellar
        mass with a Chabrier IMF.

        Arguments
        =========

          mass : Stellar mass.

        '''

        # Select the right mass regime
        if mass <= 1:
            return 0.158 * np.exp( -np.log10(mass/0.079)**2 / (2.0 * 0.69**2))
        else:
            return 0.0443 * mass * mass**(-self.alphaimf)


    ##############################################
    #                  G1 Kroupa                 #
    ##############################################
    def __g1_kroupa(self, mass):

	'''
        This function returns the number of stars having a certain stellar mass
        with a Kroupa IMF.

        Arguments
        =========

          mass : Stellar mass.

        '''

        # Select the right mass regime
        if mass < 0.08:
            return self.p0 * mass**(-0.3)
        elif mass < 0.5:
            return self.p1 * mass**(-1.3)
        else:
            return self.p1 * self.p2 * mass**(-2.3)


    ##############################################
    #                  G2 Kroupa                 #
    ##############################################
    def __g2_kroupa(self, mass):

	'''
        This function returns the total mass of stars having a certain stellar
        mass with a Kroupa IMF.

        Arguments
        =========

          mass : Stellar mass.

        '''

        # Select the right mass regime
        if mass < 0.08:
            return self.p0 * mass * mass**(-0.3)
        elif mass < 0.5:
            return self.p1 * mass * mass**(-1.3)
        else:
            return self.p1 * self.p2 * mass * mass**(-2.3)


    ##############################################
    #                   G1 FPP                   #
    ##############################################
    def __g1_fpp(self, mass):

	'''
        This function returns the number of stars having a certain stellar mass
        with a Ferrini, Pardi & Penco (1990) IMF.

        Arguments
        =========

          mass : Stellar mass.

        '''

        # Calculate the number of stars
        lgmm = np.log10(mass)
        return 2.01 * mass**(-1.52) / 10**((2.07*lgmm**2+1.92*lgmm+0.73)**0.5)


    ##############################################
    #                   G2 FPP                   #
    ##############################################
    def __g2_fpp(self, mass):

	'''
        This function returns the total mass of stars having a certain stellar
        mass with a Ferrini, Pardi & Penco (1990) IMF.

        Arguments
        =========

          mass : Stellar mass.

        '''

        # Calculate the mass of stars
        lgmm = np.log10(mass)
        return 2.01 * mass**(-0.52) / 10**((2.07*lgmm**2+1.92*lgmm+0.73)**0.5)


    ##############################################
    #             Interpolate Yields             #
    ##############################################
    def __interpolate_yields(self, Z, ymgal_t, ytables):

        '''
        This function takes input yields and interpolates them to return the
        yields at the desired metallicity (if the metallicity is not available
        in the grid).

	It also returns the func_total_ejecta function which calculates the 
        total mass ejected as a function of the stellar initial mass.

        Arguments
        =========

          Z : Current metallicity of the gas reservoir.
          ymgal_t : mass of the gas reservoir of last timestep/current gas reservoir (for 'wiersma' setting). 
          ytables : Ojbect containing the yield tables.
          type_inter : 'lin' - Simple linear interpolation.
                       'wiersma' - Interpolation method from Wiersma+ (2009)

        '''

        # Output information
        if self.iolevel >= 2:
             print 'Start interpolating yields'

        # Reduce the number of decimal places in the metallicity variable
        Z = round(Z, 6)

        # Get all available metallicities
        Z_grid = ytables.metallicities

	# Simplest case: No interpolation of any kind for the yields 
	if (self.yield_interp == 'None'):
	    
	    # to choose the grid metallicity closest to Z
            z1, z2 = self.__get_Z_bdys(Z, Z_grid)
	    if abs(Z-z1) > abs(Z-z2):
		Z_gridpoint=z2
	    else:	
		Z_gridpoint=z1
	    m_stars = ytables.get(Z=Z_gridpoint, quantity='masses') 
	    yields=[]
	    for k in range(len(m_stars)):
	    	y1 = ytables.get(Z=Z_gridpoint, M=m_stars[k], quantity='Yields') 
		yields.append(y1)

	    # func_total_ejecta is skipped for self.yield_interp == 'None', 
	    # hence return not relevant
	    def func_total_ejecta(inimass):
		return inimass
	    return m_stars, yields, func_total_ejecta
	

        # If the interpolation is linear between grid points ...
        if (self.yield_interp == 'lin') and (not self.netyields_on):

            # Get the lower and upper Z boundaries for the interpolation
            z1, z2 = self.__get_Z_bdys(Z, Z_grid)

            # Get all initial stellar masses available in each Z boundary
            m1 = ytables.get(Z=z1, quantity='masses')
            m2 = ytables.get(Z=z2, quantity='masses')
            # Get the stellar masses and the interpolated yields at Z
            m_stars, yields = self.__lin_int_yields(z1, z2, m1, m2, ytables, Z)
	    #print m_stars

        # If the "interpolation" is taken from Wiersma et al. (2009) ....   
        elif (self.yield_interp == 'wiersma') or (self.netyields_on==True):

            # Get the closest available metallicity (favouring lower boundary)
            Z_gridpoint = self.__get_Z_wiersma(Z, Z_grid)
	    print 'Take yields from Z closest to: ',Z,' found: ',Z_gridpoint
            # Get all initial masses at the selected closest metallicity
            m_stars = ytables.get(Z=Z_gridpoint, quantity='masses')

            # Get the yields corrected for the different initial abundances
            yields = self.__correct_iniabu(ymgal_t, ytables, Z_gridpoint, m_stars)

        # Output information
        if self.iolevel >= 3:
            print 'Finished interpolating yields'
            print 'Interpolation test for ',round(Z,6)
            print z1,z2,m_stars #,star_age
            print 'Yields:'
            for k in range(len(m_stars)):
                print m_stars[k], yields[k]

        # Calculate the coefficients of the linear fit regarding the total mass
        # ejected as a function of the stellar initial mass
        slope, intercept = self.__fit_mej_mini(m_stars, yields)

        # Function that returns the ejected mass as a function of stellar mass
        def func_total_ejecta(inimass):

            # Use calculated linear fit
            for h in range(len(m_stars)-1):
                if (inimass >= m_stars[h]) and (inimass <= m_stars[h+1]):
                    massejected = slope[h] * inimass + intercept[h]
                    if massejected < 0:
                        print slope[h],intercept[h]
                        print 'inimass', inimass, \
                              'problem wiht massejected', massejected
                    return massejected

            # Use the last fit for stars more massive than the ones available
            if inimass > m_stars[-1]:
                massejected = slope[-1] * inimass + intercept[-1]
                if massejected < 0:
                    print slope[h], intercept[h]
                    print 'inimass', inimass, \
                          'problem wiht massejected',massejected
                return massejected

	self.func_total_ejecta=func_total_ejecta
	self.m_stars=m_stars
	self.yields=yields
        '''
        if self.iolevel>0:
            mtests=np.arange(m_stars[0],m_stars[-1],0.1) 
            plt.figure()
            for h in range(len(slope)):
                y=slope[h]*np.array(mtests)+intercept[h]
                plt.plot(mtests,y)
                plt.title('Total mass fit')
                plt.xlabel('Minis');plt.ylabel('Meject')
        '''

        # Return the stellar masses, yields and ejecta function at Z
        return m_stars, yields, func_total_ejecta


    ##############################################
    #                 Get Z Bdys                 #
    ##############################################
    def __get_Z_bdys(self, Z, Z_grid):

        '''
        This function returns the lower and upper metallicity boundaries that
        bracket a given metallicity.  The boundary metallicities represent 
        available grid points in the input yield tables.

        Arguments
        =========

          Z : Current metallicity of the gas reservoir.
          Z_grid : Available metallicity grid points.

        '''

        # For every available metallicity ...
        for tz in Z_grid:

            # If Z is above the grid range, use max available Z
            if Z >= Z_grid[0]:
                z1 = Z_grid[0]
                z2 = Z_grid[0]
                if self.iolevel >= 2:
                    print 'Z > Zgrid'
                break

            # If Z is below the grid range, use min available Z
            if Z <= Z_grid[-1]:
                z1 = Z_grid[-1]
                z2 = Z_grid[-1]
                if self.iolevel >= 2:
                    print 'Z < Zgrid'
                break

            # If Z is exactly one of the available Z, use no boundary
            if Z == tz:
                z1 = tz
                z2 = tz
                if self.iolevel >= 2:
                    print 'Z = Zgrid'
                break

            # If Z is above the grid point at index tz ...
            if Z > tz:

                # Use the current grid point as the lower boundary
                z1 = tz

                # Use the previous grid point as the upper boundary
                z2 = Z_grid[Z_grid.index(tz) - 1]

                # Output information
                if self.iolevel >= 2:
                    print 'interpolation necessary '
                break

        # Return the boundaries
        return z1, z2


    ##############################################
    #               Lin Int Yields               #
    ##############################################
    def __lin_int_yields(self, z1, z2, m1, m2, ytables, Z):

        '''
        This function returns the stellar masses and the interpolated yields
        at a given metallicity.  It uses a linear interpolation.
        

        Arguments
        =========

          z1 : Lower metallicity boundary.
          z2 : Upper metallicity boundary.
          m1 : Stellar mass grid points at z1.
          m2 : Stellar mass grid points at z2.
          ytables : Yield tables.
          Z : Current metallicity of the gas reservoir.

        '''

        # Declaration of arrays containing the stellar masses and the
        # interpolated yields at the desired metallicity.
        m_stars = []
        yields = []
	
        # If yields need to be interpolated ...
        if not z1 == z2:

            # Extract stellar common in both Z1 and Z2
            for p in range(len(m1)):
                for k in range(len(m2)):
                    if  m1[p] == m2[k]:
                        m_stars.append(m1[p])
	    m_missing=[]
	    zs_missing=[]
	    #print 'mstars',m_stars
	    for p in range(len(m1)):
		if not m1[p] in m_stars:
	    		if abs(Z-z1) < abs(Z-z2):
				m_stars.append(m1[p])
				m_missing.append(m1[p])
				zs_missing.append(z1)
	    for p in range(len(m2)):
		if not m2[p] in m_stars:
	    		if abs(Z-z2) < abs(Z-z1):
				m_stars.append(m2[p])
				m_missing.append(m2[p])
				zs_missing.append(z2)
	    '''			
	    if (not len(m_stars) == len(m2)) or (not len(m_stars) == len(m1)):
	            print 'Warning: Different masses for different Z',z1,z2,'  in yield input.'
		    print m1,m2
                    print 'Cannot interpolate yields between metallicities!'
                    print 'Take yield for Z closest to current metallicity'
                    if abs(Z-z1) < abs(Z-z2):
                        m_stars=m1
                        z=z1
                    else:
                        m_stars=m2
                        z=z2
                    yields=[]
                    for m in m_stars:
                        yields.append(ytables.get(Z=z, M=m, quantity='Yields'))    
                    return m_stars,yields
	    '''
	    #print 'Zs: ',z1,z2,Z
	    #print 'missing masses: ',m_missing,zs_missing
	    m_stars=np.sort(m_stars)
            # Interpolate each isotope of each stellar model
            for m in m_stars:
		yields.append([])
		if m in m_missing:
			idx=m_missing.index(m)
			z_missing=zs_missing[idx]
			yi = ytables.get(Z=z_missing, M=m, quantity='Yields')
			yields[-1]=yi
			print 'Take ',m,' from ',z_missing
			#do not interpolate
		else:	
                	y1 = ytables.get(Z=z1, M=m, quantity='Yields')
                	y2 = ytables.get(Z=z2, M=m,quantity='Yields')
                	for p in range(len(y1)):
                		y1i = y1[p]
               	   		y2i = y2[p]
                    		slope = (y2i - y1i) / (z2 - z1)
                    		b = y2i - slope * z2
                    		yi = slope * Z + b
                    		yields[-1].append(yi)

        # If no need to interpolate ...
        else:

            # Copy one of the available stellar masses and yields
            m_stars = m1
            for m in m_stars:
                yields.append(ytables.get(Z=z1, M=m, quantity='Yields'))

        # Return the stellar masses and the linearly interpolated yields
        return m_stars, yields


    ##############################################
    #                Get Z Wiersma               #
    ##############################################
    def __get_Z_wiersma(self, Z, Z_grid):

        '''
        This function returns the closest available metallicity grid point 
        for a given Z.  It always favours the lower boundary.

        Arguments
        =========

          Z : Current metallicity of the gas reservoir.
          Z_grid : Available metallicity grid points.

        '''
	import decimal

        # For every available metallicity ...
        for tz in Z_grid:

            # If Z is above the grid range, use max available Z
            if Z >= Z_grid[0]:
                Z_gridpoint = Z_grid[0]
                if self.iolevel >= 2:
                    print 'Z > Zgrid'
                break

            # If Z is below the grid range, use min available Z
            if Z <= Z_grid[-1]:
                Z_gridpoint = Z_grid[-1]
                if self.iolevel >= 2:
                    print 'Z < Zgrid'
                break

            # If Z is exactly one of the available Z, use the given Z
	    # round here to precision given in yield table
            if round(Z,abs(decimal.Decimal(str(tz)).as_tuple().exponent)) == tz: #Z == tz:
                Z_gridpoint = tz#Z
                if self.iolevel >= 2:
                    print 'Z = Zgrid'
                break

            # If Z is above the grid point at index tz, use this last point
            if Z > tz:
                Z_gridpoint = tz
                if self.iolevel >= 2:
                    print 'interpolation necessary'
                break

        # Return the closest metallicity grid point
        return Z_gridpoint


    ##############################################
    #                Correct Iniabu               #
    ##############################################
    def __correct_iniabu(self, ymgal_t, ytables, Z_gridpoint, m_stars):

        '''
        This function returns yields that are corrected for the difference
        between the initial abundances used in the stellar model calculations
        and the ones in the gas reservoir at the moment of star formation.
        See Wiersma et al. (2009) for more information on this approach.

        Arguments
        =========

          mgal_t : Current mass of the gas reservoir (for 'wiersma' setting).
          ytables : Ojbect containing the yield tables.
          Z_gridpoint : Metallicity where the correction is made.
          m_stars : Stellar mass grid point at metallicity Z_gridpoint.

        '''

        # Calculate the isotope mass fractions of the gas reservoir
        X_ymgal_t = []
        for p in range(len(ymgal_t)):
            X_ymgal_t.append(ymgal_t[p] / sum(ymgal_t))
 
        if not Z_gridpoint==0: #X0 is not in popIII tables and not necessary for popIII setting
             # Get the initial abundances used for the stellar model calculation
             X0 = ytables.get(Z=Z_gridpoint, M=m_stars[0], quantity='X0')

        # Declaration of the corrected yields
        yields = []

        # For every stellar model at metallicity Z_gridpoint ...
        for m in m_stars:

            # Get its yields
            y = ytables.get(Z=Z_gridpoint, M=m, quantity='Yields')
            mfinal = ytables.get(Z=Z_gridpoint, M=m, quantity='Mfinal')
	    iso_name=ytables.get(Z=Z_gridpoint, M=m, quantity='Isotopes')

	    yi_all=[]
            # Correct every isotope and make sure the ejecta is always positive
            for p in range(len(X_ymgal_t)):
                if self.netyields_on == True:
		    if self.wiersmamod: #for Wiesma09 tests
			    # initial amount depending on the simulation Z + net production factors 
			    if (m>8) and (iso_name[p] in ['C-12','Mg-24','Fe-56']):
				yi = (X_ymgal_t[p]*(m-mfinal) + y[p])
				if iso_name[p] in ['C-12','Fe-56']:
					#print iso_name[p]
					yi = yi*0.5
				else:
					yi = yi*2.
			    else:
				yi = (X_ymgal_t[p]*(m-mfinal) + y[p])
		    else:
                    	yi = (X_ymgal_t[p]*(m-mfinal) + y[p])
                    #print yi,(m-mfinal),y[p],X_ymgal_t[p]
                else:
                    yi = y[p] + ( X_ymgal_t[p] - X0) * sum(y)
                if yi < 0:
		    if self.iolevel>0:
		    	print 'set ',iso_name[p],' for star ',m,' with ',yi,' to 0, ', \
			'netyields: ',y[p]
                    yi = 0
		yi_all.append(yi)
	 
	    # do the normalization
	    norm = (m-mfinal)/sum(yi_all)
	    yi_all= np.array(yi_all) * norm 	
	   	
            yields.append(yi_all)
	    #print 'star ',m,(m-mfinal),sum(yields[-1])
        # Return the corrected yields
        return yields


    ##############################################
    #                Mass Ejected Fit            #
    ##############################################
    def __fit_mej_mini(self, m_stars, yields):

        '''
        This function calculates and returns the coefficients of the linear fit
        regarding the total mass ejected as a function of the initial mass at
        the low-mass end of massive stars (up to 15 Mo).

        Arguments
        =========

          m_stars : Stellar mass grid point at a specific metallicity.
          yields : Stellar yields at a specific metalliticy

        '''

	# Linear fit coefficients
        slope = []
        intercept = []

	# Get the actual stellar masses and total mass ejected
	x_all = np.array(m_stars)
	y_all = np.array([np.sum(a) for a in yields])

        if self.iolevel>0:
            plt.figure()

        # Calculate the linear fit for all stellar mass bins
        for h in range(len(x_all)-1):
            x=np.array([x_all[h],x_all[h+1]])
            y=np.array([y_all[h],y_all[h+1]])
            a,b=polyfit(x=x,y=y,deg=1)
            slope.append(a)
            intercept.append(b)
            if self.iolevel>0:
                  mtests=np.arange(x[0],x[1],0.1)
                  plt.plot(mtests,slope[-1]*np.array(mtests)+intercept[-1])
                  plt.title('Total mass fit')
                  plt.xlabel('Minis');plt.ylabel('Meject')


        # Return the linear fit coefficients
        return slope, intercept


    ##############################################
    #         Interpolate Yields PopIII          #
    ##############################################
    def __interpolate_yields_pop3(self, zmetal,ymgal_t, ytables):
 
        '''
        This function returns PopIII star yields and the func_total_ejecta that
        calculates the total mass ejected as a function of the stellar initial
        mass.

        Arguments
        =========

          Z : Current metallicity of the gas reservoir (Z = 0).
          ymgal_t : mass of the gas reservoir of last timestep/current gas reservoir (for 'wiersma' setting). 
          ytables : Object containing the yield tables.

        '''	

        # Declaration of arrays containing the stellar masses and the
        # interpolated yields at the desired metallicity, which is Z = 0
        mstars = []
        yields_all = []

        # Recover PopIII stellar masses and yields
        for k in range(len(self.ytables_pop3.table_mz)):
            mini = float(self.ytables_pop3.table_mz[k].split('=')[1].split(',')[0])
            if mini > self.imf_yields_range_pop3[1]:
                break
            if mini < self.imf_yields_range_pop3[0]:
                continue
            mstars.append(mini)
            yields_all.append(self.ytables_pop3.get(Z=zmetal, \
                M=mini,quantity='Yields'))

        # Copy stellar masses and yields ..
        m_stars = mstars
        yields = yields_all

        
        if self.netyields_on==True:
            # Get the yields corrected for the different initial abundances
            yields = self.__correct_iniabu(ymgal_t, self.ytables_pop3, 0, m_stars)


        # Get the actual stellar masses and total mass ejected
        x_all = np.array(m_stars)
        y_all = np.array([np.sum(a) for a in yields])


        # Calculate the coefficients of the linear fit regarding the total mass
        # ejected as a function of the stellar initial mass 
        slope = []
        intercept = []
        for h in range(len(x_all)-1):
            x = np.array([x_all[h], x_all[h+1]])
            y = np.array([y_all[h], y_all[h+1]])
            a,b = polyfit(x=x, y=y, deg=1)   
            slope.append(a)
            intercept.append(b)

        # Function that returns the ejected mass as a function of stellar mass
        def func_total_ejecta(inimass):
            for h in range(len(m_stars)-1):
                if (inimass >= m_stars[h]) and (inimass <= m_stars[h+1]):
                    massejected = slope[h] * inimass + intercept[h]
                    if massejected < 0:
                        print slope[h], intercept[h]
                        print 'inimass', inimass, \
                              'problem wiht massejected',massejected
                    return massejected

        # Return the stellar masses, yields and ejecta function at Z = 0
        return mstars, yields_all, func_total_ejecta


    ##############################################
    #       Interpolate Lifetimes PopIII         #
    ##############################################
    def __interpolate_lifetimes_pop3(self, tables):

        '''
	This function interpolates and returns the stellar lifetimes of PopIII
        stars using given yield tables.  The PopIII star yields range boundary
        is taken into account in the interpolation.

        Argument
        ========

          tables : Input yield tables 

        '''

        # Get the available lifetimes from the input tables
        lifetimes = tables.get('Lifetime')

        # Get the available stellar masses from the input tables
        masses = []
        sim_ids = tables.get('Table (M,Z)')
        for k in range(len(sim_ids)):
            masses.append(float(sim_ids[k].split(',')[0].split('=')[1]))

        # Define the minimum and maximum mass range for the interpolation
        fit_massmin = self.imf_yields_range_pop3[0]
        fit_massmax = self.imf_yields_range_pop3[1]

        # Define the parameters of the interpolation
        resolution_mass = 2e2
        spline_degree1 = 2  # Other choices : 1, 2
        smoothing1 = 0      # Other choices : 0, 1, 2, 3
        x = np.log10(masses)
        z = np.log10(lifetimes)
        boundary = [None,None]

        # Interpolate the stellar masses to have more of them available
        s = UnivariateSpline(x, z, bbox=boundary, k=spline_degree1, s=smoothing1)
        all_masses = np.linspace(fit_massmin, fit_massmax, \
            len(range(fit_massmin, fit_massmax, 1+1)))
        all_masses = np.linspace(10, 30, len(range(10,30,1)) +1)
        all_masses = np.linspace(10, 30, len(range(10,30,1)) +21+40+20)
        all_masses = np.linspace(10, 30, len(range(10,30,1)) +21+40+20+100)

        # Get the interpolated lifetimes from the interpolated stellar masses
        lifetimes_grid = s(np.log10(np.array(all_masses)))

        # Return Z = 0, interpolated stellar masses and lifetimes
        return [np.array([0.0]), all_masses, np.array([lifetimes_grid])]


    ##############################################
    #        Interpolate Lifetimes Grid          #
    ##############################################
    def __interpolate_lifetimes_grid(self, tables):
        
        '''
        This function interpolates and returns the main sequence lifetimes
        using given input tables (NuGrid as default).  The yields range boundary
        is taken into account in the interpolation.
	 
        Argument
        ========

          tables : Input yield tables

        Return array
        ============

           [[metallicities Z1,Z2,...], [masses], [[log10(lifetimesofZ1)],
           [log10(lifetimesofZ2)],..] ]

        '''

        # Output information
        if self.iolevel >= 2:
            print 'Entering interpolate_lifetimes_grid routine'

        # Read the lifetimes from the given yield tables
        lifetimes_all = tables.get('Lifetime')
        sim_ids = tables.get('Table (M,Z)')

        # Get the available Z, M, and lifetimes
        z_all = []
        m_all = []
        mass = []
        metallicity = []
        lifetimes = []
        for k in range(len(lifetimes_all)):
            m = sim_ids[k].split(',')[0].split('=')[-1]
            z = sim_ids[k].split(',')[1].split('=')[-1][:-1]
            m_all.append(m)
            z_all.append(z)
            if not float(z) in metallicity:
                metallicity.append(float(z))
                mass.append([])
                lifetimes.append([])
            mass[-1].append(float(m))
            lifetimes[-1].append(lifetimes_all[k])
        mass_save=mass
        metallicity_save=metallicity
        lifetimes_save=lifetimes
        # Initialisation of the interpolation parameters for M vs lifetime
        fit_massmin = self.imf_yields_range[0]
        fit_massmax = self.imf_yields_range[1]
        fit_metalmin = 1e-6
        fit_metalmax = 2e-2
        resolution_mass = 2e2
        resolution_metallicity = 1e6 # 100
        spline_degree1 = 2           # 1, 2
        smoothing1 = 0               # 0, 1, 2, 3
        all_masses = []
        spline_metallicity = []

        # Fit lifetimes over the mass ranges
        for m in range(len(metallicity)):
            x=np.log10(mass[m])
            for k in range(len(mass[m])):
                if not mass[m][k] in all_masses:
                    all_masses.append(mass[m][k])
            y=len(mass[m])*[metallicity[m]]
            z=np.log10(lifetimes[m])
            boundary=[None,None]
            s = UnivariateSpline(x,z,bbox=boundary,k=spline_degree1,s=smoothing1)
            spline_metallicity.append(s)

        mmin = np.log10(fit_massmin) # min(all_masses)
        mmax = np.log10(fit_massmax) # max(all_masses)
        all_masses1 = np.linspace(fit_massmin,fit_massmax,2901) 


        # if fit over metallicity not necessary or possible, get separate fit results
	if len(metallicity)==1:
                if iolevel>0:
                        print 'Only 1 metallicity provided, no fit over lifetime' 
        	all_lifetimes=[]
        	for m in range(len(metallicity)):
            		x=all_masses1
	    		all_lifetimes.append(s[m](x))	
        		all_metallicities=metallicity
        		# Creating the return array (masses in solar mass, no log anymore)
        	zm_lifetime_grid = [np.array(all_metallicities), \
            	np.array(all_masses1),np.array(all_lifetimes)]
        	# Return array with M, t, Z (see function definition)
        	return zm_lifetime_grid

        # Declaration of the wanted metallicity step
        all_metallicities = np.array([1e-06, 2e-06, 3e-06, 4e-06, 5e-06, 6e-06, \
                            7e-06, 8e-06, 9e-06, 1e-05, 2e-05, 3e-05, 4e-05, 5e-05,\
                            6e-05, 7e-05, 8e-05, 9e-05, 0.0001, 0.0002, 0.0003, \
                            0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, \
                            0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,\
                            0.01,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018, \
                            0.019,0.02])


 		
        # Initialisation of the interpolation parameters for M vs Z ?
        # for 2 Z do linear interpolation
	if len(metallicity)==2:
		spline_degree1 = 1
                smoothing1 = 3
                if self.iolevel>0:
                        print 'Only 2 metallicities provided, do linear fit over lifetime'	
	else:	
        	spline_degree1 = 1#4           # 1, 2
        	smoothing1 = 1#3               # 0, 1, 2, 3

        # ...
        all_masses = np.linspace(fit_massmin,fit_massmax,2901)
        #all_masses1 = np.linspace(fit_massmin,fit_massmax,2901)
        all_masses = np.log10(all_masses)
        all_lifetimes = []
        for m in range(len(all_masses)):
            all_lifetimes.append([])
            y = []
            z = []
            for k in range(len(spline_metallicity)):
                z.append(spline_metallicity[k](all_masses[m]))  #same mass
                y.append(metallicity[k])
            x = [all_masses[m]]*len(y)
            s = UnivariateSpline(y,z,bbox=boundary,k=spline_degree1,s=smoothing1)
            for metal in all_metallicities:
                all_lifetimes[-1].append(s(metal))

        # Sorting
        all_lifetimes1=[]
        for k in range(len(all_lifetimes[0])):
            all_lifetimes1.append([])
            for j in range(len(all_lifetimes)):
                all_lifetimes1[-1].append(all_lifetimes[j][k])
        all_lifetimes = all_lifetimes1


        # Creating the return array (masses in solar mass, no log anymore)
        zm_lifetime_grid = [np.array(all_metallicities), \
            np.array(all_masses1),np.array(all_lifetimes)]

        # Output information
        if self.iolevel >= 2:
            print 'Routine __interpolate_lifetimes_grid'
            idx = np.where(zm_lifetime_grid[0] ==0.0001)[0][0]
            masses = zm_lifetime_grid[1]
            print 'masses',masses
            masses = all_masses1 #10**masses
            lifetimes = zm_lifetime_grid[2][idx]
            lifetimes = 10**lifetimes
            massescheck = [1.,1.65,3.,5.,7.,15.,25.]
            lifetimescheck = []
            for k in range(len(massescheck)):
                idx = np.where(masses==massescheck[k])[0][0]
                print 'mass',masses[idx]
                print lifetimes[idx]
            print 'end routine'
        if self.iolevel >= 1:
	    for k in range(len(all_metallicities)):
	        plt.plot(all_masses,all_lifetimes[k])
            for k in range(len(metallicity)):
                x=np.log10(mass_save[k])
                y=np.log10(lifetimes_save[k])
                plt.plot(x,y,marker='o',label='Z='+str(metallicity[k]),linestyle='')
                plt.xlabel('log Mini')
                plt.ylabel('log lifetime')
                plt.legend();plt.title('Lifetime fit')
        # Return array with M, t, Z (see function definition)
        return zm_lifetime_grid


    ##############################################
    #           Interpolate Lifetimes            #
    ##############################################
    def __find_lifetimes(self, metallicity, mass, lifetime=0):

        '''
        Method to find lifetimes in the grid calculated by
        the method interpolate_lifetimes_grid. The 3D fit
	is done by using the mass and metallicity dependent
	lifetime values from the 'table' input (which should be
	the nugrid table file). Only those fits are tested.
	If metallicity is above 2e-2 and below 1e-4 lifetimes
	for the 2e-2 and 1e-4 case are used respectively.

        Notes 
        =====

          !! mass and metallicity in grid are log10
          round for 4 digits in (precision):
          Motivation: for M1 (2e-2) timestep size
          with limit precision 4  the last two timesteps:
          (timestep:)  1.320E+10 1.321E+10 1.0 1.00023028502
          (timestep:)  1.321E+10 1.321E+10 1 1.0
          with limit precsion 5:
          (timestep:)  1.320E+10 1.321E+10 1.00009210765 1.00027634839
          (timestep:)  1.321E+10 1.321E+10 1 1.00009210765
          with 5 last timestep gets properly resolved
          > fourth digit is changing!
          difference in mass ~
        
        Arguments
        =========

          metallicity: Metallicity
          mass : if lifetime>0, expect array input:
                   array with [minm,maxm], with entries being in solar masses
	         else: expect float input in solar masses
          lifetime : if lifetimes is 0:
                       Find lifetime for certain mass and metallicity
                       by doing a linear fit between two closest mass points
                       in the z-m-interpolated grid
                     if lifetime > 0:
                       Find

        '''

        # Get the current lifetime grid
        zm_lifetime_grid = self.zm_lifetime_grid_current

        # Print information
        if metallicity == 0:
            if self.iolevel >= 2:
                grid= zm_lifetime_grid		
                print 'zmetal',grid[0]
                print 'masses',np.array(grid[1])
                print 'lifetimes',grid[2]
			
        precision = 5
        if lifetime > 0:
            mrangemin=mass[0]#round(np.log10(mass[0]),precision)
            mrangemax=mass[1]#round(np.log10(mass[1]),precision)
            lifetime=np.log10(lifetime)
            #for retrieving masses, in certain mass range given by 
            #mrangemin,mrangemax
            #for given metallicity Z and lifetime lifetime
            idx_z = (np.abs(zm_lifetime_grid[0]-metallicity)).argmin()
            value_found=False
	    firsttest=True
	    #goes over lifetime intervals starting from the high-lifetime end
            for upper,lower in zip(zm_lifetime_grid[2][idx_z][:-1], \
                zm_lifetime_grid[2][idx_z][1:]):

		#This first test is in case lifetime is out of the grid range,
                #which should not be and is only because of the precision. 
                #If yes, then set it to grid boundary value
		if firsttest==True:
			upper_max=max(zm_lifetime_grid[2][idx_z][:-1])
			lower_max=min(zm_lifetime_grid[2][idx_z][:-1])
			if lifetime>upper_max:
				lifetime=upper_max
			if lifetime<lower_max:
				lifetime=lower_max
		firsttest=False
		#print 'lifetimes',lower,lifetime,upper
		#if lifetime lies in lifetime interval
		lower = min(lower,upper)
		upper = max(lower,upper)
                if (lower <= lifetime) and (lifetime <= upper):
                    #print 'found range, lifetimes',lower,lifetime,upper
                    upper_t=upper
                    upper_t_idx=list(zm_lifetime_grid[2][idx_z]).index(upper_t)
                    lower_m=zm_lifetime_grid[1][upper_t_idx]
                    #check that found mass is in interval given by initial mass bdy
                    lower_t=lower
                    lower_t_idx=list(zm_lifetime_grid[2][idx_z]).index(lower_t)
                    upper_m=zm_lifetime_grid[1][lower_t_idx]
                    #print lower_m,upper_m
                    slope= (lower_m-upper_m)/(upper_t-lower_t)
                    b= upper_m - slope * lower_t
                    found_mass = round( slope*lifetime +b,precision)
                    #nprint 'found mass',mrangemin,found_mass,mrangemax
                    if mrangemin <= found_mass <= mrangemax:
                        #lower_t=lower
                        #lower_t_idx=list(zm_lifetime_grid[2][idx_z]).index(lower_t)
                        #upper_m=zm_lifetime_grid[1][lower_t_idx]
                        #print 'masss itnerval found', found_mass
                        value_found=True
                        break
		    #print 'but its not in right mass range ...continue search'
            if value_found==False:
		print 'Mass for lifetime ',10**lifetime,', not found'
		print 'Throwing error in the following...'
                #a=c
            return found_mass
        else:
	    mass_solar=mass
            idx_z = (np.abs(zm_lifetime_grid[0]-metallicity)).argmin()
	    idx=np.where(zm_lifetime_grid[1]==mass)[0]#[0]
	    #print 'Z',zm_lifetime_grid[0][idx_z]
	    if len(idx)>0:
		    idxfound=idx[0]
		    #idxfound=list(zm_lifetime_grid[1]).index(np.log10(mass))
		    found_lifetime = zm_lifetime_grid[2][idx_z][idxfound]
	    #else interpolate linearly between grid points
	    else:
		    #mass=np.log10(mass)
		    mass=float(mass)
		    for lower, upper in zip(zm_lifetime_grid[1][:-1], zm_lifetime_grid[1][1:]):
			#print lower,upper,mass
			if (lower <= mass) and (mass <= upper):
			    upper_m=upper
			    lower_m=lower
			    #print 'ower_m,upper_m',lower_m,upper_m
			    upper_m_idx=list(zm_lifetime_grid[1]).index(upper)
			    lower_m_idx=list(zm_lifetime_grid[1]).index(lower)
			    break
		    #print  zm_lifetime_grid[1]
		    #print 10**np.array(zm_lifetime_grid[2][idx_z])
		    #linear fit
		    #print 'do linear fit for',zm_lifetime_grid[0][idx_z]
		    upper_t=zm_lifetime_grid[2][idx_z][lower_m_idx]
		    lower_t=zm_lifetime_grid[2][idx_z][upper_m_idx]
		    #print 'upper_t, lower_t',10**upper_t, 10**lower_t
		    slope=(lower_t-upper_t)/(upper_m-lower_m)
		    b= upper_t - slope * lower_m
		    found_lifetime = slope *mass +b
            #idx_m = (np.abs(zm_lifetime_grid[1]-mass)).argmin()
            #found_lifetime=zm_lifetime_grid[2][idx_z][idx_m]

            return 10**found_lifetime


    ##############################################
    #               Get Metallicity              #
    ##############################################
    def _getmetallicity(self, i):

        '''
        Returns the metallicity of the gas reservoir at step i.
        Metals are defined as everything heavier than lithium.

        Argument
        ========
        
          i : Index of the timestep

        '''

        # Return the input Z if the code is forced to always use a specific Z
        if self.hardsetZ >= 0:
            zmetal = self.hardsetZ		
            return zmetal
        
        # Calculate the total mass and the mass of metals
        mgastot = 0.e0
        mmetal = 0.e0 
        nonmetals = ['H-1','H-2','H-3','He-3','He-4','Li-6','Li-7']
        for k in range(len(self.history.isotopes)): 
            mgastot = mgastot + self.ymgal[i][k]
            if not self.history.isotopes[k] in nonmetals:
                mmetal = mmetal + self.ymgal[i][k]

        # In the case where there is no gas left
        if mgastot == 0.0:
            zmetal = 0.0

        # If gas left, calculate the mass fraction of metals
        else:
            zmetal = mmetal / mgastot

        # Output information
        if self.iolevel > 0:
            error = 0
            for k in range(len(self.ymgal[i])):
                if self.ymgal[i][k] < 0:
                    print 'check current ymgal[i] ISM mass'
                    print 'ymgal[i][k]<0',self.ymgal[i][k],self.history.isotopes[k]
                    error = 1
                if error == 1:
                    import sys
                    sys.exit('ERROR: zmetal<0 in getmetal routine')

        # Return the metallicity of the gas reservoir
        return zmetal


    ##############################################
    #               Iso Abu to Elem              #
    ##############################################
    def _iso_abu_to_elem(self, yields_iso):

        '''
        This function converts isotope yields in elements and returns the result. 

        Argument
        ========

          yields_iso : List of yields (isotopes)

        '''

        # Get the list of all the elements
        elements = []
        yields_ele = []
        for iso in self.history.isotopes:
            ele = iso.split('-')[0]
            if not ele in elements:
                elements.append(ele)
                yields_ele.append(yields_iso[self.history.isotopes.index(iso)])
            else:
                yields_ele[-1] += yields_iso[self.history.isotopes.index(iso)]

        # Return the list of elements, and the associated yields
        return elements,yields_ele


    ##############################################
    #                   GetTime                  #
    ##############################################
    def _gettime(self):

        '''
        Return current time.  This is for keeping track of the computational time.

        '''

        out = 'Run time: '+str(round((t_module.time() - self.start_time),2))+"s"
        return out	


    ##############################################
    #               History CLASS                #
    ##############################################
    class __history():

        '''
        Class tracking the evolution of composition, model parameter, etc.
        Allows separation of tracking variables from original code.

        '''

        #############################
        #        Constructor        #
        #############################
        def __init__(self):

            '''
	    Initiate variables tracking.history

            '''

            self.age = []
            self.sfr = []
            self.gas_mass = []
            self.metallicity = []
            self.ism_iso_yield = []
            self.ism_iso_yield_agb = []
            self.ism_iso_yield_massive = []
            self.ism_iso_yield_1a = []
            self.isotopes = []
            self.elements = []
            self.ism_elem_yield = []
            self.ism_elem_yield_agb = []
            self.ism_elem_yield_massive = []
            self.ism_elem_yield_1a = []
            self.sn1a_numbers = []
            self.sn2_numbers = []
	    self.t_m_bdys = []
