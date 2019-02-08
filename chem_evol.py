from __future__ import print_function

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

v0.5 OCT2016: B. Cote, C. Ritter, A. Paul

Stop keeking track of version from now on.

MARCH2018: B. Cote
- Switched to Python 3
- Capability to include radioactive isotopes

JULY2018: B. Cote & R. Sarmento
- Re-wrote (improved) yield and lifetime treatment (B. Cote)
- PopIII IMF and yields update (R. Sarmento)

JAN2019: B. Cote
- Re-included radioactive isotopes with the new (improved) yield treatment

FEB2019: A. Yagüe, B. Cote
- Optimized to code to run faster

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
import sys
import re
from pylab import polyfit
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from imp import *

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

# Import the decay module for radioactive isotopes
#import decay_module


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

    dt_split_info : numpy array
        Information regarding the creation of a timestep array with varying step size.
        Array format : dt_split_info[number of conditions][0->dt,1->upper time limit]
        Exemple : dt_split_info = [[1e6,40e6],[1e8,13e9]] means the timesteps will be
                  of 1 Myr until the time reaches 40 Myr, after which the timesteps
                  will be of 100 Myr until the time reaches 13 Gyr.  The number of
                  "split" is unlimited, but the array must be in chronological order.
        Default value : [] --> Not taken into account

    imf_bdys : list
        Upper and lower mass limits of the initial mass function (IMF) [Mo].

        Default value : [0.1,100]

    imf_yields_range : list
        Initial mass of stars that contribute to stellar ejecta [Mo].

        Default value : [1,30]

    imf_type : string
        Choices : 'salpeter', 'chabrier', 'kroupa', 'alphaimf', 'lognormal'
        'alphaimf' creates a custom IMF with a single power-law covering imf_bdys.
        'lognormal' creates an IMF of the form Exp[1/(2 1^2) Log[x/charMass]^2

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

    imf_pop3_char_mass : float
        The characteristic mass in a log normal IMF distribution.

        Default value : 40.0

    high_mass_extrapolation : string
        Extrapolation technique used to extrapolate yields for stars more
        massive than the most massive model (MMM) present in the yields table.

        Choices:
          "copy" --> This will apply the yields of the most massive model
                 to all more massive stars.

          "scale" --> This will scale the yields of the most massive model
                 using the relation between the total ejected mass and
                 the initial stellar mass.  The later relation is taken
                 from the interpolation of the two most massive models.

          "extrapolate" --> This will extrapolate the yields of the most massive
                 model using the interpolation coefficients taken from
                 the interpolation of the two most massive models.

        Default value : "copy"

    iniZ : float
        Initial metallicity of the gas in mass fraction (e.g. Solar Z = 0.02).

        Choices : 0.0, 0.0001, 0.001, 0.006, 0.01, 0.02
                 (-1.0 to use non-default yield tables)

        Default value : 0.0

    Z_trans : floatRJS
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

        Choices :
                'power_law' - custom power law, set parameter with beta_pow (similar to Maoz & Mannucci 2012)

                'gauss' - gaussian DTD, set parameter with gauss_dtd

                'exp' - exponential DTD, set parameter with exp_dtd

                'maoz' - specific power law from Maoz & Mannucci (2012)

        Default value : 'power_law'

    sn1a_energy : float
        Energy ejected by single SNIa event. Units in erg.

        Default value : 1e51

    ns_merger_on : boolean
        True or False to include or exclude the contribution of neutron star mergers.

        Note : If t_nsm_coal or nsm_dtd_power is not used (see below), the delay time
        distribution of neutron star mergers is given by the standard population synthesis
        models of Dominik et al. (2012), using Z = 0.002 and Z = 0.02.  In this case, the
        total number of neutron star mergers can be tuned using f_binary and f_merger
        (see below).
        Default value : False

    f_binary : float
        Binary fraction for massive stars used to determine the total number of neutron
        star mergers in a simple stellar population.

        Default value : 1.0

    f_merger : float
        Fraction of massive star binary systems that lead to neutron star mergers in a
        simple stellar population.

        Default value : 0.0008

    beta_pow : float
        Slope of the power law for custom SN Ia rate, R = Constant * t^-beta_pow.

        Default value : -1.0

    gauss_dtd : list
        Contains parameter for the gaussian DTD: first the characteristic time [yrs] (gaussian center)
        and then the width of the distribution [yrs].

        Default value : [3.3e9,6.6e8]

    exp_dtd : float
        Characteristic delay time [yrs] for the e-folding DTD.

    nb_1a_per_m : float
        Number of SNe Ia per stellar mass formed in a simple stellar population.

        Default value : 1.0e-03

    direct_norm_1a : float
        Normalization coefficient for SNIa rate integral.

        Default: deactived but replaces the usage of teh nb_1a_per_m when its value is larger than zero.

    transitionmass : float
        Initial mass which marks the transition from AGB to massive stars [Mo].

        Default value : 8.0

    exclude_masses : list
        Contains initial masses in yield tables to be excluded from the simulation;

        Default value : []

    table : string
        Path pointing toward the stellar yield tables for massive and AGB stars.

        Default value : 'yield_tables/agb_and_massive_stars_nugrid_MESAonly_fryer12delay.txt' (NuGrid)

    sn1a_table : string
        Path pointing toward the stellar yield table for SNe Ia.

        Default value : 'yield_tables/sn1a_t86.txt' (Tielemann et al. 1986)

    nsmerger_table : string
        Path pointing toward the r-process yield tables for neutron star mergers

        Default value : 'yield_tables/r_process_rosswog_2014.txt' (Rosswog et al. 2013)

    iniabu_table : string
        Path pointing toward the table of initial abuncances in mass fraction.

        Default value : 'yield_tables/iniabu/iniab2.0E-02GN93.ppn'

    yield_interp : string
        if 'None' : no yield interpolation, no interpolation of total ejecta

        if 'lin' - Simple linear yield interpolation.

        if 'wiersma' - Interpolation method which makes use of net yields
        as used e.g. in Wiersma+ (2009); Does not require net yields.

        if netyields_on is true than makes use of given net yields
        else calculates net yields from given X0 in yield table.

        Default : 'lin'

    netyields_on : boolean
        if true assumes that yields (input from table parameter)
        are net yields.

        Default : false.

    total_ejecta_interp : boolean
        if true then interpolates total ejecta given in yield tables
                  over initial mass range.

        Default : True

    stellar_param_on : boolean
        if true reads in additional stellar parameter given in table stellar_param_table.

        Default : true in sygma and false in omega
        
    stellar_param_table: string
        Path pointoing toward the table hosting the evolution of stellar parameter
        derived from stellar evolution calculations.

        Default table : 'yield_tables/isotope_yield_table_MESA_only_param.txt'

    iolevel : int
        Specifies the amount of output for testing purposes (up to 3).

        Default value : 0

    poly_fit_dtd : list
        Array of polynomial coefficients of a customized delay-time distribution
        function for SNe Ia.  The polynome can be of any order.
        Example : [0.2, 0.3, 0.1] for rate_snIa(t) = 0.2*t**2 + 0.3*t + 0.1
        Note : Must be used with the poly_fit_range parameter (see below)

        Default value : np.array([]) --> Deactivated

    poly_fit_range : list --> [t_min,t_max]
        Time range where the customized delay-time distribution function for
        SNe Ia will be applied for a simple stellar population.

        Default value : np.array([]) --> Deactivated

    mass_sampled : list
        Stellar masses that are sampled to eject yields in a stellar population.
        Warning : The use of this parameter bypasses the IMF calculation and
        do not ensure a correlation with the star formation rate.  Each sampled
        mass will eject the exact amount of mass give in the stellar yields.

        Default value : np.array([]) --> Deactivated

    scale_cor : 2D list
        Determine the fraction of yields ejected for any given stellar mass bin.
        Example : [ [1.0,8], [0.5,100] ] means that stars with initial mass between
        0 and 8 Msu will eject 100% of their yields, and stars with initial mass
        between 8 and 100 will eject 50% of their yields.  There is no limit for
        the number of [%,M_upper_limit] arrays used.

        Default value : np.array([]) --> Deactivated

    t_nsm_coal : float
        When greater than zero, t_nsm_coal sets the delay time (since star formation)
        after which all neutron star mergers occur in a simple stellar population.

        Default value : -1 --> Deactivated

    nsm_dtd_power : 3-index array --> [t_min, t_max, slope_of_the_power_law]
        When used, nsm_dtd_power defines a delay time distribution for neutron
        star mergers in the form of a power law, for a simple stellar population.

        Exemple: [1.e7, 1.e10, -1.] --> t^-1 from 10 Myr to 10 Gyr
        Default value : [] --> Deactivated

    nb_nsm_per_m : float
        Number of neutron star mergers per stellar mass formed in a simple
        stellar population.

        Note : This parameter is only considered when t_nsm_coal or nsm_dtd_power
        is used to define the delay time of neutron star mergers.
        Default value : -1 --> Deactivated

    m_ej_nsm : float
        Mass ejected per neutron star merger event.

        Default value : 2.5e-02

    Delayed extra source
    Adding source that requires delay-time distribution (DTD) functions
    -------------------------------------------------------------------
    delayed_extra_dtd : multi-D Numpy array --> [nb_sources][nb_Z]
        nb_sources is the number of different input astrophysical site (e.g.,
        SNe Ia, neutron star mergers).
        nb_Z is the number of available metallicities.
        delayed_extra_dtd[i][j] is a 2D array in the form of
        [ number_of_times ][ 0-time, 1-rate ].

        Defalut value : np.array([]), deactivated
        
    delayed_extra_dtd_norm : multi-D Numpy array --> [nb_sources]
        Total number of delayed sources occurring per Msun formed,
        for each source and each metallicity.

        Defalut value : np.array([]), deactivated

    delayed_extra_yields : Numpy array of strings
        Path to the yields table for each source.

        Defalut value : np.array([]), deactivated

    delayed extra_yields_norm : multi-D Numpy array --> [nb_sources][nb_Z]
        Fraction of the yield table (float) that will be ejected per event,
        for each source and each metallicity. This will be the mass ejected
        per event if the yields are in mass fraction (normalized to 1).

        Defalut value : np.array([]), deactivated

    delayed_extra_stochastic : Numpy array of Boolean --> [nb_sources]
        Determine whether the DTD provided as an input needs to be
        stochastically sampled using a Monte Carlo technique.

        Defalut value : np.array([]), deactivated

    Run example
    ===========

    See sygma.py and omega.py

    '''


    ##############################################
    ##               Constructor                ##
    ##############################################
    def __init__(self, imf_type='kroupa', alphaimf=2.35, imf_bdys=[0.1,100], \
             sn1a_rate='power_law', iniZ=0.02, dt=1e6, special_timesteps=30, \
             nsmerger_bdys=[8, 100], tend=13e9, mgal=1.6e11, transitionmass=8, iolevel=0, \
             ini_alpha=True, \
             table='yield_tables/agb_and_massive_stars_nugrid_MESAonly_fryer12delay.txt', \
             use_decay_module=False, f_network='isotopes_modified.prn', f_format=1, \
             table_radio='', decay_file='', sn1a_table_radio='',\
             bhnsmerger_table_radio='', nsmerger_table_radio='',\
             hardsetZ=-1, sn1a_on=True, sn1a_table='yield_tables/sn1a_t86.txt',\
             sn1a_energy=1e51, ns_merger_on=False, bhns_merger_on=False,\
             f_binary=1.0, f_merger=0.0008, t_merger_max=1.3e10,\
             m_ej_nsm = 2.5e-02, nb_nsm_per_m=-1.0, \
             t_nsm_coal=-1.0, m_ej_bhnsm=2.5e-02, nsm_dtd_power=[],\
             bhnsmerger_table = 'yield_tables/r_process_arnould_2007.txt', \
             nsmerger_table = 'yield_tables/r_process_arnould_2007.txt',\
             iniabu_table='', extra_source_on=False, \
             extra_source_table=['yield_tables/extra_source.txt'], \
             f_extra_source=[1.0], pre_calculate_SSPs=False,\
             extra_source_mass_range=[[8,30]], \
             extra_source_exclude_Z=[[]], radio_refinement=100, \
             pop3_table='yield_tables/popIII_heger10.txt', \
             imf_bdys_pop3=[0.1,100], imf_yields_range_pop3=[10,30], \
             imf_pop3_char_mass=40.0, \
             high_mass_extrapolation='copy', \
             starbursts=[], beta_pow=-1.0,gauss_dtd=[3.3e9,6.6e8],\
             exp_dtd=2e9,nb_1a_per_m=1.0e-3,direct_norm_1a=-1,Z_trans=0.0, \
             f_arfo=1, imf_yields_range=[1,30],exclude_masses=[],\
             netyields_on=False,wiersmamod=False,yield_interp='lin',\
             total_ejecta_interp=True, tau_ferrini=False,\
             input_yields=False,t_merge=-1.0,stellar_param_on=False,\
             stellar_param_table='yield_tables/stellar_feedback_nugrid_MESAonly.txt',\
             popIII_info_fast=True, out_follows_E_rate=False, \
             t_dtd_poly_split=-1.0, delayed_extra_log=False, \
             delayed_extra_yields_log_int=False, \
             delayed_extra_log_radio=False, delayed_extra_yields_log_int_radio=False, \
             pritchet_1a_dtd=[], ism_ini=np.array([]), ism_ini_radio=np.array([]),\
             nsmerger_dtd_array=np.array([]),\
             bhnsmerger_dtd_array=np.array([]),\
             ytables_in=np.array([]), zm_lifetime_grid_nugrid_in=np.array([]),\
             isotopes_in=np.array([]), ytables_pop3_in=np.array([]),\
             zm_lifetime_grid_pop3_in=np.array([]), ytables_1a_in=np.array([]),\
             ytables_nsmerger_in=np.array([]), dt_in_SSPs=np.array([]),\
             dt_in=np.array([]),dt_split_info=np.array([]),\
             ej_massive=np.array([]), ej_agb=np.array([]),\
             ej_sn1a=np.array([]), ej_massive_coef=np.array([]),\
             ej_agb_coef=np.array([]), ej_sn1a_coef=np.array([]),\
             dt_ssp=np.array([]), poly_fit_dtd_5th=np.array([]),\
             mass_sampled_ssp=np.array([]), scale_cor_ssp=np.array([]),\
             poly_fit_range=np.array([]), SSPs_in=np.array([]),\
             delayed_extra_dtd=np.array([]), delayed_extra_dtd_norm=np.array([]), \
             delayed_extra_yields=np.array([]), delayed_extra_yields_norm=np.array([]), \
             delayed_extra_dtd_radio=np.array([]), delayed_extra_dtd_norm_radio=np.array([]), \
             delayed_extra_yields_radio=np.array([]), \
             delayed_extra_yields_norm_radio=np.array([]), \
             delayed_extra_stochastic=np.array([]), \
             ytables_radio_in=np.array([]), radio_iso_in=np.array([]), \
             ytables_1a_radio_in=np.array([]), ytables_nsmerger_radio_in=np.array([]),\
             test_clayton=np.array([])):

        # Initialize the history class which keeps the simulation in memory
        self.history = self.__history()
        self.const = self.__const()

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
        self.history.nsmerger_bdys = nsmerger_bdys
        self.history.f_binary = f_binary
        self.history.f_merger = f_merger
        self.mgal = mgal
        self.transitionmass = transitionmass
        self.iniZ = iniZ
        self.imf_bdys=imf_bdys
        self.nsmerger_bdys=nsmerger_bdys
        self.popIII_info_fast = popIII_info_fast
        self.imf_bdys_pop3=imf_bdys_pop3
        self.imf_yields_range_pop3=imf_yields_range_pop3
        self.imf_pop3_char_mass=imf_pop3_char_mass # RJS
        self.high_mass_extrapolation = high_mass_extrapolation
        self.extra_source_on = extra_source_on
        self.f_extra_source= f_extra_source
        self.extra_source_mass_range=extra_source_mass_range
        self.extra_source_exclude_Z=extra_source_exclude_Z
        self.pre_calculate_SSPs = pre_calculate_SSPs
        self.SSPs_in = SSPs_in
        self.table = table
        self.iniabu_table = iniabu_table
        self.sn1a_table = sn1a_table
        self.nsmerger_table = nsmerger_table
        self.bhnsmerger_table = bhnsmerger_table
        self.extra_source_table = extra_source_table
        self.pop3_table = pop3_table
        self.hardsetZ = hardsetZ
        self.starbursts = starbursts
        self.imf_type = imf_type
        self.alphaimf = alphaimf
        self.sn1a_on = sn1a_on
        self.sn1a_energy=sn1a_energy
        self.ns_merger_on = ns_merger_on
        self.bhns_merger_on = bhns_merger_on
        self.nsmerger_dtd_array = nsmerger_dtd_array
        self.len_nsmerger_dtd_array = len(nsmerger_dtd_array)
        self.bhnsmerger_dtd_array = bhnsmerger_dtd_array
        self.len_bhnsmerger_dtd_array = len(bhnsmerger_dtd_array)
        self.f_binary = f_binary
        self.f_merger = f_merger
        self.t_merger_max = t_merger_max
        self.m_ej_nsm = m_ej_nsm
        self.m_ej_bhnsm = m_ej_bhnsm
        self.nb_nsm_per_m = nb_nsm_per_m
        self.t_nsm_coal = t_nsm_coal
        self.nsm_dtd_power = nsm_dtd_power
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
        self.nsm_normalized = False # To avoid normalizing NS merger rate more than once
        self.bhnsm_normalized = False # To avoid normalizing BHNS merger rate more than once
        self.f_arfo = f_arfo
        self.imf_yields_range = imf_yields_range
        self.exclude_masses=exclude_masses
        self.netyields_on=netyields_on
        self.wiersmamod=wiersmamod
        self.yield_interp=yield_interp
        self.out_follows_E_rate = out_follows_E_rate
        self.total_ejecta_interp=total_ejecta_interp
        self.tau_ferrini = tau_ferrini
        self.t_merge = t_merge
        self.ism_ini = ism_ini
        self.ism_ini_radio = ism_ini_radio
        self.dt_in = dt_in
        self.dt_in_SSPs = dt_in_SSPs
        self.dt_split_info = dt_split_info
        self.t_dtd_poly_split = t_dtd_poly_split
        self.poly_fit_dtd_5th = poly_fit_dtd_5th
        self.poly_fit_range = poly_fit_range
        self.stellar_param_table = stellar_param_table
        self.stellar_param_on = stellar_param_on
        self.delayed_extra_log = delayed_extra_log
        self.delayed_extra_dtd = delayed_extra_dtd
        self.delayed_extra_dtd_norm = delayed_extra_dtd_norm
        self.delayed_extra_yields = delayed_extra_yields
        self.delayed_extra_yields_norm = delayed_extra_yields_norm
        self.delayed_extra_yields_log_int = delayed_extra_yields_log_int
        self.delayed_extra_stochastic = delayed_extra_stochastic
        self.nb_delayed_extra = len(self.delayed_extra_dtd)
        self.pritchet_1a_dtd = pritchet_1a_dtd
        self.len_pritchet_1a_dtd = len(pritchet_1a_dtd)

        # Attributes associated with radioactive species
        self.table_radio = table_radio
        self.sn1a_table_radio = sn1a_table_radio
        self.bhnsmerger_table_radio = bhnsmerger_table_radio
        self.nsmerger_table_radio = nsmerger_table_radio
        self.decay_file = decay_file
        self.len_decay_file = len(decay_file)
        self.delayed_extra_log_radio = delayed_extra_log_radio
        self.delayed_extra_dtd_radio = delayed_extra_dtd_radio
        self.delayed_extra_dtd_norm_radio = delayed_extra_dtd_norm_radio
        self.delayed_extra_yields_radio = delayed_extra_yields_radio
        self.delayed_extra_yields_norm_radio = delayed_extra_yields_norm_radio
        self.delayed_extra_yields_log_int_radio = delayed_extra_yields_log_int_radio
        self.nb_delayed_extra_radio = len(self.delayed_extra_dtd_radio)
        self.ytables_radio_in = ytables_radio_in
        self.radio_iso_in = radio_iso_in
        self.ytables_1a_radio_in = ytables_1a_radio_in
        self.ytables_nsmerger_radio_in = ytables_nsmerger_radio_in
        self.radio_massive_agb_on = False
        self.radio_sn1a_on = False
        self.radio_nsmerger_on = False
        self.radio_bhnsmerger_on = False
        self.radio_refinement = radio_refinement
        self.test_clayton = test_clayton
        self.use_decay_module = use_decay_module
        if self.use_decay_module:
            print('In construction .. decay_module deactivated.')
            return
            #self.f_network = f_network
            #self.f_format = f_format
            #self.__initialize_decay_module()

        # Normalization of the delayed extra sources
        if self.nb_delayed_extra > 0:
            self.__normalize_delayed_extra()

        # Normalization constants for the Kroupa IMF
        if imf_type == 'kroupa':
            self.p0 = 1.0
            self.p1 = 0.08**(-0.3 + 1.3)
            self.p2 = 0.5**(-1.3 + 2.3)
            self.p3 = 1**(-2.3 +2.3)

        # Define the broken power-law of Ferrini IMF approximation
        self.norm_fer  = [3.1,1.929,1.398,0.9113,0.538,0.3641,0.2972,\
                          0.2814,0.2827,0.298,0.305,0.3269,0.3423,0.3634]
        self.alpha_fer = [0.6,0.35,0.15,-0.15,-0.6,-1.05,-1.4,-1.6,-1.7,\
                          -1.83,-1.85,-1.9,-1.92,-1.94]
        self.m_up_fer  = [0.15,0.2,0.24,0.31,0.42,0.56,0.76,1.05,1.5,\
                          3.16,4.0,10.0,20.0,120]
        for i_fer in range(0,len(self.norm_fer)):
            self.alpha_fer[i_fer] = self.alpha_fer[i_fer] + 1
            self.norm_fer[i_fer] = self.norm_fer[i_fer]/(self.alpha_fer[i_fer])

        # Normalize the IMF to 1 MSun
        self.A_imf = 1.0 / self._imf(self.imf_bdys[0], self.imf_bdys[1], 2)
        self.A_imf_pop3 = 1.0 / self._imf(self.imf_bdys_pop3[0], self.imf_bdys_pop3[1], 2)

        # Parameter that determines if not enough gas is available for star formation
        self.not_enough_gas_count = 0
        self.not_enough_gas = False

        # Check for incompatible inputs - Error messages
        self.__check_inputs()
        if self.need_to_quit:
            return
        # NOTE: This if statement also needs to be in SYGMA and OMEGA!

        # Initialisation of the timesteps
        if len(self.dt_split_info) > 0: # and len(self.ej_massive) == 0:
            timesteps = self.__build_split_dt()
        else:
            timesteps = self.__get_timesteps()
        self.history.timesteps = timesteps
        self.nb_timesteps = len(timesteps)
        if self.pre_calculate_SSPs:
            self.t_ce = np.zeros(self.nb_timesteps)
            self.t_ce[0] = self.history.timesteps[0]
            for i_init in range(1,self.nb_timesteps):
                self.t_ce[i_init] = self.t_ce[i_init-1] + \
                    self.history.timesteps[i_init]

        # Define the decay properties and read radioactive tables
        if self.len_decay_file > 0:
            self.__define_decay_info()
            self.__read_radio_tables()

        # If the yield tables have already been read previously ...
        if input_yields:

            # Assign the input yields and lifetimes
            self.ytables = ytables_in
            self.zm_lifetime_grid_nugrid = zm_lifetime_grid_nugrid_in
            self.history.isotopes = isotopes_in
            self.nb_isotopes = len(self.history.isotopes)
            self.ytables_pop3 = ytables_pop3_in
            self.zm_lifetime_grid_pop3 = zm_lifetime_grid_pop3_in
            self.ytables_1a = ytables_1a_in
            self.ytables_nsmerger = ytables_nsmerger_in
            self.extra_source_on = False
            self.ytables_extra = 0

            # Assign the input yields for radioactive isotopes
            if self.len_decay_file > 0:
                self.ytables_radio = ytables_radio_in
                self.radio_iso = radio_iso_in
                self.nb_radio_iso = len(self.radio_iso)
                self.nb_new_radio_iso = len(self.radio_iso)
                self.ytables_1a_radio = ytables_1a_radio_in
                self.ytables_nsmerger_radio = ytables_nsmerger_radio_in

        # If the yield tables need to be read from the files ...
        else:

            # Read of the yield tables
            self.__read_tables()

            # Declare the interpolation coefficient arrays
            self.__declare_interpolation_arrays()

            # Interpolate the yields tables
            self.__interpolate_pop3_yields()
            self.__interpolate_massive_and_agb_yields()

            # Interpolate lifetimes
            self.__interpolate_pop3_lifetimes()
            self.__interpolate_massive_and_agb_lifetimes()

            # Calculate coefficients to interpolate masses from lifetimes
            self.__interpolate_pop3_m_from_t()
            self.__interpolate_massive_and_agb_m_from_t()

            # If radioactive isotopes are used ..
            if self.len_decay_file > 0:

                # Interpolate the radioactive yields tables
                self.__interpolate_massive_and_agb_yields(is_radio=True)

        # If SSPs needs to be pre-calculated ..
        if self.pre_calculate_SSPs:

            # Calculate all SSPs
            self.__run_all_ssps()

            # Create the arrays that will contain the interpolated isotopes
            self.ej_SSP_int = np.zeros((self.nb_steps_table,self.nb_isotopes))
            if self.len_decay_file > 0:
                self.ej_SSP_int_radio = np.zeros((self.nb_steps_table,self.nb_radio_iso))

        # Initialisation of the composition of the gas reservoir
        ymgal = self._get_iniabu()
        self.len_ymgal = len(ymgal)

        # Initialisation of the storing arrays
        mdot, ymgal, ymgal_massive, ymgal_agb, ymgal_1a, ymgal_nsm, ymgal_bhnsm,\
        ymgal_delayed_extra, mdot_massive, mdot_agb, mdot_1a, mdot_nsm,\
        mdot_bhnsm, mdot_delayed_extra, sn1a_numbers, sn2_numbers, nsm_numbers,\
        bhnsm_numbers, delayed_extra_numbers, imf_mass_ranges, \
        imf_mass_ranges_contribution, imf_mass_ranges_mtot = \
        self._get_storing_arrays(ymgal, len(self.history.isotopes))

        # Initialisation of the composition of the gas reservoir
        if len(self.ism_ini) > 0:
            for i_ini in range(0,self.len_ymgal):
                ymgal[0][i_ini] = self.ism_ini[i_ini]

        # If radioactive isotopes are used ..
        if self.len_decay_file > 0:

            # Define initial radioactive gas composition
            ymgal_radio = np.zeros(self.nb_radio_iso)

            # Initialisation of the storing arrays for radioactive isotopes
            mdot_radio, ymgal_radio, ymgal_massive_radio, ymgal_agb_radio,\
            ymgal_1a_radio, ymgal_nsm_radio, ymgal_bhnsm_radio,\
            ymgal_delayed_extra_radio, mdot_massive_radio, mdot_agb_radio,\
            mdot_1a_radio, mdot_nsm_radio, mdot_bhnsm_radio,\
            mdot_delayed_extra_radio, dummy, dummy, dummy, dummy, dummy, \
            dummy, dummy, dummy = \
            self._get_storing_arrays(ymgal_radio, self.nb_radio_iso)

            # Initialisation of the composition of the gas reservoir
            if len(self.ism_ini_radio) > 0:
                for i_ini in range(0,len(self.ism_ini_radio)):
                    ymgal_radio[0][i_ini] = self.ism_ini_radio[i_ini]

            # Define indexes to make connection between unstable/stable isotopes
            self.__define_unstab_stab_indexes()

        # Output information
        if iolevel >= 1:
             print ('Number of timesteps: ', '{:.1E}'.format(len(timesteps)))

        # Create empty arrays if on the fast mode
        if self.pre_calculate_SSPs:
            self.history.gas_mass.append(sum(ymgal[0]))
            self.history.ism_iso_yield.append(ymgal[0])
            self.history.m_locked = []
            self.history.m_locked_agb = []
            self.history.m_locked_massive = []
            self.massive_ej_rate = []
            self.sn1a_ej_rate = []

        # Add the initialized arrays to the history class
        else:
            self.history.gas_mass.append(sum(ymgal[0]))
            self.history.ism_iso_yield.append(ymgal[0])
            self.history.ism_iso_yield_agb.append(ymgal_agb[0])
            self.history.ism_iso_yield_1a.append(ymgal_1a[0])
            self.history.ism_iso_yield_nsm.append(ymgal_nsm[0])
            self.history.ism_iso_yield_bhnsm.append(ymgal_bhnsm[0])
            self.history.ism_iso_yield_massive.append(ymgal_massive[0])
            self.history.sn1a_numbers.append(0)
            self.history.nsm_numbers.append(0)
            self.history.bhnsm_numbers.append(0)
            self.history.sn2_numbers.append(0)
            self.history.m_locked = []
            self.history.m_locked_agb = []
            self.history.m_locked_massive = []

            # Keep track of the mass-loss rate of massive stars and SNe Ia
            self.massive_ej_rate = []
            for k in range(self.nb_timesteps + 1):
                self.massive_ej_rate.append(0.0)
            self.sn1a_ej_rate = []
            for k in range(self.nb_timesteps + 1):
                self.sn1a_ej_rate.append(0.0)

        # Attribute arrays and variables to the current object
        self.mdot = mdot
        self.ymgal = ymgal
        self.ymgal_massive = ymgal_massive
        self.ymgal_agb = ymgal_agb
        self.ymgal_1a = ymgal_1a
        self.ymgal_nsm = ymgal_nsm
        self.ymgal_bhnsm = ymgal_bhnsm
        self.ymgal_delayed_extra = ymgal_delayed_extra
        self.mdot_massive = mdot_massive
        self.mdot_agb = mdot_agb
        self.mdot_1a = mdot_1a
        self.mdot_nsm = mdot_nsm
        self.mdot_bhnsm = mdot_bhnsm
        self.mdot_delayed_extra = mdot_delayed_extra
        self.sn1a_numbers = sn1a_numbers
        self.nsm_numbers = nsm_numbers
        self.bhnsm_numbers = bhnsm_numbers
        self.delayed_extra_numbers = delayed_extra_numbers
        self.sn2_numbers = sn2_numbers
        self.imf_mass_ranges = imf_mass_ranges
        self.imf_mass_ranges_contribution = imf_mass_ranges_contribution
        self.imf_mass_ranges_mtot = imf_mass_ranges_mtot

        # Attribute radioactive arrays and variables to the current object
        if self.len_decay_file > 0:
            self.mdot_radio = mdot_radio
            self.ymgal_radio = ymgal_radio
            self.ymgal_massive_radio = ymgal_massive_radio
            self.ymgal_agb_radio = ymgal_agb_radio
            self.ymgal_1a_radio = ymgal_1a_radio
            self.ymgal_nsm_radio = ymgal_nsm_radio
            self.ymgal_bhnsm_radio = ymgal_bhnsm_radio
            self.ymgal_delayed_extra_radio = ymgal_delayed_extra_radio
            self.mdot_massive_radio = mdot_massive_radio
            self.mdot_agb_radio = mdot_agb_radio
            self.mdot_1a_radio = mdot_1a_radio
            self.mdot_nsm_radio = mdot_nsm_radio
            self.mdot_bhnsm_radio = mdot_bhnsm_radio
            self.mdot_delayed_extra_radio = mdot_delayed_extra_radio

        # Set the initial time and metallicity
        zmetal = self._getmetallicity(0)
        self.history.metallicity.append(zmetal)
        self.t = 0
        self.history.age.append(self.t)
        self.zmetal = zmetal

        # Define the element to isotope index connections
        self.__get_elem_to_iso_main()

        # Get coefficients for the fraction of white dwarfs fit (2nd poly)
        if not pre_calculate_SSPs:
            self.__get_coef_wd_fit()

        # Output information
        if iolevel > 0:
            print ('### Start with initial metallicity of ','{:.4E}'.format(zmetal))
            print ('###############################')

    ##############################################
    #             Get elem-to-iso Main           #
    ##############################################
    def __get_elem_to_iso_main(self):

        # Get the list of elements
        self.history.elements = []
        self.i_elem_for_iso = np.zeros(self.nb_isotopes,dtype=int)
        for i_iso in range(self.nb_isotopes):
            the_elem = self.history.isotopes[i_iso].split('-')[0]
            if not the_elem in self.history.elements:
                self.history.elements.append(the_elem)
            i_elem = self.history.elements.index(the_elem)
            self.i_elem_for_iso[i_iso] = i_elem
        self.nb_elements = len(self.history.elements)

    ##############################################
    #                 Check Inputs               #
    ##############################################
    def __check_inputs(self):

        '''
        This function checks for incompatible input entries and stops
        the simulation if needed.

        '''
        
        self.need_to_quit = False
        # Total duration of the simulation
        if self.history.tend > 1.5e10:
            print ('Error - tend must be less than or equal to 1.5e10 years.')
            self.need_to_quit = True

        # Timestep
        if self.history.dt > self.history.tend:
            print ('Error - dt must be smaller or equal to tend.')
            self.need_to_quit = True

        # Transition mass between AGB and massive stars
        #if #(self.transitionmass <= 7)or(self.transitionmass > 12):
           # print ('Error - transitionmass must be between 7 and 12 Mo.')
           # self.need_to_quit = True

        # IMF
        if not self.imf_type in ['salpeter','chabrier','kroupa','input', \
            'alphaimf','chabrieralpha','fpp', 'kroupa93', 'lognormal']:
            print ('Error - Selected imf_type is not available.')
            self.need_to_quit = True

        # IMF yields range
        #if self.imf_yields_range[0] < 1:
        #    print ('Error - imf_yields_range lower boundary must be >= 1.')
            #self.need_to_quit = True
        
        #if (self.imf_yields_range[0] >= self.imf_bdys[1]) or \
        #   (self.imf_yields_range[0] <= self.imf_bdys[0]) or \
         #  (self.imf_yields_range[1] >= self.imf_bdys[1]):
        if ((self.imf_yields_range[0] >  self.imf_bdys[1]) or \
            (self.imf_yields_range[1] < self.imf_bdys[0])):
            print ('Error - part of imf_yields_range must be within imf_bdys.')
            self.need_to_quit = True
        if (self.transitionmass<self.imf_yields_range[0])\
             or (self.transitionmass>self.imf_yields_range[1]):
            print ('Error - Transitionmass outside imf yield range')
            self.need_to_quit = True
        if self.ns_merger_on:
            if ((self.nsmerger_bdys[0] > self.imf_bdys[1]) or \
                (self.nsmerger_bdys[1] < self.imf_bdys[0])):
                print ('Error - part of nsmerger_bdys must be within imf_bdys.')
                self.need_to_quit = True

        # SN Ia delay-time distribution function
        if not self.history.sn1a_rate in \
            ['exp','gauss','maoz','power_law']:
            print ('Error - Selected sn1a_rate is not available.')
            self.need_to_quit = True

        # Initial metallicity for the gas
        #if not self.iniZ in [0.0, 0.0001, 0.001, 0.006, 0.01, 0.02]:
        #    print ('Error - Selected iniZ is not available.')
        #    self.need_to_quit = True

        # If popIII stars are used ...
        if self.iniZ == 0.0:

            # IMF and yield boundary ranges
            if (self.imf_yields_range_pop3[0] >= self.imf_bdys_pop3[1]) or \
               (self.imf_yields_range_pop3[1] <= self.imf_bdys_pop3[0]):
                print ('Error - imf_yields_range_pop3 must be within imf_bdys_pop3.')
                self.need_to_quit = True
              
            if self.netyields_on == True and self.Z_trans > 0.0:
                print ('Error - net yields setting not usable with PopIII at the moment.')
                self.need_to_quit = True

        # If input poly fit DTD, the applicable range must be specified
        if len(self.poly_fit_dtd_5th) > 0:
            if not len(self.poly_fit_range) == 2:
                print ('Error - poly_fit_range must be specified when ',\
                      'using the poly_fit_dtd_5th parameter the SNe Ia DTD.')
                self.need_to_quit = True

        if self.extra_source_on:
             lt=len(self.extra_source_table)
             lf=len(self.f_extra_source)
             lmr=len(self.extra_source_mass_range)
             #leZ=len(self.extra_source_exclude_Z)
             if (not lt == lf):
                 print ('Error - parameter extra_source_table and f_extra_source not of equal size')
                 self.need_to_quit = True
             if (not lt == lmr):
                 print ('Error - parameter extra_source_table and  extra_source_mass_range not of equal size')
                 self.need_to_quit = True
             #if  (not lt == leZ):
             #    print ('Error - parameter extra_source_table and extra_source_exclude_Z not of equal size')
             #    self.need_to_quit = True

        # Use of radioactive isotopes
        if self.len_decay_file > 0 and (len(self.table_radio) == 0 and \
           len(self.sn1a_table_radio) == 0 and len(self.bhnsmerger_table_radio) == 0 and \
           len(self.nsmerger_table_radio) == 0 and self.nb_delayed_extra_radio == 0):
            print ('Error -  At least one radioactive yields table must '+\
                  'be defined when using radioactive isotopes.')
            self.need_to_quit = True
        elif self.len_decay_file > 0:
            if self.yield_interp == 'wiersma':
                print ('Error - Radioactive isotopes cannot be used with net yields .. for now.')
                self.need_to_quit = True
            if self.Z_trans > 0.0:
                print ('Error - Radioactive isotopes cannot be used with PopIII stars .. for now.')
                self.need_to_quit = True


    ##############################################
    #               Read Decay Info              #
    ##############################################
    def __define_decay_info(self):

        '''
        This function reads decay_file and create the decay_info array
        to be used when radioactive isotopes are used.

        '''

        # Declare the decay_info array
        # decay_info[nb_unstable_iso][0] --> Unstable isotope
        # decay_info[nb_unstable_iso][1] --> Stable isotope where it decays
        # decay_info[nb_unstable_iso][2] --> Mean-live (ln2*half-life)[yr]
        self.decay_info = []

        # Open the input file
        with open(global_path + self.decay_file, 'r') as ddi:

            # For each line in the input file ..
            for line in ddi:

                # Split the line and add the information in the decay_info array
                line_split = [str(x) for x in line.split()]
                if line_split[0][0] == '&':
                    self.decay_info.append(\
                            [line_split[0].split('&')[1],\
                             line_split[1].split('&')[1],\
                             float(line_split[2].split('&')[1])/np.log(2.0)])

        # Count the number of radioactive isotopes
        self.nb_radio_iso = len(self.decay_info)
        self.nb_new_radio_iso = len(self.decay_info)

        # Close the input file
        ddi.close()


    ##############################################
    #               Read Radio Tables            #
    ##############################################
    def __read_radio_tables(self):

        '''
        This function reads the radioactive isotopes yields using the
        decay_file and decay_info parameters to define which isosoptes
        are considered.

        '''

        # Create the list of radioactive isotopes considered
        self.radio_iso = []
        for i_r in range(0,self.nb_radio_iso):
            self.radio_iso.append(self.decay_info[i_r][0])

        # Massive and AGB stars
        if len(self.table_radio) > 0:
            self.radio_massive_agb_on = True
            self.ytables_radio = ry.read_nugrid_yields(global_path+self.table_radio,\
                excludemass=self.exclude_masses, isotopes=self.radio_iso)

        # SNe Ia
        sys.stdout.flush()
        if len(self.sn1a_table_radio) > 0:
            self.radio_sn1a_on = True
            self.ytables_1a_radio = ry.read_yield_sn1a_tables( \
                global_path + self.sn1a_table_radio, self.radio_iso)

        # NS mergers
        if len(self.nsmerger_table_radio) > 0:
            self.radio_nsmerger_on = True
            self.ytables_nsmerger_radio = ry.read_yield_sn1a_tables( \
                global_path + self.nsmerger_table_radio, self.radio_iso)

        # BHNS mergers
        if len(self.bhnsmerger_table_radio) > 0:
            self.radio_bhnsmerger_on = True
            self.ytables_bhnsmerger_radio = ry.read_yield_sn1a_tables( \
                global_path + self.bhnsmerger_table_radio, self.radio_iso)

        # Delayed extra sources
        if self.nb_delayed_extra_radio > 0:
            self.ytables_delayed_extra_radio = []
            for i_syt in range(0,self.nb_delayed_extra_radio):
              self.ytables_delayed_extra_radio.append(ry.read_yield_sn1a_tables( \
              global_path + self.delayed_extra_yields_radio[i_syt], self.radio_iso))


    ##############################################
    #                 Read Tables                #
    ##############################################
    def __read_tables(self):

        '''
        This function reads the isotopes yields table for different sites

        '''

        # Massive stars and AGB stars
        if self.table[0] == '/':
            self.ytables = ry.read_nugrid_yields(\
                self.table, excludemass=self.exclude_masses)
        else:
            self.ytables = ry.read_nugrid_yields(\
                global_path+self.table, excludemass=self.exclude_masses)

        # Get the list of isotopes
        # The massive and AGB star yields set the list of isotopes
        M_temp = float(self.ytables.table_mz[0].split(',')[0].split('=')[1])
        Z_temp = float(self.ytables.table_mz[0].split(',')[1].split('=')[1][:-1])
        self.history.isotopes = self.ytables.get(\
            Z=Z_temp, M=M_temp, quantity='Isotopes')
        self.nb_isotopes = len(self.history.isotopes)

        # PopIII massive stars
        self.ytables_pop3 = ry.read_nugrid_yields( \
            global_path+self.pop3_table, self.history.isotopes, \
                excludemass=self.exclude_masses)

        # SNe Ia
        #sys.stdout.flush()
        self.ytables_1a = ry.read_yield_sn1a_tables( \
            global_path+self.sn1a_table, self.history.isotopes)

        # Neutron star mergers
        self.ytables_nsmerger = ry.read_yield_sn1a_tables( \
            global_path+self.nsmerger_table, self.history.isotopes)

        # Black hole neutron star mergers
        self.ytables_bhnsmerger = ry.read_yield_sn1a_tables( \
            global_path+self.bhnsmerger_table, self.history.isotopes)

        # Delayed-extra sources
        if self.nb_delayed_extra > 0:
          self.ytables_delayed_extra = []
          for i_syt in range(0,self.nb_delayed_extra):
            self.ytables_delayed_extra.append(ry.read_yield_sn1a_tables( \
            global_path+self.delayed_extra_yields[i_syt], self.history.isotopes))

        # Extra yields (on top of massive and AGB yields)
        if self.extra_source_on == True:

            #go over all extra sources
            self.ytables_extra =[]
            for ee in range(len(self.extra_source_table)):

               #if absolute path don't apply global_path
               if self.extra_source_table[ee][0] == '/':
                   self.ytables_extra.append( ry.read_yield_sn1a_tables( \
                        self.extra_source_table[ee], self.history.isotopes))
               else:
                   self.ytables_extra.append( ry.read_yield_sn1a_tables( \
                     global_path+self.extra_source_table[ee], self.history.isotopes))

        # Read stellar parameter. stellar_param
        if self.stellar_param_on:
            table_param=ry.read_nugrid_parameter(global_path + self.stellar_param_table)
            self.table_param=table_param

        # Get the list of mass and metallicities found in the yields tables
        self.__get_M_Z_models()


    ##############################################
    #                Get M Z Models              #
    ##############################################
    def __get_M_Z_models(self):

        '''
        Get the mass and metallicities of the input stellar yields

        '''

        # Main massive and AGB star yields
        self.Z_table = self.ytables.metallicities
        self.M_table = []
        for model in self.ytables.table_mz:
            the_Z = float(model.split(',')[1].split('=')[1].split(')')[0])
            if not the_Z == self.Z_table[0]:
                break
            self.M_table.append(float(model.split(',')[0].split('=')[1]))
        self.nb_Z_table = len(self.Z_table)
        self.nb_M_table = len(self.M_table)

        # Massive PopIII stars
        Z_table_pop3 = self.ytables_pop3.metallicities
        self.M_table_pop3 = []
        for model in self.ytables_pop3.table_mz:
            self.M_table_pop3.append(float(model.split(',')[0].split('=')[1]))
            the_Z = float(model.split(',')[1].split('=')[1].split(')')[0])
            if not the_Z == Z_table_pop3[0]:
                break
        self.nb_M_table_pop3 = len(self.M_table_pop3)


    ##############################################
    #          Interpolate Pop3 Yields           #
    ##############################################
    def __interpolate_pop3_yields(self):

        '''
        Interpolate the mass-dependent yields table of massive
        popIII yields.  This will create arrays containing interpolation
        coefficients.  The chemical evolution calculations will then
        only use these coefficients instead of the yields table.

        Interpolation laws
        ==================

          Interpolation across stellar mass M
            log10(yields) = a_M * M + b_M

          Interpolation (total mass) across stellar mass
            M_ej = a_ej * M + b_ej

        Results
        =======

          a_M and b_M coefficients
          ------------------------
            y_coef_M_pop3[i_coef][i_M_low][i_iso]
              - i_coef : 0 and 1 for a_M and b_M, respectively
              - i_M_low : Index of the lower mass limit where
                          the interpolation occurs
              - i_iso : Index of the isotope

        '''

        # For each interpolation lower-mass bin point ..
        for i_M in range(self.nb_inter_M_points_pop3-1):

            # Get the yields for the lower and upper mass models
            yields_low, yields_upp, m_ej_low, m_ej_upp, yields_ej_low,\
                    yields_ej_upp = self.__get_y_low_upp_pop3(i_M)

            # Get the interpolation coefficients a_M, b_M
            self.y_coef_M_pop3[0][i_M], self.y_coef_M_pop3[1][i_M],\
                self.y_coef_M_ej_pop3[0][i_M], self.y_coef_M_ej_pop3[1][i_M] =\
                self.__get_inter_coef_M(self.inter_M_points_pop3[i_M],\
                self.inter_M_points_pop3[i_M+1], yields_low, yields_upp,\
                m_ej_low, m_ej_upp, yields_ej_low, yields_ej_upp)


    ##############################################
    #     Interpolate Massive and AGB Yields     #
    ##############################################
    def __interpolate_massive_and_agb_yields(self, is_radio=False):

        '''
        Interpolate the metallicity- and mass-dependent yields
        table of massive and AGB stars.  This will create arrays
        containing interpolation coefficients.  The chemical
        evolution calculations will then only use these
        coefficients instead of the yields table.

        Interpolation laws
        ==================

          Interpolation across stellar mass M
            log10(yields) = a_M * M + b_M

          Interpolation (total mass) across stellar mass
            M_ej = a_ej * M + b_ej

          Interpolation of a_M and b_M across metallicity Z
            x_M    = a_Z * log10(Z) + b_Z
            x_M_ej = a_Z * log10(Z) + b_Z

          The functions first calculate a_M and b_M for each Z,
          and then interpolate these coefficients across Z.

        Results
        =======

          a_M and b_M coefficients
          ------------------------
            y_coef_M[i_coef][i_Z][i_M_low][i_iso]
              - i_coef : 0 and 1 for a_M and b_M, respectively
              - i_Z : Metallicity index available in the table
              - i_M_low : Index of the lower mass limit where
                          the interpolation occurs
              - i_iso : Index of the isotope

          a_Z and b_Z coefficients for x_M
          --------------------------------
            y_coef_Z_xM[i_coef][i_Z_low][i_M_low][i_iso]
              - i_coef : 0 and 1 for a_Z and b_Z, respectively
              - i_Z_low : Index of the lower metallicity limit where
                          the interpolation occurs
              - i_M_low : Index of the lower mass limit where
                          the interpolation occurs
              - i_iso : Index of the isotope

        Note
        ====

          self.Z_table is in decreasing order
          but y_coef_... arrays have metallicities in increasing order

        '''

        # Fill the y_coef_M array
        # For each metallicity available in the yields ..
        for i_Z_temp in range(self.nb_Z_table):

            # Get the metallicity index in increasing order
            i_Z = self.inter_Z_points.index(self.Z_table[i_Z_temp])

            # For each interpolation lower-mass bin point ..
            for i_M in range(self.nb_inter_M_points-1):

                # Get the yields for the lower and upper mass models
                yields_low, yields_upp, m_ej_low, m_ej_upp, yields_ej_low,\
                    yields_ej_upp = self.__get_y_low_upp(i_Z_temp, i_M, \
                        is_radio=is_radio)

                # Get the interpolation coefficients a_M, b_M
                if is_radio: # Ignore the total mass ejected (done when stable)
                    self.y_coef_M_radio[0][i_Z][i_M], self.y_coef_M_radio[1][i_Z][i_M],\
                        dummy, dummy = self.__get_inter_coef_M(self.inter_M_points[i_M],\
                        self.inter_M_points[i_M+1], yields_low, yields_upp,\
                        1.0, 2.0, yields_upp, yields_upp)
                else:
                    self.y_coef_M[0][i_Z][i_M], self.y_coef_M[1][i_Z][i_M],\
                        self.y_coef_M_ej[0][i_Z][i_M], self.y_coef_M_ej[1][i_Z][i_M] =\
                        self.__get_inter_coef_M(self.inter_M_points[i_M],\
                        self.inter_M_points[i_M+1], yields_low, yields_upp,\
                        m_ej_low, m_ej_upp, yields_ej_low, yields_ej_upp)

        # Fill the y_coef_Z_xM arrays
        # For each interpolation lower-metallicity point ..
        for i_Z in range(self.nb_inter_Z_points-1):

            # For each interpolation lower-mass bin point ..
            for i_M in range(self.nb_inter_M_points-1):

                # If radioactive table ..
                if is_radio:

                    # Get the interpolation coefficients a_Z, b_Z for a_M
                    self.y_coef_Z_aM_radio[0][i_Z][i_M], self.y_coef_Z_aM_radio[1][i_Z][i_M],\
                        dummy, dummy =\
                            self.__get_inter_coef_Z(self.y_coef_M_radio[0][i_Z][i_M],\
                                self.y_coef_M_radio[0][i_Z+1][i_M], self.y_coef_M_ej[0][i_Z][i_M],\
                                    self.y_coef_M_ej[0][i_Z+1][i_M], self.inter_Z_points[i_Z],\
                                        self.inter_Z_points[i_Z+1])

                    # Get the interpolation coefficients a_Z, b_Z for b_M
                    self.y_coef_Z_bM_radio[0][i_Z][i_M], self.y_coef_Z_bM_radio[1][i_Z][i_M],\
                        dummy, dummy =\
                            self.__get_inter_coef_Z(self.y_coef_M_radio[1][i_Z][i_M],\
                                self.y_coef_M_radio[1][i_Z+1][i_M], self.y_coef_M_ej[1][i_Z][i_M],\
                                    self.y_coef_M_ej[1][i_Z+1][i_M], self.inter_Z_points[i_Z],\
                                        self.inter_Z_points[i_Z+1])

                # If stable table ..
                else:

                    # Get the interpolation coefficients a_Z, b_Z for a_M
                    self.y_coef_Z_aM[0][i_Z][i_M], self.y_coef_Z_aM[1][i_Z][i_M],\
                        self.y_coef_Z_aM_ej[0][i_Z][i_M], self.y_coef_Z_aM_ej[1][i_Z][i_M] =\
                            self.__get_inter_coef_Z(self.y_coef_M[0][i_Z][i_M],\
                                self.y_coef_M[0][i_Z+1][i_M], self.y_coef_M_ej[0][i_Z][i_M],\
                                    self.y_coef_M_ej[0][i_Z+1][i_M], self.inter_Z_points[i_Z],\
                                        self.inter_Z_points[i_Z+1])

                    # Get the interpolation coefficients a_Z, b_Z for b_M
                    self.y_coef_Z_bM[0][i_Z][i_M], self.y_coef_Z_bM[1][i_Z][i_M],\
                        self.y_coef_Z_bM_ej[0][i_Z][i_M], self.y_coef_Z_bM_ej[1][i_Z][i_M] =\
                            self.__get_inter_coef_Z(self.y_coef_M[1][i_Z][i_M],\
                                self.y_coef_M[1][i_Z+1][i_M], self.y_coef_M_ej[1][i_Z][i_M],\
                                    self.y_coef_M_ej[1][i_Z+1][i_M], self.inter_Z_points[i_Z],\
                                        self.inter_Z_points[i_Z+1])


    ##############################################
    #        Declare Interpolation Arrays        #
    ##############################################
    def __declare_interpolation_arrays(self):

        '''
        Declare the arrays that will contain the interpolation
        coefficients used in the chemical evolution calculation.

        '''

        # Non-zero metallicity models
        # ===========================

        # Create the stellar mass and lifetime points in between
        # which there will be interpolations
        self.__create_inter_M_points()
        self.__create_inter_lifetime_points()

        # Create the stellar metallicity points in between
        # which there will be interpolations
        self.inter_Z_points = sorted(self.Z_table)
        self.nb_inter_Z_points = len(self.inter_Z_points)
        
        # Declare the array containing the coefficients for
        # the yields interpolation between masses (a_M, b_M)
        self.y_coef_M = np.zeros((2, self.nb_Z_table,\
            self.nb_inter_M_points-1, self.nb_isotopes))

        # Declare the array containing the coefficients for
        # the total-mass-ejected interpolation between masses (a_ej, b_ej)
        self.y_coef_M_ej = np.zeros((2, self.nb_Z_table, self.nb_inter_M_points-1))

        # Declare the array containing the coefficients for
        # the yields interpolation between metallicities (a_Z, b_Z)
        self.y_coef_Z_aM = np.zeros((2, self.nb_Z_table-1,\
            self.nb_inter_M_points-1, self.nb_isotopes))
        self.y_coef_Z_bM = np.zeros((2, self.nb_Z_table-1,\
            self.nb_inter_M_points-1, self.nb_isotopes))

        # Declare the array containing the coefficients for
        # the total-mass-ejected interpolation between metallicities (a_ej, b_ej)
        self.y_coef_Z_aM_ej = np.zeros((2, self.nb_Z_table-1, self.nb_inter_M_points-1))
        self.y_coef_Z_bM_ej = np.zeros((2, self.nb_Z_table-1, self.nb_inter_M_points-1))

        # Declare the array containing the coefficients for
        # the lifetime interpolation between masses (a_M, b_M)
        self.tau_coef_M = np.zeros((2, self.nb_Z_table, self.nb_M_table-1))
        self.tau_coef_M_inv = np.zeros((2, self.nb_Z_table, self.nb_inter_lifetime_points-1))

        # Declare the array containing the coefficients for
        # the lifetime interpolation between metallicities (a_Z, b_Z)
        self.tau_coef_Z_aM = np.zeros((2, self.nb_Z_table-1, self.nb_M_table-1))
        self.tau_coef_Z_bM = np.zeros((2, self.nb_Z_table-1, self.nb_M_table-1))
        self.tau_coef_Z_aM_inv = np.zeros((2, self.nb_Z_table-1, self.nb_inter_lifetime_points-1))
        self.tau_coef_Z_bM_inv = np.zeros((2, self.nb_Z_table-1, self.nb_inter_lifetime_points-1))

        # Zero metallicity models
        # =======================

        # Create the stellar mass and lifetime points in between
        # which there will be interpolations
        self.__create_inter_M_points_pop3()
        self.__create_inter_lifetime_points_pop3()

        # Declare the array containing the coefficients for
        # the PopIII yields interpolation between masses (a_M, b_M)
        self.y_coef_M_pop3 = np.zeros((2,\
            self.nb_inter_M_points_pop3-1, self.nb_isotopes))

        # Declare the array containing the coefficients for
        # the PopIII total-mass-ejected interpolation between masses (a_ej, b_ej)
        self.y_coef_M_ej_pop3 = np.zeros((2, self.nb_inter_M_points_pop3-1))

        # Declare the array containing the coefficients for
        # the lifetime interpolation between masses (a_M, b_M)
        self.tau_coef_M_pop3 = np.zeros((2, self.nb_M_table_pop3-1))
        self.tau_coef_M_pop3_inv = np.zeros((2, self.nb_inter_lifetime_points_pop3-1))

        # Radioactive isotopes (non-zero metallicity)
        # ===========================================
        if self.len_decay_file > 0:
        
            # Declare the array containing the coefficients for
            # the yields interpolation between masses (a_M, b_M)
            self.y_coef_M_radio = np.zeros((2, self.nb_Z_table,\
                self.nb_inter_M_points-1, self.nb_radio_iso))

            # Declare the array containing the coefficients for
            # the yields interpolation between metallicities (a_Z, b_Z)
            self.y_coef_Z_aM_radio = np.zeros((2, self.nb_Z_table-1,\
                self.nb_inter_M_points-1, self.nb_radio_iso))
            self.y_coef_Z_bM_radio = np.zeros((2, self.nb_Z_table-1,\
                self.nb_inter_M_points-1, self.nb_radio_iso))


    ##############################################
    #         Create Inter M Points Pop3         #
    ##############################################
    def __create_inter_M_points_pop3(self):

        '''
        Create the boundary stellar masses array representing
        the mass points in between which there will be yields
        interpolations.  This is for massive PopIII stars.

        '''

        # Initialize the array
        self.inter_M_points_pop3 = copy.deepcopy(self.M_table_pop3)

        # Add the lower and upper IMF yields range limits
        if not self.imf_yields_range_pop3[0] in self.M_table_pop3:
            self.inter_M_points_pop3.append(self.imf_yields_range_pop3[0])
        if not self.imf_yields_range_pop3[1] in self.M_table_pop3:
            self.inter_M_points_pop3.append(self.imf_yields_range_pop3[1])

        # Remove masses that are below or beyond the IMF yields range
        len_temp = len(self.inter_M_points_pop3)
        for i_m in range(len_temp):
            ii_m = len_temp - i_m - 1
            if self.inter_M_points_pop3[ii_m] < self.imf_yields_range_pop3[0] or\
               self.inter_M_points_pop3[ii_m] > self.imf_yields_range_pop3[1]:
                self.inter_M_points_pop3.remove(self.inter_M_points_pop3[ii_m])

        # Sort the list of masses
        self.inter_M_points_pop3 = sorted(self.inter_M_points_pop3)
        self.inter_M_points_pop3_tree = self._bin_tree(self.inter_M_points_pop3)

        # Calculate the number of interpolation mass-points
        self.nb_inter_M_points_pop3 = len(self.inter_M_points_pop3)


    ##############################################
    #           Create Inter M Points            #
    ##############################################
    def __create_inter_M_points(self):

        '''
        Create the boundary stellar masses array representing
        the mass points in between which there will be yields
        interpolations.  This is for massive and AGB stars.

        '''

        # Initialize the array
        self.inter_M_points = copy.deepcopy(self.M_table)

        # Add the lower and upper IMF yields range limits
        if not self.imf_yields_range[0] in self.M_table:
            self.inter_M_points.append(self.imf_yields_range[0])
        if not self.imf_yields_range[1] in self.M_table:
            self.inter_M_points.append(self.imf_yields_range[1])

        # Add the transition mass between AGB and massive stars
        if not self.transitionmass in self.M_table:
            self.inter_M_points.append(self.transitionmass)

        # Remove masses that are above or beyond the IMF yields range
        len_temp = len(self.inter_M_points)
        for i_m in range(len_temp):
            ii_m = len_temp - i_m - 1
            if self.inter_M_points[ii_m] < self.imf_yields_range[0] or\
               self.inter_M_points[ii_m] > self.imf_yields_range[1]:
                self.inter_M_points.remove(self.inter_M_points[ii_m])

        # Sort the list of masses
        self.inter_M_points = sorted(self.inter_M_points)
        self.inter_M_points_tree = self._bin_tree(self.inter_M_points)

        # Calculate the number of interpolation mass-points
        self.nb_inter_M_points = len(self.inter_M_points)


    ##############################################
    #             Get Y Low Upp Pop3             #
    ##############################################
    def __get_y_low_upp_pop3(self, i_M):

        '''
        Get the lower and upper boundary yields in between
        which there will be a yields interpolation.  This is
        for massive PopIII star yields.

        Argument
        ========

          i_M : Index of the lower mass limit where the
                interpolation occurs.  This is taken from
                the self.inter_M_points array.

        '''

        # If need to extrapolate on the low-mass end ..
        # =============================================
        if self.inter_M_points_pop3[i_M] < self.M_table_pop3[0]:

            # Copy the two least massive PopIII star yields
            y_tables_0 = self.ytables_pop3.get(\
                Z=0.0, M=self.M_table_pop3[0], quantity='Yields')
            y_tables_1 = self.ytables_pop3.get(\
                Z=0.0, M=self.M_table_pop3[1], quantity='Yields')

            # Extrapolate the lower boundary
            yields_low = self.scale_yields_to_M_ej(self.M_table_pop3[0],\
                self.M_table_pop3[1], y_tables_0, y_tables_1, \
                    self.inter_M_points_pop3[i_M], y_tables_0, sum(y_tables_0))

            # Take lowest-mass model for the upper boundary
            yields_upp = y_tables_0

            # Set the yields and mass for total-mass-ejected interpolation
            m_ej_low, m_ej_upp, yields_ej_low, yields_ej_upp = \
                self.M_table_pop3[0], self.M_table_pop3[1], y_tables_0, y_tables_1

        # If need to extrapolate on the high-mass end ..
        # ==============================================
        elif self.inter_M_points_pop3[i_M+1] > self.M_table_pop3[-1]:

            # Take the highest-mass model for the lower boundary
            yields_low = self.ytables_pop3.get(Z=0.0,\
                M=self.M_table_pop3[-1], quantity='Yields')

            # Extrapolate the upper boundary
            yields_upp = self.extrapolate_high_mass(\
                self.ytables_pop3, 0.0, self.inter_M_points_pop3[i_M+1])

            # Set the yields and mass for total-mass-ejected interpolation
            m_ej_low, m_ej_upp, yields_ej_low, yields_ej_upp = \
                self.inter_M_points_pop3[i_M], self.inter_M_points_pop3[i_M+1],\
                yields_low, yields_upp

        # If the mass point is the first one, is higher than the
        # least massive mass in the yields table, but is not part
        # of the yields table ..
        # =======================================================
        elif i_M == 0 and not self.inter_M_points_pop3[i_M] in self.M_table_pop3:

            # Assign the upper-mass model
            yields_upp = self.ytables_pop3.get(Z=0.0,\
                M=self.inter_M_points_pop3[i_M+1], quantity='Yields')

            # Interpolate the lower-mass model
            i_M_upp = self.M_table_pop3.index(self.inter_M_points_pop3[i_M+1])
            aa, bb, dummy, dummy = self.__get_inter_coef_M(self.M_table_pop3[i_M_upp-1],\
                self.M_table_pop3[i_M_upp], self.ytables_pop3.get(Z=0.0,\
                    M=self.M_table_pop3[i_M_upp-1], quantity='Yields'), yields_upp,\
                        1.0, 2.0, yields_upp, yields_upp)
            yields_low = 10**(aa * self.inter_M_points_pop3[i_M] + bb)

            # Set the yields and mass for total-mass-ejected interpolation
            i_M_ori = self.M_table_pop3.index(self.inter_M_points_pop3[i_M+1])
            m_ej_low, m_ej_upp, yields_ej_low, yields_ej_upp = \
                self.M_table_pop3[i_M_ori-1], self.inter_M_points_pop3[i_M+1],\
                self.ytables_pop3.get(Z=0.0, M=self.M_table_pop3[i_M_ori-1],\
                quantity='Yields'), yields_upp

        # If the mass point is the last one, is lower than the
        # most massive mass in the yields table, but is not part
        # of the yields table ..
        # ======================================================
        elif i_M == (self.nb_inter_M_points_pop3-2) and \
            not self.inter_M_points_pop3[i_M+1] in self.M_table_pop3:

            # Assign the lower-mass model
            yields_low = self.ytables_pop3.get(Z=0.0,\
                M=self.inter_M_points_pop3[i_M], quantity='Yields')

            # Interpolate the upper-mass model
            i_M_low = self.M_table_pop3.index(self.inter_M_points_pop3[i_M])
            aa, bb, dummy, dummy = self.__get_inter_coef_M(self.M_table_pop3[i_M_low],\
                self.M_table_pop3[i_M_low+1], yields_low, self.ytables_pop3.get(\
                    Z=0.0, M=self.M_table_pop3[i_M_low+1],quantity='Yields'),\
                        1.0, 2.0, yields_low, yields_low)
            yields_upp = 10**(aa * self.inter_M_points_pop3[i_M+1] + bb)

            # Set the yields and mass for total-mass-ejected interpolation
            i_M_ori = self.M_table_pop3.index(self.inter_M_points_pop3[i_M])
            m_ej_low, m_ej_upp, yields_ej_low, yields_ej_upp = \
                self.inter_M_points_pop3[i_M], self.M_table_pop3[i_M_ori+1],\
                yields_low, self.ytables_pop3.get(Z=0.0,\
                M=self.M_table_pop3[i_M_ori+1], quantity='Yields')

        # If this is an interpolation between two models
        # originally in the yields table ..
        # ==============================================
        else:

            # Get the original models
            yields_low = self.ytables_pop3.get(Z=0.0,\
                M=self.inter_M_points_pop3[i_M], quantity='Yields')
            yields_upp = self.ytables_pop3.get(Z=0.0,\
                M=self.inter_M_points_pop3[i_M+1], quantity='Yields')

            # Set the yields and mass for total-mass-ejected interpolation
            m_ej_low, m_ej_upp, yields_ej_low, yields_ej_upp = \
                self.inter_M_points_pop3[i_M], self.inter_M_points_pop3[i_M+1],\
                yields_low, yields_upp

        # Return the yields for interpolation
        return yields_low, yields_upp, m_ej_low, m_ej_upp, \
               yields_ej_low, yields_ej_upp


    ##############################################
    #               Get Y Low Upp                #
    ##############################################
    def __get_y_low_upp(self, i_Z, i_M, is_radio=False):

        '''
        Get the lower and upper boundary yields in between
        which there will be a yields interpolation.  This is
        for massive and AGB star yields.

        Argument
        ========

          i_Z : Metallicity index of the yields table
          i_M : Index of the lower mass limit where the
                interpolation occurs.  This is taken from
                the self.inter_M_points array.

        '''

        # If need to extrapolate on the low-mass end ..
        # =============================================
        if self.inter_M_points[i_M] < self.M_table[0]:

            # Copy the two least massive AGB star yields
            y_tables_0 = self.ytables.get(\
                Z=self.Z_table[i_Z], M=self.M_table[0], quantity='Yields')
            y_tables_1 = self.ytables.get(\
                Z=self.Z_table[i_Z], M=self.M_table[1], quantity='Yields')

            # If radioactive yields table ..
            if is_radio:

                # Get radioactive yields
                y_tables_0_radio = self.ytables_radio.get(\
                    Z=self.Z_table[i_Z], M=self.M_table[0], quantity='Yields')

                # Extrapolate the lower boundary (using stable yields total mass)
                yields_low = self.scale_yields_to_M_ej(self.M_table[0],\
                    self.M_table[1], y_tables_0, y_tables_1, \
                        self.inter_M_points[i_M], y_tables_0_radio, \
                            sum(y_tables_0))

                # Take lowest-mass model for the upper boundary
                yields_upp = y_tables_0_radio

            # If stable yields table ..
            else:

                # Extrapolate the lower boundary
                yields_low = self.scale_yields_to_M_ej(self.M_table[0],\
                    self.M_table[1], y_tables_0, y_tables_1, \
                        self.inter_M_points[i_M], y_tables_0, sum(y_tables_0))

                # Take lowest-mass model for the upper boundary
                yields_upp = y_tables_0

                # Set the yields and mass for total-mass-ejected interpolation
                m_ej_low, m_ej_upp, yields_ej_low, yields_ej_upp = \
                    self.M_table[0], self.M_table[1], y_tables_0, y_tables_1

        # If the upper boundary is the transition mass ..
        # ===============================================
        elif self.inter_M_points[i_M+1] == self.transitionmass:

            # Keep the lower-boundary yields
            yields_low_stable = self.ytables.get(Z=self.Z_table[i_Z],\
                M=self.inter_M_points[i_M], quantity='Yields')

            # If the transition mass is part of the yields table
            if self.transitionmass in self.M_table:

                # Prepare to use the transition-mass model
                i_M_add = 1

                # Set the yields and mass for total-mass-ejected interpolation
                m_ej_low, m_ej_upp, yields_ej_low, yields_ej_upp = \
                    self.inter_M_points[i_M], self.inter_M_points[i_M+1], \
                    yields_low_stable, self.ytables.get(Z=self.Z_table[i_Z],\
                    M=self.inter_M_points[i_M+1], quantity='Yields')

            # If the transition mass is not part of the yields table
            else:

                # Prepare to use the model after the transition mass
                i_M_add = 2

                # Set the yields and mass for total-mass-ejected interpolation
                m_ej_low, m_ej_upp, yields_ej_low, yields_ej_upp = \
                    self.inter_M_points[i_M], self.inter_M_points[i_M+2], \
                    yields_low_stable, self.ytables.get(Z=self.Z_table[i_Z],\
                    M=self.inter_M_points[i_M+2], quantity='Yields')

            # Copy the upper-boundary model used to scale the yields
            yields_tr_upp = self.ytables.get(Z=self.Z_table[i_Z],\
                M=self.inter_M_points[i_M+i_M_add], quantity='Yields')

            # If radioactive table
            if is_radio:

                # Keep the lower-boundary yields
                yields_low = self.ytables_radio.get(Z=self.Z_table[i_Z],\
                    M=self.inter_M_points[i_M], quantity='Yields')

            # If stable table
            else:

                # Keep the lower-boundary yields (stable)
                yields_low = yields_low_stable

            # Scale massive AGB yields for the upper boundary
            yields_upp = self.scale_yields_to_M_ej(\
                self.inter_M_points[i_M], self.inter_M_points[i_M+i_M_add],\
                    yields_low_stable, yields_tr_upp, self.transitionmass, \
                        yields_low, sum(yields_low_stable))

        # If the lower boundary is the transition mass ..
        # ===============================================
        elif self.inter_M_points[i_M] == self.transitionmass:

            # Keep the upper-boundary yields
            yields_upp_stable = self.ytables.get(Z=self.Z_table[i_Z],\
                M=self.inter_M_points[i_M+1], quantity='Yields')

            # If the transition mass is part of the yields table
            if self.transitionmass in self.M_table:

                # Prepare to use the transition-mass model
                i_M_add = 0

                # Set the yields and mass for total-mass-ejected interpolation
                m_ej_low, m_ej_upp, yields_ej_low, yields_ej_upp = \
                    self.inter_M_points[i_M], self.inter_M_points[i_M+1], \
                    self.ytables.get(Z=self.Z_table[i_Z],\
                    M=self.inter_M_points[i_M], quantity='Yields'),\
                    yields_upp_stable

            # If the transition mass is not part of the yields table
            else:

                # Prepare to use the model before the transition mass
                i_M_add = -1

                # Set the yields and mass for total-mass-ejected interpolation
                m_ej_low, m_ej_upp, yields_ej_low, yields_ej_upp = \
                    self.inter_M_points[i_M-1], self.inter_M_points[i_M+1], \
                    self.ytables.get(Z=self.Z_table[i_Z],\
                    M=self.inter_M_points[i_M-1], quantity='Yields'),\
                    yields_upp_stable

            # Copy the lower-boundary model used to scale the yields
            yields_tr_low = self.ytables.get(Z=self.Z_table[i_Z],\
                M=self.inter_M_points[i_M+i_M_add], quantity='Yields')

            # If radioactive table
            if is_radio:

                # Keep the lower-boundary yields
                yields_upp = self.ytables_radio.get(Z=self.Z_table[i_Z],\
                    M=self.inter_M_points[i_M+1], quantity='Yields')

            # If stable table
            else:

                # Keep the lower-boundary yields (stable)
                yields_upp = yields_upp_stable
            
            # Scale lowest massive star yields for the lower boundary
            yields_low = self.scale_yields_to_M_ej(\
                self.inter_M_points[i_M+i_M_add], self.inter_M_points[i_M+1],\
                    yields_tr_low, yields_upp_stable, self.transitionmass, \
                        yields_upp, sum(yields_upp_stable))

        # If need to extrapolate on the high-mass end ..
        # ==============================================
        elif self.inter_M_points[i_M+1] > self.M_table[-1]:

            # If radioactive table ..
            if is_radio:

                # Take the highest-mass model for the lower boundary
                yields_low = self.ytables_radio.get(Z=self.Z_table[i_Z],\
                    M=self.M_table[-1], quantity='Yields')

                # Extrapolate the upper boundary
                yields_upp = self.extrapolate_high_mass(self.ytables_radio,\
                    self.Z_table[i_Z], self.inter_M_points[i_M+1])

            # If stable table ..
            else:

                # Take the highest-mass model for the lower boundary
                yields_low = self.ytables.get(Z=self.Z_table[i_Z],\
                    M=self.M_table[-1], quantity='Yields')

                # Extrapolate the upper boundary
                yields_upp = self.extrapolate_high_mass(self.ytables,\
                    self.Z_table[i_Z], self.inter_M_points[i_M+1])

                # Set the yields and mass for total-mass-ejected interpolation
                m_ej_low, m_ej_upp, yields_ej_low, yields_ej_upp = \
                    self.inter_M_points[i_M], self.inter_M_points[i_M+1],\
                    yields_low, yields_upp

        # If the mass point is the first one, is higher than the
        # least massive mass in the yields table, but is not part
        # of the yields table ..
        # =======================================================
        elif i_M == 0 and not self.inter_M_points[i_M] in self.M_table:

            # Use the appropriate yield tables ..
            if is_radio:
                the_ytables = self.ytables_radio
            else:
                the_ytables = self.ytables

            # Assign the upper-mass model
            yields_upp = the_ytables.get(Z=self.Z_table[i_Z],\
                M=self.inter_M_points[i_M+1], quantity='Yields')

            # Interpolate the lower-mass model
            i_M_upp = self.M_table.index(self.inter_M_points[i_M+1])
            aa, bb, dummy, dummy = self.__get_inter_coef_M(self.M_table[i_M_upp-1],\
               self.M_table[i_M_upp], the_ytables.get(Z=self.Z_table[i_Z],\
                   M=self.M_table[i_M_upp-1], quantity='Yields'), yields_upp,\
                       1.0, 2.0, yields_upp, yields_upp)
            yields_low = 10**(aa * self.inter_M_points[i_M] + bb)

            # Set the yields and mass for total-mass-ejected interpolation
            if not is_radio:
                i_M_ori = self.M_table.index(self.inter_M_points[i_M+1])
                m_ej_low, m_ej_upp, yields_ej_low, yields_ej_upp = \
                    self.M_table[i_M_ori-1], self.inter_M_points[i_M+1],\
                    self.ytables.get(Z=self.Z_table[i_Z],\
                    M=self.M_table[i_M_ori-1], quantity='Yields'), yields_upp

        # If the mass point is the last one, is lower than the
        # most massive mass in the yields table, but is not part
        # of the yields table ..
        # ======================================================
        elif i_M == (self.nb_inter_M_points-2) and \
            not self.inter_M_points[i_M+1] in self.M_table:

            # Use the appropriate yield tables ..
            if is_radio:
                the_ytables = self.ytables_radio
            else:
                the_ytables = self.ytables

            # Assign the lower-mass model
            yields_low = the_ytables.get(Z=self.Z_table[i_Z],\
                M=self.inter_M_points[i_M], quantity='Yields')

            # Interpolate the upper-mass model
            i_M_low = self.M_table.index(self.inter_M_points[i_M])
            aa, bb, dummy, dummy = self.__get_inter_coef_M(self.M_table[i_M_low],\
                self.M_table[i_M_low+1], yields_low, the_ytables.get(\
                    Z=self.Z_table[i_Z], M=self.M_table[i_M_low+1],quantity='Yields'),\
                        1.0, 2.0, yields_low, yields_low)
            yields_upp = 10**(aa * self.inter_M_points[i_M+1] + bb)

            # Set the yields and mass for total-mass-ejected interpolation
            if not is_radio:
                i_M_ori = self.M_table.index(self.inter_M_points[i_M])
                m_ej_low, m_ej_upp, yields_ej_low, yields_ej_upp = \
                    self.inter_M_points[i_M], self.M_table[i_M_ori+1],\
                    yields_low, self.ytables.get(Z=self.Z_table[i_Z],\
                    M=self.M_table[i_M_ori+1], quantity='Yields')

        # If this is an interpolation between two models
        # originally in the yields table ..
        else:

            # Use the appropriate yield tables ..
            if is_radio:
                the_ytables = self.ytables_radio
            else:
                the_ytables = self.ytables

            # Get the original models
            yields_low = the_ytables.get(Z=self.Z_table[i_Z],\
                M=self.inter_M_points[i_M], quantity='Yields')
            yields_upp = the_ytables.get(Z=self.Z_table[i_Z],\
                M=self.inter_M_points[i_M+1], quantity='Yields')

            # Set the yields and mass for total-mass-ejected interpolation
            if not is_radio:
                m_ej_low, m_ej_upp, yields_ej_low, yields_ej_upp = \
                    self.inter_M_points[i_M], self.inter_M_points[i_M+1],\
                    yields_low, yields_upp

        # Return the yields for interpolation
        if is_radio:
            return yields_low, yields_upp, 1.0, 2.0, 1.0, 1.0
        else:
            return yields_low, yields_upp, m_ej_low, m_ej_upp, \
                   yields_ej_low, yields_ej_upp


    ##############################################
    #           Get Inter Coef Yields M          #
    ##############################################
    def __get_inter_coef_M(self, m_low, m_upp, yields_low, yields_upp,\
                           m_low_ej, m_upp_ej, yields_low_ej, yields_upp_ej):

        '''
        Calculate the interpolation coefficients for interpolating
        in between two given yields of different mass.

        Interpolation law
        =================

          log10(yields) = a_M * M + b_M
          M_ej = a_ej * M + b_ej

        Argument
        ========

          m_low : Lower stellar mass boundary (yields)
          m_upp : Upper stellar mass boundary (yields)
          yields_low : Yields associated to m_low (yields)
          yields_upp : Yields associated to m_upp (yields)
          m_low_ej : Lower stellar mass boundary (total mass)
          m_upp_ej : Upper stellar mass boundary (total mass)
          yields_low_ej : Yields associated to m_low (total mass)
          yields_upp_ej : Yields associated to m_upp (total mass)

        '''

        # Convert zeros into 1.0e-30
        for i_iso in range(len(yields_upp)):
            if yields_upp[i_iso] == 0.0:
                yields_upp[i_iso] = 1.0e-30
        for i_iso in range(len(yields_low)):
            if yields_low[i_iso] == 0.0:
                yields_low[i_iso] = 1.0e-30

        # Calculate the coefficients a_M
        np_log10_yields_upp = np.log10(yields_upp)
        the_a_M = (np_log10_yields_upp - np.log10(yields_low)) /\
            (m_upp - m_low)
        if m_upp == m_low:
            print('Problem in __get_inter_coef_M', m_upp, m_low)

        # Calculate the coefficients b_M
        the_b_M = np_log10_yields_upp - the_a_M * m_upp

        # Calculate the coefficients a_ej
        sum_yields_upp_ej = sum(yields_upp_ej)
        the_a_ej = (sum_yields_upp_ej - sum(yields_low_ej)) /\
            (m_upp_ej - m_low_ej)

        # Calculate the coefficients b_ej
        the_b_ej = sum_yields_upp_ej - the_a_ej * m_upp_ej

        # Return the coefficients arrays
        return the_a_M, the_b_M, the_a_ej, the_b_ej


    ##############################################
    #         Get Inter Coef Yields M Tau        #
    ##############################################
    def __get_inter_coef_M_tau(self, m_low, m_upp, tau_low, tau_upp):

        '''
        Calculate the interpolation coefficients for interpolating
        in between two given lifetimes at different mass.

        Interpolation law
        =================

          log10(tau) = a_M * log10(M) + b_M

        Argument
        ========

          m_low : Lower stellar mass boundary
          m_upp : Upper stellar mass boundary
          tau_low : Lifetime associated to m_low
          tau_upp : Lifetime associated to m_upp

        '''

        # Calculate the coefficients a_M
        np_log10_tau_upp = np.log10(tau_upp)
        np_log10_m_upp = np.log10(m_upp)
        the_a_M = (np_log10_tau_upp - np.log10(tau_low)) /\
            (np_log10_m_upp - np.log10(m_low))
        if m_upp == m_low:
            print('Problem in __get_inter_coef_M_tau', m_upp, m_low)

        # Calculate the coefficients b_M
        the_b_M = np_log10_tau_upp - the_a_M * np_log10_m_upp

        # Return the coefficients arrays
        return the_a_M, the_b_M


    ##############################################
    #            Scale Yields to M_ej            #
    ##############################################
    def scale_yields_to_M_ej(self, m_low, m_upp, yields_low, yields_upp,\
                             the_m_scale, the_yields, the_yields_m_tot):

        '''
        Scale yields according to the total ejected mass vs initial
        mass relation.  This will keep the relative chemical composition
        of the yields.

        Interpolation law
        =================

          M_ej = a * M_initial + b

        Argument
        ========

          m_low : Initial mass of the lower-mass boundary model
          m_upp : Initial mass of the upper-mass boundary model
          yields_low : Yields of the lower-mass boundary model
          yields_upp : Yields of the upper-mass boundary model
          the_m_scale : Initial mass to which the_yields will be scaled
          the_yields : Yields that need to be scaled
          the_yields_m_tot : Total mass of the yields that need to be scaled

        '''

        # Get the coefficient for the total-mass-ejected interpolation
        m_ej_low = sum(yields_low)
        m_ej_upp = sum(yields_upp)
        a_temp = (m_ej_upp - m_ej_low) / (m_upp - m_low)
        b_temp = m_ej_upp - a_temp * m_upp

        # Calculate the interpolated (or extrapolated) total ejected mass
        m_ej_temp = a_temp * the_m_scale + b_temp
        if m_ej_temp < 0.0:
            m_ej_temp = 0.0

        # Return the scaled yields
        return np.array(the_yields) * m_ej_temp / the_yields_m_tot


    ##############################################
    #            Extrapolate High Mass           #
    ##############################################
    def extrapolate_high_mass(self, table_ehm, Z_ehm, m_extra, is_radio=False):

        '''
        Extrapolate yields for stellar masses larger than what
        is provided in the yields table.

        Extrapolation choices (input parameter)
        =====================

          copy : This will apply the yields of the most massive model
                 to all more massive stars.

          scale : This will scale the yields of the most massive model
                  using the relation between the total ejected mass and
                  the initial stellar mass.  The later relation is taken
                  from the interpolation of the two most massive models.

          extrapolate : This will extrapolate the yields of the most massive
                        model using the interpolation coefficients taken from
                        the interpolation of the two most massive models.

        Arguments
        =========

          table_ehm : Yields table
          Z_ehm : Metallicity of the yields table
          m_extra : Mass to which the yields will be extrapolated

        '''

        # Get the two most massive mass
        if Z_ehm == 0.0:
            mass_m2 = self.M_table_pop3[-2]
            mass_m1 = self.M_table_pop3[-1]
        else:
            mass_m2 = self.M_table[-2]
            mass_m1 = self.M_table[-1]

        # Copy the yields of most massive model
        y_tables_m1 = table_ehm.get(Z=Z_ehm, M=mass_m1, quantity='Yields')

        # If the yields are copied ..
        if self.high_mass_extrapolation == 'copy':

            # Return the yields of the most massive model
            return y_tables_m1

        # If the yields are scaled ..
        if self.high_mass_extrapolation == 'scale':

            # Make sure we use the stable yields for scaling (table_ehm could be radio)
            y_stable_m1 = self.ytables.get(Z=Z_ehm, M=mass_m1, quantity='Yields')
            y_stable_m2 = self.ytables.get(Z=Z_ehm, M=mass_m2, quantity='Yields')

            # Calculate the scaled yields
            y_scaled = self.scale_yields_to_M_ej(mass_m2,\
                mass_m1, y_stable_m2, y_stable_m1, m_extra, y_tables_m1,\
                    sum(y_stable_m1))

            # Set to 1e-30 if yields are negative.  Do not set
            # to zero, because yields will be interpolated in log.
            if sum(y_scaled) <= 0.0:
                y_scaled = np.zeros(len(y_scaled))
                y_scaled += 1.0e-30

            # Return the scaled yields of the most massive model
            return y_scaled

        # If the yields are extrapolated ..
        if self.high_mass_extrapolation == 'extrapolate':

            # Copy the yields of the second most massive model
            y_tables_m2 = table_ehm.get(Z=Z_ehm, M=mass_m2, quantity='Yields')

            # Extrapolate the yields
            the_a, the_b, the_a_ej, the_b_ej = self.__get_inter_coef_M(\
                mass_m2, mass_m1, y_tables_m2, y_tables_m1,\
                mass_m2, mass_m1, y_tables_m2, y_tables_m1)
            y_extra = 10**(the_a * m_extra + the_b)
            m_ej_extra = the_a_ej * m_extra + the_b_ej
            y_extra = y_extra * m_ej_extra / sum(y_extra)

            # # Set to 1e-30 if yields are negative.  Do not set
            # to zero, because yields will be interpolated in log.
            for i_yy in range(len(y_extra)):
                if y_extra[i_yy] <= 0.0:
                    y_extra[i_yy] = 1.0e-30

            # Return the extrapolated yields
            return y_extra


    ##############################################
    #           Get Inter Coef Yields Z          #
    ##############################################
    def __get_inter_coef_Z(self, x_M_low, x_M_upp, \
                           x_M_ej_low, x_M_ej_upp, Z_low, Z_upp):

        '''
        Calculate the interpolation coefficients for interpolating
        the mass-interpolation coefficients in between two given
        metallicities.

        Interpolation laws
        ==================

          log10(yields) = a_M * M + b_M
          x_M = a_Z * log10(Z) + b_Z

          The function calculates a_Z and b_Z for either a_M or b_M

        Argument
        ========

          x_M_low : Lower mass-interpolation coefficient limit (yields)
          x_M_upp : Upper mass-interpolation coefficient limit (yields)
          x_M_ej_low : Lower mass-interpolation coefficient limit (total mass)
          x_M_ej_upp : Upper mass-interpolation coefficient limit (total mass)
          Z_low : Lower-metallicity limit of the interpolation
          Z_upp : Upper-metallicity limit of the interpolation

        '''

        # Copy the lower and upper metallicities
        lg_Z_low = np.log10(Z_low)
        lg_Z_upp = np.log10(Z_upp)

        # Calculate the coefficients a_Z and b_Z (yields)
        the_a_Z = (x_M_upp - x_M_low) / (lg_Z_upp - lg_Z_low)
        the_b_Z = x_M_upp - the_a_Z * lg_Z_upp

        # Calculate the coefficients a_Z and b_Z (total mass)
        the_a_Z_ej = (x_M_ej_upp - x_M_ej_low) / (lg_Z_upp - lg_Z_low)
        the_b_Z_ej = x_M_ej_upp - the_a_Z_ej * lg_Z_upp

        # Return the coefficients arrays
        return the_a_Z, the_b_Z, the_a_Z_ej, the_b_Z_ej


    ##############################################
    #         Get Inter Coef Yields Z Tau        #
    ##############################################
    def __get_inter_coef_Z_tau(self, x_M_low, x_M_upp, Z_low, Z_upp):

        '''
        Calculate the interpolation coefficients for interpolating
        the mass-interpolation lifetime coefficients in between two
        given metallicities.

        Interpolation laws
        ==================

          log10(tau) = a_M * log10(M) + b_M
          x_M = a_Z * Z + b_Z

          The function calculates a_Z and b_Z for either a_M or b_M

        Argument
        ========

          x_M_low : Lower mass-interpolation coefficient limit
          x_M_upp : Upper mass-interpolation coefficient limit
          Z_low : Lower-metallicity limit of the interpolation
          Z_upp : Upper-metallicity limit of the interpolation

        '''

        # Calculate the coefficients a_Z and b_Z (yields)
        the_a_Z = (x_M_upp - x_M_low) / (Z_upp - Z_low)
        the_b_Z = x_M_upp - the_a_Z * Z_upp

        # Return the coefficients arrays
        return the_a_Z, the_b_Z


    ##############################################
    #         Interpolate Pop3 Lifetimes         #
    ##############################################
    def __interpolate_pop3_lifetimes(self):

        '''
        Interpolate the mass-dependent lifetimes of PopIII stars.
        This will create arrays containing interpolation coefficients.
        The chemical evolution calculations will then only use these
        coefficients instead of the tabulated lifetimes.

        Interpolation laws
        ==================

          Interpolation across stellar mass M
            log10(tau) = a_M * log10(M) + b_M

        Results
        =======

          a_M and b_M coefficients
          ------------------------
            tau_coef_M_pop3[i_coef][i_M_low]
              - i_coef : 0 and 1 for a_M and b_M, respectively
              - i_M_low : Index of the lower mass limit where
                          the interpolation occurs

        Note
        ====

          self.Z_table is in decreasing order
          but y_coef_... arrays have metallicities in increasing order

        '''

        # Fill the tau_coef_M_pop3 array
        # For each interpolation lower-mass bin point ..
        for i_M in range(self.nb_M_table_pop3-1):

            # Get the lifetime for the lower and upper mass models
            tau_low = self.ytables_pop3.get(\
                M=self.M_table_pop3[i_M], Z=0.0, quantity='Lifetime')
            tau_upp = self.ytables_pop3.get(\
                M=self.M_table_pop3[i_M+1], Z=0.0, quantity='Lifetime')

            # Get the interpolation coefficients a_M, b_M
            self.tau_coef_M_pop3[0][i_M],\
                self.tau_coef_M_pop3[1][i_M] =\
                    self.__get_inter_coef_M_tau(self.M_table_pop3[i_M],\
                        self.M_table_pop3[i_M+1], tau_low, tau_upp)


    ##############################################
    #    Interpolate Massive and AGB Lifetimes   #
    ##############################################
    def __interpolate_massive_and_agb_lifetimes(self):

        '''
        Interpolate the metallicity- and mass-dependent lifetimes
        of massive and AGB stars.  This will create arrays containing
        interpolation coefficients.  The chemical evolution calculations
        will then only use these coefficients instead of the tabulated
        lifetimes.

        Interpolation laws
        ==================

          Interpolation across stellar mass M
            log10(tau) = a_M * log10(M) + b_M
            log10(M) = a_M * log10(tau) + b_M

          Interpolation of a_M and b_M across metallicity Z
            x_M = a_Z * Z + b_Z

          The functions first calculate a_M and b_M for each Z,
          and then interpolate these coefficients across Z.

        Results
        =======

          a_M and b_M coefficients
          ------------------------
            tau_coef_M[i_coef][i_Z][i_M_low]
              - i_coef : 0 and 1 for a_M and b_M, respectively
              - i_Z : Metallicity index available in the table
              - i_M_low : Index of the lower mass limit where
                          the interpolation occurs

          a_Z and b_Z coefficients for x_M
          --------------------------------
            y_coef_Z_xM_tau[i_coef][i_Z_low][i_M_low]
              - i_coef : 0 and 1 for a_Z and b_Z, respectively
              - i_Z_low : Index of the lower metallicity limit where
                          the interpolation occurs
              - i_M_low : Index of the lower mass limit where
                          the interpolation occurs

        Note
        ====

          self.Z_table is in decreasing order
          but y_coef_... arrays have metallicities in increasing order

        '''

        # Fill the tau_coef_M array
        # For each metallicity available in the yields ..
        for i_Z_temp in range(self.nb_Z_table):

            # Get the metallicity index in increasing order
            i_Z = self.inter_Z_points.index(self.Z_table[i_Z_temp])

            # For each interpolation lower-mass bin point ..
            for i_M in range(self.nb_M_table-1):

                # Get the lifetime for the lower and upper mass models
                tau_low = self.ytables.get(M=self.M_table[i_M],\
                    Z=self.inter_Z_points[i_Z], quantity='Lifetime')
                tau_upp = self.ytables.get(M=self.M_table[i_M+1],\
                    Z=self.inter_Z_points[i_Z], quantity='Lifetime')

                # Get the interpolation coefficients a_M, b_M
                self.tau_coef_M[0][i_Z][i_M],\
                    self.tau_coef_M[1][i_Z][i_M] =\
                        self.__get_inter_coef_M_tau(self.M_table[i_M],\
                            self.M_table[i_M+1], tau_low, tau_upp)

        # Fill the y_coef_Z_xM_tau arrays
        # For each interpolation lower-metallicity point ..
        for i_Z in range(self.nb_inter_Z_points-1):

            # For each interpolation lower-mass bin point ..
            for i_M in range(self.nb_M_table-1):

                # Get the interpolation coefficients a_Z, b_Z for a_M
                self.tau_coef_Z_aM[0][i_Z][i_M],\
                    self.tau_coef_Z_aM[1][i_Z][i_M] =\
                    self.__get_inter_coef_Z_tau(self.tau_coef_M[0][i_Z][i_M],\
                    self.tau_coef_M[0][i_Z+1][i_M], self.inter_Z_points[i_Z],\
                    self.inter_Z_points[i_Z+1])

                # Get the interpolation coefficients a_Z, b_Z for b_M
                self.tau_coef_Z_bM[0][i_Z][i_M],\
                    self.tau_coef_Z_bM[1][i_Z][i_M] =\
                    self.__get_inter_coef_Z_tau(self.tau_coef_M[1][i_Z][i_M],\
                    self.tau_coef_M[1][i_Z+1][i_M], self.inter_Z_points[i_Z],\
                    self.inter_Z_points[i_Z+1])


    ##############################################
    #         Interpolate Pop3 M From T          #
    ##############################################
    def __interpolate_pop3_m_from_t(self):

        '''
        Calculate the interpolation coefficients to extract
        the mass of stars based on their lifetimes.

        Interpolation laws
        ==================

          Interpolation across stellar lifetime tau
            log10(M) = a_tau * log10(tau) + b_tau

          Interpolation of a_M and b_M across metallicity Z
            x_M = a_Z * Z + b_Z

        Results
        =======

          a_tau and b_tau coefficients
          ----------------------------
            tau_coef_M_pop3_inv[i_coef][i_tau_low]
              - i_coef : 0 and 1 for a_tau and b_tau, respectively
              - i_tau_low : Index of the lower lifetime limit where
                            the interpolation occurs

          a_Z and b_Z coefficients for x_tau
          ----------------------------------
            tau_coef_Z_xM_pop3_inv[i_coef][i_Z_low][i_tau_low]
              - i_coef : 0 and 1 for a_Z and b_Z, respectively
              - i_tau_low : Index of the lower lifetime limit where
                            the interpolation occurs

        Note
        ====

          self.Z_table is in decreasing order
          but y_coef_... arrays have metallicities in increasing order

        '''

        # Declare list of lifetimes for each mass at each metallicity
        self.lifetimes_list_pop3 = np.zeros(self.nb_M_table_pop3)
        for i_M in range(self.nb_M_table_pop3):
            self.lifetimes_list_pop3[i_M] = self.ytables_pop3.get(\
                M=self.M_table_pop3[i_M], Z=0.0, quantity='Lifetime')

        # Fill the tau_coef_M_inv array
        # For each interpolation lower-lifetime bin point ..
        for i_tau in range(self.nb_inter_lifetime_points_pop3-1):

            # Get the mass for the lower and upper lifetimes
            m_tau_low = self.__get_m_from_tau_pop3(\
                    self.inter_lifetime_points_pop3[i_tau])
            m_tau_upp = self.__get_m_from_tau_pop3(\
                    self.inter_lifetime_points_pop3[i_tau+1])

            # Get the interpolation coefficients a_tau, b_tau
            # Here we use the __get_inter_coef_M_tau, be we
            # swap mass for lifetime and vice-versa
            self.tau_coef_M_pop3_inv[0][i_tau],\
                self.tau_coef_M_pop3_inv[1][i_tau] =\
                self.__get_inter_coef_M_tau(self.inter_lifetime_points_pop3[i_tau],\
                self.inter_lifetime_points_pop3[i_tau+1], m_tau_low, m_tau_upp)


    ##############################################
    #    Interpolate Massive and AGB M From T    #
    ##############################################
    def __interpolate_massive_and_agb_m_from_t(self):

        '''
        Calculate the interpolation coefficients to extract
        the mass of stars from metallicity- and mass-dependent
        lifetimes.  This will fix lifetime intervals that will
        be common to all metallicities.  This will accelerate
        the mass search during the chemical evolution calculation.

        Interpolation laws
        ==================

          Interpolation across stellar lifetime tau
            log10(M) = a_tau * log10(tau) + b_tau

          Interpolation of a_M and b_M across metallicity Z
            x_M = a_Z * Z + b_Z

        Results
        =======

          a_tau and b_tau coefficients
          ----------------------------
            tau_coef_M_inv[i_coef][i_Z][i_tau_low]
              - i_coef : 0 and 1 for a_tau and b_tau, respectively
              - i_Z : Metallicity index available in the table
              - i_tau_low : Index of the lower lifetime limit where
                            the interpolation occurs

          a_Z and b_Z coefficients for x_tau
          ----------------------------------
            tau_coef_Z_xM_inv[i_coef][i_Z_low][i_tau_low]
              - i_coef : 0 and 1 for a_Z and b_Z, respectively
              - i_Z_low : Index of the lower metallicity limit where
                          the interpolation occurs
              - i_tau_low : Index of the lower lifetime limit where
                            the interpolation occurs

        Note
        ====

          self.Z_table is in decreasing order
          but y_coef_... arrays have metallicities in increasing order

        '''

        # Declare list of lifetimes for each mass at each metallicity
        self.lifetimes_list = np.zeros((self.nb_Z_table,self.nb_M_table))
        for i_Z in range(self.nb_Z_table):
            for i_M in range(self.nb_M_table):
                self.lifetimes_list[i_Z][i_M] = self.ytables.get(\
                    M=self.M_table[i_M], Z=self.Z_table[i_Z],\
                        quantity='Lifetime')

        # Fill the tau_coef_M_inv array
        # For each metallicity available in the yields ..
        for i_Z_temp in range(self.nb_Z_table):

            # Get the metallicity index in increasing order
            i_Z = self.inter_Z_points.index(self.Z_table[i_Z_temp])

            # For each interpolation lower-lifetime bin point ..
            for i_tau in range(self.nb_inter_lifetime_points-1):

                # Get the mass for the lower and upper lifetimes
                m_tau_low = self.__get_m_from_tau(\
                        i_Z_temp, self.inter_lifetime_points[i_tau])
                m_tau_upp = self.__get_m_from_tau(\
                        i_Z_temp, self.inter_lifetime_points[i_tau+1])

                # Get the interpolation coefficients a_tau, b_tau
                # Here we use the __get_inter_coef_M_tau, be we
                # swap mass for lifetime and vice-versa
                self.tau_coef_M_inv[0][i_Z][i_tau],\
                    self.tau_coef_M_inv[1][i_Z][i_tau] =\
                    self.__get_inter_coef_M_tau(self.inter_lifetime_points[i_tau],\
                    self.inter_lifetime_points[i_tau+1], m_tau_low, m_tau_upp)

        # Fill the tau_coef_Z_inv arrays
        # For each interpolation lower-metallicity point ..
        for i_Z in range(self.nb_inter_Z_points-1):

            # For each interpolation lower-lifetime bin point ..
            for i_tau in range(self.nb_inter_lifetime_points-1):

                # Get the interpolation coefficients a_Z, b_Z for a_M
                self.tau_coef_Z_aM_inv[0][i_Z][i_tau],\
                    self.tau_coef_Z_aM_inv[1][i_Z][i_tau] =\
                    self.__get_inter_coef_Z_tau(self.tau_coef_M_inv[0][i_Z][i_tau],\
                    self.tau_coef_M_inv[0][i_Z+1][i_tau], self.inter_Z_points[i_Z],\
                    self.inter_Z_points[i_Z+1])

                # Get the interpolation coefficients a_Z, b_Z for b_M
                self.tau_coef_Z_bM_inv[0][i_Z][i_tau],\
                    self.tau_coef_Z_bM_inv[1][i_Z][i_tau] =\
                    self.__get_inter_coef_Z_tau(self.tau_coef_M_inv[1][i_Z][i_tau],\
                    self.tau_coef_M_inv[1][i_Z+1][i_tau], self.inter_Z_points[i_Z],\
                    self.inter_Z_points[i_Z+1])


    ##############################################
    #     Create Inter Lifetime Points Pop3      #
    ##############################################
    def __create_inter_lifetime_points_pop3(self):

        '''
        Create the lifetime points in between which there will be
        interpolations.  This is for PopIII stars.

        '''

        # List all lifetimes for Pop III stars
        self.inter_lifetime_points_pop3 = []
        for i_M in range(self.nb_M_table_pop3):
            the_tau = self.ytables_pop3.get(M=self.M_table_pop3[i_M],\
                      Z=0.0, quantity='Lifetime')
            if not the_tau in self.inter_lifetime_points_pop3:
                self.inter_lifetime_points_pop3.append(the_tau)
        self.nb_inter_lifetime_points_pop3 = len(self.inter_lifetime_points_pop3)

        # Sort the list to have lifetimes in increasing order
        self.inter_lifetime_points_pop3 = sorted(self.inter_lifetime_points_pop3)
        self.inter_lifetime_points_pop3_tree = self._bin_tree(
                self.inter_lifetime_points_pop3)


    ##############################################
    #       Create Inter Lifetime Points         #
    ##############################################
    def __create_inter_lifetime_points(self):

        '''
        Create the lifetime points in between which there will be
        interpolations.  This is for metallicity-dependent models.

        '''

        # List all lifetimes for all metallicities
        self.inter_lifetime_points = []
        for i_Z in range(self.nb_Z_table):
            for i_M in range(self.nb_M_table):
                the_tau = self.ytables.get(M=self.M_table[i_M],\
                          Z=self.Z_table[i_Z], quantity='Lifetime')
                if not the_tau in self.inter_lifetime_points:
                    self.inter_lifetime_points.append(the_tau)
        self.nb_inter_lifetime_points = len(self.inter_lifetime_points)

        # Sort the list to have lifetimes in increasing order
        self.inter_lifetime_points = sorted(self.inter_lifetime_points)
        self.inter_lifetime_points_tree = self._bin_tree(
                self.inter_lifetime_points)


    ##############################################
    #           Get lgM from Tau Pop3            #
    ##############################################
    def __get_m_from_tau_pop3(self, the_tau):

        '''
        Return the interpolated mass of a given lifetime.
        This is for PopIII stars

        Interpolation law
        =================

            log10(M) = a_M * log10(tau) + b_M

        Arguments
        =========

          the_tau : Lifetime [yr]

        '''

        # Find the lower-mass boundary of the interval surrounding
        # the given lifetime
        if the_tau >= self.lifetimes_list_pop3[0]:
            i_M_low = 0
        elif the_tau <= self.lifetimes_list_pop3[-1]:
            i_M_low = len(self.lifetimes_list_pop3) - 2
        else:
            i_M_low = 0
            while self.lifetimes_list_pop3[i_M_low+1] >= the_tau:
                i_M_low += 1

        # Get the interpolation coefficients
        lg_tau_low = np.log10(self.lifetimes_list_pop3[i_M_low+1])
        lg_tau_upp = np.log10(self.lifetimes_list_pop3[i_M_low])
        lg_m_low = np.log10(self.M_table_pop3[i_M_low+1])
        lg_m_upp = np.log10(self.M_table_pop3[i_M_low])
        a_temp = (lg_m_upp - lg_m_low) / (lg_tau_upp - lg_tau_low)
        b_temp = lg_m_upp - a_temp * lg_tau_upp

        # Return the interpolated mass
        return 10**(a_temp * np.log10(the_tau) + b_temp)


    ##############################################
    #             Get lgM from Tau               #
    ##############################################
    def __get_m_from_tau(self, i_Z, the_tau):

        '''
        Return the interpolated mass of a given metallicity
        that has a given lifetime.

        Interpolation law
        =================

            log10(M) = a_M * log10(tau) + b_M

        Arguments
        =========

          i_Z : Metallicity index of the yields table
          the_tau : Lifetime [yr]

        '''

        # Find the lower-mass boundary of the interval surrounding
        # the given lifetime
        if the_tau >= self.lifetimes_list[i_Z][0]:
            i_M_low = 0
        elif the_tau <= self.lifetimes_list[i_Z][-1]:
            i_M_low = len(self.lifetimes_list[i_Z]) - 2
        else:
            i_M_low = 0
            while self.lifetimes_list[i_Z][i_M_low+1] >= the_tau:
                i_M_low += 1

        # Get the interpolation coefficients
        lg_tau_low = np.log10(self.lifetimes_list[i_Z][i_M_low+1])
        lg_tau_upp = np.log10(self.lifetimes_list[i_Z][i_M_low])
        lg_m_low = np.log10(self.M_table[i_M_low+1])
        lg_m_upp = np.log10(self.M_table[i_M_low])
        a_temp = (lg_m_upp - lg_m_low) / (lg_tau_upp - lg_tau_low)
        b_temp = lg_m_upp - a_temp * lg_tau_upp

        # Return the interpolated mass
        return 10**(a_temp * np.log10(the_tau) + b_temp)


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
                    print ('Use initial abundance of ', self.iniabu_table)
                ymgal_gi = np.array(iniabu.iso_abundance(self.history.isotopes)) * \
                       self.mgal

            else:

                # Get the primordial composition of Walker et al. (1991)
                iniabu_table = 'yield_tables/iniabu/iniab_bb_walker91.txt'
                ytables_bb = ry.read_yield_sn1a_tables( \
                    global_path+iniabu_table, self.history.isotopes)

                # Assign the composition to the gas reservoir
                ymgal_gi = ytables_bb.get(quantity='Yields') * self.mgal

                # Output information
                if self.iolevel > 0:
                    print ('Use initial abundance of ', iniabu_table)

        # Already enriched gas reservoir
        else:

            # If an input iniabu table is provided ...
            if len(self.iniabu_table) > 0:
                iniabu=ry.iniabu(global_path + self.iniabu_table)
                if self.iolevel > 0:
                    print ('Use initial abundance of ', self.iniabu_table)

            # If NuGrid's yields are used ...
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
                            print ('Use initial abundance of ', \
                            ini_list[ini_Z.index(metal)])
                        break

            # Input file for the initial composition ...
            #else:
            #    iniabu=ry.iniabu(global_path + iniabu_table)
            #    print ('Use initial abundance of ', iniabu_table)

            # Assign the composition to the gas reservoir
            ymgal_gi = np.array(iniabu.iso_abundance(self.history.isotopes)) * \
                       self.mgal

        # Make sure the total mass of gas is exactly mgal
        # This is in case we have a few isotopes without H and He
        ymgal_gi = ymgal_gi * self.mgal / sum(ymgal_gi)

#        if sum(ymgal_gi) > self.mgal:
#            ymgal_gi[0] = ymgal_gi[0] - (sum(ymgal_gi) - self.mgal)

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
          if self.t_merge < (self.history.tend - 1.1):

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

          else:
            self.i_t_merger = len(timesteps_gt)-1

        # Correct the last timestep if needed
        if timesteps_gt[-1] == 0.0:
            timesteps_gt[-1] = self.history.tend - sum(timesteps_gt)

        # Return the duration of all timesteps
        return timesteps_gt


    ##############################################
    #             Get Storing Arrays             #
    ##############################################
    def _get_storing_arrays(self, ymgal, nb_iso_gsa):

        '''
        This function declares and returns all the arrays containing information
        about the evolution of the stellar ejecta, the gas reservoir, the star
        formation rate, and the number of core-collapse SNe, SNe Ia, neutron star
        mergers, and white dwarfs.

        Argument
        ========

          ymgal : Initial gas reservoir. This function extends it to all timesteps
          nb_iso_gsa : Number of isotopes (can different if stable or radio isotopes)

        '''

        # Number of timesteps and isotopes
        nb_dt_gsa = self.nb_timesteps

        # Stellar ejecta
        mdot = np.zeros((nb_dt_gsa,nb_iso_gsa))

        # Gas reservoir
        temp = copy.deepcopy(ymgal)
        ymgal = np.zeros((nb_dt_gsa+1,nb_iso_gsa))
        ymgal[0] += np.array(temp)

        # Massive stars, AGB stars, SNe Ia ejecta, and neutron star merger ejecta
        ymgal_massive = []
        ymgal_agb = []
        ymgal_1a = []
        ymgal_nsm = []
        ymgal_bhnsm = []
        ymgal_delayed_extra = []
        if self.pre_calculate_SSPs:
            mdot_massive = copy.deepcopy(mdot)
            mdot_agb     = []
            mdot_1a      = copy.deepcopy(mdot)
            mdot_nsm     = []
            mdot_bhnsm   = []
            mdot_delayed_extra = []
            sn1a_numbers = []
            nsm_numbers = []
            bhnsm_numbers = []
            sn2_numbers  = []
            self.wd_sn1a_range  = []
            self.wd_sn1a_range1 = []
            delayed_extra_numbers = []
            self.number_stars_born = []
        else:
            for k in range(nb_dt_gsa + 1):
                ymgal_massive.append(np.zeros(nb_iso_gsa))
                ymgal_agb.append(np.zeros(nb_iso_gsa))
                ymgal_1a.append(np.zeros(nb_iso_gsa))
                ymgal_bhnsm.append(np.zeros(nb_iso_gsa))
                ymgal_nsm.append(np.zeros(nb_iso_gsa))
            for iiii in range(0,self.nb_delayed_extra):
                ymgal_delayed_extra.append([])
                for k in range(nb_dt_gsa + 1):
                    ymgal_delayed_extra[iiii].append(np.zeros(nb_iso_gsa))
            mdot_massive = copy.deepcopy(mdot)
            mdot_agb     = copy.deepcopy(mdot)
            mdot_1a      = copy.deepcopy(mdot)
            mdot_nsm     = copy.deepcopy(mdot)
            mdot_bhnsm   = copy.deepcopy(mdot)
            mdot_delayed_extra = []
            for iiii in range(0,self.nb_delayed_extra):
                mdot_delayed_extra.append(copy.deepcopy(mdot))

            # Number of SNe Ia, core-collapse SNe, and neutron star mergers
            sn1a_numbers = np.zeros(nb_dt_gsa)
            nsm_numbers = np.zeros(nb_dt_gsa)
            bhnsm_numbers = np.zeros(nb_dt_gsa)
            sn2_numbers = np.zeros(nb_dt_gsa)
            self.wd_sn1a_range = np.zeros(nb_dt_gsa)
            self.wd_sn1a_range1 = np.zeros(nb_dt_gsa)
            delayed_extra_numbers = []
            for iiii in range(0,self.nb_delayed_extra):
                delayed_extra_numbers.append(np.zeros(nb_dt_gsa))

            # Star formation
            self.number_stars_born = np.zeros(nb_dt_gsa+1)

        # Related to the IMF
        self.history.imf_mass_ranges = [[]] * (nb_dt_gsa + 1)
        imf_mass_ranges = []
        imf_mass_ranges_contribution = [[]] * (nb_dt_gsa + 1)
        imf_mass_ranges_mtot = [[]] * (nb_dt_gsa + 1)

        # Return all the arrays
        return mdot, ymgal, ymgal_massive, ymgal_agb, ymgal_1a, ymgal_nsm, ymgal_bhnsm,\
               ymgal_delayed_extra, mdot_massive, mdot_agb, mdot_1a, mdot_nsm, mdot_bhnsm,\
               mdot_delayed_extra, sn1a_numbers, sn2_numbers, nsm_numbers, bhnsm_numbers,\
               delayed_extra_numbers, imf_mass_ranges, imf_mass_ranges_contribution,\
               imf_mass_ranges_mtot


    ##############################################
    #        Define Unstab. Stab. Indexes        #
    ##############################################
    def __define_unstab_stab_indexes(self):

        '''
        Create an array to make the connection between radioactive isotopes
        and the stable isotopes ther are decaying into.  For example, if Al-26
        decays into Mg-26, the array will contain the Mg-26 index in the stable
        ymgal array.

        '''

        # Declare the index connection array
        self.rs_index = [0]*self.nb_radio_iso

        # For each radioactive isotope ..
        for i_dusi in range(0,self.nb_radio_iso):

            # If stable isotope is in the main yields table ..
            if self.decay_info[i_dusi][1] in self.history.isotopes:

                # Get the radioactive and stable index
                self.rs_index[i_dusi] = \
                    self.history.isotopes.index(self.decay_info[i_dusi][1])

            # If stable isotope is not in the main yields table ..
            else:
                self.need_to_quit = True
                print ('Error - Decayed product '+self.decay_info[i_dusi][1]+\
                      ' is not in the list of considered stable isotopes.')


    ##############################################
    #               Build Split dt               #
    ##############################################
    def __build_split_dt(self):

        '''
        Create a timesteps array from the dt_split_info array.

        '''

        # Declaration of the the timestep array to be return
        dt_in_split = []

        # Initiation of the time for the upcomming simulation
        t_bsd = 0.0

        # For each split condition ...
        for i_bsd in range(0,len(self.dt_split_info)):

            # While the time still satisfies the current condition ...
            while t_bsd < self.dt_split_info[i_bsd][1]:

                # Add the timestep and update the time
                dt_in_split.append(self.dt_split_info[i_bsd][0])
                t_bsd += dt_in_split[-1]

        # Return the timesteps array
        return dt_in_split


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
        nb_m_per_z = int(len(self.ytables.table_mz) / \
                     len(self.ytables.metallicities))

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
        n_tot_prog_inv = 1.0 / self._imf(3.0,8.0,1)

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
    def _evol_stars(self, i, f_esc_yields=0.0, mass_sampled=np.array([]), \
                    scale_cor=np.array([])):

        '''
        This function executes a part of a single timestep with the simulation
        managed by either OMEGA or SYGMA.  It converts gas into stars, calculates
        the stellar ejecta of the new simple stellar population (if any), and adds
        its contribution to the total ejecta coming from all stellar populations.

        Argument
        ========

          i : Index of the current timestep
          f_esc_yields: Fraction of non-contributing stellar ejecta
          mass_sampled : Stars sampled in the IMF by an external program
          scale_cor : Envelope correction for the IMF

        '''

        # Update the time of the simulation.  Here, i is in fact the end point
        # of the current timestep which extends from i-1 to i.
        self.t += self.history.timesteps[i-1]

        # Initialisation of the mass locked into stars
        self.m_locked = 0
        self.m_locked_agb = 0
        self.m_locked_massive = 0

        # If stars are forming during the current timestep ..
        # Note: self.sfrin is calculated in SYGMA or OMEGA
        if self.sfrin > 0:

            # Limit the SFR if there is not enough gas
            if self.sfrin > 1.0:
                print ('Warning -- Not enough gas to sustain the SFH.', i)
                self.sfrin = 1.0
                self.not_enough_gas = True
                self.not_enough_gas_count += 1

            # Lock gas into stars
            f_lock_remain = 1.0 - self.sfrin
            self.__lock_gas_into_stars(i, f_lock_remain)

            # Correction if comparing with Clayton's analytical model
            # DO NOT USE unless you know why
            if not self.pre_calculate_SSPs:
                if len(self.test_clayton) > 0:
                    i_stable = self.test_clayton[0]
                    i_unst = self.test_clayton[1]
                    RR = self.test_clayton[2]
                    self.ymgal[i][i_stable] = \
                        (1.0 - self.sfrin*(1.0-RR)) * self.ymgal[i-1][i_stable]
                    self.ymgal_radio[i][i_unst] = \
                        (1.0 - self.sfrin*(1.0-RR)) * self.ymgal_radio[i-1][i_unst]

            # Add the pre-calculated SSP ejecta .. if fast mode
            if self.pre_calculate_SSPs:
                self.__add_ssp_ejecta(i)

            # Calculate stellar ejecta .. if normal mode
            else:
                self.__calculate_stellar_ejecta(i, f_esc_yields, mass_sampled, scale_cor)

        # If no star is forming during the current timestep ...
        else:

            # Use the previous gas reservoir for the current timestep
            # Done by assuming f_lock_remain = 1.0
            self.__lock_gas_into_stars(i, 1.0)

            # Initialize array containing no CC SNe for the SSP_i-1
            if self.out_follows_E_rate:
                self.ssp_nb_cc_sne = np.array([])

        # Add stellar ejecta to the gas reservoir
        # This needs to be called even if no star formation at the
        # current timestep, because older stars may still pollute
        self.__pollute_gas_with_ejecta(i)

        # Convert the mass ejected by massive stars into rate
        if not self.pre_calculate_SSPs:
            if self.history.timesteps[i-1] == 0.0:
                self.massive_ej_rate[i-1] = 0.0
                self.sn1a_ej_rate[i-1] = 0.0
            else:
                self.massive_ej_rate[i-1] = sum(self.mdot_massive[i-1]) / \
                    self.history.timesteps[i-1]
                self.sn1a_ej_rate[i-1] = sum(self.mdot_1a[i-1]) / \
                    self.history.timesteps[i-1]


    ##############################################
    #             Lock Gas Into Stars            #
    ##############################################
    def __lock_gas_into_stars(self, i, f_lock_remain):

        '''
        Correct the mass of the different gas reservoirs "ymgal"
        for the mass lock into stars.

        Argument
        ========

          i : Index of the current timestep
          f_lock_remain: Mass fraction of gas remaining after star formation

        '''

        # If this is the fast chem_evol version ..
        if self.pre_calculate_SSPs:

            # Update a limited number of gas components
            self.ymgal[i] = f_lock_remain * self.ymgal[i-1]
            if self.len_decay_file > 0:
                self.ymgal_radio[i] = f_lock_remain * self.ymgal_radio[i-1]

            # Keep track of the mass locked into stars
            self.m_locked += (1.0 - f_lock_remain) * sum(self.ymgal[i-1])

        # If this is the normal chem_evol version ..
        else:

            # Update all stable gas components
            self.ymgal[i] = f_lock_remain * self.ymgal[i-1]
            self.ymgal_massive[i] = f_lock_remain * self.ymgal_massive[i-1]
            self.ymgal_agb[i] = f_lock_remain * self.ymgal_agb[i-1]
            self.ymgal_1a[i] = f_lock_remain * self.ymgal_1a[i-1]
            self.ymgal_nsm[i] = f_lock_remain * self.ymgal_nsm[i-1]
            self.ymgal_bhnsm[i] = f_lock_remain * self.ymgal_bhnsm[i-1]
            self.m_locked += self.sfrin * sum(self.ymgal[i-1])
            for iiii in range(0,self.nb_delayed_extra):
                self.ymgal_delayed_extra[iiii][i] = \
                    f_lock_remain * self.ymgal_delayed_extra[iiii][i-1]

            # Update all radioactive gas components
            if self.len_decay_file > 0:
                self.ymgal_radio[i] = f_lock_remain * self.ymgal_radio[i-1]
            if not self.use_decay_module:
                if self.radio_massive_agb_on:
                    self.ymgal_massive_radio[i] = f_lock_remain * self.ymgal_massive_radio[i-1]
                    self.ymgal_agb_radio[i] = f_lock_remain * self.ymgal_agb_radio[i-1]
                if self.radio_sn1a_on:
                    self.ymgal_1a_radio[i] = f_lock_remain * self.ymgal_1a_radio[i-1]
                if self.radio_nsmerger_on:
                    self.ymgal_nsm_radio[i] = f_lock_remain * self.ymgal_nsm_radio[i-1]
                if self.radio_bhnsmerger_on:
                    self.ymgal_bhnsm_radio[i] = f_lock_remain * self.ymgal_bhnsm_radio[i-1]
                for iiii in range(0,self.nb_delayed_extra_radio):
                    self.ymgal_delayed_extra_radio[iiii][i] = \
                        f_lock_remain * self.ymgal_delayed_extra_radio[iiii][i-1]


    ##############################################
    #           Pollute Gas With Ejecta          #
    ##############################################
    def __pollute_gas_with_ejecta(self, i):

        '''
        Add stellar ejecta to the gas components.

        Argument
        ========

          i : Index of the current timestep

        '''

        # If this is the fast chem_evol version ..
        if self.pre_calculate_SSPs:

            # Pollute a limited number of gas components
            self.ymgal[i] += self.mdot[i-1]
            if self.len_decay_file > 0:
                self.ymgal_radio[i][:self.nb_radio_iso] += self.mdot_radio[i-1]

        # If this is the normal chem_evol version ..
        else:

            # Pollute all stable gas components
            self.ymgal[i] += self.mdot[i-1]
            self.ymgal_agb[i] += self.mdot_agb[i-1]
            self.ymgal_1a[i] += self.mdot_1a[i-1]
            self.ymgal_massive[i] += self.mdot_massive[i-1]
            self.ymgal_nsm[i] += self.mdot_nsm[i-1]
            self.ymgal_bhnsm[i] += self.mdot_bhnsm[i-1]
            if self.nb_delayed_extra > 0:
                for iiii in range(0,self.nb_delayed_extra):
                    self.ymgal_delayed_extra[iiii][i] += \
                        self.mdot_delayed_extra[iiii][i-1]

            # Pollute all radioactive gas components
            # Note: ymgal_radio[i] is treated in the decay_radio function
            # However, the contribution of individual sources must be here!
            if not self.use_decay_module:
              if self.radio_massive_agb_on:
                  self.ymgal_agb_radio[i] += \
                       self.mdot_agb_radio[i-1]
                  self.ymgal_massive_radio[i] += \
                       self.mdot_massive_radio[i-1]
              if self.radio_sn1a_on:
                  self.ymgal_1a_radio[i] += \
                       self.mdot_1a_radio[i-1]
              if self.radio_nsmerger_on:
                  self.ymgal_nsm_radio[i] += \
                       self.mdot_nsm_radio[i-1]
              if self.radio_bhnsmerger_on:
                  self.ymgal_bhnsm_radio[i] += \
                       self.mdot_bhnsm_radio[i-1]
              for iiii in range(0,self.nb_delayed_extra_radio):
                  self.ymgal_delayed_extra_radio[iiii][i] += \
                          self.mdot_delayed_extra_radio[iiii][i-1]


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
        if self.pre_calculate_SSPs:
            self.history.metallicity.append(self.zmetal)
            self.history.age.append(self.t)
            self.history.gas_mass.append(sum(self.ymgal[i]))
            self.history.ism_iso_yield.append(self.ymgal[i])
            self.history.m_locked.append(self.m_locked)
        else:
            self.history.metallicity.append(self.zmetal)
            self.history.age.append(self.t)
            self.history.gas_mass.append(sum(self.ymgal[i]))
            self.history.ism_iso_yield.append(self.ymgal[i])
            self.history.ism_iso_yield_agb.append(self.ymgal_agb[i])
            self.history.ism_iso_yield_1a.append(self.ymgal_1a[i])
            self.history.ism_iso_yield_nsm.append(self.ymgal_nsm[i])
            self.history.ism_iso_yield_bhnsm.append(self.ymgal_bhnsm[i])
            self.history.ism_iso_yield_massive.append(self.ymgal_massive[i])
            self.history.sn1a_numbers.append(self.sn1a_numbers[i-1])
            self.history.nsm_numbers.append(self.nsm_numbers[i-1])
            self.history.bhnsm_numbers.append(self.bhnsm_numbers[i-1])
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
        if self.pre_calculate_SSPs:
          for h in range(len(self.history.ism_iso_yield)):
            self.history.ism_elem_yield.append(self._iso_abu_to_elem(self.history.ism_iso_yield[h]))

        else:
          for h in range(len(self.history.ism_iso_yield)):
            self.history.ism_elem_yield.append(\
                self._iso_abu_to_elem(self.history.ism_iso_yield[h]))
            self.history.ism_elem_yield_agb.append(\
                self._iso_abu_to_elem(self.history.ism_iso_yield_agb[h]))
            self.history.ism_elem_yield_1a.append(\
                self._iso_abu_to_elem(self.history.ism_iso_yield_1a[h]))
            self.history.ism_elem_yield_nsm.append(\
                self._iso_abu_to_elem(self.history.ism_iso_yield_nsm[h]))
            self.history.ism_elem_yield_bhnsm.append(\
                self._iso_abu_to_elem(self.history.ism_iso_yield_bhnsm[h]))
            self.history.ism_elem_yield_massive.append(\
                self._iso_abu_to_elem(self.history.ism_iso_yield_massive[h]))



    ##############################################
    #          Calculate Stellar Ejecta          #
    ##############################################
    def __calculate_stellar_ejecta(self, i, f_esc_yields, mass_sampled, \
                                   scale_cor, dm_imf=0.25):

        '''
          For each upcoming timestep, including the current one,
          calculate the yields ejected by the new stellar population
          that will be deposited in the gas at that timestep.  This
          function updates the "mdot" arrays, which will eventually
          be added to the "ymgal" arrays, corresponding to the gas
          component arrays.

        Argument
        ========

          i : Index of the current timestep
          f_esc_yields: Fraction of non-contributing stellar ejecta
          mass_sampled : Stars sampled in the IMF by an external program
          scale_cor : Envelope correction for the IMF
          dm_imf : Mass interval resolution of the IMF.  Stars within a
                   specific mass interval will have the same yields

        '''

        # Select the adequate IMF properties
        if self.zmetal <= self.Z_trans:
            the_A_imf = self.A_imf_pop3
        else:
            the_A_imf = self.A_imf

        # Initialize the age of the newly-formed stars
        t_lower = 0.0

        # If the IMF is stochastically sampled ..
        if len(mass_sampled) > 0:

            # Sort the list of masses in decreasing order
            # And set the index to point to most massive one
            mass_sampled_sort = sorted(mass_sampled)[::-1]
            nb_mass_sampled = len(mass_sampled)
            stochastic_IMF = True
            i_m_sampled = 0

        # If the IMF is fully sampled ..
        else:
            stochastic_IMF = False

        # For each upcoming timesteps (including the current one) ..
        for i_cse in range(i-1, self.nb_timesteps):

            # Get the adapted IMF mass bin information
            nb_dm, new_dm_imf, m_lower = \
                self.__get_mass_bin(dm_imf, t_lower, i_cse)

            # If there are yields to be calculated ..
            if nb_dm > 0:

                # If the IMF is stochastically sampled ..
                if stochastic_IMF:

                    # For each sampled mass in that mass bin ..
                    m_upper = m_lower + new_dm_imf*nb_dm
                    while i_m_sampled < nb_mass_sampled and \
                        mass_sampled_sort[i_m_sampled] >= m_lower and \
                            mass_sampled_sort[i_m_sampled] <= m_upper:

                        # Get the yields for that star
                        the_yields = self.get_interp_yields(\
                            mass_sampled_sort[i_m_sampled], self.zmetal)

                        # Add that one star in the stellar ejecta array
                        self.__add_yields_in_mdot(1.0, the_yields, \
                            mass_sampled_sort[i_m_sampled], i_cse, i)

                        # Go to the next sampled mass
                        i_m_sampled += 1

                # If the IMF is fully sampled ..
                else:

                    # For each IMF mass bin ..
                    for i_imf_bin in range(nb_dm):

                        # Calculate lower, central, and upper masses of this bin
                        the_m_low = m_lower + i_imf_bin * new_dm_imf
                        the_m_cen = the_m_low + 0.5 * new_dm_imf
                        the_m_upp = the_m_low + new_dm_imf

                        # Get the number of stars in that mass bin
                        nb_stars = self.m_locked * the_A_imf *\
                            self._imf(the_m_low, the_m_upp, 1)

                        # Get the yields for the central stellar mass
                        the_yields = self.get_interp_yields(the_m_cen, self.zmetal)

                        # Add yields in the stellar ejecta array.  We do this at
                        # each mass bin to distinguish between AGB and massive.
                        self.__add_yields_in_mdot(nb_stars, the_yields, the_m_cen, i_cse, i)

                        # If there are radioactive isotopes
                        if self.len_decay_file > 0:

                            # Get the yields for the central stellar mass
                            the_yields = self.get_interp_yields(the_m_cen, \
                                self.zmetal, is_radio=True)

                            # Add yields in the stellar ejecta array.  We do this at
                            # each mass bin to distinguish between AGB and massive.
                            self.__add_yields_in_mdot(nb_stars, the_yields, \
                                the_m_cen, i_cse, i, is_radio=True)

            # Move the lower limit of the lifetime range to the next timestep
            t_lower += self.history.timesteps[i_cse]

        # Include the ejecta from other enrichment sources
        # such as SNe Ia, neutron star mergers, ...
        self.__add_other_sources(i)


    ##############################################
    #               Get Mass Bin                 #
    ##############################################
    def __get_mass_bin(self, dm_imf, t_lower, i_cse):

        '''
        Calculate the new IMF mass bin resolution.  This is based on
        the input resolution (dm_imf), but adapted to have an integer
        number of IMF bins that fits within the stellar mass interval
        defined by a given stellar lifetime interval.

        Arguments
        =========

          dm_imf : Mass interval resolution of the IMF.  Stars within a
                   specific mass interval will have the same yields.
          t_lower : Lower age limit of the stellar populations.
          i_cse : Index of the "future" timestep (see __calculate_stellar_ejecta).

        '''

        # Copy the adequate IMF yields range
        if self.zmetal <= self.Z_trans:
            imf_yr = self.imf_yields_range_pop3
        else:
            imf_yr = self.imf_yields_range

        # Calculate the upper age limit of the stars for that timestep
        t_upper = t_lower + self.history.timesteps[i_cse]

        # Get the lower and upper stellar mass range that will
        # contribute to the ejecta in that timestep
        m_lower = self.get_interp_lifetime_mass(t_upper, self.zmetal, is_mass=False)
        if t_lower == 0.0:
            m_upper = 1.0e30
        else:
            m_upper = self.get_interp_lifetime_mass(t_lower, self.zmetal, is_mass=False)

        # If the mass interval is outside the IMF yields range ..
        if m_lower >= imf_yr[1] or m_upper < imf_yr[0]:

            # Skip the yields calculation
            nb_dm = 0
            new_dm_imf = 0

        # If the mass interval is inside or overlapping
        # with the IMF yields range ..
        else:

            # Redefine the boundary to respect the IMF yields range
            m_lower = max(m_lower, imf_yr[0])
            m_upper = min(m_upper, imf_yr[1])

            # Calculate the new IMF resolution, which is based on the
            # input resolution, but adapted to have an integer number
            # of IMF bins that fits in the redefined stellar mass interval
            nb_dm = int(round((m_upper-m_lower)/dm_imf))
            if nb_dm < 1:
                nb_dm = 1
                new_dm_imf = m_upper - m_lower
            else:
                new_dm_imf = (m_upper - m_lower) / float(nb_dm)

        # Return the new IMF bin
        return nb_dm, new_dm_imf, m_lower


    ##############################################
    #             Get Interp Yields              #
    ##############################################
    def get_interp_yields(self, M_giy, Z_giy, is_radio=False):

        '''
        Return the interpolated yields for a star with given
        mass and metallicity

        Interpolation law
        =================

          log10(yields) = a_M * M + b_M
          x_M = a_Z * log10(Z) + bZ

        Arguments
        =========

          M_giy : Initial mass of the star
          Z_giy : Initial metallicity of the star

        Note
        ====

          self.Z_table is in decreasing order
          but y_coef_... arrays have metallicities in increasing order

        '''

        # Select the appropriate interpolation coefficients
        if is_radio:
            the_y_coef_M = self.y_coef_M_radio
            the_y_coef_Z_aM = self.y_coef_Z_aM_radio
            the_y_coef_Z_bM = self.y_coef_Z_bM_radio
        else:
            the_y_coef_M = self.y_coef_M
            the_y_coef_Z_aM = self.y_coef_Z_aM
            the_y_coef_Z_bM = self.y_coef_Z_bM

        # If the metallicity is in the PopIII regime ..
        if Z_giy <= self.Z_trans and not is_radio:

            # Find the lower-mass boundary of the interpolation
            if self.nb_inter_M_points_pop3 < 30:
                i_M_low = 0
                while M_giy > self.inter_M_points_pop3[i_M_low+1]:
                    i_M_low += 1
            else:
                i_M_low = self.inter_M_points_pop3_tree.search_left(M_giy)

            # Select the M interpolation coefficients of PopIII yields
            a_M = self.y_coef_M_pop3[0][i_M_low]
            b_M = self.y_coef_M_pop3[1][i_M_low]
            a_M_ej = self.y_coef_M_ej_pop3[0][i_M_low]
            b_M_ej = self.y_coef_M_ej_pop3[1][i_M_low]

        # If we do not use PopIII yields ..
        else:

            # Find the lower-mass boundary of the interpolation
            if self.nb_inter_M_points < 30:
                i_M_low = 0
                while M_giy > self.inter_M_points[i_M_low+1]:
                    i_M_low += 1
            else:
                i_M_low = self.inter_M_points_pop3_tree.search_left(M_giy)

            # If the metallicity is below the lowest Z available ..
            if Z_giy <= self.inter_Z_points[0]:

                # Select the M interpolation coefficients of the lowest Z
                a_M = the_y_coef_M[0][0][i_M_low]
                b_M = the_y_coef_M[1][0][i_M_low]
                if not is_radio:
                    a_M_ej = self.y_coef_M_ej[0][0][i_M_low]
                    b_M_ej = self.y_coef_M_ej[1][0][i_M_low]

            # If the metallicity is above the highest Z available ..
            elif Z_giy > self.inter_Z_points[-1]:

                # Select the M interpolation coefficients of the highest Z
                a_M = the_y_coef_M[0][-1][i_M_low]
                b_M = the_y_coef_M[1][-1][i_M_low]
                if not is_radio:
                    a_M_ej = self.y_coef_M_ej[0][-1][i_M_low]
                    b_M_ej = self.y_coef_M_ej[1][-1][i_M_low]

            # If the metallicity is within the Z interval of the yields table ..
            else:

                # Find the lower-Z boundary of the interpolation
                i_Z_low = 0
                while Z_giy > self.inter_Z_points[i_Z_low+1]:
                    i_Z_low += 1
                lg_Z_giy = np.log10(Z_giy)

                # Calculate the a coefficient for the M interpolation
                a_Z = the_y_coef_Z_aM[0][i_Z_low][i_M_low]
                b_Z = the_y_coef_Z_aM[1][i_Z_low][i_M_low]
                a_M = a_Z * lg_Z_giy + b_Z
                if not is_radio:
                    a_Z_ej = self.y_coef_Z_aM_ej[0][i_Z_low][i_M_low]
                    b_Z_ej = self.y_coef_Z_aM_ej[1][i_Z_low][i_M_low]
                    a_M_ej = a_Z_ej * lg_Z_giy + b_Z_ej

                # Calculate the b coefficient for the M interpolation
                a_Z = the_y_coef_Z_bM[0][i_Z_low][i_M_low]
                b_Z = the_y_coef_Z_bM[1][i_Z_low][i_M_low]
                b_M = a_Z * lg_Z_giy + b_Z
                if not is_radio:
                    a_Z_ej = self.y_coef_Z_bM_ej[0][i_Z_low][i_M_low]
                    b_Z_ej = self.y_coef_Z_bM_ej[1][i_Z_low][i_M_low]
                    b_M_ej = a_Z_ej * lg_Z_giy + b_Z_ej

        # Interpolate the yields
        y_interp = 10**(a_M * M_giy + b_M)

        # Calculate the correction factor to match the relation
        # between the total ejected mass and the stellar initial
        # mass.  M_ej = a * M_i + b
        if is_radio:
            f_corr = 1.0
        else:
            f_corr = (a_M_ej * M_giy + b_M_ej) / sum(y_interp)
            if f_corr < 0.0:
                f_corr = 0.0

        # Return the interpolated and corrected yields
        return y_interp * f_corr


    ##############################################
    #          Get Interp Lifetime Mass          #
    ##############################################
    def get_interp_lifetime_mass(self, the_quantity, Z_giy, is_mass=True):

        '''
        Return the interpolated lifetime of a star with a given mass
        and metallicity

        Interpolation law
        =================

          log10(lifetime) = a_M * log10(M) + b_M
          log10(M) = a_M * log10(lifetime) + b_M
          x_M = a_Z * Z + bZ

        Arguments
        =========

          the_quantity : Initial mass or lifetime of the star
          Z_giy : Initial metallicity of the star
          is_mass : True  --> the_quantity = mass
                    False --> the quantity = lifetime

        Note
        ====

          self.Z_table is in decreasing order
          but y_coef_... arrays have metallicities in increasing order

        '''

        # Define the quantity
        if is_mass:
            quantity_pop3 = self.M_table_pop3
            nb_quantity_pop3 = self.nb_M_table_pop3
            tau_coef_M_pop3 = self.tau_coef_M_pop3
            quantity = self.M_table
            nb_quantity = self.nb_M_table
            tau_coef_M = self.tau_coef_M
            tau_coef_Z_aM = self.tau_coef_Z_aM
            tau_coef_Z_bM = self.tau_coef_Z_bM
        else:
            quantity_pop3 = self.inter_lifetime_points_pop3
            quantity_pop3_tree = self.inter_lifetime_points_pop3_tree
            nb_quantity_pop3 = self.nb_inter_lifetime_points_pop3
            tau_coef_M_pop3 = self.tau_coef_M_pop3_inv
            quantity = self.inter_lifetime_points
            quantity_tree = self.inter_lifetime_points_tree
            nb_quantity = self.nb_inter_lifetime_points
            tau_coef_M = self.tau_coef_M_inv
            tau_coef_Z_aM = self.tau_coef_Z_aM_inv
            tau_coef_Z_bM = self.tau_coef_Z_bM_inv

        # If the metallicity is in the PopIII regime ..
        if Z_giy <= self.Z_trans:

            # Find the lower-quantity boundary of the interpolation
            if the_quantity > quantity_pop3[-1]:
                i_q_low = nb_quantity_pop3 - 2
            else:
                if nb_quantity_pop3 < 30:
                    i_q_low = 0
                    while the_quantity > quantity_pop3[i_q_low+1]:
                        i_q_low += 1
                else:
                    i_q_low = quantity_pop3_tree.search_left(the_quantity)

            # Select the M interpolation coefficients of PopIII yields
            a_M = tau_coef_M_pop3[0][i_q_low]
            b_M = tau_coef_M_pop3[1][i_q_low]

        # If we do not use PopIII models ..
        else:

            # Find the lower-mass boundary of the interpolation
            if the_quantity > quantity[-1]:
                i_q_low = nb_quantity - 2
            else:
                if nb_quantity < 30:
                    i_q_low = 0
                    while the_quantity > quantity[i_q_low+1]:
                        i_q_low += 1
                else:
                    i_q_low = quantity_tree.search_left(the_quantity)

            # If the metallicity is below the lowest Z available ..
            if Z_giy <= self.inter_Z_points[0]:

                # Select the M interpolation coefficients of the lowest Z
                a_M = tau_coef_M[0][0][i_q_low]
                b_M = tau_coef_M[1][0][i_q_low]

            # If the metallicity is above the highest Z available ..
            elif Z_giy > self.inter_Z_points[-1]:

                # Select the M interpolation coefficients of the highest Z
                a_M = tau_coef_M[0][-1][i_q_low]
                b_M = tau_coef_M[1][-1][i_q_low]

            # If the metallicity is within the Z interval of the yields table ..
            else:

                # Find the lower-Z boundary of the interpolation
                i_Z_low = 0
                while Z_giy > self.inter_Z_points[i_Z_low+1]:
                    i_Z_low += 1

                # Calculate the a coefficient for the M interpolation
                a_Z = tau_coef_Z_aM[0][i_Z_low][i_q_low]
                b_Z = tau_coef_Z_aM[1][i_Z_low][i_q_low]
                a_M = a_Z * Z_giy + b_Z

                # Calculate the b coefficient for the M interpolation
                a_Z = tau_coef_Z_bM[0][i_Z_low][i_q_low]
                b_Z = tau_coef_Z_bM[1][i_Z_low][i_q_low]
                b_M = a_Z * Z_giy + b_Z

        # Return the interpolate the lifetime
        return 10**(a_M * np.log10(the_quantity) + b_M)


    ##############################################
    #            Add Yields in Mdot              #
    ##############################################
    def __add_yields_in_mdot(self, nb_stars, the_yields, the_m_cen, \
                             i_cse, i, is_radio=False):

        '''
        Add the IMF-weighted stellar yields in the ejecta "mdot" arrays.
        Keep track of the contribution of low-mass and massive stars.

        Argument
        ========

          nb_stars : Number of stars in the IMF that eject the yields
          the_yields : Yields of the IMF-central-mass-bin star
          the_m_cen : Central stellar mass of the IMF bin
          i_cse : Index of the "future" timestep (see __calculate_stellar_ejecta)
          i : Index of the timestep where the stars originally formed

        '''

        # Calculate the total yields
        the_tot_yields = nb_stars * the_yields

        # If radioactive yields ..
        if is_radio:

            # Add the yields in the total ejecta array
            self.mdot_radio[i_cse] += the_tot_yields

            # Keep track of the contribution of massive and AGB stars
            if the_m_cen > self.transitionmass:
                self.mdot_massive_radio[i_cse] += the_tot_yields
            else:
                self.mdot_agb_radio[i_cse] += the_tot_yields

        # If stable yields ..
        else:

            # Add the yields in the total ejecta array
            self.mdot[i_cse] += the_tot_yields

            # Keep track of the contribution of massive and AGB stars
            if the_m_cen > self.transitionmass:
                self.mdot_massive[i_cse] += the_tot_yields
            else:
                self.mdot_agb[i_cse] += the_tot_yields

            # Count the number of core-collapse SNe
            if the_m_cen > self.transitionmass:
                self.sn2_numbers[i_cse] += nb_stars
                if self.out_follows_E_rate:
                    self.ssp_nb_cc_sne[i_cse-i-1] += nb_stars
#                    self.ssp_nb_cc_sne[i_cse-i+1] += nb_stars

            # Sum the total number of stars born in the timestep
            # where the stars originally formed
            self.number_stars_born[i] += nb_stars


    ##############################################
    #             Add Other Sources              #
    ##############################################
    def __add_other_sources(self, i):

        '''
        Add the contribution of enrichment sources other than
        massive stars (wind + SNe) and AGB stars to the ejecta
        "mdot" array.

        Argument
        ========

          i : Index of the timestep where the stars originally formed

        '''

        # Add the contribution of SNe Ia, if any ...
        if self.sn1a_on and self.zmetal > self.Z_trans:
            if not (self.imf_bdys[0] > 8 or self.imf_bdys[1] < 3):
                f_esc_yields = 0.0 # temporary, this parameters will disapear
                self.__sn1a_contribution(i, f_esc_yields)

        # Add the contribution of neutron star mergers, if any...
        if self.ns_merger_on:
            self.__nsmerger_contribution(i)

        # Add the contribution of black hole - neutron star mergers, if any...
        if self.bhns_merger_on:
            self.__bhnsmerger_contribution(i)

        # Add the contribution of delayed extra sources, if any...
        if len(self.delayed_extra_dtd) > 0:
            self.__delayed_extra_contribution(i)


    ##############################################
    #               Get Yield Factor             #
    ##############################################
    def __get_yield_factor(self, minm1, maxm1, mass_sampled, \
                           func_total_ejecta, m_table):

        '''
        This function calculates the factor that must be multiplied to
        the input stellar yields, given the mass bin implied for the
        considered timestep and the stellar masses sampled by an external
        program.
   
        Argument
        ========

          minm1 : Minimum stellar mass having ejecta in this timestep j
          maxm1 : Minimum stellar mass having ejecta in this timestep j
          mass_sampled : Stellar mass sampled by an external program
          func_total_ejecta : Relation between M_tot_ej and stellar mass
          m_table : Mass of the star in the table providing the yields

        '''

        # Initialisation of the number of stars sampled in this mass bin
        nb_sampled_stars = 0.0

        # Initialisation of the total mass ejected
        m_ej_sampled = 0.0

        # For all mass sampled ...
        for i_gyf in range(0,len(mass_sampled)):

            # If the mass is within the mass bin considered in this step ...
            if mass_sampled[i_gyf] >= minm1 and mass_sampled[i_gyf] < maxm1:

                # Add a star and cumulate the mass ejected
                m_ej_sampled += func_total_ejecta(mass_sampled[i_gyf])
                nb_sampled_stars += 1.0

            # Stop the loop if the mass bin has been covered
            if mass_sampled[i_gyf] >= maxm1:
                break

        # If no star is sampled in the current mass bin ...
        if nb_sampled_stars == 0.0:

            # No ejecta
            return 0.0, 0.0

        # If stars have been sampled ...
        else:

            # Calculate an adapted scalefactor parameter and return yield_factor
            return nb_sampled_stars, m_ej_sampled / func_total_ejecta(m_table)


    ##############################################
    #                Get Scale Cor               #
    ##############################################
    def __get_scale_cor(self, minm1, maxm1, scale_cor):

        '''
        This function calculates the envelope correction that must be
        applied to the IMF.  This correction can be used the increase
        or reduce the number of stars in a particular mass bin, without
        creating a new IMF.  It returns the scalefactor_factor, that will
        be multiplied to scalefactor (e.g., 1.0 --> no correction)
   
        Argument
        ========

          minm1 : Minimum stellar mass having ejecta in this timestep j
          maxm1 : Minimum stellar mass having ejecta in this timestep j
          scale_cor : Envelope correction for the IMF

        '''

        # Initialization of the scalefactor correction factor
        scalefactor_factor = 0.0

        # Calculate the width of the stellar mass bin
        m_bin_width_inv = 1.0 / (maxm1 - minm1)

        # Cumulate the number of overlaped array bins
        nb_overlaps = 0

        # For each mass bin in the input scale_cor array ...
        for i_gsc in range(0,len(scale_cor)):

            # Copy the lower-mass limit of the current array bin
            if i_gsc == 0:
                m_low_temp = 0.0
            else:
                m_low_temp = scale_cor[i_gsc-1][0]

            # If the array bin overlaps the considered stellar mass bin ...
            if (scale_cor[i_gsc][0] > minm1 and scale_cor[i_gsc][0] <= maxm1)\
              or (m_low_temp > minm1 and m_low_temp < maxm1)\
              or (scale_cor[i_gsc][0] >= maxm1 and m_low_temp <= minm1):

                # Calculate the stellar bin fraction covered by the array bin
                frac_temp = (min(maxm1, scale_cor[i_gsc][0]) - \
                            max(minm1, m_low_temp)) * m_bin_width_inv

                # Cumulate the correction
                scalefactor_factor += frac_temp * scale_cor[i_gsc][1]

                # Increment the number of overlaps
                nb_overlaps += 1

        # Warning is no overlap
        if nb_overlaps == 0:
            print ('!!Warning - No overlap with scale_cor!!')

        # Return the scalefactor correction factor
        return scalefactor_factor


    ##############################################
    #                  Decay Radio               #
    ##############################################
    def _decay_radio(self, i):

        '''
        This function decays radioactive isotopes present in the
        radioactive gas component and add the stable decayed product
        inside the stable gas component.  This is using a simple
        decay routine where an unstable isotope decay to only one
        stable isotope.

        Argument
        ========

          i : Index of the current timestep.
          Reminder, here 'i' is the upper-time boundary of the timestep

        '''

        # Nb of refinement steps
        nb_ref = int(self.radio_refinement)
        nb_ref_fl = float(nb_ref)

        # Copy the duration of the timestep (duration of the decay)
        dt_decay = self.history.timesteps[i-1] / nb_ref_fl

        # For each radioactive isotope ..
        for i_dr in range(0,self.nb_radio_iso):

            # Keep track of the mass before the decay
            m_copy = self.ymgal_radio[i][i_dr] + self.mdot_radio[i-1][i_dr]

            # Get the mass added to the gas
            m_added = self.mdot_radio[i-1][i_dr] / nb_ref_fl

            # If there is something to decay ..
            if m_copy > 0.0:

                # Declare variable to keep track of the decayed mass
                m_decay = 0.0

                # For each refinement step ..
                for i_loop in range(nb_ref):

                    # Add ejecta and decay the isotope
                    self.ymgal_radio[i][i_dr] += m_added
                    m_prev = copy.deepcopy(self.ymgal_radio[i][i_dr])
                    self.ymgal_radio[i][i_dr] *= \
                        np.exp((-1.0)*dt_decay/self.decay_info[i_dr][2])

                    # Cumulate the decayed mass
                    m_decay += m_prev - self.ymgal_radio[i][i_dr]

                # Add the decayed stable isotope in the stable gas
                self.ymgal[i][self.rs_index[i_dr]] += m_decay

                # Calculate the fraction left over in the radioactive gas
                f_remain = self.ymgal_radio[i][i_dr] / m_copy

                # Correct the contribution of different sources
                if not self.pre_calculate_SSPs:
                    self.ymgal_massive_radio[i][i_dr] *= f_remain
                    self.ymgal_agb_radio[i][i_dr] *= f_remain
                    self.ymgal_1a_radio[i][i_dr] *= f_remain
                    self.ymgal_nsm_radio[i][i_dr] *= f_remain
                    self.ymgal_bhnsm_radio[i][i_dr] *= f_remain
                    for iiii in range(0,self.nb_delayed_extra_radio):
                        self.ymgal_delayed_extra_radio[iiii][i][i_dr] *= f_remain


    ##############################################
    #           Initialize Decay Module          #
    ##############################################
    def __initialize_decay_module(self):

        '''
        This function import and initialize the decay module
        used to decay unstable isotopes.  Declare arrays used
        for the communication between the fortran decay code
        and NuPyCEE

        '''

        # Import and declare the decay module
        #import decay_module
        decay_module.initialize(self.f_network, self.f_format, global_path)

        # Declare the element names used to return the charge number Z
        # Index 0 needs to be NN!  H needs to be index 1!
        self.element_names = ['NN', 'H', 'He', 'Li', 'Be', 'B', 'C', \
            'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P',\
            'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',\
            'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As',\
            'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',\
            'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',\
            'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',\
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',\
            'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',\
            'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr',\
            'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',\
            'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db',\
            'Sg', 'Bh', 'Hs', 'Mt', 'Uun', 'Uuu', 'Uub', 'zzz', \
            'Uuq']

        # Number is isotope entry in the fortran decay module
        self.len_iso_module = len(decay_module.iso.z)

        # Find the isotope name associated with each isotope entry
        self.iso_decay_module = ['']*self.len_iso_module
        for i_iso in range(self.len_iso_module):
          if decay_module.iso.z[i_iso] == 0:
            self.iso_decay_module[i_iso] = 'NN-1'
          else:
            self.iso_decay_module[i_iso] = \
              self.element_names[decay_module.iso.z[i_iso]] + '-' + \
                str(decay_module.iso.z[i_iso]+decay_module.iso.n[i_iso])

        # Year to second conversion
        self.yr_to_sec = 3.154e+7


    ##############################################
    #           Decay Radio With Module          #
    ##############################################
    def _decay_radio_with_module(self, i):

        '''
        This function decays radioactive isotopes present in the
        radioactive gas component and add the stable decayed product
        inside the stable gas component.  This is using the decay
        module to account for all decay channels.

        Argument
        ========

          i : Index of the current timestep.
          Reminder, here 'i' is the upper-time boundary of the timestep

        '''

        # Nb of refinement steps
        nb_ref = int(self.radio_refinement)
        nb_ref_fl = float(nb_ref)

        # Copy the duration of the timestep (duration of the decay)
        dt_decay = self.history.timesteps[i-1] / nb_ref_fl

        # Keep track of the mass before the decay
        sum_ymgal_temp = sum(self.ymgal_radio[i])
        sum_mdot_temp = sum(self.mdot_radio[i-1])

        # If there is something to decay ..
        if sum_ymgal_temp > 0.0 or sum_mdot_temp > 0.0:

            # Get the mass added to the gas at each refined timesteps
            m_added = self.mdot_radio[i-1][:self.nb_radio_iso] / nb_ref_fl

            # Declare variable to keep track of the decayed mass
            m_decay = 0.0

            # For each refinement step ..
            for i_loop in range(nb_ref):

                # Add ejecta
                self.ymgal_radio[i][:self.nb_radio_iso] += m_added

                # Call the decay module
                self.__run_decay_module(i, dt_decay)


    ##############################################
    #              Run Decay Module              #
    ##############################################
    def __run_decay_module(self, i, dt_decay):

        '''
        Decay the current radioactive abundances using
        the decay module.

        Argument
        ========

          i : Index of the current timestep.
          dt_decay: Duration of the decay [yr]

        '''

        # Get the initial abundances of radioactive isotopes
        init_abun = self.__get_init_abun_decay(i)

        # Call the decay module
        decay_module.run_decay(dt_decay*self.yr_to_sec, 1, init_abun)

        # For each relevant isotope in the decay module ..
        need_resize = False
        for i_iso in range(self.len_iso_module):
            if decay_module.iso.abundance[i_iso] > 0.0 or init_abun[i_iso] > 0.0:

                # Replace the unstable component by the decayed product
                if self.iso_decay_module[i_iso] in self.radio_iso:
                    k_temp = self.radio_iso.index(self.iso_decay_module[i_iso])
                    self.ymgal_radio[i][k_temp] = \
                        copy.deepcopy(decay_module.iso.abundance[i_iso])

                # Add decayed product to the stable component
                elif self.iso_decay_module[i_iso] in self.history.isotopes:
                    k_temp = self.history.isotopes.index(self.iso_decay_module[i_iso])
                    self.ymgal[i][k_temp] += decay_module.iso.abundance[i_iso]

                # If this is a new so-far-unacounted isotope ..
                else:
                       
                    # Add the new isotope name
                    self.radio_iso.append(self.iso_decay_module[i_iso])

                    # Add the entry and the abundance of the new isotope
                    self.ymgal_radio = np.concatenate((self.ymgal_radio,\
                        np.zeros((1,self.nb_timesteps+1)).T), axis=1)
                    self.ymgal_radio[i][-1] = \
                        copy.deepcopy(decay_module.iso.abundance[i_iso])
                    need_resize = True
 
        # Resize all radioactive arrays
        if need_resize:
            self.nb_new_radio_iso = len(self.radio_iso)


    ##############################################
    #             Get Init Abun Decay            #
    ##############################################
    def __get_init_abun_decay(self, i):

        '''
        Calculate and return the initial abundance of radioactive
        isotopes in the format required by the fortran decay module.

        Argument
        ========

          i : Index of the current timestep.

        '''

        # Initially set abundances to zero
        init_abun_temp = np.zeros(self.len_iso_module)

        # For each radioactive isotope ..
        for i_iso in range(self.nb_new_radio_iso):

            # Find the isotope index for the decay module
            i_temp = self.iso_decay_module.index(self.radio_iso[i_iso])

            # Copy the mass of the isotope
            init_abun_temp[i_temp] = copy.deepcopy(self.ymgal_radio[i][i_iso])

        # Return the initial abundance
        return init_abun_temp


    ##############################################
    #             SN Ia Contribution             #
    ##############################################
    def __sn1a_contribution(self, i, f_esc_yields):

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
        tables_Z = sorted(self.ytables_1a.metallicities,reverse=True)
        if self.radio_sn1a_on:
            tables_Z_radio = sorted(self.ytables_1a_radio.metallicities,reverse=True)

        # Pick the metallicity
        for tz in tables_Z:
            if self.zmetal <= tables_Z[-1]:
                yields1a = self.ytables_1a.get(Z=tables_Z[-1], quantity='Yields')
                break
            if self.zmetal >= tables_Z[0]:
                yields1a = self.ytables_1a.get(Z=tables_Z[0], quantity='Yields')
                break
            if self.zmetal > tz:
                yields1a = self.ytables_1a.get(Z=tz, quantity='Yields')
                break

        # Pick the metallicity (for radioactive yields)
        if self.radio_sn1a_on:
            for tz in tables_Z_radio:
                if self.zmetal <= tables_Z_radio[-1]:
                    yields1a_radio = \
                        self.ytables_1a_radio.get(Z=tables_Z_radio[-1], quantity='Yields')
                    break
                if self.zmetal >= tables_Z_radio[0]:
                    yields1a_radio = \
                        self.ytables_1a_radio.get(Z=tables_Z_radio[0], quantity='Yields')
                    break
                if self.zmetal > tz:
                    yields1a_radio = self.ytables_1a_radio.get(Z=tz, quantity='Yields')
                    break

        # If the selected SN Ia rate depends on the number of white dwarfs ...
        if self.history.sn1a_rate == 'exp' or \
           self.history.sn1a_rate == 'gauss' or \
           self.history.sn1a_rate == 'maoz' or \
           self.history.sn1a_rate == 'power_law':

           # Get the lifetimes of the considered stars (if needed ...)
           if len(self.poly_fit_dtd_5th) == 0:
                lifetime_min = self.inter_lifetime_points[0]

        # Normalize the SN Ia rate if not already done
        if len(self.poly_fit_dtd_5th) > 0 and not self.normalized:
            self.__normalize_poly_fit()
        if self.history.sn1a_rate == 'exp' and not self.normalized:
            self.__normalize_efolding(lifetime_min)
        elif self.history.sn1a_rate == 'gauss' and not self.normalized:
            self.__normalize_gauss(lifetime_min)
        elif (self.history.sn1a_rate == 'maoz' or \
            self.history.sn1a_rate == 'power_law') and not self.normalized:
            self.__normalize_maoz(lifetime_min)

        # Initialisation of the cumulated time and number of SNe Ia
        sn1a_output = 0
        tt = 0

        # For every upcoming timestep j, starting with the current one ...
        for j in range(i-1, self.nb_timesteps):

            # Set the upper and lower time boundary of the timestep j
            timemin = tt
            tt += self.history.timesteps[j]
            timemax = tt

            # For an input polynomial DTD ...
            if len(self.poly_fit_dtd_5th) > 0:

                # If no SN Ia ...
                if timemax < self.poly_fit_range[0] or \
                   timemin > self.poly_fit_range[1]:
                    n1a = 0.0

                # If SNe Ia occur during this timestep j ...
                else:

                    # Calculate the number of SNe Ia and white dwarfs (per Mo)
                    wd_number = 0.0 # Could be calculated if needed
                    n1a = self.__poly_dtd(timemin, timemax)

            # If we use Chris Pritchet's prescription ...
            #elif self.len_pritchet_1a_dtd > 0:

                # If no SN Ia ...
            #    if timemax < self.pritchet_1a_dtd[0] or \
            #       timemin > self.pritchet_1a_dtd[[1]:
            #        n1a = 0.0

                # If SNe Ia occur during this timestep j ...
            #    else:

                    # Calculate the number of SNe Ia and white dwarfs (per Mo)
            #        wd_number = 0.0 # Could be calculated if needed
            #        n1a = self.__pritchet_dtd(timemin, timemax)
                    
            # For other DTDs ...
            else:

              # Calculate the number of SNe Ia if with Vogelsberger SN Ia rate
              if self.history.sn1a_rate=='vogelsberger':
                  n1a = self.__vogelsberger13(timemin, timemax)

              # No SN Ia if the minimum current stellar lifetime is too long
              if lifetime_min > timemax:
                  n1a = 0

              # If SNe Ia occur during this timestep j ...
              else:

                  # Set the lower time limit for the integration
                  if timemin < lifetime_min:
                      timemin = lifetime_min

                  # For an exponential SN Ia rate ...
                  if self.history.sn1a_rate == 'exp':
       
                      # Calculate the number of SNe Ia and white dwarfs (per Mo)
                      n1a, wd_number = self.__efolding(timemin, timemax)

                  # For a power law SN Ia rate ...
                  elif self.history.sn1a_rate == 'maoz' or \
                       self.history.sn1a_rate == 'power_law':

                      # Calculate the number of SNe Ia and white dwarfs (per Mo)
                      n1a, wd_number = self.__maoz12_powerlaw(timemin, timemax)

                  # For a gaussian SN Ia rate ...
                  elif self.history.sn1a_rate == 'gauss':

                      # Calculate the number of SNe Ia and white dwarfs (per Mo)
                      n1a, wd_number = self.__gauss(timemin, timemax)

                  # Cumulate the number of white dwarfs in the SN Ia mass range
                  self.wd_sn1a_range[j] += (wd_number * self.m_locked)

            # Convert number of SNe Ia per Mo into real number of SNe Ia
            n1a = n1a * self.m_locked

            # Cumulate the number of SNe Ia
            self.sn1a_numbers[j] += n1a

            # add SNIa energy
            if self.sn1a_on and self.stellar_param_on:
                idx=self.stellar_param_attrs.index('SNIa energy')
                self.stellar_param[idx][j] = self.stellar_param[idx][j] + \
                    n1a * self.sn1a_energy/(self.history.timesteps[j]*self.const.syr)

            # Output information
            if sn1a_output == 0 :
                if self.iolevel >= 2:
                    print ('SN1a (pop) start to contribute at time ', \
                          '{:.3E}'.format((timemax)))
                sn1a_output = 1

            # Add the contribution of SNe Ia to the timestep j
            f_contr_yields = 1.0 - f_esc_yields
            self.mdot[j] = self.mdot[j] +  n1a * f_contr_yields * yields1a
            self.mdot_1a[j] = self.mdot_1a[j] + n1a * f_contr_yields * yields1a
            if self.radio_sn1a_on:
                self.mdot_radio[j] += n1a * f_contr_yields * yields1a_radio
                self.mdot_1a_radio[j] += n1a * f_contr_yields * yields1a_radio


    #############################################
    #            NS Merger Contribution         #
    #############################################
    def __nsmerger_contribution(self, i):
        '''
        This function calculates the contribution of neutron star mergers
        on the stellar ejecta and adds it to the mdot array.

        Arguments
        =========

            i : index of the current timestep

        '''

        # Get NS merger yields
        tables_Z = self.ytables_nsmerger.metallicities
        for tz in tables_Z:
            if self.zmetal > tz:
                yieldsnsm = self.ytables_nsmerger.get(Z=tz, quantity='Yields')
                break
            if self.zmetal <= tables_Z[-1]:
                yieldsnsm = self.ytables_nsmerger.get(Z=tables_Z[-1], quantity='Yields')
                break

        # Get NS merger radioactive yields
        if self.radio_nsmerger_on:
            tables_Z_radio = self.ytables_nsmerger_radio.metallicities
            for tz in tables_Z_radio:
                if self.zmetal > tz:
                    yieldsnsm_radio = \
                        self.ytables_nsmerger_radio.get(Z=tz, quantity='Yields')
                    break
                if self.zmetal <= tables_Z_radio[-1]:
                    yieldsnsm_radio = \
                        self.ytables_nsmerger_radio.get(Z=tables_Z_radio[-1], quantity='Yields')
                    break

        # initialize variables which cumulate in loop
        tt = 0

        # Normalize ...
        if not self.nsm_normalized:
            self.__normalize_nsmerger(1) # NOTE: 1 is a dummy variable right now

        # For every upcoming timestep j, starting with the current one...
        for j in range(i-1, self.nb_timesteps):

            # Set the upper and lower time boundary of the timestep j
            timemin = tt
            tt += self.history.timesteps[j]
            timemax = tt

            # Stop if the SSP no more NS merger occurs
            if timemin >= self.t_merger_max:
                break

            # Calculate the number of NS mergers per stellar mass
            nns_m = self.__nsmerger_num(timemin, timemax)

            # Calculate the number of NS mergers in the current SSP
            nns_m = nns_m * self.m_locked
            self.nsm_numbers[j] += nns_m

            # Add the contribution of NS mergers to the timestep j
            self.mdot[j] = np.array(self.mdot[j]) + \
                np.array(nns_m * self.m_ej_nsm * yieldsnsm)
            self.mdot_nsm[j] = np.array(self.mdot_nsm[j]) + \
                np.array(nns_m * self.m_ej_nsm * yieldsnsm)
            if self.radio_nsmerger_on:
                self.mdot_radio[j] += nns_m * self.m_ej_nsm * yieldsnsm_radio
                self.mdot_nsm_radio[j] += nns_m * self.m_ej_nsm * yieldsnsm_radio


    ##############################################
    #               NS merger number             #
    ##############################################
    def __nsmerger_num(self, timemin, timemax):

        '''
        This function returns the number of neutron star mergers occurring within a given time
        interval using the Dominik et al. (2012) delay-time distribution function.
        
        Arguments
        =========
        
            timemin : Lower boundary of time interval.
            timemax : Upper boundary of time interval.

        '''

        # If an input DTD array is provided ...
        if self.len_nsmerger_dtd_array > 0:
            
            # Find the lower and upper Z boundaries
            if self.zmetal <= self.Z_nsmerger[0]:
                i_Z_low = 0
                i_Z_up  = 0
            elif self.zmetal >= self.Z_nsmerger[-1]:
                i_Z_low = -1
                i_Z_up  = -1
            else:
                i_Z_low = 0
                i_Z_up  = 1
                while self.zmetal > self.Z_nsmerger[i_Z_up]:
                    i_Z_low += 1
                    i_Z_up  += 1

            # Get the number of NSMs at the lower Z boundary
            nb_NSMs_low = self.__get_nb_nsm_array(timemin, timemax, i_Z_low)
            
            # Return the number of NSM .. if no interpolation is needed
            if i_Z_up == i_Z_low:
                return nb_NSMs_low

            # Interpolate the number of NSMs .. if needed
            else:
                nb_NSMs_up = self.__get_nb_nsm_array(timemin, timemax, i_Z_up)
                lg_Z_low   = np.log10(self.Z_nsmerger[i_Z_low])
                lg_Z_up    = np.log10(self.Z_nsmerger[i_Z_up])
                lg_Z_metal = np.log10(self.zmetal)
                a = (nb_NSMs_up - nb_NSMs_low) / (lg_Z_up - lg_Z_low)
                b = nb_NSMs_low - a * lg_Z_low
                return a * lg_Z_metal + b

        # If all NSMs occur after a time t_NSM_coal ...
        if self.t_nsm_coal > 0.0:

            # Return all NSMs if t_NSM_coal is in the current time interval
            if timemin <= self.t_nsm_coal and self.t_nsm_coal < timemax:
                return self.nb_nsm_per_m
            else:
                return 0.0

        # If the NSM DTD is a power law ...
        if len(self.nsm_dtd_power) > 0:

            # Copy the power law characteristics
            t_min_temp = self.nsm_dtd_power[0]
            t_max_temp = self.nsm_dtd_power[1]
            alpha_temp = self.nsm_dtd_power[2]

            # Return the number of NSMs
            if timemax < t_min_temp or timemin > t_max_temp:
                return 0.0
            elif alpha_temp == -1.0:
                return self.A_nsmerger * \
                  (np.log(min(t_max_temp,timemax)) - np.log(max(t_min_temp,timemin)))
            else:
                return self.A_nsmerger / (1.0+alpha_temp) * \
                  (min(t_max_temp,timemax)**(1.0+alpha_temp) - \
                    max(t_min_temp,timemin)**(1.0+alpha_temp))

        # Values of bounds on the piecewise DTDs, in Myr
        lower = 10
        a02bound = 22.2987197486
        a002bound = 39.7183036496
        #upper = 10000

        # convert time bounds into Myr, since DTD is in units of Myr
        timemin = timemin/1.0e6
        timemax = timemax/1.0e6

        # initialise the number of neutron star mergers in the current time interval
        nns_m = 0.0

        # Integrate over solar metallicity DTD
        if self.zmetal >= 0.019:

            # Define a02 DTD fit parameters
            a = -0.0138858377011
            b = 1.0712569392
            c = -32.1555682584
            d = 468.236521089
            e = -3300.97955814
            f = 9019.62468302
            a_pow = 1079.77358975

            # Manually compute definite integral values over DTD with bounds timemin and timemax
            # DTD doesn't produce until 10 Myr
            if timemax < lower:
                nns_m = 0.0

            # if timemin is below 10 Myr and timemax is in the first portion of DTD
            elif timemin < lower and timemax <= a02bound:
                up = ((a/6.)*(timemax**6))+((b/5.)*(timemax**5))+((c/4.)*(timemax**4))+((d/3.)*(timemax**3))+((e/2.)*(timemax**2))+(f*timemax)
                down = ((a/6.)*(lower**6))+((b/5.)*(lower**5))+((c/4.)*(lower**4))+((d/3.)*(lower**3))+((e/2.)*(lower**2))+(f*lower)
                nns_m = up - down

            # if timemin is below 10 Myr and timemax is in the power law portion of DTD
            elif timemin < lower and timemax >= a02bound:
                up1 = a_pow * np.log(timemax)
                down1 = a_pow * np.log(a02bound)
                up = up1 - down1
                up2 = ((a/6.)*(a02bound**6))+((b/5.)*(a02bound**5))+((c/4.)*(a02bound**4))+((d/3.)*(a02bound**3))+((e/2.)*(a02bound**2))+(f*a02bound)
                down2 = ((a/6.)*(lower**6))+((b/5.)*(lower**5))+((c/4.)*(lower**4))+((d/3.)*(lower**3))+((e/2.)*(lower**2))+(f*lower)
                down = up2 - down2
                nns_m = up + down # + because we are adding the contribution of the two integrals on either side of the piecewise discontinuity

            # if both timemin and timemax are in initial portion of DTD
            elif timemin >= lower and timemax <= a02bound:
                up = ((a/6.)*(timemax**6))+((b/5.)*(timemax**5))+((c/4.)*(timemax**4))+((d/3.)*(timemax**3))+((e/2.)*(timemax**2))+(f*timemax)
                down = ((a/6.)*(timemin**6))+((b/5.)*(timemin**5))+((c/4.)*(timemin**4))+((d/3.)*(timemin**3))+((e/2.)*(timemin**2))+(f*timemin)
                nns_m = up - down

            # if timemin is in initial portion of DTD and timemax is in power law portion
            elif timemin <= a02bound and timemax > a02bound:
                up1 = a_pow * np.log(timemax)
                down1 = a_pow * np.log(a02bound)
                up = up1 - down1
                up2 = ((a/6.)*(a02bound**6))+((b/5.)*(a02bound**5))+((c/4.)*(a02bound**4))+((d/3.)*(a02bound**3))+((e/2.)*(a02bound**2))+(f*a02bound)
                down2 = ((a/6.)*(timemin**6))+((b/5.)*(timemin**5))+((c/4.)*(timemin**4))+((d/3.)*(timemin**3))+((e/2.)*(timemin**2))+(f*timemin)
                down = up2 - down2
                nns_m = up + down # + because we are adding the contribution of the two integrals on either side of the piecewise discontinuity

            # if both timemin and timemax are in power law portion of DTD
            elif timemin > a02bound:
                up = a_pow * np.log(timemax)
                down = a_pow * np.log(timemin)
                nns_m = up - down

            # normalize
            nns_m *= self.A_nsmerger_02

        # Integrate over 0.1 solar metallicity
        elif self.zmetal <= 0.002:

            # Define a002 DTD fit parameters
            a = -2.88192413434e-5
            b = 0.00387383125623
            c = -0.20721471544
            d = 5.64382310405
            e = -82.6061154979
            f = 617.464778362
            g = -1840.49386605
            a_pow = 153.68106991

            # Manually compute definite integral values over DTD with bounds timemin and timemax, procedurally identical to a02 computation above
            if timemax < lower:
                nns_m = 0.0
            elif timemin < lower and timemax <= a002bound:
                up = ((a/7.)*(timemax**7))+((b/6.)*(timemax**6))+((c/5.)*(timemax**5))+((d/4.)*(timemax**4))+((e/3.)*(timemax**3))+((f/2.)*(timemax**2))+(g*timemax)
                down = ((a/7.)*(lower**7))+((b/6.)*(lower**6))+((c/5.)*(lower**5))+((d/4.)*(lower**4))+((e/3.)*(lower**3))+((f/2.)*(lower**2))+(g*lower)
                nns_m = up - down
            elif timemin < lower and timemax >= a002bound:
                up1 = a_pow * np.log(timemax)
                down1 = a_pow * np.log(a002bound)
                up = up1 - down1
                up2 = ((a/7.)*(a002bound**7))+((b/6.)*(a002bound**6))+((c/5.)*(a002bound**5))+((d/4.)*(a002bound**4))+((e/3.)*(a002bound**3))+((f/2.)*(a002bound**2))+(g*a002bound)
                down2 = ((a/7.)*(lower**7))+((b/6.)*(lower**6))+((c/5.)*(lower**5))+((d/4.)*(lower**4))+((e/3.)*(lower**3))+((f/2.)*(lower**2))+(g*lower)
                down = up2 - down2
                nns_m = up + down # + because we are adding the contribution of the two integrals on either side of the piecewise discontinuity
            elif timemin >= lower and timemax <= a002bound:
                up = ((a/7.)*(timemax**7))+((b/6.)*(timemax**6))+((c/5.)*(timemax**5))+((d/4.)*(timemax**4))+((e/3.)*(timemax**3))+((f/2.)*(timemax**2))+(g*timemax)
                down = ((a/7.)*(timemin**7))+((b/6.)*(timemin**6))+((c/5.)*(timemin**5))+((d/4.)*(timemin**4))+((e/3.)*(timemin**3))+((f/2.)*(timemin**2))+(g*timemin)
                nns_m = up - down
            elif timemin <= a002bound and timemax > a002bound:
                up1 = a_pow * np.log(timemax)
                down1 = a_pow * np.log(a002bound)
                up = up1 - down1
                up2 = ((a/7.)*(a002bound**7))+((b/6.)*(a002bound**6))+((c/5.)*(a002bound**5))+((d/4.)*(a002bound**4))+((e/3.)*(a002bound**3))+((f/2.)*(a002bound**2))+(g*a002bound)
                down2 = ((a/7.)*(timemin**7))+((b/6.)*(timemin**6))+((c/5.)*(timemin**5))+((d/4.)*(timemin**4))+((e/3.)*(timemin**3))+((f/2.)*(timemin**2))+(g*timemin)
                down = up2 - down2
                nns_m = up + down # + because we are adding the contribution of the two integrals on either side of the piecewise discontinuity
            elif timemin > a002bound:
                up = a_pow*np.log(timemax)
                down = a_pow*np.log(timemin)
                nns_m = up - down

            # normalize
            nns_m *= self.A_nsmerger_002

        # Interpolate between the two metallicities
        else:

            # Define a002 DTD fit parameters
            a = -2.88192413434e-5
            b = 0.00387383125623
            c = -0.20721471544
            d = 5.64382310405
            e = -82.6061154979
            f = 617.464778362
            g = -1840.49386605
            a_pow = 153.68106991

            # 0.1 solar metallicity integration
            if timemax < lower:
                nns_m002 = 0.0
            elif timemin < lower and timemax <= a002bound:
                up = ((a/7.)*(timemax**7))+((b/6.)*(timemax**6))+((c/5.)*(timemax**5))+((d/4.)*(timemax**4))+((e/3.)*(timemax**3))+((f/2.)*(timemax**2))+(g*timemax)
                down = ((a/7.)*(lower**7))+((b/6.)*(lower**6))+((c/5.)*(lower**5))+((d/4.)*(lower**4))+((e/3.)*(lower**3))+((f/2.)*(lower**2))+(g*lower)
                nns_m002 = up - down
            elif timemin < lower and timemax >= a002bound:
                up1 = a_pow * np.log(timemax)
                down1 = a_pow * np.log(a002bound)
                up = up1 - down1
                up2 = ((a/7.)*(a002bound**7))+((b/6.)*(a002bound**6))+((c/5.)*(a002bound**5))+((d/4.)*(a002bound**4))+((e/3.)*(a002bound**3))+((f/2.)*(a002bound**2))+(g*a002bound)
                down2 = ((a/7.)*(lower**7))+((b/6.)*(lower**6))+((c/5.)*(lower**5))+((d/4.)*(lower**4))+((e/3.)*(lower**3))+((f/2.)*(lower**2))+(g*lower)
                down = up2 - down2
                nns_m002 = up + down # + because we are adding the contribution of the two integrals on either side of the piecewise discontinuity
            elif timemin >= lower and timemax <= a002bound:
                up = ((a/7.)*(timemax**7))+((b/6.)*(timemax**6))+((c/5.)*(timemax**5))+((d/4.)*(timemax**4))+((e/3.)*(timemax**3))+((f/2.)*(timemax**2))+(g*timemax)
                down = ((a/7.)*(timemin**7))+((b/6.)*(timemin**6))+((c/5.)*(timemin**5))+((d/4.)*(timemin**4))+((e/3.)*(timemin**3))+((f/2.)*(timemin**2))+(g*timemin)
                nns_m002 = up - down
            elif timemin <= a002bound and timemax > a002bound:
                up1 = a_pow * np.log(timemax)
                down1 = a_pow * np.log(a002bound)
                up = up1 - down1
                up2 = ((a/7.)*(a002bound**7))+((b/6.)*(a002bound**6))+((c/5.)*(a002bound**5))+((d/4.)*(a002bound**4))+((e/3.)*(a002bound**3))+((f/2.)*(a002bound**2))+(g*a002bound)
                down2 = ((a/7.)*(timemin**7))+((b/6.)*(timemin**6))+((c/5.)*(timemin**5))+((d/4.)*(timemin**4))+((e/3.)*(timemin**3))+((f/2.)*(timemin**2))+(g*timemin)
                down = up2 - down2
                nns_m002 = up + down # + because we are adding the contribution of the two integrals on either side of the piecewise discontinuity
            elif timemin > a002bound:
                up = a_pow*np.log(timemax)
                down = a_pow*np.log(timemin)
                nns_m002 = up - down

            # Define a02 DTD fit parameters
            a = -0.0138858377011
            b = 1.0712569392
            c = -32.1555682584
            d = 468.236521089
            e = -3300.97955814
            f = 9019.62468302
            a_pow = 1079.77358975

            # solar metallicity integration
            if timemax < lower:
                nns_m02 = 0.0
            elif timemin < lower and timemax <= a02bound:
                up = ((a/6.)*(timemax**6))+((b/5.)*(timemax**5))+((c/4.)*(timemax**4))+((d/3.)*(timemax**3))+((e/2.)*(timemax**2))+(f*timemax)
                down = ((a/6.)*(lower**6))+((b/5.)*(lower**5))+((c/4.)*(lower**4))+((d/3.)*(lower**3))+((e/2.)*(lower**2))+(f*lower)
                nns_m02 = up - down
            elif timemin < lower and timemax >= a02bound:
                up1 = a_pow * np.log(timemax)
                down1 = a_pow * np.log(a02bound)
                up = up1 - down1
                up2 = ((a/6.)*(a02bound**6))+((b/5.)*(a02bound**5))+((c/4.)*(a02bound**4))+((d/3.)*(a02bound**3))+((e/2.)*(a02bound**2))+(f*a02bound)
                down2 = ((a/6.)*(lower**6))+((b/5.)*(lower**5))+((c/4.)*(lower**4))+((d/3.)*(lower**3))+((e/2.)*(lower**2))+(f*lower)
                down = up2 - down2
                nns_m02 = up + down # + because we are adding the contribution of the two integrals on either side of the piecewise discontinuity
            elif timemin >= lower and timemax <= a02bound:
                up = ((a/6.)*(timemax**6))+((b/5.)*(timemax**5))+((c/4.)*(timemax**4))+((d/3.)*(timemax**3))+((e/2.)*(timemax**2))+(f*timemax)
                down = ((a/6.)*(timemin**6))+((b/5.)*(timemin**5))+((c/4.)*(timemin**4))+((d/3.)*(timemin**3))+((e/2.)*(timemin**2))+(f*timemin)
                nns_m02 = up - down
            elif timemin <= a02bound and timemax > a02bound:
                up1 = a_pow * np.log(timemax)
                down1 = a_pow * np.log(a02bound)
                up = up1 - down1
                up2 = ((a/6.)*(a02bound**6))+((b/5.)*(a02bound**5))+((c/4.)*(a02bound**4))+((d/3.)*(a02bound**3))+((e/2.)*(a02bound**2))+(f*a02bound)
                down2 = ((a/6.)*(timemin**6))+((b/5.)*(timemin**5))+((c/4.)*(timemin**4))+((d/3.)*(timemin**3))+((e/2.)*(timemin**2))+(f*timemin)
                down = up2 - down2
                nns_m02 = up + down # + because we are adding the contribution of the two integrals on either side of the piecewise discontinuity
            elif timemin > a02bound:
                up = a_pow * np.log(timemax)
                down = a_pow * np.log(timemin)
                nns_m02 = up - down

            # normalize
            nns_m02 *= self.A_nsmerger_02
            nns_m002 *= self.A_nsmerger_002

            # interpolate between nns_m002 and nns_m02
            metallicities = np.asarray([0.002, 0.02])
            nsm_array = np.asarray([nns_m002, nns_m02])
            nns_m = np.interp(self.zmetal, metallicities, nsm_array)

        # return the number of neutron star mergers produced in this time interval
        return nns_m


    ##############################################
    #              Get Nb NSM Array              #
    ##############################################
    def __get_nb_nsm_array(self, timemin, timemax, i_Z_temp):
        '''
        This function returns the number of NSMs that occur within
        a specific time interval for the input DTD array.
        
        Arguments
        =========

            timemin : Lower time intervall of the OMEGA timestep
            timemax : Upper time intervall of the OMEGA timestep
            i_Z_temp : Index of the considered Z in the DTD array

        '''

        # If there are some NSMs ...
        nb_NSMs_temp = 0.0
        if timemin < max(self.nsmerger_dtd_array[i_Z_temp][0]) and \
           timemax > min(self.nsmerger_dtd_array[i_Z_temp][0]):
                
            # Find the lower time boundary of the first input interval
            i_t_low = 0
            while timemin > self.nsmerger_dtd_array[i_Z_temp][0][i_t_low+1]:
                i_t_low += 1

            # While the current input interval is still within timemin - timemax ...
            while timemax > self.nsmerger_dtd_array[i_Z_temp][0][i_t_low]:

                # Cumulate the number of NSMs
                dt_NSM_temp = \
                    min(timemax, self.nsmerger_dtd_array[i_Z_temp][0][i_t_low+1]) - \
                    max(timemin, self.nsmerger_dtd_array[i_Z_temp][0][i_t_low])
                nb_NSMs_temp += \
                    self.nsmerger_dtd_array[i_Z_temp][1][i_t_low] * dt_NSM_temp

                # Go to the next interval
                i_t_low += 1

        # Return the number of NSMs
        return nb_NSMs_temp


    ##############################################
    #               NS Merger Rate               #
    ##############################################
    def __nsmerger_rate(self, t):
        '''
        This function returns the rate of neutron star mergers occurring at a given
        stellar lifetime. It uses the delay time distribution
        of Dominik et al. (2012).
        
        Arguments
        =========

            t : lifetime of stellar population in question
            Z : metallicity of stellar population in question

        '''
        # if solar metallicity...
        if self.zmetal == 0.02:

            # piecewise defined DTD
            if t < 25.7:
                func = (-0.0138858377011*(t**5))+(1.10712569392*(t**4))-(32.1555682584*(t**3))+(468.236521089*(t**2))-(3300.97955814*t)+(9019.62468302)
            elif t >= 25.7:
                func = 1079.77358975/t

        # if 0.1 solar metallicity...
        elif self.zmetal == 0.002:

            # piecewise defined DTD
            if t < 45.76:
                func = ((-2.88192413434e-5)*(t**6))+(0.00387383125623*(t**5))-(0.20721471544*(t**4))+(5.64382310405*(t**3))-(82.6061154979*(t**2))+(617.464778362*t)-(1840.49386605)
            elif t >= 45.76:
                func = 153.68106991 / t

        # return the appropriate NS merger rate for time t
        return func


    ##############################################
    #           NS merger normalization          #
    ##############################################
    def __normalize_nsmerger(self, lifetime_min):
        '''
        This function normalizes the Dominik et al. (2012) delay time distribution
        to appropriately compute the total number of neutron star mergers in an SSP.

        Arguments
        =========
        
            lifetime_min : minimum stellar lifetime

        '''
        # Compute the number of massive stars (NS merger progenitors)
        N = self._imf(self.nsmerger_bdys[0], self.nsmerger_bdys[1], 1)   # IMF integration

        # Compute total mass of system
        M = self._imf(self.imf_bdys[0], self.imf_bdys[1], 2)

        # multiply number by fraction in binary systems
        N *= self.f_binary / 2.

        # multiply number by fraction which will form neutron star mergers
        N *= self.f_merger

        # Define the number of NSM per Msun formed .. if not already given
        if self.nb_nsm_per_m < 0.0:
            self.nb_nsm_per_m = N / M

        # Calculate the normalization constants for Z_o and 0.1Z_o
        self.A_nsmerger_02 = N / ((196.4521885+6592.893564)*M)
        self.A_nsmerger_002 = N / ((856.0742532+849.6301493)*M)

        # Initialization for the input DTD .. if chosen
        if self.len_nsmerger_dtd_array > 0:
            self.Z_nsmerger   = np.zeros(self.len_nsmerger_dtd_array)
            for i_dtd in range(0,self.len_nsmerger_dtd_array):
                self.Z_nsmerger[i_dtd] = self.nsmerger_dtd_array[i_dtd][2]
                if max(self.nsmerger_dtd_array[i_dtd][0]) < self.history.tend:
                    self.nsmerger_dtd_array[i_dtd][0].append(2.*self.history.tend)
                    self.nsmerger_dtd_array[i_dtd][1].append(0.0)

        # Calculate the normalization of the power law .. if chosen
        elif len(self.nsm_dtd_power) > 0:
            t_min_temp = self.nsm_dtd_power[0]
            t_max_temp = self.nsm_dtd_power[1]
            alpha_temp = self.nsm_dtd_power[2]
            if alpha_temp == -1.0:
                self.A_nsmerger = self.nb_nsm_per_m / \
                    ( np.log(t_max_temp) - np.log(t_min_temp) )
            else:
                self.A_nsmerger = self.nb_nsm_per_m * (1.0+alpha_temp) / \
                    ( t_max_temp**(1.0+alpha_temp) - t_min_temp**(1.0+alpha_temp) )

        # Ensure normalization only occurs once
        self.nsm_normalized = True


    #############################################
    #           BHNS Merger Contribution        #
    #############################################
    def __bhnsmerger_contribution(self, i):
        '''
        This function calculates the contribution of BH-NS mergers on the stellar ejecta
        and adds it to the mdot array.

        Arguments
        =========

            i : index of the current timestep

        '''

        # Get BHNS merger yields
        tables_Z = self.ytables_bhnsmerger.metallicities
        for tz in tables_Z:
            if self.zmetal > tz:
                yieldsbhnsm = self.ytables_bhnsmerger.get(Z=tz, quantity='Yields')
                break
            if self.zmetal <= tables_Z[-1]:
                yieldsbhnsm = self.ytables_bhnsmerger.get(Z=tables_Z[-1], quantity='Yields')
                break

        # initialize variables which cumulate in loop
        tt = 0

        # Normalize ...
        if not self.bhnsm_normalized:
            self.__normalize_bhnsmerger()

        # For every upcoming timestep j, starting with the current one...
        for j in range(i-1, self.nb_timesteps):

            # Set the upper and lower time boundary of the timestep j
            timemin = tt
            tt += self.history.timesteps[j]
            timemax = tt

            # Stop if the SSP no more BHNS merger occurs
            #if timemin >= self.t_bhns_merger_max:
            #    break

            # Calculate the number of BHNS mergers per unit of stellar mass formed
            nbhns_m = self.__bhnsmerger_num(timemin, timemax)

            # Calculate the number of BHNS mergers in the current SSP
            nbhns_m = nbhns_m * self.m_locked
            self.bhnsm_numbers[j] += nbhns_m

            # Add the contribution of NS mergers to the timestep j
            self.mdot_bhnsm[j] = np.array(self.mdot_bhnsm[j]) + np.array(nbhns_m * self.m_ej_bhnsm * yieldsbhnsm)
            self.mdot[j] = np.array(self.mdot[j]) + np.array(nbhns_m * self.m_ej_bhnsm * yieldsbhnsm)


    ##############################################
    #              BHNS merger number            #
    ##############################################
    def __bhnsmerger_num(self, timemin, timemax):

        '''
        This function returns the number of BH-NS mergers, per units of stellar mass
        formed, occurring within a given time interval using a delay-time distribution
        function.
        
        Arguments
        =========
        
            timemin : Lower boundary of time interval.
            timemax : Upper boundary of time interval.

        '''

        # If an input DTD array is provided ...
        if self.len_bhnsmerger_dtd_array > 0:
            
            # Find the lower and upper Z boundaries
            if self.zmetal <= self.Z_bhnsmerger[0]:
                i_Z_low = 0
                i_Z_up  = 0
            elif self.zmetal >= self.Z_bhnsmerger[-1]:
                i_Z_low = -1
                i_Z_up  = -1
            else:
                i_Z_low = 0
                i_Z_up  = 1
                while self.zmetal > self.Z_bhnsmerger[i_Z_up]:
                    i_Z_low += 1
                    i_Z_up  += 1

            # Get the number of BHNSMs at the lower Z boundary
            nb_BHNSMs_low = self.__get_nb_bhnsm_array(timemin, timemax, i_Z_low)
            
            # Return the number of BHNSM .. if no interpolation is needed
            if i_Z_up == i_Z_low:
                return nb_BHNSMs_low

            # Interpolate the number of BHNSMs .. if needed
            else:
                nb_BHNSMs_up = self.__get_nb_bhnsm_array(timemin, timemax, i_Z_up)
                lg_Z_low   = np.log10(self.Z_bhnsmerger[i_Z_low])
                lg_Z_up    = np.log10(self.Z_bhnsmerger[i_Z_up])
                lg_Z_metal = np.log10(self.zmetal)
                a = (nb_BHNSMs_up - nb_BHNSMs_low) / (lg_Z_up - lg_Z_low)
                b = nb_BHNSMs_low - a * lg_Z_low
                return a * lg_Z_metal + b

        # Return zero if no DTD is selected
        else:
            return 0.0


    ##############################################
    #             Get Nb BHNSM Array             #
    ##############################################
    def __get_nb_bhnsm_array(self, timemin, timemax, i_Z_temp):
        '''
        This function returns the number of BHNSMs that occur within
        a specific time interval for the input DTD array.
        
        Arguments
        =========

            timemin : Lower time intervall of the OMEGA timestep
            timemax : Upper time intervall of the OMEGA timestep
            i_Z_temp : Index of the considered Z in the DTD array

        '''

        # If there are some BHNSMs ...
        nb_BHNSMs_temp = 0.0
        if timemin < max(self.bhnsmerger_dtd_array[i_Z_temp][0]) and \
           timemax > min(self.bhnsmerger_dtd_array[i_Z_temp][0]):
                
            # Find the lower time boundary of the first input interval
            i_t_low = 0
            while timemin > self.bhnsmerger_dtd_array[i_Z_temp][0][i_t_low+1]:
                i_t_low += 1

            # While the current input interval is still within timemin - timemax ...
            while timemax > self.bhnsmerger_dtd_array[i_Z_temp][0][i_t_low]:

                # Cumulate the number of NSMs
                dt_BHNSM_temp = \
                    min(timemax, self.bhnsmerger_dtd_array[i_Z_temp][0][i_t_low+1]) - \
                    max(timemin, self.bhnsmerger_dtd_array[i_Z_temp][0][i_t_low])
                nb_BHNSMs_temp += \
                    self.bhnsmerger_dtd_array[i_Z_temp][1][i_t_low] * dt_BHNSM_temp

                # Go to the next interval
                i_t_low += 1

        # Return the number of NSMs
        return nb_BHNSMs_temp


    ##############################################
    #         BHNS Merger Normalization          #
    ##############################################
    def __normalize_bhnsmerger(self):
        '''
        This function normalizes the delay time distribution of BH-NS merger
        to appropriately compute the total number of BH-NS mergers in an SSP.

        '''

        # Calculate the normalization of the input DTD .. if chosen
        if self.len_bhnsmerger_dtd_array > 0:
            self.Z_bhnsmerger   = np.zeros(self.len_bhnsmerger_dtd_array)
            for i_dtd in range(0,self.len_bhnsmerger_dtd_array):
                self.Z_bhnsmerger[i_dtd] = self.bhnsmerger_dtd_array[i_dtd][2]
                if max(self.bhnsmerger_dtd_array[i_dtd][0]) < self.history.tend:
                    self.bhnsmerger_dtd_array[i_dtd][0].append(2.*self.history.tend)
                    self.bhnsmerger_dtd_array[i_dtd][1].append(0.0)
        
        # Ensure normalization only occurs once
        self.bhnsm_normalized = True


    #############################################
    #           Normalize Delayed Extra         #
    #############################################
    def __normalize_delayed_extra(self):

        '''
        This function normalize the DTD of all input delayed extra sources.

        '''

        # Create the normalization factor array
        self.delayed_extra_dtd_A_norm = []

        # Create the un-normalized maximum dtd value array
        self.delayed_extra_dtd_max = []

        # For each delayed source ...
        for i_e_nde in range(0,self.nb_delayed_extra):
          self.delayed_extra_dtd_A_norm.append([])
          self.delayed_extra_dtd_max.append([])

          # For each metallicity ...
          for i_Z_nde in range(0,len(self.delayed_extra_dtd[i_e_nde])):

            # Copy the lower and upper time boundaries
            t_low_nde = self.delayed_extra_dtd[i_e_nde][i_Z_nde][0][0]
            t_up_nde  = self.delayed_extra_dtd[i_e_nde][i_Z_nde][-1][0]

            # Integrate the entire DTD
            N_tot_nde = self.__delayed_extra_num(t_low_nde,t_up_nde,i_e_nde,i_Z_nde)

            # Assigne the normalization factor
            self.delayed_extra_dtd_A_norm[i_e_nde].append(\
                self.delayed_extra_dtd_norm[i_e_nde][i_Z_nde] / N_tot_nde)

            # Find the max DTD value (un-normalized)
            #yy_max_temp = 0.0
            #for i_t_nde in range(len(self.delayed_extra_dtd[i_e_nde][i_Z_nde])):
            #    if self.delayed_extra_dtd[i_e_nde][i_Z_nde][i_t_nde][1] > yy_max_temp:
            #        yy_max_temp = self.delayed_extra_dtd[i_e_nde][i_Z_nde][i_t_nde][1]
            #self.delayed_extra_dtd_max[i_e_nde].append(yy_max_temp)


    #############################################
    #          Delayed Extra Contribution       #
    #############################################
    def __delayed_extra_contribution(self, i):

        '''
        This function calculates the contribution of delayed extra source
        and adds it to the mdot array.

        Arguments
        =========

            i : index of the current timestep

        '''

        # For each delayed extra source ...
        for i_extra in range(0,self.nb_delayed_extra):

            # Get the yields and metallicity indexes of the considered source
            Z_extra, yextra_low, yextra_up, iZ_low, iZ_up = \
                self.__get_YZ_delayed_extra(i_extra)
         
            # Initialize age of the latest SSP, which cumulate in loop
            tt = 0

            # Define the ultimate min and max times
            tmax_extra = max(self.delayed_extra_dtd[i_extra][iZ_low][-1][0],\
                             self.delayed_extra_dtd[i_extra][iZ_up ][-1][0])
            tmin_extra = min(self.delayed_extra_dtd[i_extra][iZ_low][0][0],\
                             self.delayed_extra_dtd[i_extra][iZ_up ][0][0])

            # For every upcoming timestep j, starting with the current one...
            for j in range(i-1, self.nb_timesteps):

              # Set the upper and lower time boundary of the timestep j
              timemin = tt
              tt += self.history.timesteps[j]
              timemax = tt

              # Stop if the SSP do not contribute anymore to the delayed extra source
              if timemin >= tmax_extra:
                  break

              # If the is someting to eject during this timestep j ...
              if timemax > tmin_extra:

                # Get the total number of sources and yields (interpolated)
                nb_sources_extra_tot, yields_extra_interp = \
                    self.__get_nb_y_interp(timemin, timemax, i_extra, iZ_low, iZ_up,\
                        yextra_low, yextra_up, Z_extra)

                # Calculate the number of sources in the current SSP (not per Msun)
                self.delayed_extra_numbers[i_extra][j] += nb_sources_extra_tot

                # Add the contribution of the sources to the timestep j
                self.mdot_delayed_extra[i_extra][j] = \
                    np.array(self.mdot_delayed_extra[i_extra][j]) + yields_extra_interp
                self.mdot[j] = np.array(self.mdot[j]) + yields_extra_interp


    #############################################
    #             Get YZ Delayed Extra          #
    #############################################
    def __get_YZ_delayed_extra(self, i_extra):

        '''
        This function returns the yields, metallicities, and Z boundary indexes
        for a considered delayed extra source (according to the ISM metallicity).

        Arguments
        =========

          i_extra : index of the extra source

        '''

        # Get the number of metallicities considered source (in decreasing order)
        Z_extra =  self.ytables_delayed_extra[i_extra].metallicities
        nb_Z_extra = len(Z_extra)

        # Set the Z indexes if only one metallicity is provided
        if nb_Z_extra == 1:
            iZ_low = iZ_up = 0

        # If several metallicities are provided ...
        else:

            # Search the input metallicity interval that englobe self.zmetal
            # Copy the lowest input Z is zmetal is lower (same for higher than highest)
            if self.zmetal <= Z_extra[-1]:
                iZ_low = iZ_up = -1
            elif self.zmetal >= Z_extra[0]:
                iZ_low = iZ_up = 0
            else:
                iZ_low = 1
                iZ_up = 0
                while self.zmetal < Z_extra[iZ_low]:
                    iZ_low += 1
                    iZ_up += 1

        # Get the yields table for the lower and upper Z boundaries
        yextra_low = self.ytables_delayed_extra[i_extra].get( \
            Z=Z_extra[iZ_low], quantity='Yields')
        yextra_up  = self.ytables_delayed_extra[i_extra].get( \
            Z=Z_extra[iZ_up],  quantity='Yields')

        # Return the metallicities and the yields and Z boundaries
        return Z_extra, yextra_low, yextra_up, iZ_low, iZ_up


    #############################################
    #               Get Nb Y Interp             #
    #############################################
    def __get_nb_y_interp(self, timemin, timemax, i_extra, iZ_low, iZ_up,\
                          yextra_low, yextra_up, Z_extra):

        '''
        This function returns the yields, metallicities, and Z boundary indexes
        for a considered delayed extra source (according to the ISM metallicity).

        Arguments
        =========

          timemin : Lower boundary of the time interval.
          timemax : Upper boundary of the time interval.
          i_extra : Index of the extra source.
          iZ_low  : Lower index of the provided input Z in the delayed extra yields table.
          iZ_up   : Upper index of the provided input Z in the delayed extra yields table.
          yextra_low : Delayed extra yields of the lower Z.
          yextra_up  : Delayed extra yields of the upper Z.
          Z_extra : List of provided Z in the delayed extra yields table.

        '''

        # Calculate the number of sources per unit of Msun formed (lower Z)
        nb_sources_low = self.__delayed_extra_num(timemin, timemax, i_extra, iZ_low)

        # Normalize the number of sources (still per unit of Msun formed)
        # This needs to be before calculating ejecta_Z_low!
        nb_sources_low *= self.delayed_extra_dtd_A_norm[i_extra][iZ_low]

        # Calculate the total ejecta (yields) for the lower Z
        ejecta_Z_low = np.array(nb_sources_low * self.m_locked *
            yextra_low * self.delayed_extra_yields_norm[i_extra][iZ_low])

        # If we do not need to interpolate between Z
        if iZ_up == iZ_low:

            # Return the total number of sources and ejecta for the lower Z
            return nb_sources_low * self.m_locked, ejecta_Z_low

        # If we need to interpolate between Z
        else:

            # Calculate the number of sources per unit of Msun formed (upper Z)
            nb_sources_up = self.__delayed_extra_num(timemin, timemax, i_extra, iZ_up)

            # Normalize the number of sources (still per unit of Msun formed)
            # This needs to be before calculating ejecta_Z_up!
            nb_sources_up *= self.delayed_extra_dtd_A_norm[i_extra][iZ_up]

            # Calculate the total ejecta (yields) for the lower Z
            ejecta_Z_up = np.array(nb_sources_up * self.m_locked *
                yextra_up * self.delayed_extra_yields_norm[i_extra][iZ_up])

            # Interpolate the number of sources (N = aa*log10(Z) + bb)
            aa = (nb_sources_up - nb_sources_low) / \
                 (np.log10(Z_extra[iZ_up]) - np.log10(Z_extra[iZ_low]))
            bb = nb_sources_up - aa * np.log10(Z_extra[iZ_up])
            nb_sources_interp = aa * np.log10(self.zmetal) + bb

            # Convert yields into log if needed ..
            if self.delayed_extra_yields_log_int:
                for i_iso_temp in range(self.nb_isotopes):
                    if ejecta_Z_low[i_iso_temp] == 0.0:
                        ejecta_Z_low[i_iso_temp] = -50.0
                    else:
                        ejecta_Z_low[i_iso_temp] = np.log10(ejecta_Z_low[i_iso_temp])
                    if ejecta_Z_up[i_iso_temp] == 0.0:
                        ejecta_Z_up[i_iso_temp] = -50.0
                    else:
                        ejecta_Z_up[i_iso_temp] = np.log10(ejecta_Z_up[i_iso_temp])

            # Interpolate the yields (Y = aa*log10(Z) + bb)
            aa = (ejecta_Z_up - ejecta_Z_low) / \
                 (np.log10(Z_extra[iZ_up]) - np.log10(Z_extra[iZ_low]))
            bb = ejecta_Z_up - aa * np.log10(Z_extra[iZ_up])
            ejecta_interp = aa * np.log10(self.zmetal) + bb

            # Convert interpolated yields back into linear scale if needed ..
            if self.delayed_extra_yields_log_int:
                for i_iso_temp in range(self.nb_isotopes):
                    ejecta_interp[i_iso_temp] = 10**(ejecta_interp[i_iso_temp])

            # Return the total number of sources and ejecta for the interpolation
            return nb_sources_interp * self.m_locked, ejecta_interp


    #############################################
    #              Delayed Extra Num            #
    #############################################
    def __delayed_extra_num(self, timemin, timemax, i_extra, i_ZZ):

        '''
        This function returns the integrated number of delayed extra source within
        a given OMEGA time interval for a given source and metallicity

        Arguments
        =========

          timemin : Lower boundary of the OMEGA time interval.
          timemax : Upper boundary of the OMEGA time interval.
          i_extra : Index of the extra source.
          iZZ     : Index of the provided input Z in the delayed extra yields table.

        '''

        # Initialize the number of sources that occur between timemin and timemax
        N_den = 0

        # Search the lower boundary input time interval
        i_search = 0
        while timemin > self.delayed_extra_dtd[i_extra][i_ZZ][i_search+1][0]:
        #while timemin > self.delayed_extra_dtd[i_extra][0][i_search+1][0]:
            i_search += 1

        # Copie the current time
        t_cur = max(self.delayed_extra_dtd[i_extra][i_ZZ][0][0], timemin)
        timemax_cor = min(timemax,self.delayed_extra_dtd[i_extra][i_ZZ][-1][0])

        # While the is still time to consider in the OMEGA timestep ...
        while abs(timemax_cor - t_cur) > 0.01:

            # Integrate the DTD
            t_min_temp = max(t_cur,self.delayed_extra_dtd[i_extra][i_ZZ][i_search][0])
            t_max_temp = min(timemax_cor,self.delayed_extra_dtd[i_extra][i_ZZ][i_search+1][0])
            N_den += self.__integrate_delayed_extra_DTD(\
                t_min_temp, t_max_temp, i_extra, i_ZZ, i_search)

            # Go to the next delayed input timestep
            t_cur += t_max_temp - t_min_temp
            i_search += 1

        # Return the number of occuring sources
        return N_den


    #############################################
    #        Integrate Delayed Extra DTD        #
    #############################################
    def __integrate_delayed_extra_DTD(self, t_min_temp, t_max_temp, i_extra, i_ZZ, i_search):

        '''
        This function returns the integrated number of delayed extra source within
        a given time interval for a given source and metallicity.

        Note: There is no normalization here, as this function is actualy used for
              the normalization process.

        Arguments
        =========

          t_min_temp : Lower boundary of the delayed extra input time interval.
          t_max_temp : Upper boundary of the delayed extra time interval.
          i_extra  : Index of the extra source.
          iZZ      : Index of the provided input Z in the delayed extra yields table.
          i_search : Index of the lower input timestep interval

        '''

        # If we integrate in the log-log space
        # Rate = R = bt^a --> logR = a*logt + logb
        if self.delayed_extra_log:

          # Copy the boundary conditions of the input DTD interval
          lg_t_max_tmp = np.log10(self.delayed_extra_dtd[i_extra][i_ZZ][i_search+1][0])
          lg_t_min_tmp = np.log10(self.delayed_extra_dtd[i_extra][i_ZZ][i_search][0])
          lg_R_max_tmp = np.log10(self.delayed_extra_dtd[i_extra][i_ZZ][i_search+1][1])
          lg_R_min_tmp = np.log10(self.delayed_extra_dtd[i_extra][i_ZZ][i_search][1])

          # Calculate the coefficients "a" and "b"
          a_ided = (lg_R_max_tmp - lg_R_min_tmp) / (lg_t_max_tmp - lg_t_min_tmp)
          b_ided = 10**(lg_R_max_tmp - a_ided * lg_t_max_tmp)

          # If not a power law with an index of -1
          if a_ided > -0.999999 or a_ided < -1.0000001:

              # Integrate
              N_ided = (b_ided / (a_ided+1.0)) * \
                  (t_max_temp**(a_ided+1.0) - t_min_temp**(a_ided+1.0))
           
          # If a power law with an index of -1
          else:

              # Integrate with a natural logarithm
              N_ided = b_ided * (np.log(t_max_temp) - np.log(t_min_temp))

        # If we integrate NOT in the log-log space
        # Rate = R = a * t + b
        else:

          # Copy the boundary conditions of the input DTD interval
          t_max_tmp = self.delayed_extra_dtd[i_extra][i_ZZ][i_search+1][0]
          t_min_tmp = self.delayed_extra_dtd[i_extra][i_ZZ][i_search][0]
          R_max_tmp = self.delayed_extra_dtd[i_extra][i_ZZ][i_search+1][1]
          R_min_tmp = self.delayed_extra_dtd[i_extra][i_ZZ][i_search][1]

          # Calculate the coefficients "a" and "b"
          a_ided = (R_max_tmp - R_min_tmp) / (t_max_tmp - t_min_tmp)
          b_ided = R_max_tmp - a_ided * t_max_tmp

          # Integrate
          N_ided = 0.5 * a_ided * (t_max_temp**2 - t_min_temp**2) + \
              b_ided * (t_max_temp - t_min_temp)

        # Return the number of extra sources
        return N_ided


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
            print ('!!!!!IMPORTANT!!!!')
            print ('With Vogelsberger SNIa implementation selected mass', \
                  'range not possible.')
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
        the_m_ts = self.get_interp_lifetime_mass(t_s, self.zmetal, is_mass=False)
        return float(max(minm_prog1a, the_m_ts))


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
        mlim = self.get_interp_lifetime_mass(t, self.zmetal, is_mass=False)

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
    def __maoz12_powerlaw(self, timemin, timemax):

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
                         t_temp_low**(self.beta_pow+1.0)) / (self.beta_pow+1.0)

                # Add the number of SNe Ia (with the wrong units)
                n1a += temp_int

            # Add the right units
            n1a = n1a * self.A_maoz * 4.0e-13 / 10**(9.0*self.beta_pow)

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
    def __efolding(self, timemin, timemax):

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
    def __normalize_efolding(self, lifetime_min):

        '''
        This function normalizes the SN Ia rate of a gaussian.
   
        Argument
        ========

          lifetime_min : Minimum stellar lifetime.

        '''

        # Set the maximum mass of progenitors of SNe Ia
        maxm_prog1a = 8.0
        if maxm_prog1a > self.imf_bdys[1]:
            maxm_prog1a = self.imf_bdys[1]

        # Maximum time of integration
        ageofuniverse = 1.3e10

        # Calculate the normalisation constant
        self.A_exp = self.nb_1a_per_m / dblquad(self.__exp_sn_rate, \
            lifetime_min, ageofuniverse, \
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
        return self.__wd_number(m,t) * 1.0 / np.sqrt(2.0 * np.pi * sigma**2) * \
            np.exp(-(t - tau)**2 / (2.0 * sigma**2))


    ##############################################
    #                Wiersma09 Gauss             #
    ##############################################
    def __gauss(self, timemin, timemax):

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
    def __normalize_gauss(self, lifetime_min):

        '''
        This function normalizes the SN Ia rate of a gaussian (similar to Wiersma09).
   
        Argument
        ========

          lifetime_min : Minimum stellar lifetime.

        '''

        # Set the maximum mass of progenitors of SNe Ia
        maxm_prog1a = 8.0
        if maxm_prog1a > self.imf_bdys[1]:
            maxm_prog1a = self.imf_bdys[1]

        # Maximum time of integration
        ageofuniverse = 1.3e10

        # Calculate the normalisation constant
        self.A_gauss = self.nb_1a_per_m / dblquad(self.__gauss_sn_rate, \
            lifetime_min, ageofuniverse, \
            lambda x:self.__spline1(x), lambda x:maxm_prog1a)[0]

        # Avoid renormalizing during the next timesteps
        self.normalized = True


    ##############################################
    #                Normalize Maoz              #
    ##############################################
    def __normalize_maoz(self, lifetime_min):

        '''
        This function normalizes the SN Ia rate of Maoz or any power law.
   
        Argument
        ========

          lifetime_min : Minimum stellar lifetime.

        '''

        # Set the maximum mass of progenitors of SNe Ia
#        maxm_prog1a = 8.0
#        if maxm_prog1a > self.imf_bdys[1]:
#            maxm_prog1a = self.imf_bdys[1]

        # Maximum time of integration
#        ageofuniverse = 1.3e10

        # Calculate the normalisation constant
#        self.A_maoz = self.nb_1a_per_m / quad(self.__maoz_sn_rate_int, \
#            lifetime_min, ageofuniverse)[0]
#        print (self.A_maoz)

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
        self.A_maoz = self.nb_1a_per_m * 10**(9.0*self.beta_pow) / 4.0e-13 / \
                   (temp_3_0 - temp_8_0 + temp_13gys)

        # Avoid renormalizing during the next timesteps
        self.normalized = True


    ##############################################
    #                  Poly DTD                  #
    ##############################################
    def __poly_dtd(self, timemin, timemax):

        '''
        This function returns the total number of SNe Ia (per Mo formed) for
        a given time interval.  It uses an input DTD polynomial function of
        any order.
   
        Arguments
        =========

          timemin : Lower limit of the time (age) interval
          timemax : Upper limit of the time (age) interval

        '''

        # Initialization of the integrated DTD with upper and lower mass limit
        int_poly_up = 0.0
        int_poly_low = 0.0

        # Set the upper and lower time limit of the integration
        t_up_int = min(timemax, self.poly_fit_range[1])
        t_low_int = max(timemin, self.poly_fit_range[0])

        # If this is a split poly DTD ...
        if self.t_dtd_poly_split > 0.0:

            # If in the first section ...
            if t_up_int <= self.t_dtd_poly_split:

              # For each order of the polynomial fit ...
              for i_npf in range(0,len(self.poly_fit_dtd_5th[0])):

                # Cumulate with the upper and lower limits
                exp_poly = len(self.poly_fit_dtd_5th[0]) - i_npf - 1.0
                int_poly_up += self.poly_fit_dtd_5th[0][i_npf] * \
                    t_up_int**(exp_poly+1.0) / (exp_poly+1.0)
                int_poly_low += self.poly_fit_dtd_5th[0][i_npf] * \
                    t_low_int**(exp_poly+1.0) / (exp_poly+1.0)

            # If in the second section ...
            elif t_low_int >= self.t_dtd_poly_split:

              # For each order of the polynomial fit ...
              for i_npf in range(0,len(self.poly_fit_dtd_5th[1])):

                # Cumulate with the upper and lower limits
                exp_poly = len(self.poly_fit_dtd_5th[1]) - i_npf - 1.0
                int_poly_up += self.poly_fit_dtd_5th[1][i_npf] * \
                    t_up_int**(exp_poly+1.0) / (exp_poly+1.0)
                int_poly_low += self.poly_fit_dtd_5th[1][i_npf] * \
                    t_low_int**(exp_poly+1.0) / (exp_poly+1.0)

            # If overlap ...
            else:

              # For each order of the polynomial fit ...
              for i_npf in range(0,len(self.poly_fit_dtd_5th[0])):

                # Cumulate with the upper and lower limits
                exp_poly = len(self.poly_fit_dtd_5th[0]) - i_npf - 1.0
                int_poly_up += self.poly_fit_dtd_5th[0][i_npf] * \
                    self.t_dtd_poly_split**(exp_poly+1.0) / (exp_poly+1.0)
                int_poly_low += self.poly_fit_dtd_5th[0][i_npf] * \
                    t_low_int**(exp_poly+1.0) / (exp_poly+1.0)
                exp_poly = len(self.poly_fit_dtd_5th[1]) - i_npf - 1.0
                int_poly_up += self.poly_fit_dtd_5th[1][i_npf] * \
                    t_up_int**(exp_poly+1.0) / (exp_poly+1.0)
                int_poly_low += self.poly_fit_dtd_5th[1][i_npf] * \
                    self.t_dtd_poly_split**(exp_poly+1.0) / (exp_poly+1.0)

        # If this is not a split poly DTD ...
        else:

            # For each order of the polynomial fit ...
            for i_npf in range(0,len(self.poly_fit_dtd_5th)):

                # Cumulate with the upper and lower limits
                exp_poly = len(self.poly_fit_dtd_5th) - i_npf - 1.0
                int_poly_up += self.poly_fit_dtd_5th[i_npf] * \
                    t_up_int**(exp_poly+1.0) / (exp_poly+1.0)
                int_poly_low += self.poly_fit_dtd_5th[i_npf] * \
                    t_low_int**(exp_poly+1.0) / (exp_poly+1.0)
 
        # Return the number of SNe Ia n this time bin
        if (int_poly_up - int_poly_low) < 0.0: # can happen since it's a fit
            return 0.0
        else:
            return self.A_poly * (int_poly_up - int_poly_low)


    ##############################################
    #              Normalize Poly Fit            #
    ##############################################
    def __normalize_poly_fit(self):

        '''
        This function normalizes the polynomial input DTD function.  Can
        be any polynomial order.

        '''

        # Initialization of the integrated DTD with upper and lower mass limit
        int_poly_up = 0.0
        int_poly_low = 0.0

        # If it is a split poly DTD ...
        if self.t_dtd_poly_split > 0.0:

          # For each order of the polynomial fit ...
          for i_npf in range(0,len(self.poly_fit_dtd_5th[0])):

              # Cumulate with the upper and lower limits
              exp_poly = len(self.poly_fit_dtd_5th[0]) - i_npf - 1.0
              int_poly_up += self.poly_fit_dtd_5th[0][i_npf] * \
                  self.t_dtd_poly_split**(exp_poly+1.0) / (exp_poly+1.0)
              int_poly_low += self.poly_fit_dtd_5th[0][i_npf] * \
                  self.poly_fit_range[0]**(exp_poly+1.0) / (exp_poly+1.0)
              exp_poly = len(self.poly_fit_dtd_5th[1]) - i_npf - 1.0
              int_poly_up += self.poly_fit_dtd_5th[1][i_npf] * \
                  self.poly_fit_range[1]**(exp_poly+1.0) / (exp_poly+1.0)
              int_poly_low += self.poly_fit_dtd_5th[1][i_npf] * \
                  self.t_dtd_poly_split**(exp_poly+1.0) / (exp_poly+1.0)

        # If it not is a split poly DTD ...
        else:

            # For each order of the polynomial fit ...
            for i_npf in range(0,len(self.poly_fit_dtd_5th)):

                # Cumulate with the upper and lower limits
                exp_poly = len(self.poly_fit_dtd_5th) - i_npf - 1.0
                int_poly_up += self.poly_fit_dtd_5th[i_npf] * \
                    self.poly_fit_range[1]**(exp_poly+1.0) / (exp_poly+1.0)
                int_poly_low += self.poly_fit_dtd_5th[i_npf] * \
                    self.poly_fit_range[0]**(exp_poly+1.0) / (exp_poly+1.0)

        # Calculate the normalization constant
        self.A_poly = self.nb_1a_per_m / (int_poly_up - int_poly_low)

        # Avoid renormalizing during the next timesteps
        self.normalized = True

    ##############################################
    #              element list                  #
    ##############################################
    def _i_elem_lists(self, elem):
        '''
        Finds and returns the list of indices for isotopes of
        element 'elem'. Also returns a list of the indices for
        H and He to facility metallicity calculations.
        
        Arguments
        =========

        elem  : a string identifying the element requested.

        Returns 2 lists
        =========

        indices of isotopes of elem,
        indices of isotopes of H and He

        '''
        # Declare the list of isotope indexes associated with this element
        i_iso_list = []
        # Declare the list of isotope indexes associated with H and He
        i_H_He_list = []
        
        # Find the isotopes associated with this element
        for i_iso in range(self.nb_isotopes):
            if self.history.isotopes[i_iso].split('-')[0] == elem:
                i_iso_list.append(i_iso)
            if 'H-' in self.history.isotopes[i_iso] or 'He-' in self.history.isotopes[i_iso]:
                i_H_He_list.append(i_iso)
        return i_iso_list, i_H_He_list


    ##############################################
    #           Compute metal fraction           #
    ##############################################
    def Z_x(self, elem, t_step=-1):
        '''
        Compute the metal fraction for a list of elements.
        The metal fraction is defined as mass_element/mass_metals.
        
        Arguments
        =========

        elem     : the name of the element to use. All isotopes
                   will be found.
        t_step   : the indx of the time step to do the calculation.
                   if t_step = -1, or not specified, the last
                   time_step is used

        Returns
        =========

        mass fraction of metals for all isotopes identified by
        i_iso_list as a single number

        '''
        
        # Get the list of isotopes indices for element elem
        # along with a list of indices for H and He
        i_iso_list, i_H_He_list = self._i_elem_lists(elem)
        
        if len(i_iso_list) == 0:
            print("Element {} not found. Returning -1".format(elem))
        if t_step > self.nb_timesteps:
            print("t_step must be < nb_timesteps")
            return -1.0
        if t_step == -1:
            t_step = self.nb_timesteps
        
        # Calculate the total mass of gas at that timestep
        m_tot   = self.ymgal[t_step].sum()
        m_Z_tot = m_tot
        # Calculate the total mass of metals at that timestep
        for i_iso in range(len(i_H_He_list)):
            m_Z_tot = m_Z_tot - self.ymgal[t_step][i_H_He_list[i_iso]]
            
        # Make sure there is something in the gas reservoir ..
        if m_Z_tot > 0.0:
            # Sum the mass of each isotope associated with the desired element
            m_tot_elem = 0.0
            for i_iso in range(len(i_iso_list)):
                m_tot_elem += self.ymgal[t_step][ i_iso_list[i_iso] ]
            
            # Calculate the mass fraction of metals
            return m_tot_elem / m_Z_tot
        else:
            return 0.0

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
                print ('Warning in _imf function')
                print ('mmin:',mmin)
                print ('mmax',mmax)
                print ('mmin>mmax')
                print ('Assume mmin == mmax')
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
                self.imfnorm = 1.0 / quad(self.__g2_costum, \
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

        # Kroupa 1993 - IMF
        elif self.imf_type=='kroupa93':

            # Choose the right option
            if inte == 0:
                return self.imfnorm * self.__g1_kroupa93_alphaimf(mass)
            if inte == 1:
                return quad(self.__g1_kroupa93_alphaimf, mmin, mmax)[0]
            if inte == 2:
                return quad(self.__g2_kroupa93_alphaimf, mmin, mmax)[0]
            if inte == -1:
                self.imfnorm = 1.0 / quad(self.__g2_kroupa93_alphaimf, \
                    self.imf_bdys[0], self.imf_bdys[1])[0]

        # Kroupa IMF
        elif self.imf_type=='kroupa':

            # Choose the right option
            if inte == 0:
                return self.imfnorm * self.__g1_kroupa(mass)
            if inte == 1:
                return self.__integrate_g1_kroupa(mmin, mmax)
            if inte == 2:
                return quad(self.__g2_kroupa, mmin, mmax)[0]
            if inte == -1:
                self.imfnorm = 1.0 / quad(self.__g2_kroupa, \
                    self.imf_bdys[0], self.imf_bdys[1])[0]

        elif self.imf_type == 'lognormal': # RJS
            # Choose the right option
            if inte == 0:
                return self.imfnorm * self.__g1_log_normal(mass)
            if inte == 1:
                return quad(self.__g1_log_normal, mmin, mmax)[0]
            if inte == 2:
                return quad(self.__g2_log_normal, mmin, mmax)[0]
            if inte == -1:
                self.imfnorm = 1.0 / quad(self.__g2_log_normal, \
                    self.imf_bdys[0], self.imf_bdys[1])[0]

        # Ferrini, Pardi & Penco (1990)
        elif self.imf_type=='fpp':

            # Choose the right option
            if inte == 0:
                return self.imfnorm * self.__g1_fpp(mass)
            if inte == 1:
                return quad(self.__g1_fpp, mmin, mmax)[0]
            if inte == 2:
                #return quad(self.__g2_fpp, mmin, mmax)[0]
 
                #if mmin < 0.8:
                #    print ('!!Error - Ferrini IMF not fitted below 0.8 Msun!!')

                # Find the lower mass bin
                i_fer = 0
                while mmin >= self.m_up_fer[i_fer]:
                    i_fer += 1

                # Integrate this mass bin ...
                imf_int = 0.0
                imf_int += self.norm_fer[i_fer] * \
                       (min(mmax,self.m_up_fer[i_fer])**self.alpha_fer[i_fer]\
                           - mmin**self.alpha_fer[i_fer])

                # For the remaining mass bin ...
                if not mmax <= self.m_up_fer[i_fer]:
                  for i_fer2 in range((i_fer+1),len(self.m_up_fer)):
                    if mmax >= self.m_up_fer[i_fer2-1]:
                      imf_int += self.norm_fer[i_fer2] * \
                          (min(mmax,self.m_up_fer[i_fer2])**self.alpha_fer[i_fer2]\
                           - self.m_up_fer[i_fer2-1]**self.alpha_fer[i_fer2])

                # Return the integration
                return imf_int

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
    #               G1 Log Normal                #
    ##############################################
    def __g1_log_normal(self, mass):

        '''
        This function returns the number of stars having a certain stellar mass
        with a log normal IMF with characteristic mass self.imf_pop3_char_mass.

        Arguments
        =========

          mass : Stellar mass.

          ** future add, sigma... assuming sigma = 1 for now **

        '''
        # Select the right alpha index
        return np.exp(-1.0/2.0 * np.log(mass/self.imf_pop3_char_mass)**2) * 1/mass


    ##############################################
    #                G2 Log Normal                #
    ##############################################
    def __g2_log_normal(self, mass):

        '''
        This function returns the total mass of stars having a certain initial
        mass with a log normal IMF with characteristic mass self.imf_pop3_char_mass.

        Arguments
        =========

          mass : Stellar mass.

          ** future add, sigma... assuming sigma = 1 for now **

        '''
        return np.exp(-1.0/2.0 * np.log(mass/self.imf_pop3_char_mass)**2)


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
    #            G1 Kroupa93 AlphaIMF            #
    ##############################################
    def __g1_kroupa93_alphaimf(self, mass):

        '''
        This function returns the number of stars having a certain stellar mass
        with a Kroupa et al. (1993) IMF.

        Arguments
        =========

          mass : Stellar mass.

        '''

        # Select the right mass regime
        if mass < 0.5:
            return 0.035 * mass**(-1.3)
        elif mass < 1.0:
            return 0.019 * mass**(-2.2)
        else:
            return 0.019 * mass**(-2.7)


    ##############################################
    #            G2 Kroupa93 AlphaIMF            #
    ##############################################
    def __g2_kroupa93_alphaimf(self, mass):

        '''
        This function returns the total mass of stars having a certain stellar
        mass with a Kroupa et al. (1993) IMF.

        Arguments
        =========

          mass : Stellar mass.

        '''

        # Select the right mass regime
        if mass < 0.5:
            return 0.035 * mass * mass**(-1.3)
        elif mass < 1.0:
            return 0.019 * mass * mass**(-2.2)
        else:
            return 0.019 * mass * mass**(-2.7)


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
    #            Integrate G1 Kroupa             #
    ##############################################
    def __integrate_g1_kroupa(self, mmin, mmax):

        '''
        This function returns the integration of the Kroupa (2001)
        IMF. Number of stars.

        Arguments
        =========

          mmin : Lower-boundary mass of the integration
          mmax : Upper-boundary mass

        '''

        # Declare the integral result
        integral_sum = 0.0

        # Integrate the lower-mass regime if needed
        # 1.42857 = 1.0 / 0.7
        if mmin < 0.08:
            integral_sum += self.p0 * 1.42857 * \
                            ( min(mmax,0.08)**0.7 - mmin**0.7 )

        # Integrate the intermediate-mass regime if needed
        if mmax > 0.08 and mmin < 0.5:
        # 3.33333 = 1.0 / 0.3
            integral_sum += self.p1 * 3.33333 * \
                            ( max(mmin,0.08)**(-0.3) - min(mmax,0.5)**(-0.3) )

        # Integrate the high-mass regime if needed
        if mmax > 0.5:
        # 0.769231 = 1.0 / 1.3
            integral_sum += self.p1*self.p2 * 0.769231 * \
                            ( max(mmin,0.5)**(-1.3) - mmax**(-1.3) )

        # Return the integral of all mass regime combined
        return integral_sum


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
                    print ('Z > Zgrid')
                break

            # If Z is below the grid range, use min available Z
            if Z <= Z_grid[-1]:
                Z_gridpoint = Z_grid[-1]
                if self.iolevel >= 2:
                    print ('Z < Zgrid')
                break

            # If Z is exactly one of the available Z, use the given Z
            # round here to precision given in yield table
            if round(Z,abs(decimal.Decimal(str(tz)).as_tuple().exponent)) == tz: #Z == tz:
                Z_gridpoint = tz#Z
                if self.iolevel >= 2:
                    print ('Z = Zgrid')
                break

            # If Z is above the grid point at index tz, use this last point
            if Z > tz:
                Z_gridpoint = tz
                if self.iolevel >= 2:
                    print ('interpolation necessary')
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
        Note that tabulated net yields are not required for this approach.

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
            #print ('test Z: ',Z_gridpoint,' M: ',m)
            mfinal = ytables.get(Z=Z_gridpoint, M=m, quantity='Mfinal')
            iso_name=ytables.get(Z=Z_gridpoint, M=m, quantity='Isotopes')

            yi_all=[]
            # Correct every isotope and make sure the ejecta is always positive
            for p in range(len(X_ymgal_t)):
                #assume your yields are net yields
                if (self.netyields_on==True):
                    if self.wiersmamod: #for Wiesma09 tests
                            # initial amount depending on the simulation Z + net production factors
                            if (m>8) and (iso_name[p] in ['C-12','Mg-24','Fe-56']):
                                yi = (X_ymgal_t[p]*(m-mfinal) + y[p]) #total yields, Eq. 4 in Wiersma09
                                if iso_name[p] in ['C-12','Fe-56']:
                                        #print ('M=',m,' Reduce ',iso_name[p],' by 0.5 ',yi,yi*0.5)
                                        yi = yi*0.5
                                else:
                                        #print ('M=',m,' Multiply ',iso_name[p],' by 2.')
                                        yi = yi*2.
                            else:
                                yi = (X_ymgal_t[p]*(m-mfinal) + y[p])
                    else:
                        yi = (X_ymgal_t[p]*(m-mfinal) + y[p])
                    #print (yi,(m-mfinal),y[p],X_ymgal_t[p])
                else:
                    #assume your yields are NOT net yields
                    #if iso_name[p] in ['C-12']:
                        #print  ('C12: Current gas fraction and X0: ',X_ymgal_t[p],X0[p])
                        #introduce relative correction check of term X_ymgal_t[p] - X0[p]
                        #since small difference (e.g. due to lack of precision in X0) can
                        #lead to big differences in yi; yield table X0 has only limited digits
                        relat_corr=abs(X_ymgal_t[p] - X0[p])/X_ymgal_t[p]
                        if (relat_corr - 1.)>1e-3:
                                yi = y[p] + ( X_ymgal_t[p] - X0[p]) * (m-mfinal) #sum(y) #total yields yi, Eq. 7 in Wiersma09
                        else:
                                yi = y[p]
                if yi < 0:
                    if self.iolevel>0:
                        if abs(yi/y[p])>0.1:
                                print (iso_name[p],'star ',m,' set ',yi,' to 0, ', \
                                'netyields: ',y[p],'Xsim: ',X_ymgal_t[p],X0[p])
                    yi = 0
                yi_all.append(yi)

            # we do not do the normalization
            #norm = (m-mfinal)/sum(yi_all)
            yi_all= np.array(yi_all) #* norm
            yields.append(yi_all)
            # save calculated net yields and corresponding masses
            self.history.netyields=yields
            self.history.netyields_masses=m_stars

            #print ('star ',m,(m-mfinal),sum(yields[-1]))
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

        import matplotlib.pyplot as plt
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
                    print ('check current ymgal[i] ISM mass')
                    print ('ymgal[i][k]<0',self.ymgal[i][k],self.history.isotopes[k])
                    error = 1
                if error == 1:
                    sys.exit('ERROR: zmetal<0 in getmetal routine')

        # Return the metallicity of the gas reservoir
        return zmetal

    ##############################################
    #               Iso Abu to Elem              #
    ##############################################
    def _iso_abu_to_elem(self, yields_iso):

        '''
        This function converts isotope yields in elements and returns the result.

        '''

        # Combine isotopes into elements
        yields_ele = np.zeros(self.nb_elements)
        for i_iso in range(self.nb_isotopes):
            yields_ele[self.i_elem_for_iso[i_iso]] += yields_iso[i_iso]

        # Return the list of elements, and the associated yields
        return yields_ele


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
    #                Run All SSPs                #
    ##############################################
    def __run_all_ssps(self):

        '''
        Create a SSP with SYGMA for each metallicity available in the yield tables.
        Each SSP has a total mass of 1 Msun, is it can easily be re-normalized.

        '''

        # Copy the metallicities and put them in increasing order
        self.Z_table_SSP = copy.deepcopy(self.ytables.metallicities)
        self.Z_table_first_nzero = min(self.Z_table_SSP)
        if self.popIII_info_fast and self.iniZ <= 0.0 and self.Z_trans > 0.0:
            self.Z_table_SSP.append(0.0)
        self.Z_table_SSP = sorted(self.Z_table_SSP)
        self.nb_Z_table_SSP = len(self.Z_table_SSP)

        # If the SSPs are not given as an input ..
        if len(self.SSPs_in) == 0:

          import sygma

          # Define the SSP timesteps
          len_dt_SSPs = len(self.dt_in_SSPs)
          if len_dt_SSPs == 0:
              dt_in_ras = self.history.timesteps
              len_dt_SSPs = self.nb_timesteps
          else:
              dt_in_ras = self.dt_in_SSPs

          # Declare the SSP ejecta arrays [Z][dt][iso]
          self.ej_SSP = np.zeros((self.nb_Z_table_SSP,len_dt_SSPs,self.nb_isotopes))
          if self.len_decay_file > 0:
              self.ej_SSP_radio = \
                  np.zeros((self.nb_Z_table_SSP,len_dt_SSPs,self.nb_radio_iso))

          # For each metallicity ...
          for i_ras in range(0,self.nb_Z_table_SSP):

              # Use a dummy iniabu file if the metallicity is not zero
              if self.Z_table_SSP[i_ras] == 0:
                  iniabu_t = ''
                  hardsetZ2 = self.hardsetZ
              else:
                  iniabu_t='yield_tables/iniabu/iniab2.0E-02GN93.ppn'
                  hardsetZ2 = self.Z_table_SSP[i_ras]

              # Run a SYGMA simulation (1 Msun SSP)
              sygma_inst = sygma.sygma(pre_calculate_SSPs=False, \
                 imf_type=self.imf_type, alphaimf=self.alphaimf, \
                 imf_bdys=self.history.imf_bdys, sn1a_rate=self.history.sn1a_rate, \
                 iniZ=self.Z_table_SSP[i_ras], dt=self.history.dt, \
                 special_timesteps=self.special_timesteps, \
                 nsmerger_bdys=self.nsmerger_bdys, tend=self.history.tend, \
                 mgal=1.0, transitionmass=self.transitionmass, \
                 table=self.table, hardsetZ=hardsetZ2, \
                 sn1a_on=self.sn1a_on, sn1a_table=self.sn1a_table, \
                 sn1a_energy=self.sn1a_energy, ns_merger_on=self.ns_merger_on, \
                 bhns_merger_on=self.bhns_merger_on, f_binary=self.f_binary, \
                 f_merger=self.f_merger, t_merger_max=self.t_merger_max, \
                 m_ej_nsm=self.m_ej_nsm, nsm_dtd_power=self.nsm_dtd_power,\
                 m_ej_bhnsm=self.m_ej_bhnsm, bhnsmerger_table=self.bhnsmerger_table, \
                 nsmerger_table=self.nsmerger_table, iniabu_table=iniabu_t, \
                 extra_source_on=self.extra_source_on, nb_nsm_per_m=self.nb_nsm_per_m, \
                 t_nsm_coal=self.t_nsm_coal, extra_source_table=self.extra_source_table, \
                 f_extra_source=self.f_extra_source, \
                 extra_source_mass_range=self.extra_source_mass_range, \
                 extra_source_exclude_Z=self.extra_source_exclude_Z, \
                 pop3_table=self.pop3_table, imf_bdys_pop3=self.imf_bdys_pop3, \
                 imf_yields_range_pop3=self.imf_yields_range_pop3, \
                 starbursts=self.starbursts, beta_pow=self.beta_pow, \
                 gauss_dtd=self.gauss_dtd, exp_dtd=self.exp_dtd, \
                 nb_1a_per_m=self.nb_1a_per_m, direct_norm_1a=self.direct_norm_1a, \
                 imf_yields_range=self.imf_yields_range, \
                 exclude_masses=self.exclude_masses, netyields_on=self.netyields_on, \
                 wiersmamod=self.wiersmamod, yield_interp=self.yield_interp, \
                 stellar_param_on=self.stellar_param_on, \
                 t_dtd_poly_split=self.t_dtd_poly_split, \
                 stellar_param_table=self.stellar_param_table, \
                 tau_ferrini=self.tau_ferrini, delayed_extra_log=self.delayed_extra_log, \
                 dt_in=dt_in_ras, nsmerger_dtd_array=self.nsmerger_dtd_array, \
                 bhnsmerger_dtd_array=self.bhnsmerger_dtd_array, \
                 poly_fit_dtd_5th=self.poly_fit_dtd_5th, \
                 poly_fit_range=self.poly_fit_range, \
                 delayed_extra_dtd=self.delayed_extra_dtd, \
                 delayed_extra_dtd_norm=self.delayed_extra_dtd_norm, \
                 delayed_extra_yields=self.delayed_extra_yields, \
                 delayed_extra_yields_norm=self.delayed_extra_yields_norm, \
                 table_radio=self.table_radio, decay_file=self.decay_file, \
                 sn1a_table_radio=self.sn1a_table_radio, \
                 bhnsmerger_table_radio=self.bhnsmerger_table_radio, \
                 nsmerger_table_radio=self.nsmerger_table_radio, \
                 delayed_extra_log_radio=self.delayed_extra_log_radio, \
                 delayed_extra_yields_log_int_radio=self.delayed_extra_yields_log_int_radio, \
                 ism_ini_radio=self.ism_ini_radio, \
                 delayed_extra_dtd_radio=self.delayed_extra_dtd_radio, \
                 delayed_extra_dtd_norm_radio=self.delayed_extra_dtd_norm_radio, \
                 delayed_extra_yields_radio=self.delayed_extra_yields_radio, \
                 delayed_extra_yields_norm_radio=self.delayed_extra_yields_norm_radio, \
                 ytables_radio_in=self.ytables_radio_in, radio_iso_in=self.radio_iso_in, \
                 ytables_1a_radio_in=self.ytables_1a_radio_in, \
                 ytables_nsmerger_radio_in=self.ytables_nsmerger_radio_in)

              # Copy the ejecta arrays from the SYGMA simulation
              self.ej_SSP[i_ras] = sygma_inst.mdot
              if self.len_decay_file > 0:
                  self.ej_SSP_radio[i_ras] = sygma_inst.mdot_radio

              # If this is the last Z entry ..
              if i_ras == self.nb_Z_table_SSP - 1:

                  # Keep in memory the number of timesteps in SYGMA
                  self.nb_steps_table = len(sygma_inst.history.timesteps)
                  self.dt_ssp = sygma_inst.history.timesteps

                  # Keep the time of ssp in memory
                  self.t_ssp = np.zeros(self.nb_steps_table)
                  self.t_ssp[0] = self.dt_ssp[0]
                  for i_ras in range(1,self.nb_steps_table):
                      self.t_ssp[i_ras] = self.t_ssp[i_ras-1] + self.dt_ssp[i_ras]

              # Clear the memory
              del sygma_inst

          # Calculate the interpolation coefficients (between metallicities)
          self.__calculate_int_coef()

        # If the SSPs are given as an input ..
        else:

            # Copy the SSPs
            self.ej_SSP = self.SSPs_in[0]
            self.nb_steps_table = len(self.ej_SSP[0])
            self.ej_SSP_coef = self.SSPs_in[1]
            self.dt_ssp = self.SSPs_in[2]
            self.t_ssp = self.SSPs_in[3]
            if len(self.SSPs_in) > 4:
                self.ej_SSP_radio = self.SSPs_in[4]
                self.ej_SSP_coef_radio = self.SSPs_in[5]
                self.decay_info = self.SSPs_in[6]
                self.len_decay_file = self.SSPs_in[7]
                self.nb_radio_iso = len(self.decay_info)
                self.nb_new_radio_iso = len(self.decay_info)
            del self.SSPs_in


    ##############################################
    #            Calculate Int. Coef.            #
    ##############################################
    def __calculate_int_coef(self):

        '''
        Calculate the interpolation coefficients of each isotope between the
        different metallicities for every timestep considered in the SYGMA
        simulations.  ejecta = a * log(Z) + b

        self.ej_x[Z][step][isotope][0 -> a, 1 -> b], where the Z index refers
        to the lower metallicity boundary of the interpolation.

        '''

        # Declare the interpolation coefficients arrays
        self.ej_SSP_coef = \
            np.zeros((2,self.nb_Z_table_SSP,self.nb_steps_table,self.nb_isotopes))
        if self.len_decay_file > 0:
            self.ej_SSP_coef_radio = \
                np.zeros((2,self.nb_Z_table_SSP,self.nb_steps_table,self.nb_radio_iso))

        # For each metallicity interval ...
        for i_cic in range(0,self.nb_Z_table_SSP-1):

          # If the metallicity is not zero ...
          if not self.Z_table_SSP[i_cic] == 0.0:

            # Calculate the log(Z) for the boundary metallicities
            logZ_low = np.log10(self.Z_table_SSP[i_cic])
            logZ_up = np.log10(self.Z_table_SSP[i_cic+1])
            dif_logZ_inv = 1.0 / (logZ_up - logZ_low)

            # For each step ...
            for j_cic in range(0,self.nb_steps_table):

              # For every stable isotope ..
              for k_cic in range(0,self.nb_isotopes):

                # Copy the isotope mass for the boundary metallicities
                iso_low = self.ej_SSP[i_cic][j_cic][k_cic]
                iso_up = self.ej_SSP[i_cic+1][j_cic][k_cic]
               
                # Calculate the "a" and "b" coefficients
                self.ej_SSP_coef[0][i_cic][j_cic][k_cic] = \
                    (iso_up - iso_low) * dif_logZ_inv
                self.ej_SSP_coef[1][i_cic][j_cic][k_cic] = iso_up - \
                    self.ej_SSP_coef[0][i_cic][j_cic][k_cic] * logZ_up

              # For every radioactive isotope ..
              if self.len_decay_file > 0:
                for k_cic in range(0,self.nb_radio_iso):

                    # Copy the isotope mass for the boundary metallicities
                    iso_low = self.ej_SSP_radio[i_cic][j_cic][k_cic]
                    iso_up = self.ej_SSP_radio[i_cic+1][j_cic][k_cic]
               
                    # Calculate the "a" and "b" coefficients
                    self.ej_SSP_coef_radio[0][i_cic][j_cic][k_cic] = \
                        (iso_up - iso_low) * dif_logZ_inv
                    self.ej_SSP_coef_radio[1][i_cic][j_cic][k_cic] = iso_up - \
                        self.ej_SSP_coef_radio[0][i_cic][j_cic][k_cic] * logZ_up


    ##############################################
    #               Add SSP Ejecta               #
    ##############################################
    def __add_ssp_ejecta(self, i):

        '''
        Distribute the SSP ejecta.  The SSP that forms during this step
        is treated by interpolating SYGMA results that were kept in memory.
        The SSp still deposit its ejecta in the upcoming timesteps, as in the
        original chem_evol.py class.

        Argument
        ========

          i : Index of the current timestep

        '''

        # Interpolate the SSP ejecta
        self.__interpolate_ssp_ej( i-1 )

        # Copy the initial simulation step that will be increased
        i_sim = i-1
        if i_sim == 0:
            t_form = 0.0
        else:
            t_form = self.t_ce[i_sim-1]

        # For each SSP step ...
        for i_ssp in range(0,self.nb_steps_table):

            # Declare the array that contains the time covered by
            # the SSP step on each simulation step
            time_frac = []

            # Keep the initial current simulation step in memory
            i_sim_low = i_sim

            # While all simulation steps covered by the SSP step
            # have not been treated ...
            not_complete = True
            while not_complete:

                # Calculate the time lower-boundary of the SSP time bin
                if i_ssp == 0:
                    t_low_ssp = 0.0
                else:
                    t_low_ssp = self.t_ssp[i_ssp-1]
                if i_sim == 0:
                    t_low_ce = 0.0
                else:
                    t_low_ce = self.t_ce[i_sim-1]

                # Calculate the time covered by the SSP step on
                # the considered simulation step
                time_frac.append( \
                  min((self.t_ce[i_sim]-t_form), self.t_ssp[i_ssp]) - \
                  max((t_low_ce-t_form), t_low_ssp))

                # If all the simulations steps has been covered ...
                if (self.t_ce[i_sim]-t_form) >= self.t_ssp[i_ssp] or \
                  (i_sim + 1) == self.nb_timesteps:

                    # Stop the while loop
                    not_complete = False

                # If we still need to cover simulation steps ...
                else:

                    # Move to the next one
                    i_sim += 1

            # Convert the time into time fraction
            dt_temp_inv = 1.0 / (self.t_ssp[i_ssp] - t_low_ssp)
            for i_tf in range(0,len(time_frac)):
                time_frac[i_tf] = time_frac[i_tf] * dt_temp_inv

            # For each simulation step ...
            for j_ase in range(0,len(time_frac)):

                # Add the ejecta
                self.mdot[i_sim_low+j_ase] += \
                    self.ej_SSP_int[i_ssp] * time_frac[j_ase]
                if self.len_decay_file > 0:
                    self.mdot_radio[i_sim_low+j_ase] += \
                        self.ej_SSP_int_radio[i_ssp] * time_frac[j_ase]

            # Break is the end of the simulation is reached
            if (i_sim + 1) == self.nb_timesteps:
                break
            

    ##############################################
    #             Interpolate SSP Ej.            #
    ##############################################
    def __interpolate_ssp_ej(self, i):

        '''
        Interpolate all the isotope for each step to create a SSP with the
        wanted metallicity.  The ejecta is scale to the mass of the SSP.

        Arguments
        =========

          i : Index of the current timestep

        '''

        # Use the lowest metallicity
        if self.zmetal <= self.Z_trans:
            self.ej_SSP_int = self.ej_SSP[0] * self.m_locked
            if self.len_decay_file > 0:
                self.ej_SSP_int_radio = self.ej_SSP_radio[0] * self.m_locked

        # Use the highest metallicity
        elif self.zmetal >= self.Z_table_SSP[-1]:
            self.ej_SSP_int = self.ej_SSP[-1] * self.m_locked
            if self.len_decay_file > 0:
                self.ej_SSP_int_radio = self.ej_SSP_radio[-1] * self.m_locked

        # If the metallicity is between Z_trans and lowest non-zero Z_table ..
        elif self.zmetal <= self.Z_table_first_nzero:
            if self.Z_table_SSP[0] == self.Z_table_first_nzero:
                self.ej_SSP_int = self.ej_SSP[0] * self.m_locked
            else:
                self.ej_SSP_int = self.ej_SSP[1] * self.m_locked
            if self.len_decay_file > 0:
                if self.Z_table_SSP[0] == self.Z_table_first_nzero:
                    self.ej_SSP_int_radio = self.ej_SSP_radio[0] * self.m_locked
                else:
                    self.ej_SSP_int_radio = self.ej_SSP_radio[1] * self.m_locked

        # If we need to interpolate the ejecta ...
        else:

            # Find the metallicity lower boundary
            i_Z_low = 0
            while self.zmetal >= self.Z_table_SSP[i_Z_low+1]:
                i_Z_low += 1

            # Calculate the log of the current gas metallicity
            log_Z_cur = np.log10(self.zmetal)

            # Calculate the time left to the simulation
            t_left_ce = self.t_ce[-1] - self.t_ce[i]

            # For each step and each isotope ...
            for j_ise in range(0,self.nb_steps_table):

                # Interpolate the isotopes
                self.ej_SSP_int[j_ise] = (self.ej_SSP_coef[0][i_Z_low][j_ise] * \
                    log_Z_cur + self.ej_SSP_coef[1][i_Z_low][j_ise]) * self.m_locked
                if self.len_decay_file > 0:
                    self.ej_SSP_int_radio[j_ise] = (\
                        self.ej_SSP_coef_radio[0][i_Z_low][j_ise] * log_Z_cur + \
                            self.ej_SSP_coef_radio[1][i_Z_low][j_ise]) * self.m_locked

                # Break if the SSP time exceed the simulation time
                if self.t_ssp[j_ise] > t_left_ce:
                  break


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
            self.ism_iso_yield_nsm = []
            self.ism_iso_yield_bhnsm = []
            self.isotopes = []
            self.elements = []
            self.ism_elem_yield = []
            self.ism_elem_yield_agb = []
            self.ism_elem_yield_massive = []
            self.ism_elem_yield_1a = []
            self.ism_elem_yield_nsm = []
            self.ism_elem_yield_bhnsm = []
            self.sn1a_numbers = []
            self.nsm_numbers = []
            self.bhnsm_numbers = []
            self.sn2_numbers = []
            self.t_m_bdys = []


    class __const():

        '''
        Holds the physical constants.
        Please add further constants if required.

        '''

        #############################
        #        Constructor        #
        #############################
        def __init__(self):

            '''
            Initiate variables tracking.history

            '''


            self.syr = 31536000 #seconds in a year
            self.c= 2.99792458e10 #speed of light in vacuum (cm s^-1)
            self.pi = 3.1415926535897932384626433832795029e0
            self.planck_h  = 6.62606896e-27 # Planck's constant (erg s)
            self.ev2erg = 1.602176487e-12 # electron volt (erg)
            self.rsol = 6.9598e10 # solar radius (cm)
            self.lsol = 3.8418e33 #erg/s
            self.msol =  1.9892e33  # solar mass (g)
            self.ggrav = 6.67428e-8 #(g^-1 cm^3 s^-2)


    ##############################################
    #               BinTree CLASS                #
    ##############################################
    class _bin_tree():

        '''
        Class for the construction and search in a binary tree.

        '''

        #############################
        #        Constructor        #
        #############################
        def __init__(self, sorted_array):

            '''
            Initialize the balanced tree

            '''

            self.head = self._create_tree(sorted_array)

        #############################
        #        Tree creation      #
        #############################
        def _create_tree(self, sorted_array, index = 0):
            '''
            Create the tree itself

            '''

            # Sort edge cases
            len_array = len(sorted_array)
            if len_array == 0:
                return None
            elif len_array == 1:
                return self._node(sorted_array[0], index)

            # Find middle value and index, introduce them
            # and recursively create the children
            mid_index = len_array//2
            mid_value = sorted_array[mid_index]
            new_node = self._node(mid_value, mid_index + index)
            new_node.lchild = self._create_tree(sorted_array[0:mid_index], index)
            new_node.rchild = self._create_tree(sorted_array[mid_index + 1:],
                    mid_index + 1 + index)

            return new_node

        #############################
        #  Wraper Tree search left  #
        #############################
        def search_left(self, value):
            '''
            Wrapper for search_left
            Search for the rightmost index lower or equal than value

            '''

            # Call function and be careful with lowest case
            index = self._search_left_rec(value, self.head)
            if index is None:
                return 0

            return index

        #############################
        #        Tree search left   #
        #############################
        def _search_left_rec(self, value, node):
            '''
            Search for the rightmost index lower or equal than value

            '''

            # Sort edge case
            if node is None:
                return None

            # If to the left, we can always return the index, even if None.
            # If to the right and none, we return current index, as it will
            # be the closest to value from the left
            if value < node.value:
                return self._search_left_rec(value, node.lchild)
            else:
                index = self._search_left_rec(value, node.rchild)
                if index is None:
                    return node.index

                return index


        ##############################################
        #               Node CLASS                   #
        ##############################################
        class _node():

            '''
            Class for the bin_tree nodes.

            '''

            #############################
            #        Constructor        #
            #############################
            def __init__(self, value, index):

                '''
                Initialize the constructor

                '''

                self.value = value
                self.index = index
                self.lchild = None
                self.rchild = None
