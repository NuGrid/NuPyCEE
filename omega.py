# coding=utf-8
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

'''

GCE OMEGA (One-zone Model for the Evolution of Galaxies) module


Functionality
=============

This tool allows one to simulate the chemical evolution of single-zone galaxies.
Having the star formation history as one of the input parameters, OMEGA can
target local galaxies by using observational data found in the literature.


Made by
=======

FEB2015: C. Ritter, B. Cote

MAY2015: B. Cote

The code inherits the chem_evol class, which contains common functions shared by
SYGMA and OMEGA.  The code in chem_evol has been developed by :

v0.1 NOV2013: C. Fryer, C. Ritter

v0.2 JAN2014: C. Ritter

v0.3 APR2014: C. Ritter, J. F. Navarro, F. Herwig, C. Fryer, E. Starkenburg,
              M. Pignatari, S. Jones, K. Venn1, P. A. Denissenkov &
              the NuGrid collaboration

v0.4 FEB2015: C. Ritter, B. Cote

v0.5 MAR2015: B. Cote

v0.6 OCT2016: B. Cote

Stop keeking track of version from now on.

MARCH2018: B. Cote
- Switched to Python 3
- Capability to include radioactive isotopes

FEB2019: A. Yagüe, B. Cote
- Optimized to code to run faster


Note
====
Please do not use "tabs" when introducing new lines of code.


Usage
=====

Import the module:

>>> import omega as o

Get help:

>>> help o

Get more information:

>>> o.omega?

Create a custom galaxy (closed box):

>>> o1 = o.omega(cte_sfr=1.0, mgal=1.5e10)

Simulate a known galaxy (open box):

>>> o2 = o.omega(galaxy='sculptor', in_out_control=True, mgal=1e6, mass_loading=8, in_out_ratio=1.5)

Analysis functions: See the Sphinx documentation

'''

# Standard packages
import copy
import math
import random
import os

# Define where is the working directory
# This is where the NuPyCEE code will be extracted
nupy_path = os.path.dirname(os.path.realpath(__file__))

# Import NuPyCEE codes
import NuPyCEE.sygma as sygma
from NuPyCEE.chem_evol import *

class omega( chem_evol ):

    '''
    Input parameters (OMEGA)
    ================

    Important : By default, a closed box model is always assumed.

    galaxy : string
        Name of the target galaxy.  By using a known galaxy, the code
        automatically selects the corresponding star formation history, stellar
        mass, and total mass (when available).  By using 'none', the user has
        perfect control of these three last parameters.

        Choices : 'milky_way', 'milky_way_cte', 'sculptor', 'carina', 'fornax',
        'none'

        Default value : 'none'

        Special note : The 'milky_way_cte' option uses the Milky Way's
        characteristics, but with a constant star formation history.

    cte_sfr : float
        Constant star formation history in [Mo/yr].

        Default value : 1.0

    rand_sfh : float
        Maximum possible ratio between the maximum and the minimum values of a star
        formation history that is randomly generated.

        Default value : 0.0 (deactivated)

        Special note : A value greater than zero automatically generates a random
        star formation history, which pypasses the use of the cte_sfr parameter.

    sfh_file : string
        Path to a file containing an input star formation history.  The first and
        second columns must be the age of the galaxy in [yr] and the star
        formation rate in [Mo/yr].

        Default value : 'none' (deactivated)

        Special note : When a path is specified, it by passes the cte_sfr and the
        and_sfh parameters.

    stellar_mass_0 : float
        Current stellar mass of the galaxy, in [Mo], at the end of the simulation.

        Default value : -1.0 (you need to specify a value with unknown galaxies)

    in_out_control : boolean
        The in_out_control implementation enables control of the outflow and
        the inflow rates independently by using constant values (see outflow_rate
        and inflow_rate) or by using a mass-loading factor that connects the
        rates to the star formation history (see mass_loading and in_out_ratio).

        Default value : False (deactivated)

    mass_loading : float
        Ratio between the outflow rate and the star formation rate.

        Default value : 1.0

    outflow_rate : float
        Constant outflow rate in [Mo/yr].

        Default value : -1.0 (deactivated)

        Special note : A value greater or equal to zero activates the constant
        utflow mode, which bypasses the use of the mass_loading parameter.

    in_out_ratio : float
        Used in : in_out_control mode

        Ratio between the inflow rate and the outflow rate.  This parameter is
        used to calculate the inflow rate, not the outflow rate.

        Default value : 1.0

    inflow_rate : float
        Used in : in_out_control mode
        Constant inflow rate in [Mo/yr].

        Default value : -1.0 (deactivated)

        Special note : A value greater or equal to zero activates the constant
        inflow mode, which bypasses the use of the in_out_ratio parameter.

    SF_law : boolean
        The SF_law inplementation assumes a Kennicutt-Schmidt star formation law
        and combines it to the known input star formation history in order to
        derive the mass of the gas reservoir at every timestep.

        Default value : False (deactivated)

    sfe : float
        Used in : SF_law and DM_evolution modes

        Star formation efficiency present in the Kennicutt-Schmidt law.

        Default value : 0.1

    f_dyn : float
        Used in : SF_law and DM_evolution modes
        Scaling factor used to calculate the star formation timescale present in
        the Kennicutt-Schmidt law.  We assume that this timescale is equal to a
        fraction of the dynamical timescale of the virialized system (dark and
        baryonic matter), t_star = f_dyn * t_dyn.

        Default value : 0.1

    m_DM_0 : float
        Used in : SF_law and DM_evolution modes
        Current dark matter halo mass of the galaxy, in [Mo], at the end of the
        simulations.

        Default value : 1.0e+11

    t_star : float
        Used in : SF_law and DM_evolution modes
        Star formation timescale, in [yr], used in the Kennicutt-Schmidt law.
        Default value = -1.0 (deactivated)

        Special note : A positive value activates the use of this parameter,
        which bypasses the f_dyn parameter.

    DM_evolution : boolean
        The DM_evolution implementation is an extension of the SF_law option.
        In addition to using a Kennicutt-Schmidt star formation law, it assumes
        an evolution in the total mass of the galaxy as function of time.  With
        this prescription, the mass-loading factor has a mass dependency.  The
        mass_loading parameter then only represents the final value at the end
        of the simulation.

        Default value : False (deactivated)

    exp_ml : float
        Used in : DM_evolution mode

        Exponent of the mass dependency of the mass-loading factor.  This last
        factor is proportional to M_vir**(-exp_ml/3), where M_vir is the sum of
        dark and baryonic matter.

        Default value : 2.0

    ================
    '''

    #Combine docstrings from chem_evol with sygma docstring
    __doc__ = __doc__+chem_evol.__doc__


    ##############################################
    ##               Constructor                ##
    ##############################################
    def __init__(self, galaxy='none', in_out_control=False, SF_law=False, \
                 DM_evolution=False, f_dyn=0.1, sfe=0.01, outflow_rate=-1.0, \
                 inflow_rate=-1.0, rand_sfh=0.0, cte_sfr=1.0, m_DM_0=1.0e12, \
                 omega_0=0.32, omega_b_0=0.05, lambda_0=0.68, H_0=67.11, \
                 mass_loading=1.0, t_star=-1.0, sfh_file='none', in_out_ratio=1.0, \
                 stellar_mass_0=-1.0, z_dependent=True, exp_ml=2.0, beta_crit=1.0, \
                 skip_zero=False, redshift_f=0.0, long_range_ref=False,\
                 f_s_enhance=1.0, m_gas_f=-1.0, cl_SF_law=False, \
                 external_control=False, calc_SSP_ej=False, t_sf_z_dep = 1.0, \
                 m_crit_on=False, norm_crit_m=8.0e+09, mass_frac_SSP=0.5, \
                 sfh_array_norm=-1.0, imf_rnd_sampling=False, r_gas_star=-1.0, \
                 cte_m_gas = -1.0, DM_array=[], sfh_array=[], \
                 m_inflow_array=[], m_gas_array=[], mdot_ini=[], mdot_ini_t=[], \
                 r_vir_array=[], mass_sampled=[], scale_cor=[], \
                 mass_sampled_ssp=[], m_tot_ISM_t_in=[], \
                 m_inflow_X_array=[], dt_in_SSPs=[], **kwargs):

        # Get the name of the instance
        import traceback
        (filename,line_number,function_name,text)=traceback.extract_stack()[-2]
        self.inst_name = text[:text.find('=')].strip()

        # Overwrite default chem_evol parameters (if needed)
        if not "iniZ" in kwargs:
            kwargs["iniZ"] = 0.0
        if not "mgal" in kwargs:
            kwargs["mgal"] = 8.0e10

        # Call the init function of the class inherited by SYGMA
        chem_evol.__init__(self, **kwargs)

        # Quit if something bad happened in chem_evol ..
        if self.need_to_quit:
            return

        # Announce the beginning of the simulation
        if not self.print_off:
            print ('OMEGA run in progress..')
        start_time = t_module.time()
        self.start_time = start_time

        # Attributes for chem_evol
        self.kwargs = kwargs

        # Attribute the input parameters to the current OMEGA object
        self.galaxy = galaxy
        self.in_out_control = in_out_control
        self.SF_law = SF_law
        self.DM_evolution = DM_evolution
        self.f_dyn = f_dyn
        self.sfe = sfe
        self.outflow_rate = outflow_rate
        self.inflow_rate = inflow_rate
        self.rand_sfh = rand_sfh
        self.cte_sfr = cte_sfr
        self.m_DM_0 = m_DM_0
        self.mass_loading = mass_loading
        self.t_star = t_star
        self.sfh_file = sfh_file
        self.in_out_ratio = in_out_ratio
        self.stellar_mass_0 = stellar_mass_0
        self.z_dependent = z_dependent
        self.exp_ml = exp_ml
        self.DM_too_low = False
        self.skip_zero = skip_zero
        self.redshift_f = redshift_f
        self.long_range_ref = long_range_ref
        self.m_crit_on = m_crit_on
        self.norm_crit_m = norm_crit_m
        self.sfh_array_norm = sfh_array_norm
        self.DM_array = DM_array
        self.sfh_array = sfh_array
        self.mdot_ini = mdot_ini
        self.mdot_ini_t = mdot_ini_t
        self.r_gas_star = r_gas_star
        self.m_gas_f = m_gas_f
        self.cl_SF_law = cl_SF_law
        self.external_control = external_control
        self.mass_sampled = mass_sampled
        self.scale_cor = scale_cor
        self.imf_rnd_sampling = imf_rnd_sampling
        self.cte_m_gas = cte_m_gas
        self.t_sf_z_dep = t_sf_z_dep
        self.m_tot_ISM_t_in = m_tot_ISM_t_in
        self.m_inflow_array = m_inflow_array
        self.len_m_inflow_array = len(m_inflow_array)
        self.m_inflow_X_array = m_inflow_X_array
        self.len_m_inflow_X_array = len(m_inflow_X_array)
        self.m_gas_array = m_gas_array
        self.len_m_gas_array = len(m_gas_array)
        self.beta_crit = beta_crit
        self.r_vir_array = r_vir_array
        self.calc_SSP_ej = calc_SSP_ej
        self.mass_frac_SSP = -1.0
        self.mass_frac_SSP_in = mass_frac_SSP
        self.dt_in_SSPs = dt_in_SSPs

        # Set cosmological parameters - default is Planck 2013 (used in Caterpillar)
        self.omega_0   = omega_0   # Current mass density parameter
        self.omega_b_0 = omega_b_0 # Current baryonic mass density parameter
        self.lambda_0  = lambda_0  # Current dark energy density parameter
        self.H_0       = H_0       # Hubble constant [km s^-1 Mpc^-1]

        # Look for errors in the input parameters
        if self.__check_inputs_omega():
            return

        # Calculate the number of CC SNe per Msun formed (if needed)
        if self.out_follows_E_rate:
            self.__calc_ccsne_per_m()

        # Pre-calculate SSPs (if needed)
        if self.pre_calculate_SSPs:
            self.__run_SSPs()

        # Calculate random IFM sampling parameters (if needed)
        if self.imf_rnd_sampling:
            self.__calc_imf_rnd_param()

        # Define whether the open box scenario is used or not
        if self.in_out_control or self.SF_law or self.DM_evolution:
            self.open_box = True
        else:
            self.open_box = False

        # Refine timesteps (if needed)
        if self.SF_law or self.DM_evolution:
            self.__refine_timesteps()

        # Declare arrays used to follow the evolution of the galaxy
        self.__declare_evol_arrays()

        # Calculate the average mass fraction ejected by SSPs
        # !! This function needd to be before self.__initialize_gal_prop() !!
        self.__calc_mass_frac_SSP()

        # Set the general properties of the selected galaxy
        self.__initialize_gal_prop()

        # Fill arrays used to follow the evolution
        self.__fill_evol_arrays()

        # Read the primordial composition of the inflow gas
        if self.in_out_control or self.SF_law or self.DM_evolution:
            prim_comp_table = os.path.join('yield_tables', 'iniabu',\
                    'iniab_bb_walker91.txt')
            self.prim_comp = ry.read_yields_Z(os.path.join(nupy_path,\
                    prim_comp_table), isotopes=self.history.isotopes)

        # Add the stellar ejecta coming from external galaxies that just merged
        if len(self.mdot_ini) > 0:
            self.__add_ext_mdot()

        # Initialisation of the composition of the gas reservoir
        if len(self.ism_ini) > 0:
            for i_ini in range(0,self.len_ymgal):
                self.ymgal[0][i_ini] = self.ism_ini[i_ini]

        # Copy the outflow-vs-SFR array and re-initialize for delayed outflow
        if self.out_follows_E_rate:
            self.outflow_test = np.sum(self.m_outflow_t)
            self.m_outflow_t_vs_SFR = copy.copy(self.m_outflow_t)
            for i_ofer in range(0,self.nb_timesteps):
                self.m_outflow_t[i_ofer] = 0.0

        # Run the simulation if not controled by an external code
        if not self.external_control:
            self.__run_simulation(self.mass_sampled, self.scale_cor)


    ##############################################
    #             Check Inputs OMEGA             #
    ##############################################
    def __check_inputs_omega(self):

        '''
        This function checks for incompatible input entries, and stops
        the simulation if needed.

        '''

        # Initialize whether or not the code should abord
        abord = False

        # Input galaxy
        if not self.galaxy in ['none', 'milky_way', 'milky_way_cte', \
                               'sculptor', 'fornax', 'carina']:
            print ('Error - Selected galaxy not available.')
            abord = True

        # Random SFH
        if self.rand_sfh > 0.0 and self.stellar_mass_0 < 0.0:
            print ('Error - You need to choose a current stellar mass.')
            abord = True

        # Inflow control when non-available
        if self.in_out_control and (self.SF_law or self.DM_evolution):
            print ('Error - Cannot control inflows and outflows when SF_law or'\
                  'DM_evolution is equal to True.')
            abord = True

        # Inflow and outflow control when the dark matter mass if evolving
        if (self.outflow_rate >= 0.0 or self.inflow_rate >= 0.0) and \
            self.DM_evolution:
            print ('Error - Cannot fix inflow and outflow rates when the mass'\
                  'of the dark matter halo is evolving.')
            abord = True

        # Inflow array when input
        if self.len_m_inflow_array > 0:
            if not self.len_m_inflow_array == self.nb_timesteps:
                print ('Error - len(m_inflow_array) needs to equal nb_timesteps.')
                abord = True

        # Inflow X array when input
        if self.len_m_inflow_X_array > 0:
            if not self.len_m_inflow_X_array == self.nb_timesteps:
                print ('Error - len(m_inflow_X_array) needs to equal nb_timesteps.')
                abord = True
            if not len(self.m_inflow_X_array[0]) == self.nb_isotopes:
                print ('Error - len(m_inflow_X_array[i]) needs to equal nb_isotopes.')
                abord = True

        # Mgas array when input
        if self.len_m_gas_array > 0:
            if not self.len_m_gas_array == (self.nb_timesteps+1):
                print ('Error - len(m_gas_array) needs to equal nb_timesteps+1.')
                abord = True

        # Return whether or not the code should abord
        return abord


    ##############################################
    #             Calc CCSNe per M               #
    ##############################################
    def __calc_ccsne_per_m(self):

        '''
        Calculate the number of core-collapse SNe that will occur
        in total per units of stellar mass formed.

        '''

        # IMF constant for a 1 Msun PopIII stellar population
        A_pop3 = 1.0 / self._imf(self.imf_bdys_pop3[0], self.imf_bdys_pop3[1],2)
        A = 1.0 / self._imf(self.imf_bdys[0], self.imf_bdys[1],2)

        # Number of CC SNe per stellar mass formed
        self.nb_ccsne_per_m_pop3 = \
            A_pop3 * self._imf(self.imf_yields_range_pop3[0], \
                self.imf_yields_range_pop3[1],1)
        self.nb_ccsne_per_m = \
            A * self._imf(self.transitionmass,\
                self.imf_yields_range[1],1)


    ##############################################
    #                 Run SSPs                   #
    ##############################################
    def __run_SSPs(self):

        '''
        Run all stellar populations at all metallicties.

        '''

        # Calculate all SSPs
        self.__run_all_ssps()

        # Declare the arrays that will contain the interpolated isotopes
        self.ej_SSP_int = np.zeros((self.nb_steps_table, self.nb_isotopes))
        if self.len_decay_file > 0 or self.use_decay_module:
            self.ej_SSP_int_radio = np.zeros((self.nb_steps_table, self.nb_radio_iso))


    ##############################################
    #            Calc IMF rnd Param              #
    ##############################################
    def __calc_imf_rnd_param(self):

        '''
        Calculate parameters that will be used when the IFM is randomly sampled.

        '''

        # Print info about the IMF sampling
        self.m_pop_max = 1.0e4
        print ('IMF random sampling for SSP with M < ',self.m_pop_max)

        # Calculate the stellar mass associated with the
        # highest IMF value (needed for Monte Carlo)
        # This only samples massive stars
        self.A_rdm = 1.0 / self.transitionmass**(-2.3)
        self.m_frac_massive_rdm = self.A_rdm * \
            self._imf(self.transitionmass, self.imf_bdys[1], 2)


    ##############################################
    #             Refine Timesteps               #
    ##############################################
    def __refine_timesteps(self):

        '''
        Refine the timesteps and increase the sizes of relevant arrays if needed.

        '''

        # Initialize evolution arrays that will determine the refinement needs
        self.t_SF_t = np.zeros(self.nb_timesteps+1)
        self.redshift_t = np.zeros(self.nb_timesteps+1)

        # Fill the evolution arrays        
        self.calculate_redshift_t()
        self.__calculate_t_SF_t()

        # Define whether refinement is needed
        need_t_raf = False
        for i_raf in range(self.nb_timesteps):
            if self.history.timesteps[i_raf] > self.t_SF_t[i_raf] / self.sfe:
                need_t_raf = True
                break

        # Refine timesteps (if needed)
        if need_t_raf:
            if self.long_range_ref:
                self.__rafine_steps_lr()
            else:
                self.__rafine_steps()

        # Re-Create entries for the mass-loss rate of massive stars
        self.massive_ej_rate = np.zeros(self.nb_timesteps+1)
        self.sn1a_ej_rate = np.zeros(self.nb_timesteps+1)


    ##############################################
    #                Refine Steps                #
    ##############################################
    def __rafine_steps(self):

        '''
        This function increases the number of timesteps if the star formation
        will eventually consume all the gas, which occurs when dt > (t_star/sfe).

        '''

        # Declaration of the new timestep array
        if not self.print_off:
            print ('..Time refinement..')
        new_dt = []

        # For every timestep ...
        for i_rs in range(0,len(self.history.timesteps)):

            # Calculate the critical time delay
            t_raf = self.t_SF_t[i_rs] / self.sfe

            # If the step needs to be refined ...
            if self.history.timesteps[i_rs] > t_raf:

                # Calculate the split factor
                nb_split = int(self.history.timesteps[i_rs] / t_raf) + 1

                # Split the step
                for i_sp_st in range(0,nb_split):
                    new_dt.append(self.history.timesteps[i_rs]/nb_split)

            # If ok, don't change anything
            else:
                new_dt.append(self.history.timesteps[i_rs])

        # Update the timestep information
        self.nb_timesteps = len(new_dt)
        self.history.timesteps = new_dt

        # Update self.history.age
        self.history.age = [0]
        for ii in range(self.nb_timesteps):
            self.history.age.append(self.history.age[-1] + new_dt[ii])
        self.history.age = np.array(self.history.age)

        # If a timestep needs to be added to be synchronized with
        # the external program managing merger trees ...
        if self.t_merge > 0.0:

            # Find the interval where the step needs to be added
            i_temp = 0
            t_temp = new_dt[0]
            while t_temp / self.t_merge < 0.9999999:
                i_temp += 1
                t_temp += new_dt[i_temp]

            # Keep the t_merger index in memory
            self.i_t_merger = i_temp

        # Update/redeclare all the arrays (stable isotopes)
        ymgal = self._get_iniabu()
        self.len_ymgal = len(ymgal)
        self.mdot, self.ymgal, self.ymgal_massive, self.ymgal_agb, \
        self.ymgal_1a, self.ymgal_nsm, \
        self.ymgal_delayed_extra, self.mdot_massive, \
        self.mdot_agb, self.mdot_1a, self.mdot_nsm, \
        self.mdot_delayed_extra, \
        self.sn1a_numbers, self.sn2_numbers, self.nsm_numbers, \
        self.delayed_extra_numbers, self.imf_mass_ranges, \
        self.imf_mass_ranges_contribution, self.imf_mass_ranges_mtot = \
        self._get_storing_arrays(ymgal, len(self.history.isotopes))

        # Update/redeclare all the arrays (unstable isotopes)
        if self.len_decay_file > 0 or self.use_decay_module:
            ymgal_radio = np.zeros(self.nb_radio_iso)

            # Initialisation of the storing arrays for radioactive isotopes
            self.mdot_radio, self.ymgal_radio, self.ymgal_massive_radio, \
            self.ymgal_agb_radio, self.ymgal_1a_radio, self.ymgal_nsm_radio, \
            self.ymgal_delayed_extra_radio, \
            self.mdot_massive_radio, self.mdot_agb_radio, self.mdot_1a_radio, \
            self.mdot_nsm_radio, \
            self.mdot_delayed_extra_radio, dummy, dummy, dummy, dummy, dummy, \
            dummy, dummy, dummy = \
            self._get_storing_arrays(ymgal_radio, self.nb_radio_iso)

        # Recalculate the simulation time (used in chem_evol)
        self.t_ce = []
        self.t_ce.append(self.history.timesteps[0])
        for i_init in range(1,self.nb_timesteps):
          self.t_ce.append(self.t_ce[i_init-1] + self.history.timesteps[i_init])


    ##############################################
    #              Rafine Steps LR               #
    ##############################################
    def __rafine_steps_lr(self):

        '''
        This function increases the number of timesteps if the star formation
        will eventually consume all the gas, which occurs when dt > (t_star/sfe).

        '''

        # Declaration of the new timestep array
        if not self.print_off:
            print ('..Time refinement (long range)..')
        new_dt = []

        # For every timestep ...
        for i_rs in range(0,len(self.history.timesteps)):

            # Calculate the critical time delay
            t_raf = self.t_SF_t[i_rs] / self.sfe

            # If the step needs to be refined ...
            if self.history.timesteps[i_rs] > t_raf:

                # Calculate the number of remaining steps
                nb_step_rem = len(self.history.timesteps) - i_rs
                t_rem = 0.0
                for i_rs in range(0,len(self.history.timesteps)):
                    t_rem += self.history.timesteps[i_rs]

                # Calculate the split factor
                nb_split = int(t_rem / t_raf) + 1

                # Split the step
                for i_sp_st in range(0,nb_split):
                    new_dt.append(t_rem/nb_split)

                # Quit the for loop
                break

            # If ok, don't change anything
            else:
                new_dt.append(self.history.timesteps[i_rs])

        # Update the timestep information
        self.nb_timesteps = len(new_dt)
        self.history.timesteps = new_dt

        # Update self.history.age
        self.history.age = [0]
        for ii in range(self.nb_timesteps):
            self.history.age.append(self.history.age[-1] + new_dt[ii])
        self.history.age = np.array(self.history.age)

        # If a timestep needs to be added to be synchronized with
        # the external program managing merger trees ...
        if self.t_merge > 0.0:

            # Find the interval where the step needs to be added
            i_temp = 0
            t_temp = new_dt[0]
            while t_temp / self.t_merge < 0.9999999:
                i_temp += 1
                t_temp += new_dt[i_temp]

            # Keep the t_merger index in memory
            self.i_t_merger = i_temp

        # Update/redeclare all the arrays (stable isotopes)
        ymgal = self._get_iniabu()
        self.len_ymgal = len(ymgal)
        self.mdot, self.ymgal, self.ymgal_massive, self.ymgal_agb, \
        self.ymgal_1a, self.ymgal_nsm, \
        self.ymgal_delayed_extra, self.mdot_massive, \
        self.mdot_agb, self.mdot_1a, self.mdot_nsm, \
        self.mdot_delayed_extra, \
        self.sn1a_numbers, self.sn2_numbers, self.nsm_numbers, \
        self.delayed_extra_numbers, self.imf_mass_ranges, \
        self.imf_mass_ranges_contribution, self.imf_mass_ranges_mtot = \
        self._get_storing_arrays(ymgal, len(self.history.isotopes))

        # Update/redeclare all the arrays (unstable isotopes)
        if self.len_decay_file > 0 or self.use_decay_module:
            ymgal_radio = np.zeros(self.nb_radio_iso)

            # Initialisation of the storing arrays for radioactive isotopes
            self.mdot_radio, self.ymgal_radio, self.ymgal_massive_radio, \
            self.ymgal_agb_radio, self.ymgal_1a_radio, self.ymgal_nsm_radio, \
            self.ymgal_delayed_extra_radio, \
            self.mdot_massive_radio, self.mdot_agb_radio, self.mdot_1a_radio, \
            self.mdot_nsm_radio, \
            self.mdot_delayed_extra_radio, dummy, dummy, dummy, dummy, dummy, \
            dummy, dummy, dummy = \
            self._get_storing_arrays(ymgal_radio, self.nb_radio_iso)

        # Recalculate the simulation time (used in chem_evol)
        self.t_ce = []
        self.t_ce.append(self.history.timesteps[0])
        for i_init in range(1,self.nb_timesteps):
          self.t_ce.append(self.t_ce[i_init-1] + self.history.timesteps[i_init])


    ##############################################
    #             Calc Mass Frac SSP             #
    ##############################################
    def __calc_mass_frac_SSP(self):

        '''
        Calculate the average mass fraction returned by stellar populations.
        This refers to the ratio between the ejected mass and the initial
        stellar population mass.

        '''

        # If the mass fraction is self-calculated from SSPs ..
        if self.calc_SSP_ej:

            # Set the stellar population mass to 1 Msun
            self.kwargs["mgal"] = 1.0

            # Define the 5 metallicities that will be used
            Z = [0.02, 0.01, 0.006, 0.001, 0.0001]

            # Run SYGMA with different metallicities and cumulate ejected mass
            self.mass_frac_SSP = 0.0
            for i_Z_SSP in range(0,len(Z)):
                self.kwargs["iniZ"] = Z[i_Z_SSP]
                s_inst = sygma.sygma(**self.kwargs)
                self.mass_frac_SSP += np.sum(s_inst.ymgal[-1])

            # Calculate the average mass fraction returned
            self.mass_frac_SSP = self.mass_frac_SSP / len(Z)
            print ('Average SSP mass fraction returned = ',self.mass_frac_SSP)

        # Take the input value if SSP calculations are not used
        else:
            self.mass_frac_SSP = self.mass_frac_SSP_in


    ##############################################
    #            Declare Evol Arrays             #
    ##############################################
    def __declare_evol_arrays(self):

        '''
        This function declares the arrays used to follow the evolution of the
        galaxy regarding its growth and the exchange of gas with its surrounding.

        '''

        # Arrays with specific values at every timestep
        self.sfr_input = np.zeros(self.nb_timesteps+1) # Star formation rate [Mo yr^-1]
        self.m_DM_t = np.zeros(self.nb_timesteps+1) # Mass of the dark matter halo
        self.r_vir_DM_t= np.zeros(self.nb_timesteps+1) # Virial radius of the dark matter halo
        self.v_vir_DM_t= np.zeros(self.nb_timesteps+1) # Virial velocity of the halo
        self.m_tot_ISM_t = np.zeros(self.nb_timesteps+1) # Mass of the ISM in gas
        self.m_outflow_t = np.zeros(self.nb_timesteps) # Mass of the outflow at every timestep
        self.eta_outflow_t = np.zeros(self.nb_timesteps) # Mass-loading factor == M_outflow / SFR
        self.t_SF_t = np.zeros(self.nb_timesteps+1) # Star formation timescale at every timestep
        self.m_crit_t = np.zeros(self.nb_timesteps+1) # Critital ISM mass below which no SFR
        self.redshift_t = np.zeros(self.nb_timesteps+1) # Redshift associated to every timestep
        self.m_inflow_t = np.zeros(self.nb_timesteps) # Mass of the inflow at every timestep


    ##############################################
    #            Initialize Gal Prop             #
    ##############################################
    def __initialize_gal_prop(self):

        '''
        This function sets the properties of the selected galaxy, such as its
        SFH, its total mass, and its stellar mass.

        '''

        # No specific galaxy - Use input parameters
        if self.galaxy == 'none':

            #If an array is used for the SFH ..
            if len(self.sfh_array) > 0:
                self.__copy_sfr_array()

            # If an input file is used for the SFH ...
            elif not self.sfh_file == 'none':
                self.__copy_sfr_input(self.sfh_file)

            # If a star formation law is used in a closed box ...
            elif self.cl_SF_law and not self.open_box:
                self.__calculate_sfe_cl()

            # If a random SFH is chosen ...
            elif self.rand_sfh > 0.0:
                self.__generate_rand_sfh()

            # If the SFH is constant ...
            else:
                for i_cte_sfr in range(0, self.nb_timesteps+1):
                    self.sfr_input[i_cte_sfr] = self.cte_sfr

        # Milky Way galaxy ...
        elif self.galaxy == 'milky_way' or self.galaxy == 'milky_way_cte':

            # Set the current dark and stellar masses (corrected for mass loss)
            self.m_DM_0 = 1.0e12
            self.stellar_mass_0 = 5.0e10

            # Read Chiappini et al. (2001) SFH
            if self.galaxy == 'milky_way':
                self.__copy_sfr_input('stellab_data/milky_way_data/sfh_mw_cmr01.txt')

            # Read constant SFH
            else:
                self.__copy_sfr_input('stellab_data/milky_way_data/sfh_cte.txt')

        # Sculptor dwarf galaxy ...
        elif self.galaxy == 'sculptor':

            # Set the current dark and stellar masses (corrected for mass loss)
            self.m_DM_0 = 1.5e9
            self.stellar_mass_0 = 7.8e6
            self.stellar_mass_0 = self.stellar_mass_0 * (1-self.mass_frac_SSP)

            # Read deBoer et al. (2012) SFH
            self.__copy_sfr_input('stellab_data/sculptor_data/sfh_deBoer12.txt')

        # Fornax dwarf galaxy ...
        elif self.galaxy == 'fornax':

            # Set the current dark and stellar masses (corrected for mass loss)
            self.m_DM_0 = 7.08e8
            self.stellar_mass_0 = 4.3e7
            self.stellar_mass_0 = self.stellar_mass_0 * (1-self.mass_frac_SSP)

            # Read deBoer et al. (2012) SFH
            self.__copy_sfr_input('stellab_data/fornax_data/sfh_fornax_deboer_et_al_2012.txt')

        # Carina dwarf galaxy ...
        elif self.galaxy == 'carina':

            # Set the current dark and stellar masses (corrected for mass loss)
            self.m_DM_0 = 3.4e6
            self.stellar_mass_0 = 1.07e6
            self.stellar_mass_0 = self.stellar_mass_0 * (1-self.mass_frac_SSP)

            # Read deBoer et al. (2014) SFH
            self.__copy_sfr_input('stellab_data/carina_data/sfh_deBoer14.txt')

        # Interpolate the last timestep
        if len(self.sfr_input) > 3:
            aa = (self.sfr_input[-2] - self.sfr_input[-3])/\
                 self.history.timesteps[-2]
            bb = self.sfr_input[-2]- (self.history.tend-self.history.timesteps[-1])*aa
            self.sfr_input[-1] = aa*self.history.tend + bb

        # Keep the SFH in memory
        self.history.sfr_abs = self.sfr_input


    ##############################################
    ##             Copy SFR Array               ##
    ##############################################
    def __copy_sfr_array(self):

        '''
        See copy_sfr_input() for more info.

        '''

        # Variable to keep track of the OMEGA's timestep
        i_dt_csa = 0
        t_csa = 0.0
        nb_dt_csa = self.nb_timesteps + 1

        # Variable to keep track of the total stellar mass from the input SFH
        m_stel_sfr_in = 0.0

        # For every timestep given in the array (starting at the second step)
        for i_csa in range(1,len(self.sfh_array)):

            # Calculate the SFR interpolation coefficient
            a_sfr = (self.sfh_array[i_csa][1] - self.sfh_array[i_csa-1][1]) / \
                    (self.sfh_array[i_csa][0] - self.sfh_array[i_csa-1][0])
            b_sfr = self.sfh_array[i_csa][1] - a_sfr * self.sfh_array[i_csa][0]

            # While we stay in the same time bin ...
            while t_csa <= self.sfh_array[i_csa][0]:

                # Interpolate the SFR
                self.sfr_input[i_dt_csa] = a_sfr * t_csa + b_sfr

                # Cumulate the stellar mass formed
                m_stel_sfr_in += self.sfr_input[i_dt_csa] * \
                    self.history.timesteps[i_dt_csa]

                # Exit the loop if the array is full
                if i_dt_csa >= nb_dt_csa:
                    break

                # Calculate the new time
                t_csa += self.history.timesteps[i_dt_csa]
                i_dt_csa += 1

            # Exit the loop if the array is full
            if (i_dt_csa + 1) >= nb_dt_csa:
                break

        # If the array has been read completely, but the sfr_input array is
        # not full, fil the rest of the array with the last read value
        if self.sfh_array[-1][1] == 0.0:
            sfr_temp = 0.0
        else:
            sfr_temp = self.sfr_input[i_dt_csa-1]
        while i_dt_csa < nb_dt_csa - 1:
            self.sfr_input[i_dt_csa] = sfr_temp
            m_stel_sfr_in += self.sfr_input[i_dt_csa] * \
                self.history.timesteps[i_dt_csa]
            t_csa += self.history.timesteps[i_dt_csa]
            i_dt_csa += 1

        # Normalise the SFR in order to be consistent with the integrated
        # input star formation array (no mass loss considered!)
        if self.sfh_array_norm > 0.0:
            norm_sfr_in = self.sfh_array_norm / m_stel_sfr_in
            for i_csa in range(0, nb_dt_csa):
                self.sfr_input[i_csa] = self.sfr_input[i_csa] * norm_sfr_in

        # Fill the missing last entry (extention of the last timestep, for tend)
        # Since we don't know dt starting at tend, it is not part of m_stel_sfr_in
        self.sfr_input[-1] = self.sfr_input[-2]


    ##############################################
    ##             Calculate SFE Cl.            ##
    ##############################################
    def __calculate_sfe_cl(self):

        '''
        Calculate the star formation efficiency and the initial mass of gas
        for a closed box model, given the final gas mass and the current
        stellar mass.

        '''

        # Get the average return gas fraction of SSPs
        if self.mass_frac_SSP == -1.0:
            f_ej = 0.35
        else:
            f_ej = self.mass_frac_SSP

        # If the gas-to-stellar mass ratio is the selected input ...
        if self.r_gas_star > 0.0:

            # Calculate the final mass of gas
            self.m_gas_f = self.r_gas_star * self.stellar_mass_0

            # Calculate the initial mass of gas
            m_gas_ini = self.m_gas_f + self.stellar_mass_0

        # If the final mass of gas is the selected input ...
        elif self.m_gas_f > 0.0:

            # Calculate the initial mass of gas
            m_gas_ini = self.m_gas_f + self.stellar_mass_0

        # If the initial mass of gas is the selected input ...
        else:

            # Use the input value for the initial mass of gas
            m_gas_ini = self.mgal

            # Calculate the final mass of gas
            self.m_gas_f = m_gas_ini - self.stellar_mass_0

        # Verify if the final mass of gas is negative
        if self.m_gas_f < 0.0:
            self.not_enough_gas = True
            sfe_gcs = 1.0e-10
            print ('!!Error - Try to have a negative final gas mass!!')

        if not self.not_enough_gas:

          # Scale the initial mass of all isotopes
          scale_m_tot = m_gas_ini / np.sum(self.ymgal[0])
          for k_cm in range(len(self.ymgal[0])):
              self.ymgal[0][k_cm] = self.ymgal[0][k_cm] * scale_m_tot

          # Initialization for finding the right SFE
          sfe_gcs = 1.8e-10
          sfe_max = 1.0
          sfe_min = 0.0
          m_gas_f_try = self.__get_m_gas_f(m_gas_ini, sfe_gcs, f_ej)

          # While the SFE is not the right one ...
          while abs(m_gas_f_try - self.m_gas_f) > 0.01:

            # If the SFE needs to be increased ...
            if (m_gas_f_try / self.m_gas_f) > 1.0:

              # Set the lower limit of the SFE interval
              sfe_min = sfe_gcs

              # If an upper limit is already defined ...
              if sfe_max < 1.0:

                # Set the SFE to the middle point of the interval
                sfe_gcs = (sfe_max + sfe_gcs) * 0.5

              # If an upper limit is not already defined ...
              else:

                # Try a factor of 2
                sfe_gcs = sfe_gcs * 2.0

            # If the SFE needs to be decreased ...
            else:

              # Set the upper limit of the SFE interval
              sfe_max = sfe_gcs

              # If a lower limit is already defined ...
              if sfe_min > 0.0:

                # Set the SFE to the middle point of the interval
                sfe_gcs = (sfe_min + sfe_gcs) * 0.5

              # If a lower limit is not already defined ...
              else:

                # Try a factor of 2
                sfe_gcs = sfe_gcs * 0.5

            # Get the approximated final mass of gas
            m_gas_f_try = self.__get_m_gas_f(m_gas_ini, sfe_gcs, f_ej)

        # Keep the SFE in memory
        self.sfe_gcs = sfe_gcs


    ##############################################
    ##               Get M_gas_f                ##
    ##############################################
    def __get_m_gas_f(self, m_gas_ini, sfe_gcs, f_ej):

        '''
        Return the final mass of gas, given the initial mass of the gas
        reservoir and the star formation efficiency.  The function uses
        a simple star formation law in the form of SFR(t) = sfe * M_gas(t)

        '''

        # Initialisation of the integration
        m_gas_loop = m_gas_ini
        t_gmgf = 0.0

        # For every timestep ...
        for i_gmgf in range(0,self.nb_timesteps):

            # Calculate the new mass of gass
            t_gmgf += self.history.timesteps[i_gmgf]

            #self.sfr_input[i_gmgf] = sfe_gcs * m_gas_loop
            m_gas_loop -= sfe_gcs * (1-f_ej) * m_gas_loop * \
                self.history.timesteps[i_gmgf]

        # Return the final mass of gas
        return m_gas_loop


    ##############################################
    #              Copy SFR Input                #
    ##############################################
    def __copy_sfr_input(self, path_sfh_in):

        '''
        This function reads a SFH input file and interpolates its values so it
        can be inserted in the array "sfr_input", which contains the SFR for each
        OMEGA timestep.

        Note
        ====

          The input file does not need to have constant time step lengths, and
          does not need to have the same number of timesteps as the number of
          OMEGA timesteps.

        Important
        =========

          In OMEGA and SYGMA, t += timestep[i] is the first thing done in the main
          loop.  The loop calculates what happened between the previous t and the
          new t.  This means the mass of stars formed must be SFR(previous t) *
          timestep[i].  Therefore, sfr_input[i] IS NOT the SFR at time t +=
          timestep[i], but rather the SFR at previous time which is used for the
          current step i.

        Argument
        ========

          path_sfh_in : Path of the input SFH file.

        '''

        # Variable to keep track of the OMEGA timestep
        nb_dt_csi = self.nb_timesteps + 1
        i_dt_csi = 0
        t_csi = 0.0 # Not timesteps[0] because sfr_input[0] must be
                    # used from t = 0 to t = timesteps[0]

        # Variable to keep track of the total stellar mass from the input SFH
        m_stel_sfr_in = 0.0

        # Open the file containing the SFR vs time
        with open(os.path.join(nupy_path, path_sfh_in), 'r') as sfr_file:

            # Read the first line  (col 0 : t, col 1 : SFR)
            line_1_str = sfr_file.readline()
            parts_1 = [float(x) for x in line_1_str.split()]

            # For every remaining line ...
            for line_2_str in sfr_file:

                # Extract data
                parts_2 = [float(x) for x in line_2_str.split()]

                # Calculate the interpolation coefficients (SFR = a*t + b)
                a_csi = (parts_2[1] - parts_1[1]) / (parts_2[0] - parts_1[0])
                b_csi = parts_1[1] - a_csi * parts_1[0]

                # While we stay in the same time bin ...
                while t_csi <= parts_2[0]:

                    # Calculate the right SFR for the specific OMEGA timestep
                    #self.sfr_input[i_dt_csi] = a_csi * t_csi + b_csi

                    # Calculate the average SFR for the specific OMEGA timestep
                    if i_dt_csi < self.nb_timesteps:
                        self.sfr_input[i_dt_csi] = a_csi * (t_csi + \
                            self.history.timesteps[i_dt_csi] * 0.5) + b_csi
                    else:
                        self.sfr_input[i_dt_csi] = a_csi * t_csi + b_csi

                    # Cumulate the mass of stars formed
                    if i_dt_csi < nb_dt_csi - 1:
                        m_stel_sfr_in += self.sfr_input[i_dt_csi] * \
                            self.history.timesteps[i_dt_csi]

                        # Calculate the new time
                        t_csi += self.history.timesteps[i_dt_csi]

                    # Go to the next time step
                    i_dt_csi += 1

                    # Exit the loop if the array is full
                    if i_dt_csi >= nb_dt_csi:
                        break

                # Exit the loop if the array is full
                if i_dt_csi >= nb_dt_csi:
                    break

                # Copie the last read line
                parts_1 = copy.copy(parts_2)

        # Close the file
        sfr_file.close()

        # If the file has been read completely, but the sfr_input array is
        # not full, fill the rest of the array with the last read value
        while i_dt_csi < nb_dt_csi:
            self.sfr_input[i_dt_csi] = self.sfr_input[i_dt_csi-1]
            if i_dt_csi < nb_dt_csi - 1:
                m_stel_sfr_in += self.sfr_input[i_dt_csi] * \
                    self.history.timesteps[i_dt_csi]
            i_dt_csi += 1

        # Normalise the SFR in order to be consistent with the input current
        # stellar mass (if the stellar mass is known)
        if self.stellar_mass_0 > 0.0:
            norm_sfr_in = self.stellar_mass_0 / ((1-self.mass_frac_SSP) * m_stel_sfr_in)
            for i_csi in range(0, nb_dt_csi):
                self.sfr_input[i_csi] = self.sfr_input[i_csi] * norm_sfr_in


    ##############################################
    #             Generate Rand SFH              #
    ##############################################
    def __generate_rand_sfh(self):

        '''
        This function generates a random SFH. This should only be used for
        testing purpose in order to look at how the uncertainty associated to the
        SFH can affects the results.

        The self.rand_sfh sets the maximum ratio between the maximum and the
        minimum values for the SFR.  This parameter sets how "bursty" or constant
        a SFH is.  self.rand_sfh = 1 means a constant SFH.

        '''

        # Variable to keep track of the total stellar mass from the random SFH
        m_stel_sfr_in = 0.0

        # For each timestep
        for i_csi in range(0,self.nb_timesteps+1):

            self.sfr_input[i_csi] = random.randrange(1,self.rand_sfh+1)

            # Cumulate the total mass of stars formed
            if i_csi < self.nb_timesteps:
                m_stel_sfr_in += self.sfr_input[i_csi] * \
                    self.history.timesteps[i_csi]

        # Normalise the SFR in order to be consistent with the input
        # current stellar mass
        norm_sfr_in = self.stellar_mass_0 / ((1-self.mass_frac_SSP) * m_stel_sfr_in)
        for i_csi in range(0,self.nb_timesteps+1):
            self.sfr_input[i_csi] = self.sfr_input[i_csi] * norm_sfr_in


    ##############################################
    #              Fill Evol Arrays              #
    ##############################################
    def __fill_evol_arrays(self):

        '''
        This function fills the arrays used to follow the evolution of the
        galaxy regarding its growth and the exchange of gas with its surrounding.

        '''

        # Execute this function only if needed
        if self.in_out_control or self.SF_law or self.DM_evolution:

            # Calculate the redshift for every timestep, if needed
            self.calculate_redshift_t()

            # Calculate the mass of the dark matter halo at every timestep
            self.__calculate_m_DM_t()

            # Calculate the virial radius and velocity at every timestep
            self.calculate_virial()

            # Calculate the critical, mass below which no SFR, at every dt
            self.__calculate_m_crit_t()

            # Calculate the star formation timescale at every timestep
            self.__calculate_t_SF_t()

            # Calculate the gas mass of the ISM at every timestep
            self.__calculate_m_tot_ISM_t()

            # Calculate the mass-loading factor and ouflow mass at every timestep
            self.__calculate_outflow_t()


    ##############################################
    #               Get t From z                 #
    ##############################################
    def __get_t_from_z(self, z_gttfz):

        '''
        This function returns the age of the Universe at a given redshift.

        Argument
        ========

          z_gttfz : Redshift that needs to be converted into age.

        '''

        # Return the age of the Universe
        temp_var = math.sqrt((self.lambda_0/self.omega_0)/(1.0+z_gttfz)**3)
        x_var = math.log( temp_var + math.sqrt( temp_var**2 + 1.0 ) )
        return 2.0 / ( 3.0 * self.H_0 * math.sqrt(self.lambda_0)) * \
               x_var * 9.77793067e11


    ##############################################
    #               Get z From t                 #
    ##############################################
    def __get_z_from_t(self, t_gtzft):

        '''
        This function returns the redshift of a given Universe age.

        Argument
        ========

          t_gtzft : Age of the Universe that needs to be converted into redshift.

        '''

        # Return the redshift
        temp_var = 1.5340669e-12 * self.lambda_0**0.5 * self.H_0 * t_gtzft
        return (self.lambda_0 / self.omega_0)**0.3333333 / \
                math.sinh(temp_var)**0.66666667 - 1.0


    ##############################################
    #           Calculate redshift(t)            #
    ##############################################
    def calculate_redshift_t(self):

        '''
        This function calculates the redshift associated to every timestep
        assuming that 'tend' represents redshift zero.

        '''

        # Calculate the current age of the Universe (LambdaCDM - z = 0)
        current_age_czt = self.__get_t_from_z(self.redshift_f)

        # Calculate the age of the Universe when the galaxy forms
        age_formation_czt = current_age_czt - self.history.tend

        # Initiate the age of the galaxy
        t_czt = 0.0

        # Initialize the linear interpolation coefficients
        self.redshift_t_coef = np.zeros((self.nb_timesteps,2))

        #For each timestep
        for i_czt in range(0, self.nb_timesteps+1):

            #Calculate the age of the Universe at that time [yr]
            age_universe_czt = age_formation_czt + t_czt

            #Calculate the redshift at that time
            self.redshift_t[i_czt] = self.__get_z_from_t(age_universe_czt)

            # Calculate the interpolation coefficients
            # z = self.redshift_t_coef[0] * t + self.redshift_t_coef[1]
            if i_czt > 0:
                self.redshift_t_coef[i_czt-1][0] = \
                    (self.redshift_t[i_czt]-self.redshift_t[i_czt-1]) / \
                        self.history.timesteps[i_czt-1]
                self.redshift_t_coef[i_czt-1][1] = self.redshift_t[i_czt] - \
                    self.redshift_t_coef[i_czt-1][0] * t_czt

            #Udpate the age of the galaxy [yr]
            if i_czt < self.nb_timesteps:
                t_czt += self.history.timesteps[i_czt]

        #Correction for last digit error (e.g. z = -2.124325345e-8)
        if self.redshift_t[-1] < 0.0:
            self.redshift_t[-1] = 0.0


    ##############################################
    #                Run All SSPs                #
    ##############################################
    def __run_all_ssps(self):

        '''
        Create a SSP with SYGMA for each metallicity available in the yield tables.
        Each SSP has a total mass of 1 Msun, is it can easily be re-normalized.

        '''

        # Set (modify) parameters that are common to each SSP calculation
        self.kwargs["mgal"] = 1.0
        self.kwargs["pre_calculate_SSPs"] = False

        # Copy the metallicities and put them in increasing order
        self.Z_table_SSP = copy.copy(self.ytables.Z_list)
        self.Z_table_first_nzero = min(self.Z_table_SSP)
        if self.popIII_info_fast and self.iniZ <= 0.0 and self.Z_trans > 0.0:
            self.Z_table_SSP.append(0.0)
        self.Z_table_SSP = sorted(self.Z_table_SSP)
        self.nb_Z_table_SSP = len(self.Z_table_SSP)

        # If the SSPs are not given as an input ..
        if len(self.SSPs_in) == 0:

          # Define the SSP timesteps
          len_dt_SSPs = len(self.dt_in_SSPs)
          if len_dt_SSPs == 0:
              self.kwargs["dt_in"] = self.history.timesteps
              len_dt_SSPs = self.nb_timesteps
          else:
              self.kwargs["dt_in"] = self.dt_in_SSPs

          # Declare the SSP ejecta arrays [Z][dt][iso]
          self.ej_SSP = np.zeros((self.nb_Z_table_SSP, len_dt_SSPs, self.nb_isotopes))
          if self.len_decay_file > 0 or self.use_decay_module:
              self.ej_SSP_radio = \
                  np.zeros((self.nb_Z_table_SSP, len_dt_SSPs, self.nb_radio_iso))

          # For each metallicity ...
          for i_ras in range(0,self.nb_Z_table_SSP):

              # Use a dummy iniabu file if the metallicity is not zero
              if self.Z_table_SSP[i_ras] == 0:
                  self.kwargs["iniabu_table"] = ''
                  self.kwargs["hardsetZ"] = self.hardsetZ
              else:
                  self.kwargs["iniabu_table"] = 'yield_tables/iniabu/iniab2.0E-02GN93.ppn'
                  self.kwargs["hardsetZ"] = self.Z_table_SSP[i_ras]

              # Set metallicity for SYGMA run
              self.kwargs["iniZ"] = self.Z_table_SSP[i_ras]

              # Run a SYGMA simulation (1 Msun SSP)
              sygma_inst = sygma.sygma(**self.kwargs)

              # Copy the ejecta arrays from the SYGMA simulation
              self.ej_SSP[i_ras] = sygma_inst.mdot
              if self.len_decay_file > 0 or self.use_decay_module:
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
        if self.len_decay_file > 0 or self.use_decay_module:
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
              if self.len_decay_file > 0 or self.use_decay_module:
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
    #              Calculate M_DM(t)             #
    ##############################################
    def __calculate_m_DM_t(self):

        '''
        This functions calculates the mass of the dark matter halo at each
        timestep.

        '''

        # If the mass of the dark matter halo is kept at a constant value ...
        if not self.DM_evolution:

            # If the dark matter evolution is an input array ...
            if len(self.DM_array) > 0:

                # Copy the input values
                self.copy_DM_input()

            # Use the current value for every timestep
            else:
                for i_cmdt in range(0, self.nb_timesteps+1):
                    self.m_DM_t[i_cmdt] = self.m_DM_0

        # If the mass of the dark matter halo evolves with time ...
        else:

          # If the dark matter evolution is an input array ...
          if len(self.DM_array) > 0:

            # Copy the input values
            self.copy_DM_input()

          # If the dark matter evolution is taken from Millenium simulations ...
          else:

            # Calculate the polynomial coefficient for the evolution of
            # the dark matter mass
            poly_up_dm, poly_low_dm = self.__get_DM_bdy()

            # For each timestep ...
            for i_cmdt in range(0, self.nb_timesteps+1):

                # Calculate the lower and upper dark matter mass boundaries
                log_m_dm_up = poly_up_dm[0] * self.redshift_t[i_cmdt]**3 + \
                    poly_up_dm[1] * self.redshift_t[i_cmdt]**2 + poly_up_dm[2] * \
                        self.redshift_t[i_cmdt] + poly_up_dm[3]
                log_m_dm_low = poly_low_dm[0] * self.redshift_t[i_cmdt]**3 + \
                    poly_low_dm[1] * self.redshift_t[i_cmdt]**2 + poly_low_dm[2]*\
                        self.redshift_t[i_cmdt] + poly_low_dm[3]

                # If the dark matter mass is too low for the available fit ...
                if self.DM_too_low:

                    # Scale the fit using the current input mass
                    self.m_DM_t[i_cmdt] = 10**log_m_dm_low * \
                                            self.m_DM_0 / 10**poly_low_dm[3]

                # If the dark matter mass can be interpolated
                else:

                    # Use a linear interpolation with the log of the mass
                    a = (log_m_dm_up - log_m_dm_low) / \
                        (poly_up_dm[3] - poly_low_dm[3])
                    b = log_m_dm_up - a * poly_up_dm[3]
                    self.m_DM_t[i_cmdt] = 10**( a * math.log10(self.m_DM_0) + b )

            # If the simulation does not stop at redshift zero ...
            if not self.redshift_f == 0.0:

                # Scale the DM mass (because the fits look at M_DM_0 at z=0)
                m_dm_scale = self.m_DM_0 / self.m_DM_t[-1]
                for i_cmdt in range(0, self.nb_timesteps+1):
                    self.m_DM_t[i_cmdt] = self.m_DM_t[i_cmdt] * m_dm_scale

        # Create the interpolation coefficients
        # M_DM = self.m_DM_t_coef[0] * t + self.m_DM_t_coef[1]
        self.m_DM_t_coef = np.zeros((self.nb_timesteps,2))
        for i_cmdt in range(0, self.nb_timesteps):
            self.m_DM_t_coef[i_cmdt][0] = (self.m_DM_t[i_cmdt+1] - \
                self.m_DM_t[i_cmdt]) / self.history.timesteps[i_cmdt]
            self.m_DM_t_coef[i_cmdt][1] = self.m_DM_t[i_cmdt] - \
                self.m_DM_t_coef[i_cmdt][0] * self.history.age[i_cmdt]


    ##############################################
    #               Copy DM Input                #
    ##############################################
    def copy_DM_input(self):

        '''
        This function interpolates the DM masses from an input array
        and add the masses to the corresponding OMEGA step

        '''

        # Variable to keep track of the OMEGA's timestep
        i_dt_csa = 0
        t_csa = 0.0
        nb_dt_csa = self.nb_timesteps

        # If just one entry ...
        if len(self.DM_array) == 1:
            self.m_DM_t[i_dt_csa] = self.DM_array[0][1]
            i_dt_csa += 1

        # If DM values need to be interpolated ...
        else:

          # For every timestep given in the array (starting at the second step)
          for i_csa in range(1,len(self.DM_array)):

            # Calculate the DM interpolation coefficient
            a_DM = (self.DM_array[i_csa][1] - self.DM_array[i_csa-1][1]) / \
                    (self.DM_array[i_csa][0] - self.DM_array[i_csa-1][0])
            b_DM = self.DM_array[i_csa][1] - a_DM * self.DM_array[i_csa][0]

            # While we stay in the same time bin ...
            while t_csa <= self.DM_array[i_csa][0]:

                # Interpolate the SFR
                self.m_DM_t[i_dt_csa] = a_DM * t_csa + b_DM

                # Exit the loop if the array is full
                if i_dt_csa >= nb_dt_csa:
                    break

                # Calculate the new time
                t_csa += self.history.timesteps[i_dt_csa]
                i_dt_csa += 1

            # Exit the loop if the array is full
            if i_dt_csa >= nb_dt_csa:
                break

        # If the array has been read completely, but the DM array is
        # not full, fil the rest of the array with the last input value
        while i_dt_csa < nb_dt_csa+1:
            self.m_DM_t[i_dt_csa] = self.DM_array[-1][1]
            #self.m_DM_t[i_dt_csa] = self.m_DM_t[i_dt_csa-1]
            i_dt_csa += 1


    ##############################################
    #              Copy R_vir Input              #
    ##############################################
    def copy_r_vir_input(self):

        '''
        This function interpolates the R_vir from an input array
        and add the radius to the corresponding OMEGA step

        '''

        # Variable to keep track of the OMEGA's timestep
        i_dt_csa = 0
        t_csa = 0.0
        nb_dt_csa = self.nb_timesteps

        # If just one entry ...
        if len(self.r_vir_array) == 1:
            self.r_vir_DM_t[i_dt_csa] = self.r_vir_array[0][1]
            i_dt_csa += 1

        # If r_vir values need to be interpolated ...
        else:

          # For every timestep given in the array (starting at the second step)
          for i_csa in range(1,len(self.r_vir_array)):

            # Calculate the DM interpolation coefficient
            a_r_vir = (self.r_vir_array[i_csa][1] - self.r_vir_array[i_csa-1][1]) / \
                    (self.r_vir_array[i_csa][0] - self.r_vir_array[i_csa-1][0])
            b_r_vir = self.r_vir_array[i_csa][1] - a_r_vir * self.r_vir_array[i_csa][0]

            # While we stay in the same time bin ...
            while t_csa <= self.r_vir_array[i_csa][0]:

                # Interpolate r_vir
                self.r_vir_DM_t[i_dt_csa] = a_r_vir * t_csa + b_r_vir

                # Exit the loop if the array is full
                if i_dt_csa >= nb_dt_csa:
                    break

                # Calculate the new time
                t_csa += self.history.timesteps[i_dt_csa]
                i_dt_csa += 1

            # Exit the loop if the array is full
            if i_dt_csa >= nb_dt_csa:
                break

        # If the array has been read completely, but the r_vir array is
        # not full, fil the rest of the array with the last input value
        while i_dt_csa < nb_dt_csa+1:
            self.r_vir_DM_t[i_dt_csa] = self.r_vir_array[-1][1]
            #self.r_vir_DM_t[i_dt_csa] = self.r_vir_DM_t[i_dt_csa-1]
            i_dt_csa += 1

        # Create the interpolation coefficients
        # R_vir = self.r_vir_DM_t_coef[0] * t + self.r_vir_DM_t_coef[1]
        self.r_vir_DM_t_coef = np.zeros((self.nb_timesteps,2))
        for i_cmdt in range(0, self.nb_timesteps):
            self.r_vir_DM_t_coef[i_cmdt][0] = (self.r_vir_DM_t[i_cmdt+1] - \
                self.r_vir_DM_t[i_cmdt]) / self.history.timesteps[i_cmdt]
            self.r_vir_DM_t_coef[i_cmdt][1] = self.r_vir_DM_t[i_cmdt] - \
                self.r_vir_DM_t_coef[i_cmdt][0] * self.history.age[i_cmdt]


    ##############################################
    #                  Get DM Bdy                #
    ##############################################
    # Return the fit coefficients for the interpolation of the dark matter mass
    def __get_DM_bdy(self):

        '''
        This function calculates and returns the fit coefficients for the
        interpolation of the evolution of the dark matter mass as a function
        of time.

        '''

        # Open the file containing the coefficient of the 3rd order polynomial fit
        with open(os.path.join(nupy_path, "m_dm_evolution", "poly3_fits.txt"),\
                'r') as m_dm_file:

            # Read the first line
            line_str = m_dm_file.readline()
            parts_1 = [float(x) for x in line_str.split()]

            # If the input dark matter mass is higher than the ones provided
            # by the fits ...
            if math.log10(self.m_DM_0) > parts_1[3]:

                # Use the highest dark matter mass available
                parts_2 = copy.copy(parts_1)
                print ('Warning - Current dark matter mass too high for' \
                      'the available fits.')

            # If the input dark matter mass is in the available range ...
            # Find the mass boundary for the interpolation.
            else:

                # For every remaining line ...
                for line_str in m_dm_file:

                    # Extract data
                    parts_2 = [float(x) for x in line_str.split()]

                    # If the read mass is lower than the input dark matter mass
                    if math.log10(self.m_DM_0) > parts_2[3]:
                        # Exit the loop and use the current interpolation boundary
                        break

                    # Copy the current read line
                    parts_1 = copy.copy(parts_2)

            # Keep track if the input dark matter mass is too low ...
            if parts_1[3] == parts_2[3]:
                 self.DM_too_low = True

        #Close the file
        m_dm_file.close()

        return parts_1, parts_2


    ##############################################
    #              Calculate Virial              #
    ##############################################
    def calculate_virial(self):

        # If R_vir needs to be calculated ..
        if len(self.r_vir_array) == 0:

            # Average current mass density of the Universe [Mo Mpc^-3]
            rho_0_uni = 3.7765e10

            # For each timestep ...
            for i_cv in range(0,len(self.history.timesteps)+1):

                # Calculate the virial radius of the dark matter halo [kpc]
                self.r_vir_DM_t[i_cv] = 1.0e3 * 0.106078 * \
                    (self.m_DM_t[i_cv] / rho_0_uni)**0.3333333 / \
                    (1 + self.redshift_t[i_cv])

        # If R_vir is provided as an input ..
        else:

            # Use the input array and synchronize the timesteps
            self.copy_r_vir_input()

        # For each timestep ...
        for i_cv in range(0,len(self.history.timesteps)+1):

            #Calculate the virial velocity of the dark matter "particles" [km/s]
            self.v_vir_DM_t[i_cv] = ( 4.302e-6 * self.m_DM_t[i_cv] / \
                self.r_vir_DM_t[i_cv] )** 0.5

        # Create the interpolation coefficients
        # R_vir = self.r_vir_DM_t_coef[0] * t + self.r_vir_DM_t_coef[1]
        self.r_vir_DM_t_coef = np.zeros((self.nb_timesteps,2))
        for i_cmdt in range(0, self.nb_timesteps):
            self.r_vir_DM_t_coef[i_cmdt][0] = (self.r_vir_DM_t[i_cmdt+1] - \
                self.r_vir_DM_t[i_cmdt]) / self.history.timesteps[i_cmdt]
            self.r_vir_DM_t_coef[i_cmdt][1] = self.r_vir_DM_t[i_cmdt] - \
                self.r_vir_DM_t_coef[i_cmdt][0] * self.history.age[i_cmdt]

        # Create the interpolation coefficients
        # v_vir = self.v_vir_DM_t_coef[0] * t + self.v_vir_DM_t_coef[1]
        self.v_vir_DM_t_coef = np.zeros((self.nb_timesteps,2))
        for i_cmdt in range(0, self.nb_timesteps):
            self.v_vir_DM_t_coef[i_cmdt][0] = (self.v_vir_DM_t[i_cmdt+1] - \
                self.v_vir_DM_t[i_cmdt]) / self.history.timesteps[i_cmdt]
            self.v_vir_DM_t_coef[i_cmdt][1] = self.v_vir_DM_t[i_cmdt] - \
                self.v_vir_DM_t_coef[i_cmdt][0] * self.history.age[i_cmdt]


    ##############################################
    #             Calculate M_crit_t             #
    ##############################################
    def __calculate_m_crit_t(self):

        # Calculate the real constant
#        m_crit_final = self.norm_crit_m * (0.1/2000.0) * \
#            (self.v_vir_DM_t[-1] * self.r_vir_DM_t[-1])
#        the_constant = m_crit_final / ((0.1/2000.0) * \
#            (self.v_vir_DM_t[-1] * self.r_vir_DM_t[-1])**self.beta_crit)
        the_constant = self.norm_crit_m

        #For each timestep ...
        for i_ctst in range(0,len(self.history.timesteps)+1):

            # If m_crit_t is wanted ...
            if self.m_crit_on:

                # Calculate the critical mass (Croton et al. 2006 .. modified)
                self.m_crit_t[i_ctst] = the_constant * (0.1/2000.0) * \
                    (self.v_vir_DM_t[i_ctst] * self.r_vir_DM_t[i_ctst])**self.beta_crit

            # If m_crit_t is not wanted ...
            else:

                # Set the critical mass to zero
                self.m_crit_t[i_ctst] = 0.0

        # Create the interpolation coefficients
        # M_crit = self.m_crit_t_coef[0] * t + self.m_crit_t_coef[1]
        self.m_crit_t_coef = np.zeros((self.nb_timesteps,2))
        for i_cmdt in range(0, self.nb_timesteps):
            self.m_crit_t_coef[i_cmdt][0] = (self.m_crit_t[i_cmdt+1] - \
                self.m_crit_t[i_cmdt]) / self.history.timesteps[i_cmdt]
            self.m_crit_t_coef[i_cmdt][1] = self.m_crit_t[i_cmdt] - \
                self.m_crit_t_coef[i_cmdt][0] * self.history.age[i_cmdt]


    ##############################################
    #             Calculate t_SF(t)              #
    ##############################################
    def __calculate_t_SF_t(self):

        '''
        This function calculates the star formation timescale at every timestep.

        '''

        # Execute this function only if needed
        if self.SF_law or self.DM_evolution:

            # If the star formation timescale is kept constant ...
            if self.t_star > 0:

                # Use the same value for every timestep
                for i_ctst in range(0, self.nb_timesteps+1):
                    self.t_SF_t[i_ctst] = self.t_star

            # If the timescale follows the halo dynamical time ...
            else:

                # Set the timescale to a fraction of the halo dynamical time
                # See White & Frenk (1991); Springel et al. (2001)
                for i_ctst in range(0, self.nb_timesteps+1):

                    # If the dark matter mass is evolving ...
                    if self.DM_evolution:
                        self.t_SF_t[i_ctst] = self.f_dyn * 0.1 * (1 + \
                            self.redshift_t[i_ctst])**((-1.5)*self.t_sf_z_dep) \
                                / self.H_0 * 9.7759839e11

                    # If the dark matter mass is not evolving ...
                    else:
                        self.t_SF_t[i_ctst] = self.f_dyn * 0.1 / self.H_0 * \
                                9.7759839e11

        # Create the interpolation coefficients
        # SF_t = self.t_SF_t_coef[0] * t + self.t_SF_t_coef[1]
        self.t_SF_t_coef = np.zeros((self.nb_timesteps,2))
        for i_cmdt in range(0, self.nb_timesteps):
            self.t_SF_t_coef[i_cmdt][0] = (self.t_SF_t[i_cmdt+1] - \
                self.t_SF_t[i_cmdt]) / self.history.timesteps[i_cmdt]
            self.t_SF_t_coef[i_cmdt][1] = self.t_SF_t[i_cmdt] - \
                self.t_SF_t_coef[i_cmdt][0] * self.history.age[i_cmdt]


    ##############################################
    #           Calculate M_tot ISM(t)           #
    ##############################################
    def __calculate_m_tot_ISM_t(self):

        '''
        This function calculates the mass of the gas reservoir at every
        timestep using a classical star formation law.

        '''

        # If the evolution of the mass of the ISM is an input ...
        if len(self.m_tot_ISM_t_in) > 0:

            # Copy and adjust the input array for the OMEGA timesteps
            self.__copy_m_tot_ISM_input()

        # If the ISM has a constant mass ...
        elif self.cte_m_gas > 0.0:

            # For each timestep ...
            for i_cm in range(0, self.nb_timesteps+1):
                self.m_tot_ISM_t[i_cm] = self.cte_m_gas

        # If the mass of gas is tighted to the SFH ...
        elif self.SF_law or self.DM_evolution:

            # For each timestep ...
            for i_cm in range(0, self.nb_timesteps+1):

                # If it's the last timestep ... use the previous sfr_input
                if i_cm == self.nb_timesteps:

                    # Calculate the total mass of the ISM using the previous SFR
                    self.m_tot_ISM_t[i_cm] = self.sfr_input[i_cm-1] * \
                        self.t_SF_t[i_cm] / self.sfe + self.m_crit_t[i_cm]

                # If it's not the last timestep ...
                else:

                    # Calculate the total mass of the ISM using the current SFR
                    self.m_tot_ISM_t[i_cm] = self.sfr_input[i_cm] * \
                        self.t_SF_t[i_cm] / self.sfe + self.m_crit_t[i_cm]

        # If the IO model ...
        elif self.in_out_control:
            self.m_tot_ISM_t[0] = self.mgal

        # Scale the initial gas reservoir that was already set
        scale_m_tot = self.m_tot_ISM_t[0] / np.sum(self.ymgal[0])
        for k_cm in range(len(self.ymgal[0])):
            self.ymgal[0][k_cm] = self.ymgal[0][k_cm] * scale_m_tot


    ##############################################
    #            Copy M_tot_ISM Input            #
    ##############################################
    def __copy_m_tot_ISM_input(self):

        '''
        This function interpolates the gas masses from an input array
        and add the masses to the corresponding OMEGA step

        '''

        # Variable to keep track of the OMEGA's timestep
        i_dt_csa = 0
        t_csa = 0.0
        nb_dt_csa = self.nb_timesteps + 1

        # If just one entry ...
        if len(self.m_tot_ISM_t_in) == 1:
            self.m_tot_ISM_t[i_dt_csa] = self.m_tot_ISM_t_in[0][1]
            i_dt_csa += 1

        # If DM values need to be interpolated ...
        else:

          # For every timestep given in the array (starting at the second step)
          for i_csa in range(1,len(self.m_tot_ISM_t_in)):

            # Calculate the DM interpolation coefficient
            a_gas = (self.m_tot_ISM_t_in[i_csa][1] - self.m_tot_ISM_t_in[i_csa-1][1]) / \
                    (self.m_tot_ISM_t_in[i_csa][0] - self.m_tot_ISM_t_in[i_csa-1][0])
            b_gas = self.m_tot_ISM_t_in[i_csa][1] - a_gas * self.m_tot_ISM_t_in[i_csa][0]

            # While we stay in the same time bin ...
            while t_csa <= self.m_tot_ISM_t_in[i_csa][0]:

                # Interpolate the SFR
                self.m_tot_ISM_t[i_dt_csa] = a_gas * t_csa + b_gas

                # Exit the loop if the array is full
                if i_dt_csa >= nb_dt_csa:
                    break

                # Calculate the new time
                t_csa += self.history.timesteps[i_dt_csa]
                i_dt_csa += 1

            # Exit the loop if the array is full
            if i_dt_csa >= nb_dt_csa:
                break

        # If the array has been read completely, but the DM array is
        # not full, fil the rest of the array with the last read value
        while i_dt_csa < nb_dt_csa:
            self.m_tot_ISM_t[i_dt_csa] = self.m_tot_ISM_t_in[-1][1]
            i_dt_csa += 1


    ##############################################
    #              Calculate Outflow             #
    ##############################################
    def __calculate_outflow_t(self):

        '''
        This function calculates the mass-loading factor and the mass of outflow
        at every timestep.

        '''

        # If the outflow rate is kept constant ...
        if self.outflow_rate >= 0.0:

            # Use the input value for each timestep
            for i_ceo in range(0, self.nb_timesteps):
                self.eta_outflow_t[i_ceo] = self.outflow_rate / \
                    self.sfr_input[i_ceo]
                self.m_outflow_t[i_ceo] = self.outflow_rate * \
                    self.history.timesteps[i_ceo]

        # If the outflow rate is connected to the SFR ...
        else:

            # If the mass of the dark matter halo is not evolving
            if not self.DM_evolution:

                #For each timestep ...
                for i_ceo in range(0, self.nb_timesteps):

                    # Use the input mass-loading factor
                    self.eta_outflow_t[i_ceo] = self.mass_loading
                    self.m_outflow_t[i_ceo] = self.eta_outflow_t[i_ceo] * \
                        self.sfr_input[i_ceo] * self.history.timesteps[i_ceo]

            # If the mass of the dark matter halo is evolving
            else:

                # Use the input mass-loading factor to normalize the evolution
                # of this factor as a function of time
                eta_norm = self.mass_loading * \
                    self.m_DM_0**(self.exp_ml*0.33333)* \
                        (1+self.redshift_f)**(0.5*self.exp_ml)

                # For each timestep ...
                for i_ceo in range(0, self.nb_timesteps):

                    # Calculate the mass-loading factor with redshift dependence
                    if self.z_dependent:
                        self.eta_outflow_t[i_ceo] = eta_norm * \
                            self.m_DM_t[i_ceo]**((-0.3333)*self.exp_ml) * \
                                (1+self.redshift_t[i_ceo])**(-(0.5)*self.exp_ml)

                    # Calculate the mass-loading factor without redshift dependence
                    else:
                        self.eta_outflow_t[i_ceo] = eta_norm * \
                            self.m_DM_t[i_ceo]**((-0.3333)*self.exp_ml)

                    # Calculate the outflow mass during the current timestep
                    self.m_outflow_t[i_ceo] = self.eta_outflow_t[i_ceo] * \
                        self.sfr_input[i_ceo] * self.history.timesteps[i_ceo]

        # Create the interpolation coefficients
        # eta = self.eta_outflow_t_coef[0] * t + self.eta_outflow_t_coef[1]
        self.eta_outflow_t_coef = np.zeros((self.nb_timesteps,2))
        for i_cmdt in range(self.nb_timesteps-1):
            self.eta_outflow_t_coef[i_cmdt][0] = (self.eta_outflow_t[i_cmdt+1] - \
                self.eta_outflow_t[i_cmdt]) / self.history.timesteps[i_cmdt]
            self.eta_outflow_t_coef[i_cmdt][1] = self.eta_outflow_t[i_cmdt] - \
                self.eta_outflow_t_coef[i_cmdt][0] * self.history.age[i_cmdt]
        self.eta_outflow_t_coef[-1][0] = self.eta_outflow_t_coef[-2][0]
        self.eta_outflow_t_coef[-1][1] = self.eta_outflow_t_coef[-2][1]


    ##############################################
    #                Add Ext. M_dot              #
    ##############################################
    def __add_ext_mdot(self):

        '''
        This function adds the stellar ejecta of external galaxies that
        just merged in the mdot array of the current galaxy.  This function
        assumes that the times and the number of timesteps of the merging
        galaxies are different from the current galaxy.

        Notes
        =====

            i_ext : Step index in the "external" merging mdot array
            i_cur : Step index in the "current" galaxy mdot array
            t_cur_prev : Lower time limit in the current i_cur bin
            t_cur : Upper time limit in the current i_cur bin

            M_dot_ini has an extra slot in the isotopes for the time,
            which is t = 0.0 for i_ext = 0.

        '''

        # For every merging galaxy (every branch of a merger tree)
        for i_merg in range(0,len(self.mdot_ini)):

            # Initialisation of the local variables
            i_ext = 0
            i_cur = 0
            t_cur_prev = 0.0
            t_cur = self.history.timesteps[0]
            t_ext_prev = 0.0
            t_ext = self.mdot_ini_t[i_merg][i_ext+1]

            # While the external ejecta has not been fully transfered...
            len_mdot_ini_i_merg = len(self.mdot_ini[i_merg])
            while i_ext < len_mdot_ini_i_merg and i_cur < self.nb_timesteps:

                # While we need to change the external time bin ...
                while t_ext <= t_cur:

                    # Calculate the overlap time between ext. and cur. bins
                    dt_trans = t_ext - max([t_ext_prev, t_cur_prev])

                    # Calculate the mass fraction that needs to be transfered
                    f_dt = dt_trans / (t_ext - t_ext_prev)

                    # Transfer all isotopes in the current mdot array
                    self.mdot[i_cur] += self.mdot_ini[i_merg][i_ext] * f_dt

                    # Move to the next external bin
                    i_ext += 1
                    if i_ext == (len_mdot_ini_i_merg):
                        break
                    t_ext_prev = t_ext
                    t_ext = self.mdot_ini_t[i_merg][i_ext+1]

                # Quit the loop if all external bins have been considered
                if i_ext == (len_mdot_ini_i_merg):
                    break

                # While we need to change the current time bin ...
                while t_cur < t_ext:

                    # Calculate the overlap time between ext. and cur. bins
                    dt_trans = t_cur - max([t_ext_prev, t_cur_prev])

                    # Calculate the mass fraction that needs to be transfered
                    f_dt = dt_trans / (t_ext - t_ext_prev)

                    # Transfer all isotopes in the current mdot array
                    self.mdot[i_cur] += self.mdot_ini[i_merg][i_ext] * f_dt

                    # Move to the next current bin
                    i_cur += 1
                    if i_cur == self.nb_timesteps:
                        break
                    t_cur_prev = t_cur
                    t_cur += self.history.timesteps[i_cur]


    ##############################################
    #                Run Simulation              #
    ##############################################
    def __run_simulation(self, mass_sampled=np.array([]), \
                         scale_cor=np.array([])):

        '''
        This function calculates the evolution of the chemical abundances of a
        galaxy as a function of time.

        Argument
        ========

          mass_sampled : Stars sampled in the IMF by an external program.
          scale_cor : Envelope correction for the IMF.

        '''

        # For every timestep i considered in the simulation ...
        for i in range(1, self.nb_timesteps+1):

            # If the IMF must be sampled ...
            if self.imf_rnd_sampling and self.m_pop_max >= \
               (self.sfr_input[i-1] * self.history.timesteps[i-1]):

                # Get the sampled masses
                mass_sampled = self._get_mass_sampled(\
                    self.sfr_input[i-1] * self.history.timesteps[i-1])

            # No mass sampled if using the full IMF ...
            else:
                mass_sampled = np.array([])

            # Run a timestep using the input SFR
            self.run_step(i, self.sfr_input[i-1], \
                mass_sampled=mass_sampled, scale_cor=scale_cor)

        # Calculate the last SFR at the end point of the simulation
        if self.cl_SF_law and not self.open_box:
            self.history.sfr_abs[-1] = self.sfe_gcs * np.sum(self.ymgal[i])


    ##############################################
    #                   Run Step                 #
    ##############################################
    def run_step(self, i, sfr_rs, m_added = np.array([]), m_lost = 0.0, \
                 no_in_out = False, f_esc_yields=0.0, mass_sampled=np.array([]),
                 scale_cor=np.array([])):

        '''
        This function calculates the evolution of one single step in the
        chemical evolution.

        Argument
        ========

          i : Index of the timestep.
          sfr_rs : Input star formation rate [Mo/yr] for the step i.
          m_added : Mass (and composition) added for the step i.
          m_lost : Mass lost for the step i.
          no_in_out : Cancel the open box "if" statement if True
          f_esc_yields: Fraction of non-contributing stellar ejecta
          mass_sampled : Stars sampled in the IMF by an external program.
          scale_cor : Envelope correction for the IMF.

        '''

        # Make sure that the the number of timestep is not exceeded
        if not i == (self.nb_timesteps+1):

            # For testing ..
            if i == 1:
                self.sfr_test = sfr_rs

            # Calculate the current mass fraction of gas converted into stars,
            # but only if the star formation rate is not followed
            # within a self-consistent integration scheme.
            if not self.use_external_integration:
                self.__cal_m_frac_stars(i, sfr_rs)
            else:
                self.sfrin = sfr_rs # [Msun/yr]
                self.m_locked = self.sfrin * self.history.timesteps[i-1]

            # Run the timestep i (!need to be right after __cal_m_frac_stars!)
            self._evol_stars(i, f_esc_yields, mass_sampled, scale_cor)

            # Decay radioactive isotopes
            if not self.use_external_integration:
                if self.use_decay_module:
                    self._decay_radio_with_module(i)
                elif self.len_decay_file > 0:
                    self._decay_radio(i)

            # Delay outflow is needed (following SNe rather than SFR) ...
            if self.out_follows_E_rate:
                self.__delay_outflow(i)

            # Add the incoming gas (if any)
            if not self.use_external_integration:
                len_m_added = len(m_added)
                for k_op in range(0, len_m_added):
                    self.ymgal[i][k_op] += m_added[k_op]

            # If no integration scheme is used to advance the system ..
            if not self.use_external_integration:

                # If gas needs to be removed ...
                if m_lost > 0.0:

                    # Calculate the gas fraction lost
                    f_lost = m_lost / sum(self.ymgal[i])
                    if f_lost > 1.0:
                        f_lost = 1.0
                        if not self.print_off:
                            print ('!!Warning -- Remove more mass than available!!')

                    # Remove the mass for each isotope
                    f_lost_2 = (1.0 - f_lost)
                    self.ymgal[i] = f_lost_2 * self.ymgal[i]

                    # Radioactive isotopes lost
                    if self.len_decay_file > 0 or self.use_decay_module:
                        self.ymgal_radio[i] = f_lost_2 * self.ymgal_radio[i]
                    if not self.pre_calculate_SSPs:
                        self.ymgal_agb[i] = f_lost_2 * self.ymgal_agb[i]
                        self.ymgal_1a[i] = f_lost_2 * self.ymgal_1a[i]
                        self.ymgal_nsm[i] = f_lost_2 * self.ymgal_nsm[i]
                        self.ymgal_massive[i] = f_lost_2 * self.ymgal_massive[i]
                        for iiii in range(0,self.nb_delayed_extra):
                            self.ymgal_delayed_extra[iiii][i] = \
                                f_lost_2 * self.ymgal_delayed_extra[iiii][i]
                        # Radioactive isotopes lost
                        if self.len_decay_file > 0 or self.use_decay_module:
                            if self.radio_massive_agb_on:
                                self.ymgal_massive_radio[i] = f_lost_2 * self.ymgal_massive_radio[i]
                                self.ymgal_agb_radio[i] = f_lost_2 * self.ymgal_agb_radio[i]
                            if self.radio_sn1a_on:
                                self.ymgal_1a_radio[i] = f_lost_2 * self.ymgal_1a_radio[i]
                            if self.radio_nsmerger_on:
                                self.ymgal_nsm_radio[i] = f_lost_2 * self.ymgal_nsm_radio[i]
                            for iiii in range(0,self.nb_delayed_extra_radio):
                                self.ymgal_delayed_extra_radio[iiii][i] = \
                                    f_lost_2 * self.ymgal_delayed_extra_radio[iiii][i]

                # If the open box scenario is used (and it is not skipped) ...
                if self.open_box and (not no_in_out):

                    # Calculate the total mass of the gas reservoir at timstep i
                    # after the star formation and the stellar ejecta
                    m_tot_current = sum(self.ymgal[i])

                    # Add inflows
                    if self.len_m_inflow_X_array > 0.0:
                        self.ymgal[i] += self.m_inflow_X_array[i-1]
                        m_inflow_current = self.m_inflow_array[i-1]
                        self.m_inflow_t[i-1] = float(m_inflow_current)
                    else:

                        # Get the current mass of inflow
                        m_inflow_current = self.__get_m_inflow(i, m_tot_current)

                        # Add primordial gas coming with the inflow
                        if m_inflow_current > 0.0:
                            ym_inflow = self.prim_comp.get(quantity='Yields', Z=0.0) * \
                                        m_inflow_current
                            for k_op in range(0, self.nb_isotopes):
                                self.ymgal[i][k_op] += ym_inflow[k_op]

                    # Calculate the fraction of gas removed by the outflow
                    if not (m_tot_current + m_inflow_current) == 0.0:
                        if self.len_m_gas_array > 0:
                            self.m_outflow_t[i-1] = (m_tot_current + m_inflow_current) - self.m_gas_array[i]
                            frac_rem = self.m_outflow_t[i-1] / (m_tot_current + m_inflow_current)
                            if frac_rem < 0.0:
                                frac_rem = 0.0
                                # Add primordial gas coming with the inflow
                                self.m_outflow_t[i-1] = 0.0
                                ym_inflow = self.prim_comp.get(quantity='Yields', Z=0.0) * \
                                            (-1.0) * self.m_outflow_t[i-1]
                                for k_op in range(0, self.nb_isotopes):
                                    self.ymgal[i][k_op] += ym_inflow[k_op]
                        else:
                            frac_rem = self.m_outflow_t[i-1] / \
                                (m_tot_current + m_inflow_current)
                    else:
                        frac_rem = 0.0

                    # Limit the outflow mass to the amount of available gas
                    if frac_rem > 1.0:
                        frac_rem = 1.0
                        self.m_outflow_t[i-1] = m_tot_current + m_inflow_current
                        if not self.print_off:
                            print ('Warning - '\
                              'Outflows eject more mass than available.  ' \
                              'It has been reduced to the amount of available gas.')

                    # Remove mass from the ISM because of the outflow
                    self.ymgal[i] *= (1.0 - frac_rem)
                    if self.len_decay_file > 0 or self.use_decay_module:
                        self.ymgal_radio[i]  *= (1.0 - frac_rem)
                    if not self.pre_calculate_SSPs:
                        self.ymgal_agb[i] *= (1.0 - frac_rem)
                        self.ymgal_1a[i] *= (1.0 - frac_rem)
                        self.ymgal_nsm[i] *= (1.0 - frac_rem)
                        self.ymgal_massive[i] *= (1.0 - frac_rem)
                        for iiii in range(0,self.nb_delayed_extra):
                            self.ymgal_delayed_extra[iiii][i] *= (1.0 - frac_rem)
                        # Radioactive isotopes lost
                        if self.len_decay_file > 0 or self.use_decay_module:
                            if self.radio_massive_agb_on:
                                self.ymgal_massive_radio[i] *= (1.0 - frac_rem)
                                self.ymgal_agb_radio[i] *= (1.0 - frac_rem)
                            if self.radio_sn1a_on:
                                self.ymgal_1a_radio[i] *= (1.0 - frac_rem)
                            if self.radio_nsmerger_on:
                                self.ymgal_nsm_radio[i] *= (1.0 - frac_rem)
                            for iiii in range(0,self.nb_delayed_extra_radio):
                                self.ymgal_delayed_extra_radio[iiii][i] *= (1.0 - frac_rem)

                # Get the new metallicity of the gas and update history class
                self.zmetal = self._getmetallicity(i)
                self._update_history(i)

                # If this is the last step ...
                if i == self.nb_timesteps:

                    # Do the final update of the history class
                    self._update_history_final()

                    # Add the evolution arrays to the history class
                    self.history.m_tot_ISM_t = self.m_tot_ISM_t
                    self.history.eta_outflow_t = self.eta_outflow_t

                    # If external control ...
                    if self.external_control:
                        self.history.sfr_abs[i] = self.history.sfr_abs[i-1]

                    # Calculate the total mass of gas
                    self.m_stel_tot = 0.0
                    for i_tot in range(0,len(self.history.timesteps)):
                        self.m_stel_tot += self.history.sfr_abs[i_tot] * \
                            self.history.timesteps[i_tot]
                    if self.m_stel_tot > 0.0:
                        self.m_stel_tot = 1.0 / self.m_stel_tot
                    self.f_m_stel_tot = []
                    m_temp = 0.0
                    for i_tot in range(0,len(self.history.timesteps)):
                        m_temp += self.history.sfr_abs[i_tot] * \
                            self.history.timesteps[i_tot]
                        self.f_m_stel_tot.append(m_temp*self.m_stel_tot)
                    self.f_m_stel_tot.append(self.f_m_stel_tot[-1])

                    # Announce the end of the simulation
                    print ('   OMEGA run completed -',self._gettime())

        # Error message
        else:
            print ('The simulation is already over.')


    ##############################################
    #              Get Mass Sampled              #
    ##############################################
    def _get_mass_sampled(self, m_pop):

        '''
        This function samples randomly the IMF using a Monte Carlo
        approach and returns an array with all masses sampled (not
        in increasing or decreasing orders).

        Argument
        ========

          m_pop : Mass of the considered stellar population

        '''

        # Initialization of the sampling arrays
        mass_sampled_gms = []
        m_tot_temp = 0.0

        # Define the sampling precision in Msun
        precision = 0.01 * m_pop * self.m_frac_massive_rdm

        # Copy the lower and upper mass limit of the IMF
        m_low_imf = self.transitionmass
        m_up_imf  = self.imf_bdys[1]
        dm_temp = m_up_imf - m_low_imf

        # While the total stellar mass is not formed ...
        while abs(m_tot_temp - m_pop) > precision:

            # Choose randomly a (m,nb) coordinate
            rand_m = m_low_imf + np.random.random_sample()*dm_temp
            rand_y = np.random.random_sample()

            # If the coordinate is below the IMF curve
            if rand_y <= (self.A_rdm * rand_m**(-2.3)):

                # Add the stellar mass only if it doesn't
                # form to much mass compared to m_pop
                if (m_tot_temp + rand_m) - m_pop <= precision:
                    mass_sampled_gms.append(rand_m)
                    m_tot_temp += rand_m

            # Stop if cannot fit a massive star
            if abs(m_tot_temp - m_pop) < self.transitionmass:
                break

        # Return the stellar masses sampled using Monte Carlo
        return mass_sampled_gms


    ##############################################
    #              Cal M Frac Stars              #
    ##############################################
    def __cal_m_frac_stars(self, i, sfr_rs):

        '''
        This function calculates the mass fraction of the gas reservoir that
        is converted into stars at a given timestep.

        Argument
        ========

          i : Index of the timestep.
          sfr_rs : Star formation rate [Mo/yr] for the timestep i

        '''

        # If the SFR is calculated from a star formation law (closed box)
        if self.cl_SF_law and not self.open_box:
            self.history.sfr_abs[i-1] = self.sfe_gcs * np.sum(self.ymgal[i-1])
            self.sfrin = self.history.sfr_abs[i-1] * self.history.timesteps[i-1]

        else:
            # Calculate the total mass of stars formed during this timestep
            self.sfrin = sfr_rs * self.history.timesteps[i-1]
            self.history.sfr_abs[i-1] = sfr_rs

        # Calculate the mass fraction of gas converted into stars
        mgal_tot = 0.0
        for k_ml in range(0, self.nb_isotopes):
            mgal_tot += self.ymgal[i-1][k_ml]
        if mgal_tot <= 0.0:
            self.sfrin = 0.0
        else:
            self.sfrin = self.sfrin / mgal_tot

        # Modify the history of SFR if there is not enough gas
        if self.sfrin > 1.0:
           self.history.sfr_abs[i-1] = mgal_tot / self.history.timesteps[i-1]


    ##############################################
    #                Delay Outflow               #
    ##############################################
    def __delay_outflow(self, i):

        '''
        This function convert the instantaneous outflow rate (vs SFR) into a delayed
        rate where Mout follows the number of CC SNe.

        Argument
        ========

          i : Index of the timestep.


        '''

        # Calculate the 1 / (total number of CC SNe in the SSP)
        if self.m_locked <= 0.0:
            nb_cc_sne_inv = 1.0e+30
        elif self.zmetal <= 0.0 and self.Z_trans > 0.0:
            nb_cc_sne_inv = 1.0 / (self.nb_ccsne_per_m_pop3 * self.m_locked)
        else:
            nb_cc_sne_inv = 1.0 / (self.nb_ccsne_per_m * self.m_locked)

        # Calculate the fraction of CC SNe in each future timesteps
        len_ssp_nb_cc_sne = len(self.ssp_nb_cc_sne)
        f_nb_cc = np.zeros(len_ssp_nb_cc_sne, np.float64)
        for i_nb_cc in range(0,len_ssp_nb_cc_sne):
            f_nb_cc[i_nb_cc] = self.ssp_nb_cc_sne[i_nb_cc] * nb_cc_sne_inv

        # Copy the original instanteneous mass outflow [Msun]
        m_out_inst = self.m_outflow_t_vs_SFR[i-1]

        # For each future timesteps including the current one ...
        for i_do in range(0,len_ssp_nb_cc_sne):

            # Add the delayed mass outflow
            #print (i, i_do, i+i_do, len(self.m_outflow_t))
            self.m_outflow_t[i-1+i_do] += m_out_inst * f_nb_cc[i_do]


    ##############################################
    #                Get M Inflow                #
    ##############################################
    def __get_m_inflow(self, i, m_tot_current):

        '''
        This function calculates and returns the inflow mass at a given timestep

        Argument
        ========

          i : Index of the timestep.
          m_tot_current : Total mass of the gas reservoir at step i

        '''

        # If an inflow mass in given at each timestep as an input ...
        if self.len_m_inflow_array > 0:
            m_inflow_current = self.m_inflow_array[i-1]

        # If the constant inflow rate is kept constant ...
        elif self.inflow_rate >= 0.0:

            # Use the input rate to calculate the inflow mass
            # Note : i-1 --> current timestep, see __copy_sfr_input()
            m_inflow_current = self.inflow_rate * self.history.timesteps[i-1]

        # If the inflow rate follows the outflow rate ...
        elif self.in_out_control:

            # Use the input scale factor to calculate the inflow mass
            if self.out_follows_E_rate:
                m_inflow_current = self.in_out_ratio * self.m_outflow_t_vs_SFR[i-1]
            else:
                m_inflow_current = self.in_out_ratio * self.m_outflow_t[i-1]

        # If the inflow rate is calculated from the main equation ...
        else:

            # If SFR = 0 and we do not want to use the main equation ..
            if self.sfr_input[i] == 0 and self.skip_zero:
                m_inflow_current = 0.0
            else:

                # Calculate the mass of the inflow
                m_inflow_current = self.m_tot_ISM_t[i] - \
                    m_tot_current + self.m_outflow_t[i-1]

                # If the inflow mass is negative ...
                if m_inflow_current < 0.0:

                    # Convert the negative inflow into positive outflow
                    if not self.skip_zero:
                        self.m_outflow_t[i-1] += (-1.0) * m_inflow_current
                        if not self.print_off:
                            print ('Warning - Negative inflow.  ' \
                              'The outflow rate has been increased.', i)

                    # Assume no inflow
                    m_inflow_current = 0.0

        # Keep the mass of inflow in memory
        self.m_inflow_t[i-1] = float(m_inflow_current)

        return m_inflow_current


###############################################################################################
######################## Here start the analysis methods ######################################
###############################################################################################

    ##############################################
    #              Get Isolation Time            #
    ##############################################
    def get_isolation_time(self, isotope, value, time_sun, reac_dictionary = None):

        '''
        Return the isolation time in years for the desired isotope
        or ratio of isotopes.

        Parameters
        ----------
        -isotope: A string with the isotope name or ratio. For
        example: 'Fe-60/Al-26'
        -value: The decayed value of the isotope or ratio.
        -time_sun: The galaxy time in years when the ISM value
        should be taken. This value is then decayed to value and
        the needed amount of time is the isolation time returned.

        '''

        # Build the reac_dictionary if it's not provided
        if reac_dictionary is None:
            if self.len_decay_file > 0:
                reac_dictionary = {}

                # The information stored in decay_info is...
                # decay_info[nb_radio_iso][0] --> Unstable isotope
                # decay_info[nb_radio_iso][1] --> Stable isotope where it decays
                # decay_info[nb_radio_iso][2] --> Mean-life (half-life/ln2)[yr]

                # Build the network
                for elem in self.decay_info:

                    # Get names for reaction
                    targ = elem[0]; prod = elem[1]; rate = 1 / elem[2]

                    # Add reaction, create a lambda object
                    reaction = lambda: None
                    reaction.target = targ
                    reaction.products = [prod]
                    reaction.rate = rate

                    if targ in reac_dictionary:
                        reac_dictionary[targ].append(reaction)
                    else:
                        reac_dictionary[targ] = [reaction]
            else:
                s = "This routine needs either a reac_dictionary passed from "
                s += "OMEGA+ or the decay information written in the decay file."
                print(s)
                return None

        # Check whether this is an isotope or a ratio of isotopes
        splt = isotope.split("/")
        if len(splt) == 1:
            isotope2 = None
        else:
            isotope, isotope2 = splt

        # Get indices and decay rates for isotopes
        stableList = list(self.history.isotopes)
        radioList = list(self.radio_iso)

        rate = 0
        if isotope in radioList:
            indx = radioList.index(isotope)
            gas = np.transpose(self.ymgal_radio)[indx]

            for reac in reac_dictionary[isotope]:
                rate += reac.rate

        elif isotope in stableList:
            indx = stableList.index(isotope)
            gas = np.transpose(self.ymgal)[indx]
        else:
            print("Isotope {} not in omega".format(isotope))
            return None

        rate2 = 0
        if isotope2 is not None:
            if isotope2 in radioList:
                indx2 = radioList.index(isotope2)
                gas2 = np.transpose(self.ymgal_radio)[indx2]

                for reac in reac_dictionary[isotope2]:
                    rate2 += reac.rate

            elif isotope2 in stableList:
                indx2 = stableList.index(isotope2)
                gas2 = np.transpose(self.ymgal)[indx2]
            else:
                print("Isotope {} not in omega".format(isotope2))
                return None

        # Calculate the equivalent rate
        # (doesn't matter what situation, always works)
        rateEq = rate - rate2

        # Look for the lowest time index
        timesArr = self.history.age
        for ii in range(len(self.history.age)):
            if timesArr[ii] >= time_sun:
                i1 = ii - 1
                i2 = ii
                break

        # Take the value of the isotopes at the time time_sun
        lt1 = np.log(timesArr[i1]); lt2 = np.log(timesArr[i2])
        lg1 = np.log(gas[i1]); lg2 = np.log(gas[i2])
        mm = (lg2 - lg1)/(lt2 - lt1)
        valueISM = np.exp(mm*(np.log(time_sun) - lt1) + lg1)
        if isotope2 is not None:
            lg1 = np.log(gas2[i1]); lg2 = np.log(gas2[i2])
            mm = (lg2 - lg1)/(lt2 - lt1)
            valueISM2 = np.exp(mm*(np.log(time_sun) - lt1) + lg1)

            valueISM /= valueISM2

        # Now just return the isolation time
        return -np.log(value/valueISM)/rateEq


#### trueman edits

    def mass_frac_plot(self,fig=0,species=['all'],sources=['agb','massive','1a'],\
                      cycle=-1, solar_ref='Asplund_et_al_2009',yscale='log'):

        '''
        fractional contribution from each stellar source towards the galactic total relative to solar

        Parameters
        ----------

        species : array of strings
             isotope or element name,
             e.g. ['H-1','He-4','Fe','Fe-56']
             default = ['all']
        sources : array of strings
             specifies the stellar sources to plot,
             e.g. ['agb','massive','1a']
        cycle : float
             specifies cycle number to plot,
             e.g. 'cycle=-1' will plot last cycle
        solar_ref : string
             the solar abundances used as a reference
             default is Asplund et al. 2009
             'Asplund_et_al_2009'
             'Anders_Grevesse_1989'
             'Grevesse_Noels_1993'
             'Grevesse_Sauval_1998'
             'Lodders_et_al_2009'
        yscale: string
             choose y axis scale
             'log' or 'linear'

        Examples
        ---------

        >>> s.plot(['all']['agb','massive','1a'],
               cycle=-1, solar_ref='Lodders', yscale='log')

        '''

        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        f = open(os.path.join(nupy_path, 'stellab_data',\
            'solar_normalization', str(solar_ref) + '.txt'), 'r')

        g = open(os.path.join(nupy_path, 'stellab_data',\
        'solar_normalization', 'element_mass.txt'), 'r')

        h = open(os.path.join(nupy_path, 'stellab_data',\
        'solar_normalization', 'Asplund_et_al_2009_iso.txt'), 'r')

        lines=f.readlines()
        lines_g=g.readlines()
        lines_h=h.readlines()
        ele_mass = []
        ele_nam = []
        abu_sol = []
        ele_sol = []
        iso_nam =[]
        iso_frac =[]

        # items taken from Asplund
        # keys = element symbol, values = logarithmic solar abundnace
        for i in lines:
            ele_sol.append(i.split()[1])
            abu_sol.append(float(i.split()[2]))
        f.close()
        sol_dict = dict(zip(ele_sol, abu_sol))

        # items taken from online data table
        # keys = element symbol, values = element mass number
        for j in lines_g:
            ele_mass.append(float(j.split()[0]))
            ele_nam.append(j.split()[2])
        g.close()
        ele_dict = dict(zip(ele_nam, ele_mass))

        # items taken from Asplund
        # keys = isotope symbol, values = relative number fraction of isotope
        for k in lines_h:
            iso_nam.append(k.split()[0])
            iso_frac.append(float(k.split()[1])/100)
        h.close()
        iso_frac_dict = dict(zip(iso_nam, iso_frac))

        # Create a dictionary with keys = element symbol
        # and vals = solar mass fraction
        ele_mass_frac = {}
        for ele,mass in ele_dict.items():
            for el,abu in sol_dict.items():
                if ele == el:
                    ele_mass_frac.update([(ele,10**(abu-12)*mass*0.7381)])

        # Normalise the above dictionary so that mass fractions
        # sum to unity
        tot_mass_frac = sum(ele_mass_frac.values())
        for ele,frac in ele_mass_frac.items():
            sol_dict.update([(ele,frac/tot_mass_frac)])

        # Create a dictionary with keys = isotope
        # vals = (mass fraction)/(isotope mass)
        new = {}
        for ele,mass in ele_dict.items():
            for iso,frac in iso_frac_dict.items():
                if ele == iso.split('-',1)[0]:
                    new.update([(iso,frac/mass)])

        # Create a dictionary with keys = isotope
        # vals = contribution towards total element mass fraction from each isotope
        weighted_iso_frac={}
        for ele,frac in sol_dict.items():
            for iso,fracs in new.items():
                if ele == iso.split('-',1)[0]:
                    weighted_iso_frac.update([
        (iso,frac*fracs*float(iso.split('-',1)[-1]))])

        species_mass_frac_sol_dict = weighted_iso_frac
        species_mass_frac_sol_dict.update(sol_dict)

        # Remove species which have no solar mass data
        remove_keys = []
        for key,val in species_mass_frac_sol_dict.items():
            if val < 10e-30:
                remove_keys.append(key)

        for i in remove_keys:
            if i in species_mass_frac_sol_dict:
                del species_mass_frac_sol_dict[i]

        iso_mass_gal = dict(zip(self.history.isotopes, self.ymgal[cycle]))
        ele_dum=[]
        for iso,mass in iso_mass_gal.items(): # create a list of the elements
            ele = (iso.split('-',1)[0])       # from list of isotopes
            ele_dum.append(ele)

        elements = np.unique(ele_dum)
        ele_mass_gal = np.zeros(len(elements))
        i=0

        # add the mass contribution from each isotope to
        # make the total element mass
        for el in elements:
            for iso,mass in iso_mass_gal.items():
                mass = float(mass)
                if el == iso.split('-', 1)[0]:
                    ele_mass_gal[i] += mass
            i+=1

        # create a dictionary which has keys = element/isotope and
        # vals = mass of species
        ele_mass_gal_dict = dict(zip(elements, ele_mass_gal))
        species_mass_gal = iso_mass_gal
        species_mass_gal.update(ele_mass_gal_dict)

        iso_mass_agb = dict(zip(self.history.isotopes, self.ymgal_agb[cycle]/sum(self.ymgal[cycle])))

        ele_mass_agb = np.zeros(len(elements))
        i=0

        for el in elements:
            for iso,mass in iso_mass_agb.items():
                mass = float(mass)
                if el == iso.split('-',1)[0]:
                    ele_mass_agb[i] += mass
            i+=1
        ele_mass_agb_dict = dict(zip(elements, ele_mass_agb))
        species_mass_agb = iso_mass_agb
        species_mass_agb.update(ele_mass_agb_dict)

        # This dictionary contains the mass fraction contribution toward the total
        # by AGB's for each species
        species_frac_agb = {k: species_mass_agb[k] / species_mass_gal[k]
        for k in species_mass_agb if k in species_mass_gal}

        iso_mass_massive = dict(zip(self.history.isotopes, self.ymgal_massive[cycle]/sum(self.ymgal[cycle])))

        ele_mass_massive = np.zeros(len(elements))
        i=0

        for el in elements:
            for iso,mass in iso_mass_massive.items():
                mass = float(mass)
                if el == iso.split('-',1)[0]:
                    ele_mass_massive[i] += mass
            i+=1
        ele_mass_massive_dict = dict(zip(elements, ele_mass_massive))
        species_mass_massive = iso_mass_massive
        species_mass_massive.update(ele_mass_massive_dict)

        # This dictionary contains the mass fraction contribution toward the total
        # by SN1a's for each species
        species_frac_massive = {k: species_mass_massive[k] / species_mass_gal[k]
        for k in species_mass_massive if k in species_mass_gal}

        iso_mass_1a = dict(zip(self.history.isotopes, self.ymgal_1a[cycle]/sum(self.ymgal[cycle])))

        ele_mass_1a = np.zeros(len(elements))
        i=0

        for el in elements:
            for iso,mass in iso_mass_1a.items():
                mass = float(mass)
                if el == iso.split('-',1)[0]:
                    ele_mass_1a[i] += mass
            i+=1
        ele_mass_1a_dict = dict(zip(elements, ele_mass_1a))
        species_mass_1a = iso_mass_1a
        species_mass_1a.update(ele_mass_1a_dict)

        # This dictionary contains the mass fraction contribution toward the total
        # by SN1a's for each species
        species_frac_1a = {k: species_mass_1a[k] / species_mass_gal[k]
        for k in species_mass_1a if k in species_mass_gal}

        map_str_dic = {
        "agb":species_mass_agb,
        "1a":species_mass_1a,
        "massive":species_mass_massive,
        }

        source_proper_name = {
        "agb":'AGB',
        "1a":'SN1a',
        "massive":'Massive Stars',
        }

        colors = ['blue', 'orange', 'grey','navy','green']

        if species == ['all']:
            species = species_mass_frac_sol_dict.keys()

        bar_bottom=[]
        for spec in species:
            bar_bottom.append(0)

        h=0
        j=0
        labels=[]
        legend_elements=[]
        for source in sources:
            labels.append(source_proper_name[source])
            legend_elements.append(Patch(facecolor=colors[j],
                                    label=labels[j]))
            ii=0
            for spe in species:
                if spe in species_mass_agb:
                    source_frac = map_str_dic[source][spe]
                    val = source_frac/species_mass_frac_sol_dict[spe]
                    plt.bar(spe, val,bottom=bar_bottom[ii], color=colors[j])
                    bar_bottom[ii]+=val
                    ii+=1
            j+=1
        plt.legend(handles=legend_elements)
        plt.axhline(y=1, ls='--', color='k')
        plt.title('Galaxy Age: '+str(round(self.history.age[cycle]/10**6, 1))+' Myr')
        plt.xticks(rotation=90, fontsize=12)
        plt.yscale(yscale)
        plt.ylabel('$X/X_{\odot}$')
        plt.tick_params(right=True)


    def plot_mass(self,fig=0,specie='C',source='all',norm=False,label='',shape='',marker='',color='',markevery=20,multiplot=False,return_x_y=False,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14,show_legend=True):

        '''
        mass evolution (in Msun) of an element or isotope vs time.


        Parameters
        ----------


        specie : string
             isotope or element name, in the form 'C' or 'C-12'
        source : string
             Specifies if yields come from
             all sources ('all'), including
             AGB+SN1a, massive stars. Or from
             distinctive sources:
             only agb stars ('agb'), only
             SN1a ('SN1a'), or only massive stars
             ('massive')
        norm : boolean
             If True, normalize to current total ISM mass
        label : string
             figure label
        marker : string
             figure marker
        shape : string
             line style
        color : string
             color of line
        fig : string,float
             to name the plot figure
        show_legend : boolean
             Default True. Show or not the legend

        Examples
        ----------

        >>> s.plot('C-12')

        '''

        import matplotlib
        import matplotlib.pyplot as plt

        yaxis=specie
        if len(label)<1:
            label=yaxis
            if source=='agb':
                label=yaxis+', AGB'
            if source=='massive':
                label=yaxis+', Massive'
            if source=='sn1a':
                label=yaxis+', SNIa'

        #Reserved for plotting
        if not return_x_y:
            shape,marker,color=self.__msc(source,shape,marker,color)

        x=self.history.age
        self.x=x
        y=[]
        yields_evol_all=self.history.ism_elem_yield
        if yaxis in self.history.elements:
            if source == 'all':
                yields_evol=self.history.ism_elem_yield
            elif source =='agb':
                yields_evol=self.history.ism_elem_yield_agb
            elif source == 'sn1a':
                yields_evol=self.history.ism_elem_yield_1a
            elif source == 'massive':
                yields_evol=self.history.ism_elem_yield_massive
            idx=self.history.elements.index(yaxis)
        elif yaxis in self.history.isotopes:
            if source == 'all':
                yields_evol=self.history.ism_iso_yield
            elif source =='agb':
                yields_evol=self.history.ism_iso_yield_agb
            elif source == 'sn1a':
                yields_evol=self.history.ism_iso_yield_1a
            elif source == 'massive':
                yields_evol=self.history.ism_iso_yield_massive
            idx=self.history.isotopes.index(yaxis)
        else:
            print ('Isotope or element not available')
            return 0
        for k in range(0,len(yields_evol)):
            if norm == False:
                y.append(yields_evol[k][idx])
            else:
                y.append( yields_evol[k][idx]/yields_evol_all[k][idx])


        x=x[1:]
        y=y[1:]
        if multiplot==True:
                return x,y

        #Reserved for plotting
        if not return_x_y:
           plt.figure(fig, figsize=(fsize[0],fsize[1]))
           plt.xscale('log')
           plt.xlabel('log-scaled age [yrs]')
           if norm == False:
               plt.ylabel('yields [Msun]')
               plt.yscale('log')
           else:
               plt.ylabel('(IMF-weighted) fraction of ejecta')
        self.y=y

        #If x and y need to be returned ...
        if return_x_y:
            return x, y

        else:
            if show_legend:
                plt.plot(x,y,label=label,linestyle=shape,marker=marker,color=color,markevery=markevery)
                plt.legend()
            else:
                plt.plot(x,y,linestyle=shape,marker=marker,color=color,markevery=markevery)
            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)
            plt.xlim(self.history.dt,self.history.tend)
            #return x,y
            #self.save_data(header=['Age[yrs]',specie],data=[x,y])


    def plot_massfrac(self,fig=2,xaxis='age',yaxis='O-16',source='all',norm='no',label='',shape='',marker='',color='',markevery=20,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14):

        '''
        Plots mass fraction of isotope or element
        vs either time or other isotope or element.

        Parameters
        ----------
        xaxis : string
            either 'age' for time
            or isotope name, in the form e.g. 'C-12'
        yaxis : string
            isotope name, in the same form as for xaxis

        source : string
            Specifies if yields come from
            all sources ('all'), including
            AGB+SN1a, massive stars. Or from
            distinctive sources:
            only agb stars ('agb'), only
            SN1a ('SN1a'), or only massive stars
            ('massive')

        norm : string
            if 'no', no normalization of mass fraction
            if 'ini', normalization ot initial abundance
        label : string
             figure label
        marker : string
             figure marker
        shape : string
             line style
        color : string
             color of line
        fig : string,float
             to name the plot figure


        Examples
        ----------

        >>> s.plot_massfrac('age','C-12')

        '''

        import matplotlib
        import matplotlib.pyplot as plt

        if len(label)<1:
                label=yaxis

        shape,marker,color=self.__msc(source,shape,marker,color)

        plt.figure(fig, figsize=(fsize[0],fsize[1]))

        #Input X-axis
        if '-' in xaxis:
        #to test the different contributions
            if source == 'all':
               yields_evol=self.history.ism_iso_yield
            elif source =='agb':
               yields_evol=self.history.ism_iso_yield_agb
            elif source == 'sn1a':
               yields_evol=self.history.ism_iso_yield_1a
            elif source == 'massive':
               yields_evol=self.history.ism_iso_yield_massive
               iso_idx=self.history.isotopes.index(xaxis)
            x=[]
            for k in range(1,len(yields_evol)):
               if norm=='no':
                   x.append(yields_evol[k][iso_idx]/np.sum(yields_evol[k]))
               if norm=='ini':
                   x.append(yields_evol[k][iso_idx]/np.sum(yields_evol[k])/yields_evol[0][iso_idx])
            plt.xlabel('log-scaled X('+xaxis+')')
            plt.xscale('log')
        elif 'age' == xaxis:
            x=self.history.age#[1:]
            plt.xscale('log')
            plt.xlabel('log-scaled Age [yrs]')
        elif 'Z' == xaxis:
            x=self.history.metallicity#[1:]
            plt.xlabel('ISM metallicity')
            plt.xscale('log')
        elif xaxis in self.history.elements:
            if source == 'all':
                yields_evol=self.history.ism_elem_yield
            elif source =='agb':
                yields_evol=self.history.ism_elem_yield_agb
            elif source == 'sn1a':
                yields_evol=self.history.ism_elem_yield_1a
            elif source == 'massive':
                yields_evol=self.history.ism_elem_yield_massive
            iso_idx=self.history.elements.index(xaxis)
            x=[]
            for k in range(1,len(yields_evol)):
                if norm=='no':
                    x.append(yields_evol[k][iso_idx]/np.sum(yields_evol[k]))
                if norm=='ini':
                    x.append(yields_evol[k][iso_idx]/np.sum(yields_evol[k])/yields_evol[0][iso_idx])
                    print (yields_evol[0][iso_idx])

            plt.xlabel('log-scaled X('+xaxis+')')
            plt.xscale('log')


        #Input Y-axis
        if '-' in yaxis:
            #to test the different contributions
            if source == 'all':
                yields_evol=self.history.ism_iso_yield
            elif source =='agb':
                yields_evol=self.history.ism_iso_yield_agb
            elif source == 'sn1a':
                yields_evol=self.history.ism_iso_yield_1a
            elif source == 'massive':
                yields_evol=self.history.ism_iso_yield_massive
            iso_idx=self.history.isotopes.index(yaxis)
            y=[]
            #change of xaxis array 'age 'due to continue statement below
            if xaxis=='age':
                x_age=x
                x=[]
            for k in range(1,len(yields_evol)):
                if np.sum(yields_evol[k]) ==0:
                    continue
                if norm=='no':
                    y.append(yields_evol[k][iso_idx]/np.sum(yields_evol[k]))
                if norm=='ini':
                    y.append(yields_evol[k][iso_idx]/np.sum(yields_evol[k])/yields_evol[0][iso_idx])

                x.append(x_age[k])
            plt.ylabel('X('+yaxis+')')
            self.y=y
            plt.yscale('log')
        elif 'Z' == yaxis:
            y=self.history.metallicity
            plt.ylabel('ISM metallicity')
            plt.yscale('log')
        elif yaxis in self.history.elements:
            if source == 'all':
                yields_evol=self.history.ism_elem_yield
            elif source =='agb':
                yields_evol=self.history.ism_elem_yield_agb
            elif source == 'sn1a':
                yields_evol=self.history.ism_elem_yield_1a
            elif source == 'massive':
                yields_evol=self.history.ism_elem_yield_massive
            iso_idx=self.history.elements.index(yaxis)
            y=[]
            #change of xaxis array 'age 'due to continue statement below
            if xaxis=='age':
                x_age=x
                x=[]
            for k in range(1,len(yields_evol)):
                if np.sum(yields_evol[k]) ==0:
                    continue
                if norm=='no':
                    y.append(yields_evol[k][iso_idx]/np.sum(yields_evol[k]))
                if norm=='ini':
                    y.append(yields_evol[k][iso_idx]/np.sum(yields_evol[k])/yields_evol[0][iso_idx])

                x.append(x_age[k])
            #plt.yscale('log')
            plt.ylabel('X('+yaxis+')')
            self.y=y
            plt.yscale('log')
        #To prevent 0 +log scale
        if 'age' == xaxis:
                x=x[1:]
                y=y[1:]
                plt.xlim(self.history.dt,self.history.tend)
        plt.plot(x,y,label=label,linestyle=shape,marker=marker,color=color,markevery=markevery)
        plt.legend()
        ax=plt.gca()
        self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)
        self.save_data(header=[xaxis,yaxis],data=[x,y])

    def plot_spectro(self,fig=3,xaxis='age',yaxis='[Fe/H]',source='all',label='',shape='-',marker='o',color='k',markevery=100,show_data=False,show_sculptor=False,show_legend=True,return_x_y=False,sub_plot=False,linewidth=3,sub=1,plot_data=False,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14,only_one_iso=False,solar_ab='',sfr_thresh=0.0,m_formed_thresh=1.0,solar_norm=''):
        '''
        Plots elements in spectroscopic notation:

        Parameters
        ----------

        xaxis : string
            Elements spectroscopic notation e.g. [Fe/H]
            if 'age': time evolution in years
        yaxis : string
                Elements in spectroscopic notation, e.g. [C/Fe]
        source : string
                If yields come from
                all sources use 'all' (include
                AGB+SN1a, massive stars.)

                If yields come from distinctive source:
                only agb stars use 'agb', only
                SN1a ('SN1a'), or only massive stars
                ('massive')

        label : string
             figure label
        marker : string
             figure marker
        shape : string
             line style
        color : string
             color of line
        fig : string,float
             to name the plot figure

        Examples
        ----------
        >>> plt.plot_spectro('[Fe/H]','[C/Fe]')

        '''

        import matplotlib
        import matplotlib.pyplot as plt

        #Error message if there is the "subplot" has not been provided
        if sub_plot and sub == 1:
            print ('!! Error - You need to use the \'sub\' parameter and provide the frame for the plot !!')
            return

        #Operations associated with plot visual aspects
        if not return_x_y and not sub_plot:

            if len(label)<1:
                    label=yaxis

            shape,marker,color=self.__msc(source,shape,marker,color)

        #Set the figure output
        if sub_plot:
            plt_ps = sub
        else:
            plt_ps = plt


        #Access solar abundance
        if len(solar_ab) > 0:
            iniabu=ry.iniabu(os.path.join(nupy_path, solar_ab))
        else:
            iniabu=ry.iniabu(os.path.join(nupy_path, 'yield_tables', 'iniabu',\
                    'iniab2.0E-02GN93.ppn'))

        # If a solar normalization is used ..
        if len(solar_norm) > 0:

            # Mass number of the most abundant isotopes
            el_norm = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',\
                       'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',\
                       'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',\
                       'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag',\
                       'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce',\
                       'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',\
                       'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',\
                       'Pb', 'Bi', 'Th', 'U']
            mn_norm = [1.0000200000000001, 3.999834, 6.924099999999999, 9.0, 10.801,\
                       12.011061999999999, 14.00229, 16.004379000000004, 19.0, 20.13891,\
                       23.0, 24.3202, 27.0, 28.108575891424106, 31.0, 32.094199999999994,\
                       35.4844, 36.30859999999999, 39.13588999999999, 40.115629999999996,\
                       45.0, 47.9183, 50.9975, 52.05541, 55.0, 55.90993, 59.0, 58.759575,\
                       63.6166, 65.4682, 69.79784000000001, 72.69049999999999, 75.0, 79.0421,\
                       79.9862, 83.88084, 85.58312, 87.71136299999999, 89.0, 91.31840000000001,\
                       93.0, 96.05436999999998, 101.1598, 103.0, 106.5111, 107.96322,\
                       112.508, 114.9142, 118.8077, 121.85580000000002, 127.69839999999999,\
                       127.0, 131.28810000000001, 133.0, 137.42162, 138.99909000000002,\
                       140.20986, 141.0, 144.33351666483335, 150.4481, 152.0438, 157.3281,\
                       159.0, 162.57152999999997, 165.0, 167.32707000000002, 169.0,\
                       173.11618838116186, 175.02820514102572, 178.54163, 180.99988000000002,\
                       183.89069999999998, 186.28676, 190.28879711202887, 192.254,\
                       195.11347999999998, 197.0, 200.6297, 204.40952, 207.34285, 209.0,\
                       232.0, 237.27134]

            # Solar values
            sol_values_el = []
            sol_values_ab = []

            # Open the data file
            with open(os.path.join(nupy_path, 'stellab_data',\
                    'solar_normalization', solar_norm + '.txt'), 'r') as data_file:

                # For every line (for each element) ...
                i_elem = 0
                for line_1_str in data_file:

                    # Split the line
                    split = [str(x_split) for x_split in line_1_str.split()]

                    # Copy the element and the solar value
                    sol_values_el.append(str(split[1]))
                    sol_values_ab.append(float(split[2]))

                    # Go to the next element
                    i_elem += 1

            # Close the file
            data_file.close()

        x_ini_iso=iniabu.iso_abundance(self.history.isotopes)
        elements = self.history.elements
        x_ini = self._iso_abu_to_elem(x_ini_iso)

        #to test the different contributions
        if source == 'all':
            yields_evol=self.history.ism_elem_yield
        elif source =='agb':
            yields_evol=self.history.ism_elem_yield_agb
        elif source == 'sn1a':
            yields_evol=self.history.ism_elem_yield_1a
        elif source == 'massive':
            yields_evol=self.history.ism_elem_yield_massive

        #Operations associated with plot visual aspects
        if not return_x_y and not sub_plot:
            plt.figure(fig, figsize=(fsize[0],fsize[1]))


        if 'age' == xaxis:
            x=self.history.age

            #Operations associated with plot visual aspects
            if not return_x_y and not sub_plot:
                #plt.xscale('log')
                plt.xlabel('Age [yr]')

            self.x=x
        else:

            xaxis_elem1=xaxis.split('/')[0][1:]
            xaxis_elem2=xaxis.split('/')[1][:-1]

            #print (xaxis_elem1, xaxis_elem2)

            #X-axis ini values
            x_elem1_ini=x_ini[elements.index(xaxis_elem1)]
            x_elem2_ini=x_ini[elements.index(xaxis_elem2)]

            #X-axis gce values
            elem_idx1=self.history.elements.index(xaxis_elem1)
            elem_idx2=self.history.elements.index(xaxis_elem2)

            # Take only 56-Fe is needed
            if only_one_iso:
                if yaxis_elem1 == 'Fe':
                    yields_evol = self.history.ism_iso_yield
                    elem_idx1 = self.history.isotopes.index('Fe-56')

            x=[]
            for k in range(0,len(yields_evol)):
                if np.sum(yields_evol[k]) ==0:
                    continue
                #in case no contribution during timestep
                x1=yields_evol[k][elem_idx1]/np.sum(yields_evol[k])
                x2=yields_evol[k][elem_idx2]/np.sum(yields_evol[k])
                if x1 <= 0.0 or x2 <= 0.0:
                    spec = -30.0
                else:
                  if len(solar_norm) > 0:
                      index_1 = el_norm.index(xaxis_elem1)
                      index_2 = el_norm.index(xaxis_elem2)
                      i_xx_sol_1 = sol_values_el.index(xaxis_elem1)
                      i_xx_sol_2 = sol_values_el.index(xaxis_elem2)
                      aaa = np.log10(yields_evol[k][elem_idx1]*mn_norm[index_2] /\
                            (yields_evol[k][elem_idx2]*mn_norm[index_1]))
                      bbb = sol_values_ab[i_xx_sol_1] - sol_values_ab[i_xx_sol_2]
                      spec= aaa - bbb
                  else:
                    spec=np.log10(x1/x2) - np.log10(x_elem1_ini/x_elem2_ini)
                x.append(spec)
            #Operations associated with plot visual aspects
            if not return_x_y and not sub_plot:
                plt.xlabel(xaxis)
            self.x=x


        yaxis_elem1=yaxis.split('/')[0][1:]
        yaxis_elem2=yaxis.split('/')[1][:-1]

        #y=axis ini values
        x_elem1_ini=x_ini[elements.index(yaxis_elem1)]
        x_elem2_ini=x_ini[elements.index(yaxis_elem2)]
        #print ('Fe_sol = ',x_elem1_ini,' , H_sol = ',x_elem2_ini)

        #Y-axis gce values
        elem_idx1=self.history.elements.index(yaxis_elem1)
        elem_idx2=self.history.elements.index(yaxis_elem2)

        # Take only 56-Fe is needed
        if only_one_iso:
            if yaxis_elem1 == 'Fe':
                yields_evol = self.history.ism_iso_yield
                elem_idx1 = self.history.isotopes.index('Fe-56')
            elif yaxis_elem2 == 'Fe':
                yields_evol = self.history.ism_iso_yield
                elem_idx2 = self.history.isotopes.index('Fe-56')

        y=[]
        if xaxis=='age':
            x_age=x
            x=[]
        for k in range(0,len(yields_evol)):
            if np.sum(yields_evol[k]) ==0:
                continue
            #in case no contribution during timestep
            x1=yields_evol[k][elem_idx1]/np.sum(yields_evol[k])
            x2=yields_evol[k][elem_idx2]/np.sum(yields_evol[k])
            if x1 <= 0.0 or x2 <= 0.0:
                spec = -30.0
            else:
                #print ('Fe_sim = ',x1, ' , H_sim = ',x2)
                if len(solar_norm) > 0:
                    index_1 = el_norm.index(yaxis_elem1)
                    index_2 = el_norm.index(yaxis_elem2)
                    i_xx_sol_1 = sol_values_el.index(yaxis_elem1)
                    i_xx_sol_2 = sol_values_el.index(yaxis_elem2)
                    aaa = np.log10(yields_evol[k][elem_idx1]*mn_norm[index_2] /\
                          (yields_evol[k][elem_idx2]*mn_norm[index_1]))
                    bbb = sol_values_ab[i_xx_sol_1] - sol_values_ab[i_xx_sol_2]
                    spec= aaa - bbb
                else:
                  spec=np.log10(x1/x2) - np.log10(x_elem1_ini/x_elem2_ini)
            y.append(spec)
            if xaxis=='age':
                x.append(x_age[k])
        if len(y)==0:
            print ('Y values all zero')
        #Operations associated with plot visual aspects
        if not return_x_y and not sub_plot:
            plt.ylabel(yaxis)
        self.y=y
        #To prevent 0 +log scale
        if 'age' == xaxis:
            x=x[1:]
            y=y[1:]
            #Operations associated with plot visual aspects
            if not return_x_y and not sub_plot:
                plt.xlim(self.history.dt,self.history.tend)

        #Remove values when SFR is zero, that is when no element is locked into stars
        i_rem = len(y) - 1
        if m_formed_thresh < 1.0 or sfr_thresh > 0.0:
            while self.history.sfr_abs[i_rem] <= sfr_thresh or \
               self.f_m_stel_tot[i_rem] >= m_formed_thresh:
                del y[-1]
                del x[-1]
                i_rem -= 1
        else:
            while self.history.sfr_abs[i_rem] == 0.0:
                del y[-1]
                del x[-1]
                i_rem -= 1

        if not plot_data:

          # Filtrate bad value
          x_temp = []
          y_temp = []
          for i_temp in range(0,len(x)):
              if np.isfinite(x[i_temp]) and np.isfinite(y[i_temp])\
                 and x[i_temp] > -20.0 and y[i_temp] > -20.0:
                  x_temp.append(x[i_temp])
                  y_temp.append(y[i_temp])
          x = x_temp
          y = y_temp

          #If this function is supposed to return the x, y arrays only ...
          if return_x_y:

            return x, y

          #If this is a sub-figure managed by an external module
          elif sub_plot:

            if self.galaxy == 'none':
                if show_legend:
                    sub.plot(x,y,linestyle=shape,label=label,marker=marker,color=color,markevery=markevery,linewidth=linewidth)
                else:
                    sub.plot(x,y,linestyle=shape,marker=marker,color=color,markevery=markevery,linewidth=linewidth)
            else:
                if show_legend:
                    sub.plot(x,y,linestyle=shape,label=label,marker=marker,color=color,markevery=markevery,linewidth=linewidth)
                else:
                    sub.plot(x,y,linestyle=shape,marker=marker,color=color,markevery=markevery,linewidth=linewidth)

          #If this function is supposed to plot ...
          else:

            #Plot a thicker line for specific galaxies, since they must be visible with all the obs. data
            if self.galaxy == 'none':
                if show_legend:
                    plt.plot(x,y,linestyle=shape,label=label,marker=marker,color=color,markevery=markevery,linewidth=linewidth)
                else:
                    plt.plot(x,y,linestyle=shape,marker=marker,color=color,markevery=markevery,linewidth=linewidth)
            else:
                if show_legend:
                    plt.plot(x,y,linestyle=shape,label=label,marker=marker,color=color,markevery=markevery,linewidth=linewidth)
                else:
                    plt.plot(x,y,linestyle=shape,marker=marker,color=color,markevery=markevery,linewidth=linewidth)

            #plt.plot([1.93,2.123123,3.23421321321],[4.123123132,5.214124142,6.11111],linestyle='--')
            #plt.plot([1.93,2.123123,3.23421321321],[4.123123132,5.214124142,6.11111],linestyle='--')
            if len(label)>0:
                plt.legend()
            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)
            #self.save_data(header=[xaxis,yaxis],data=[x,y])


    def plot_totmasses(self,fig=4,source='all',norm='no',label='',shape='',marker='',color='',markevery=20,log=True,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14):
        '''
        Plots gas mass in fraction of total mass vs time.

        Parameters
        ----------

        norm : string
            normalization, either 'no' for no normalization (total gass mass in solar masses),

            for normalization to the initial gas mass (mgal) with 'ini',
            for normalization to the current total gas mass 'current'.
            The latter case makes sense when comparing different
            sources (see below)

        source : string
            specificies if yields come from
            all sources ('all'), including
            AGB+SN1a, massive stars. Or from
            distinctive sources:
            only agb stars ('agb'), only
            SN1a ('sn1a'), or only massive stars
            ('massive')
        log : boolean
            if true plot logarithmic y axis
        label : string
             figure label
        marker : string
             figure marker
        shape : string
             line style
        color : string
             color of line
        fig : string,float
             to name the plot figure

        Examples
        ----------
        >>> s.plot_totmasses()


        '''

        import matplotlib
        import matplotlib.pyplot as plt

        #if len(label)<1:
        #        label=mass+', '+source


        plt.figure(fig, figsize=(fsize[0],fsize[1]))

        #Assume isotope input

        xaxis='age'
        if source =='all':
            if len(label)==0:
                        label='All'
        if source == 'agb':
            if len(label)==0:
                        label='AGB'
        if source =='massive':
            if len(label)==0:
                        label='Massive'
        if source =='sn1a':
            if len(label)==0:
                label='SNIa'

        shape,marker,color=self.__msc(source,shape,marker,color)

        if 'age' == xaxis:
            x_all=self.history.age#[1:]
            plt.xscale('log')
            plt.xlabel('log-scaled '+xaxis+' [yrs]')
            #self.x=x

        gas_mass=self.history.gas_mass

        #to test the different contributions
        if source == 'all':
            gas_evol=self.history.gas_mass
        else:
            if source =='agb':
                yields_evol=self.history.ism_elem_yield_agb
            elif source == 'sn1a':
                yields_evol=self.history.ism_elem_yield_1a
            elif source == 'massive':
                yields_evol=self.history.ism_elem_yield_massive
            gas_evol=[]
            for k in range(len(yields_evol)):
                gas_evol.append(np.sum(yields_evol[k]))

        ism_gasm=[]
        star_m=[]
        x=[]
        #To prevent 0 +log scale
        if 'age' == xaxis:
                x_all=x_all[1:]
                gas_evol=gas_evol[1:]
                gas_mass=gas_mass[1:]
        for k in range(0,len(gas_evol)):
            if (gas_evol[k]==0) or (gas_mass[k]==0):
                continue
            x.append(x_all[k])
            if norm=='ini':
                ism_gasm.append(gas_evol[k]/self.history.mgal)
                star_m.append((self.history.mgal-gas_evol[k])/self.history.mgal)
            if norm == 'current':
                if not self.history.gas_mass[k] ==0.:
                      ism_gasm.append(gas_evol[k]/gas_mass[k])
                      star_m.append((self.history.mgal-gas_evol[k])/gas_mass[k])
            #else:
             #   ism_gasm.append(0.)
              #  star_m.append(0.)
            elif norm == 'no':
                ism_gasm.append(gas_evol[k])
                star_m.append(self.history.mgal-gas_evol[k])

        mass = 'gas'
        #TODO This is a quick fix to remove input mass option (should rewrite the whole function)

        if mass == 'gas':
            y=ism_gasm
        if mass == 'stars':
            y=star_m
        plt.plot(x,y,linestyle=shape,marker=marker,markevery=markevery,color=color,label=label)
        if len(label)>0:
            plt.legend()
        if norm=='current':
            plt.ylim(0,1.2)
        if not norm=='no':
            if mass=='gas':
                plt.ylabel('mass fraction')
                plt.title('Gas mass as a fraction of total gas mass')
            else:
                plt.ylabel('mass fraction')
                plt.title('Star mass as a fraction of total star mass')
        else:
            if mass=='gas':
                plt.ylabel('ISM gas mass [Msun]')
            else:
                plt.ylabel('mass locked in stars [Msun]')

            if mass=='gas':
                plt.ylabel('ISM gas mass [Msun]')
            else:
                plt.ylabel('Mass locked in stars [Msun]')

        if log==True:
            plt.yscale('log')
            if not norm=='no':
                plt.ylim(1e-4,1.2)
        ax=plt.gca()
        self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)
        plt.xlim(self.history.dt,self.history.tend)
        #self.save_data(header=['age','mass'],data=[x,y])


    def plot_sn_distr(self,fig=5,rate=True,rate_only='',xaxis='time',fraction=False,label1='SNIa',label2='SN2',shape1=':',shape2='--',marker1='o',marker2='s',color1='k',color2='b',markevery=20,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14):

        '''
        Plots the SN1a distribution:
        The evolution of the number of SN1a and SN2
        #Add numbers/dt numbers/year...

        Parameters
        ----------
        rate : boolean
            if true, calculate rate [1/century]
            else calculate numbers
        fraction ; boolean
            if true, ignorate rate and calculate number fraction of SNIa per WD
        rate_only : string
            if empty string, plot both rates (default)

            if 'sn1a', plot only SN1a rate

            if 'sn2', plot only SN2 rate
        xaxis: string
            if 'time' : time evolution
            if 'redshift': experimental! use with caution; redshift evolution
        label : string
             figure label
        marker : string
             figure marker
        shape : string
             line style
        color : string
             color of line
        fig : string,float
             to name the plot figure

        Examples
        ----------
        >>> s.plot_sn_distr()

        '''

        import matplotlib
        import matplotlib.pyplot as plt

        #For Wiersma09
        Hubble_0=73.
        Omega_lambda=0.762
        Omega_m=0.238

        figure=plt.figure(fig, figsize=(fsize[0],fsize[1]))
        age=self.history.age
        sn1anumbers=self.history.sn1a_numbers#[:-1]
        sn2numbers=self.history.sn2_numbers
        if xaxis=='redshift':
                print ('this features is not tested yet.')
                return 0
                age,idx=self.__time_to_z(age,Hubble_0,Omega_lambda,Omega_m)
                age=[0]+age
                plt.xlabel('Redshift z')
                timesteps=self.history.timesteps[idx-1:]
                sn2numbers=sn2numbers[idx:]
                sn1anumbers=sn1anumbers[idx:]
        else:
                plt.xlabel('Log-scaled age [yrs]')
                #plt.xscale('log')
        if rate and not fraction:
            if xaxis=='redshift':
                sn1a_rate=np.array(sn1anumbers)/ (np.array(timesteps)/100.)
                sn2_rate=np.array(sn2numbers)/ (np.array(timesteps)/100.)
            else:
                sn1a_rate=np.array(sn1anumbers[1:])/ (np.array(self.history.timesteps)/100.)
                sn2_rate=np.array(sn2numbers[1:])/ (np.array(self.history.timesteps)/100.)
            sn1a_rate1=[]
            sn2_rate1=[]
            age=age[1:]
            age_sn1a=[] #age[1:]
            age_sn2=[]
            #correct sn1a rate
            for k in range(len(sn1a_rate)):
                if sn1a_rate[k]>0:
                        sn1a_rate1.append(sn1a_rate[k])
                        age_sn1a.append(age[k])

            for k in range(len(sn2_rate)):
                if sn2_rate[k]>0:
                        sn2_rate1.append(sn2_rate[k])
                        age_sn2.append(age[k])

            if len(rate_only)==0:
                    x=[age_sn2,age_sn1a]
                    y=[sn1a_rate1,sn2_rate1]
                    plt.plot(age_sn1a,sn1a_rate1,linestyle=shape1,color=color1,label=label1,marker=marker1,markevery=markevery)
                    plt.plot(age_sn2,sn2_rate1,linestyle=shape2,color=color2,label=label2,marker=marker2,markevery=markevery)
                    plt.ylabel('SN rate [century$^{-1}$]')
            if rate_only=='sn1a':
                    x=age_sn1a
                    y= sn1a_rate1
                    plt.plot(age_sn1a,sn1a_rate1,linestyle=shape1,color=color1,label=label1,marker=marker1,markevery=markevery)
                    plt.ylabel('SNIa rate [century$^{-1}$]')
            if rate_only=='sn2':
                    x=age_sn2
                    y=sn2_rate
                    plt.plot(age_sn2,sn2_rate1,linestyle=shape2,color=color2,label=label2,marker=marker2,markevery=markevery)
                    plt.ylabel('SN2 rate [century$^{-1}$]')

        else:
            #if xaxis=='redshift':
                        #sn1_numbers=np.array(sn1anumbers)/ (np.array(timesteps)/100.)
                        #sn2_numbers=np.array(sn2numbers)/ (np.array(timesteps)/100.)
            #True: #else:
                        #sn1a_rate=np.array(sn1anumbers[1:])/ (np.array(self.history.timesteps)/100.)
                        #sn2_rate=np.array(sn2numbers[1:])/ (np.array(self.history.timesteps)/100.)

            sn1a_numbers=sn1anumbers[1:]
            sn2_numbers=sn2numbers[1:]
            sn1a_numbers1=[]
            sn2_numbers1=[]
            age_sn1a=[]
            age_sn2=[]
            age=age[1:]
            for k in range(len(sn1a_numbers)):
                if sn1a_numbers[k]>0:
                        sn1a_numbers1.append(sn1a_numbers[k])
                        age_sn1a.append(age[k])
            for k in range(len(sn2_numbers)):
                if sn2_numbers[k]>0:
                        sn2_numbers1.append(sn2_numbers[k])
                        age_sn2.append(age[k])

            if fraction:
                age=self.history.age
                ratio=[]
                age1=[]
                for k in range(len(self.wd_sn1a_range1)):
                    if self.wd_sn1a_range1[k]>0:
                        ratio.append(self.history.sn1a_numbers[1:][k]/self.wd_sn1a_range1[k])
                        age1.append(age[k])
                plt.plot(age1,ratio)
                plt.yscale('log')
                #plt.xscale('log')
                plt.ylabel('Number of SNIa going off per WD born')
                label='SNIafractionperWD';label='sn1a '+label
                x=age1
                y=ratio
                self.save_data(header=['age',label],data=[x,y])
                return
            else:
                    if len(rate_only)==0:
                            x=[age_sn1a,age_sn2]
                            y=[sn1a_numbers,sn2_numbers]
                            plt.plot(age_sn1a,sn1a_numbers1,linestyle=shape1,color=color1,label=label1,marker=marker1,markevery=markevery)
                            plt.plot(age_sn2,sn2_numbers1,linestyle=shape2,color=color2,label=label2,marker=marker2,markevery=markevery)
                            plt.ylabel('SN numbers')
                    if rate_only=='sn1a':
                            x= age1
                            y= sn1anumbers1
                            plt.plot(age_sn1a,sn1a_numbers1,linestyle=shape1,color=color1,label=label1,marker=marker1,markevery=markever)
                            plt.ylabel('SN numbers')
                    if rate_only=='sn2':
                            x= age[1:]
                            y= sn2numbers[1:]
                            plt.plot(age_sn2,sn2_numbers1,linestyle=shape2,color=color2,label=label2,marker=marker2,markevery=markevery)
                            plt.ylabel('SN numbers')

        plt.legend(loc=1)
        plt.yscale('log')
        ax=plt.gca()
        self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)
        if rate:
            label='rate'
        else:
            label='number'
        if len(rate_only)==0:
                if rate:
                        label='rate'
                else:
                        label='number'
                self.save_data(header=['age','SNIa '+label,'age','CCSN '+label],data=[x[0],y[0],x[1],y[1]])
        else:
                if rate_only=='sn1a':
                        label='sn1a '+label
                else:
                        label='ccsn '+label
                #self.save_data(header=['age',label],data=[x,y])

    def save_data(self,header=[],data=[],filename='plot_data.txt'):
        '''
            Writes data into a text file. data entries
            can have different lengths
        '''
        out=' '
        #header
        for i in range(len(header)):
            out+= ('&'+header[i]+((10-len(header[i]))*' '))
        #data
        out+='\n'
        max_data=[]
        for t in range(len(data)):
            max_data.append(len(data[t]))
        for t in range(max(max_data)):
            #out+=('&'+'{:.3E}'.format(data[t]))
            #out+=(' &'+'{:.3E}'.format(frac_yields[t]))
            for i in range(len(data)):
                if t>(len(data[i])-1):
                        out+=(' '*len( ' &'+ '{:.3E}'.format(0)))
                else:
                        out+= ( ' &'+ '{:.3E}'.format(data[i][t]))
            #out+=( ' &'+ '{:.3E}'.format(mtot_gas[t]))
            out+='\n'
        #import os.path
        #if os.path.isfile(filename)
        #overwrite existing file for now
        f1=open(filename,'w')
        f1.write(out)
        f1.close()


    ##############################################
    #          Plot Star Formation Rate          #
    ##############################################
    def plot_star_formation_rate(self,fig=6,fraction=False,source='all',marker='',shape='',color='',label='',abs_unit=True,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14):
        '''

        Plots the star formation rate over time.
        Shows fraction of ISM mass which transforms into stars.

        Parameters
        ----------

        fraction : boolean
           if true: fraction of ISM which transforms into stars;
           else: mass of ISM which goes into stars

        source : string
            either 'all' for total star formation rate; 'agb' for AGB and 'massive' for massive stars
            WARNING! Do not use in OMEGA, only use it in SYGMA.

        marker : string
            marker type
        shape : string
            line shape type
        fig: figure id

        Examples
        ----------
        >>> s.star_formation_rate()

        '''

        import matplotlib
        import matplotlib.pyplot as plt

        if (len(marker)==0 and len(shape)==0) and len(color)==0:
            shape,marker,color=self.__msc(source,shape,marker,color)
        plt.figure(fig, figsize=(fsize[0],fsize[1]))
        #maybe a histogram for display the SFR?
        if False:
                age=self.history.age
                #age=[0.1]+self.history.age[1:-1]
                sfr=self.history.sfr
                plt.xlabel('Log-scaled age [yrs]')
                #plt.xscale('log')
                mean_age=[]
                color='r'
                label='aa'
                age_bdy=[]
                for k in range(len(age)-1):
                        mean_age.append(1) #age[k+1]-age[k])
                for k in range(len(age)):
                        age_bdy.append(age[k])
                print ('age',len(mean_age))
                print ('weights',len(sfr))
                print ('bdy',len(age_bdy))
                print (sfr[:5])
                p1 =plt.hist(mean_age, bins=age_bdy,weights=sfr,facecolor=color,color=color,alpha=0.5,label=label)

        #Plot the SFR in [Mo yr^-1]
        if abs_unit:

            #Define visual aspects
            shape,marker,color=self.__msc(source,shape,marker,color)

            #Recover the SFR by dividing the mass locked into stars, at
            #each timestep, by the duration of the timestep
            age=self.history.age
            #sfr_plot = self.history.m_locked / self.history.timesteps
            sfr_plot = self.history.sfr_abs

            #Label and display axis
            plt.xlabel('Age [yrs]')
            plt.ylabel('SFR [Mo/yr]')

            #Plot
            plt.plot(age[:-1],sfr_plot[:-1],label=label,marker=marker,color=color,linestyle=shape)

            #self.save_data(header=['age','SFR'],data=[age,sfr_plot])

        #Plot the mass fraction of gas available converted into stars
        elif fraction:
                sfr=self.history.sfr
                if source=='all':
                        label='all'
                elif source=='agb':
                        sfr=np.array(sfr)*np.array(self.history.m_locked_agb)/np.array(self.history.m_locked)
                        label='AGB'
                elif source=='massive':
                        masslocked=self.history.m_locked_massive
                        sfr=np.array(sfr)*np.array(self.history.m_locked_massive)/np.array(self.history.m_locked)
                        label='Massive'
                age=self.history.age[:-1]
                print (len(age),len(sfr))
                plt.xlabel('Age [yrs]')
                plt.plot(age,sfr,label=label,marker=marker,linestyle=shape)
                plt.ylabel('Fraction of current gas mass into stars')
                #self.save_data(header=['age','SFR'],data=[age,sfr])

        #Plot the mass converted into stars
        else:
                if source=='all':
                        masslocked=self.history.m_locked
                        label='all'
                elif source=='agb':
                        masslocked=self.history.m_locked_agb
                        label='AGB'
                elif source=='massive':
                        masslocked=self.history.m_locked_massive
                        label='Massive'
                age=self.history.age[1:]
                plt.plot(age,masslocked,marker=marker,linestyle=shape,label=label)
                plt.xlabel('Age [yrs]')
                plt.ylabel('ISM Mass transformed into stars')
                plt.xscale('log');plt.yscale('log')
                plt.legend()

        ax=plt.gca()
        self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)

        #print ('Total mass transformed in stars, total mass transformed in AGBs, total mass transformed in massive stars:')
        #print (np.sum(self.history.m_locked),np.sum(self.history.m_locked_agb),np.sum(self.history.m_locked_massive))


    def __time_to_z(self,time,Hubble_0,Omega_lambda,Omega_m):
        '''
        Time to redshift conversion

        '''
        import time_to_redshift as tz
        #transform into Gyr
        time_new=[]
        firstidx=True
        for k in range(len(time)):
                if time[k]>=8e8:
                        time_new.append(time[k]/1.0e9)
                        if firstidx:
                                index=k
                                firstidx=False
        return tz.t_to_z(time_new,Hubble_0,Omega_lambda,Omega_m),index

    def __msc(self,source,shape,marker,color):

        '''
        Function checks if either of shape,color,marker
        is set. If not then assign in each case
        a unique property related to the source ('agb','massive'..)
        to the variable and returns all three
        '''

        if source=='all':
                shape1='-'
                marker1='o'
                color1='k'
        elif source=='agb':
                shape1='--'
                marker1='s'
                color1='r'
        elif source=='massive':
                shape1='-.'
                marker1='D'
                color1='b'
        elif source =='sn1a':
                shape1=':'
                marker1='x'
                color1='g'

        if len(shape)==0:
                shape=shape1
        if len(marker)==0:
                marker=marker1
        if len(color)==0:
                color=color1

        return shape,marker,color


    ##############################################
    #                Fig Standard                #
    ##############################################
    def __fig_standard(self,ax,fontsize=8,labelsize=8,lwtickboth=[6,2],lwtickmajor=[10,3],markersize=8,rspace=0.6,bspace=0.15, legend_fontsize=14):

        '''
        Internal function in order to get standardized figure font sizes.
        It is used in the plotting functions.

        '''

        import matplotlib
        import matplotlib.pyplot as plt

        plt.legend(loc=2,prop={'size':legend_fontsize})
        plt.rcParams.update({'font.size': fontsize})
        ax.yaxis.label.set_size(labelsize)
        ax.xaxis.label.set_size(labelsize)
        #ax.xaxis.set_tick_params(width=2)
        #ax.yaxis.set_tick_params(width=2)
        ax.tick_params(length=lwtickboth[0],width=lwtickboth[1],which='both')
        ax.tick_params(length=lwtickmajor[0],width=lwtickmajor[1],which='major')
        #Add that line below at some point
        #ax.xaxis.set_tick_params(width=2)
        #ax.yaxis.set_tick_params(width=2)
        if len(ax.lines)>0:
                for h in range(len(ax.lines)):
                        ax.lines[h].set_markersize(markersize)
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),markerscale=0.8,fontsize=legend_fontsize)

        plt.subplots_adjust(right=rspace)
        plt.subplots_adjust(bottom=bspace)


    ##############################################
    #              Plot Mass-Loading             #
    ##############################################
    def plot_mass_loading(self,fig=8,marker='',shape='',\
            color='',label='Mass-loading',fsize=[10,4.5],fontsize=14,rspace=0.6,\
            bspace=0.15,labelsize=15,legend_fontsize=14):

        '''
        This function plots the mass-loading factor, which is the ratio
        between the mass outflow rate and the star formation rate, as a
        function of time.

        Parameters
        ----------

        fig : string, float
             Figure name.
        marker : string
             Figure marker.
        shape : string
             Line style.
        color : string
             Line color.
        label : string
             Figure label.
        fsize : 2D float array
             Figure dimension/size.
        fontsize : integer
             Font size of the numbers on the X and Y axis.
        rspace : float
             Extra space on the right for the legend.
        bspace : float
             Extra space at the bottom for the Y axis label.
        labelsize : integer
             Font size of the X and Y axis labels.
        legend_fontsize : integer
                Font size of the legend.

        Examples
        ----------

        >>> o1.plot_mass_loading(label='Mass-Loading Factor')

        '''

        import matplotlib
        import matplotlib.pyplot as plt

        # Define variable only used for plot visual
        source='all'

        #Plot only if in the open box scenario
        if self.open_box:

            #Define visual aspects
            shape,marker,color=self.__msc(source,shape,marker,color)
            plt.figure(fig, figsize=(fsize[0],fsize[1]))

            #Display axis labels
            plt.xlabel('Age [yrs]')
            plt.ylabel('Mass-loading factor')

            #Copy data for x and y axis
            age = self.history.age[:-1]
            mass_lo = self.history.eta_outflow_t

            #Plot data
            plt.plot(age,mass_lo,color=color,label=label,\
                     marker=marker,linestyle=shape)

            #Save plot
            #self.save_data(header=['age','mass-loading'],data=[age,mass_lo])

            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,\
                rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)

        else:
            print ('Not available with a closed box.')


    ##############################################
    #              Plot Outflow Rate             #
    ##############################################
    def plot_outflow_rate(self,fig=9,marker='',shape='',\
            color='',label='Outflow',fsize=[10,4.5],fontsize=14,rspace=0.6,\
            bspace=0.15,labelsize=15,legend_fontsize=14):

        '''
        This function plots the mass outflow rate as a function of time
        in units of [Mo/yr].

        Parameters
        ----------

        fig : string, float
             Figure name.
        marker : string
             Figure marker.
        shape : string
             Line style.
        color : string
             Line color.
        label : string
             Figure label.
        fsize : 2D float array
             Figure dimension/size.
        fontsize : integer
             Font size of the numbers on the X and Y axis.
        rspace : float
             Extra space on the right for the legend.
        bspace : float
             Extra space at the bottom for the Y axis label.
        labelsize : integer
             Font size of the X and Y axis labels.
        legend_fontsize : integer
                Font size of the legend.

        Examples
        ----------

        >>> o1.plot_outflow_rate(label='Outflow Rate')

        '''

        import matplotlib
        import matplotlib.pyplot as plt

        # Define variable only used for plot visual
        source='all'

        #Plot only if in the open box scenario
        if self.open_box:

            #Define visual aspects
            shape,marker,color=self.__msc(source,shape,marker,color)
            plt.figure(fig, figsize=(fsize[0],fsize[1]))

            #Display axis labels
            plt.xlabel('Age [yrs]')
            plt.ylabel('Outflow rate [Mo/yr]')

            #Copy data for x and y axis
            age = self.history.age[:-1]
            outflow_plot = []
            for i_op in range(0,len(age)):
                outflow_plot.append(self.m_outflow_t[i_op] / \
                    self.history.timesteps[i_op])
            outflow_plot[0] = 0.0

            #Plot data
            plt.plot(age,outflow_plot,label=label,marker=marker,\
                     color=color,linestyle=shape)

            #Save plot
            #self.save_data(header=['age','outflow rate'],data=[age,outflow_plot])

            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,\
                rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)

        else:
            print ('Not available with a closed box.')


    ##############################################
    #              Plot Inflow Rate              #
    ##############################################
    def plot_inflow_rate(self,fig=10,marker='',shape='',color='',\
            label='Inflow',fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,\
            labelsize=15,legend_fontsize=14):

        '''
        This function plots the mass inflow rate as a function of time
        in units of [Mo/yr].

        Parameters
        ----------

        fig : string, float
             Figure name.
        marker : string
             Figure marker.
        shape : string
             Line style.
        color : string
             Line color.
        label : string
             Figure label.
        fsize : 2D float array
             Figure dimension/size.
        fontsize : integer
             Font size of the numbers on the X and Y axis.
        rspace : float
             Extra space on the right for the legend.
        bspace : float
             Extra space at the bottom for the Y axis label.
        labelsize : integer
             Font size of the X and Y axis labels.
        legend_fontsize : integer
             Font size of the legend.

        Examples
        ----------

        >>> o1.plot_inflow_rate(label='Inflow Rate')

        '''

        import matplotlib
        import matplotlib.pyplot as plt

        # Define variable only used for plot visual
        source='all'

        #Plot only if in the open box scenario
        if self.open_box:

            #Define visual aspects
            shape,marker,color=self.__msc(source,shape,marker,color)
            plt.figure(fig, figsize=(fsize[0],fsize[1]))

            #Display axis labels
            plt.xlabel('Age [yrs]')
            plt.ylabel('Inflow rate [Mo/yr]')

            #Copy data for x and y axis
            age = self.history.age[:-1]
            inflow_plot = []
            for i_op in range(0,len(age)):
                inflow_plot.append(self.m_inflow_t[i_op] / \
                    self.history.timesteps[i_op])

            #Plot data
            plt.plot(age,inflow_plot,label=label,marker=marker,\
                     color=color,linestyle=shape)

            #Save plot
            #self.save_data(header=['age','inflow rate'],data=[age,inflow_plot])

            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,\
                rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)

        else:
            print ('Not available with a closed box.')


    ##############################################
    #              Plot Dark Matter              #
    ##############################################
    def plot_dark_matter(self,fig=11,marker='',shape='',\
            color='',label='Dark matter',fsize=[10,4.5],fontsize=14,rspace=0.6, \
            bspace=0.15,labelsize=15,legend_fontsize=14):

        '''
        This function plots the dark matter halo mass of the galaxy as a
        function of time.

        Parameters
        ----------

        fig : string, float
             Figure name.
        marker : string
             Figure marker.
        shape : string
             Line style.
        color : string
             Line color.
        label : string
             Figure label.
        fsize : 2D float array
             Figure dimension/size.
        fontsize : integer
             Font size of the numbers on the X and Y axis.
        rspace : float
             Extra space on the right for the legend.
        bspace : float
             Extra space at the bottom for the Y axis label.
        labelsize : integer
             Font size of the X and Y axis labels.
        legend_fontsize : integer
             Font size of the legend.

        Examples
        ----------

        >>> o1.plot_dark_matter(label='Dark Matter')

        '''

        import matplotlib
        import matplotlib.pyplot as plt

        # Define variable only used for plot visual
        source='all'

        # Plot only if in the open box scenario
        if self.open_box:

            # Define visual aspects
            shape,marker,color=self.__msc(source,shape,marker,color)
            plt.figure(fig, figsize=(fsize[0],fsize[1]))

            #Display axis labels
            plt.xlabel('Age [yrs]')
            plt.ylabel('Dark matter halo mass [Mo]')

            #Copy data for x and y axis
            age = self.history.age
            DM_plot = self.m_DM_t

            #Plot data
            plt.plot(age,DM_plot,label=label,marker=marker,\
                     color=color,linestyle=shape)

            #Save plot
            #self.save_data(header=['age','dark matter halo mass'],\
            #               data=[age,DM_plot])

            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,\
                rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)

        else:
            print ('Not available with a closed box.')


    ##############################################
    #             Plot SF Timescale              #
    ##############################################
    def plot_sf_timescale(self,fig=15,marker='',shape='',color='',\
            label='',fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,\
            labelsize=15,legend_fontsize=14):

        '''
        This function plots the star formation timescale as a function of time
        in units of [yr].

        Parameters
        ----------

        fig : string, float
             Figure name.
        marker : string
             Figure marker.
        shape : string
             Line style.
        color : string
             Line color.
        label : string
             Figure label.
        fsize : 2D float array
             Figure dimension/size.
        fontsize : integer
             Font size of the numbers on the X and Y axis.
        rspace : float
             Extra space on the right for the legend.
        bspace : float
             Extra space at the bottom for the Y axis label.
        labelsize : integer
             Font size of the X and Y axis labels.
        legend_fontsize : integer
             Font size of the legend.

        Examples
        ----------

        >>> o1.plot_sf_timescale(label='Star Formation Timescale')

        '''

        import matplotlib
        import matplotlib.pyplot as plt

        # Define variable only used for plot visual
        source='all'

        #Plot only if in the open box scenario
        if self.open_box:

            #Define visual aspects
            shape,marker,color=self.__msc(source,shape,marker,color)
            plt.figure(fig, figsize=(fsize[0],fsize[1]))

            #Display axis labels
            plt.xlabel('Age [yrs]')
            plt.ylabel('Star formation timescale [yr]')

            #print (len(self.history.age))
            #print (len(self.history.t_SF_t))

            #Copy data for x and y axis
            age = self.history.age
            t_SF_plot = self.t_SF_t

            #Plot data
            plt.plot(age,t_SF_plot,label=label,marker=marker,\
                     color=color,linestyle=shape)

            #Save plot
            #self.save_data(header=['age','star formation timescale'],\
            #                       data=[age,t_SF_plot])

            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,\
                rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)

        else:
            print ('Not available with a closed box.')


    ##############################################
    #               Plot Redshift                #
    ##############################################
    def plot_redshift(self,fig=16,marker='',shape='',\
            color='',label='Redshift',fsize=[10,4.5],fontsize=14,rspace=0.6,\
            bspace=0.15,labelsize=15,legend_fontsize=14):

        '''
        This function plots the redshift a function of time.

        Parameters
        ----------

        fig : string, float
             Figure name.
        marker : string
             Figure marker.
        shape : string
             Line style.
        color : string
             Line color.
        label : string
             Figure label.
        fsize : 2D float array
             Figure dimension/size.
        fontsize : integer
             Font size of the numbers on the X and Y axis.
        rspace : float
             Extra space on the right for the legend.
        bspace : float
             Extra space at the bottom for the Y axis label.
        labelsize : integer
             Font size of the X and Y axis labels.
        legend_fontsize : integer
             Font size of the legend.

        Examples
        ----------

        >>> o1.plot_redshift(label='Redshift')

        '''

        import matplotlib
        import matplotlib.pyplot as plt

        # Define variable only used for plot visual
        source='all'

        #Plot only if in the open box scenario
        if self.open_box:

            #Define visual aspects
            shape,marker,color=self.__msc(source,shape,marker,color)
            plt.figure(fig, figsize=(fsize[0],fsize[1]))

            #Display axis labels
            plt.xlabel('Age [yrs]')
            plt.ylabel('Redshift')

            #Copy data for x and y axis
            age = self.history.age
            redshift_plot = self.redshift_t

            #Plot data
            plt.plot(age,redshift_plot,label=label,marker=marker,\
                     color=color,linestyle=shape)

            #Save plot
            #self.save_data(header=['age','redshift'],data=[age,redshift_plot])

            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,\
                rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)

        else:
            print ('Not available with a closed box.')


    ###################################################
    #                  Plot Iso Ratio                 #
    ###################################################
    def plot_iso_ratio(self,return_x_y=False, grain_notation=False,
        xaxis='age',yaxis='C-12/C-13',\
        solar_ab='yield_tables/iniabu/iniab2.0E-02GN93.ppn',\
        solar_iso='stellab_data/solar_normalization/Asplund_et_al_2009_iso.txt',\
        fig=18,source='all',marker='',shape='',\
        color='',label='',fsize=[10,4.5],fontsize=14,rspace=0.6,\
        bspace=0.15,labelsize=15,legend_fontsize=14):

        '''
        This function plots the evolution of an isotopic ratio as a function
        of time, as a function of an abundance ratio, or as a function of
        another isotopic ratio.  Isotopic ratios are given in the
        delta notation and abundances ratios are given in spectroscopic notation.

        Parameters
        ----------

        return_x_y : boolean
             If False (default), show the plot.  If True, return two arrays containing
             the X and Y axis data, respectively.
        grain_notation : boolean
             If False (default), show mass ratio.  If True, show ratio in the delta notation
        xaxis : string
             X axis, either 'age', an abundance ratio such as '[Fe/H]', or an
             isotopic ratio such as 'C-12/C-13'.
        yaxis : string
             Y axis, isotopic ratio such as 'C-12/C-13'.
        solar_ab : string
             Path to the solar abundances used to normalize abundance ratios.
        solar_iso : string
             Path to the solar isotopes used to normalize the isotopic ratios.
        fig : string, float
             Figure name.
        source : string
            Specificies if yields come from
            all sources ('all') or from
            distinctive sources:
            only AGB stars ('agb'), only
            SN Ia ('sn1a'), or only massive stars ('massive')
        marker : string
             Figure marker.
        shape : string
             Line style.
        color : string
             Line color.
        label : string
             Figure label.
        fsize : 2D float array
             Figure dimension/size.
        fontsize : integer
             Font size of the numbers on the X and Y axis.
        rspace : float
             Extra space on the right for the legend.
        bspace : float
             Extra space at the bottom for the Y axis label.
        labelsize : integer
             Font size of the X and Y axis labels.
        legend_fontsize : integer
             Font size of the legend.

        Examples
        ----------

        >>> o1.plot_iso_ratio(xaxis='[Fe/H]', yaxis='C-12/C-13')

        '''

        import matplotlib
        import matplotlib.pyplot as plt

        # Marker on the plots
        markevery=500

        # Declaration of the array to plot
        x = []
        y = []

        #Access solar abundance
        iniabu=ry.iniabu(os.path.join(nupy_path, solar_ab))
        x_ini_iso=iniabu.iso_abundance(self.history.isotopes)
        elements = self.history.elements
        x_ini=self._iso_abu_to_elem(x_ini_iso)

        # Get the wanted source
        if source == 'all':
            yields_evol=self.history.ism_iso_yield
            yields_evol_el=self.history.ism_elem_yield
        elif source =='agb':
            yields_evol=self.history.ism_iso_yield_agb
            yields_evol_el=self.history.ism_elem_yield_agb
        elif source == 'sn1a':
            yields_evol=self.history.ism_iso_yield_1a
            yields_evol_el=self.history.ism_elem_yield_1a
        elif source == 'massive':
            yields_evol=self.history.ism_iso_yield_massive
            yields_evol_el=self.history.ism_elem_yield_massive
        else:
            print ('!! Source not valid !!')
            return

        # Verify the X-axis
        xaxis_ratio = False
        if xaxis == 'age':
            x = self.history.age[1:]
        elif xaxis[0] == '[' and xaxis[-1] == ']':
            xaxis_elem1 =xaxis.split('/')[0][1:]
            xaxis_elem2 =xaxis.split('/')[1][:-1]
            if not xaxis_elem1 in self.history.elements and \
               not xaxis_elem2 in self.history.elements:
                 print ('!! Elements in xaxis are not valid !!')
                 return

            #X-axis ini values
            x_elem1_ini=x_ini[elements.index(xaxis_elem1)]
            x_elem2_ini=x_ini[elements.index(xaxis_elem2)]

            #X-axis gce values
            elem_idx1=self.history.elements.index(xaxis_elem1)
            elem_idx2=self.history.elements.index(xaxis_elem2)

            for k in range(0,len(yields_evol_el)):
                if np.sum(yields_evol_el[k]) == 0:
                    continue
                #in case no contribution during timestep
                x1=yields_evol_el[k][elem_idx1]/np.sum(yields_evol_el[k])
                x2=yields_evol_el[k][elem_idx2]/np.sum(yields_evol_el[k])
                spec=np.log10(x1/x2) - np.log10(x_elem1_ini/x_elem2_ini)
                x.append(spec)
        else:
            xaxis_ratio = True
            x_1 = ''
            x_2 = ''
            is_x_1 = True
            for ix in range(0,len(xaxis)):
                if xaxis[ix] == '/' and is_x_1:
                    is_x_1 = False
                else:
                    if is_x_1:
                        x_1 += xaxis[ix]
                    else:
                        x_2 += xaxis[ix]
            if len(x_2) == 0:
                print ('!! xaxis not valid.  Need to be \'age\' or a ratio !!')
                return

        # Verify the Y-axis
        y_1 = ''
        y_2 = ''
        is_y_1 = True
        for iy in range(0,len(yaxis)):
            if yaxis[iy] == '/' and is_y_1:
                is_y_1 = False
            else:
                if is_y_1:
                    y_1 += yaxis[iy]
                else:
                    y_2 += yaxis[iy]
        if len(y_2) == 0:
            print ('!! yaxis not valid.  Need to be a ratio !!')
            return

        # Get the isotopes for X-axis
        if xaxis_ratio:
          if x_1 in self.history.isotopes and x_2 in self.history.isotopes:
            idx_1=self.history.isotopes.index(x_1)
            idx_2=self.history.isotopes.index(x_2)
            x1_elem = str(x_1.split('-')[0])
            x2_elem = str(x_2.split('-')[0])
            x1_at_nb = float(x_1.split('-')[1])
            x2_at_nb = float(x_2.split('-')[1])
          else:
            print ('!! Isotopes in xaxis are not valid !!')
            return

        # Get the isotopes for Y-axis
        if y_1 in self.history.isotopes and y_2 in self.history.isotopes:
            idy_1=self.history.isotopes.index(y_1)
            idy_2=self.history.isotopes.index(y_2)
            y1_elem = str(y_1.split('-')[0])
            y2_elem = str(y_2.split('-')[0])
            y1_at_nb = float(y_1.split('-')[1])
            y2_at_nb = float(y_2.split('-')[1])
        else:
            print ('!! Isotopes in yaxis are not valid !!')
            return

        # Set the default label .. if not defined
        if len(label)<1:
            label=yaxis
            if source=='agb':
                label=yaxis+', AGB'
            elif source=='massive':
                label=yaxis+', Massive'
            elif source=='sn1a':
                label=yaxis+', SNIa'

        # If delta notation ..
        if grain_notation:

            # Get the solar values
            if xaxis_ratio:
                x1_sol, x2_sol, y1_sol, y2_sol = \
                    self.__read_solar_iso(solar_iso, x_1, x_2, y_1, y_2)
            else:
                x1_sol, x2_sol, y1_sol, y2_sol = \
                    self.__read_solar_iso(solar_iso, '', '', y_1, y_2)

            # Calculate the isotope ratios
            for k in range(0,len(yields_evol)):
                if xaxis_ratio:
                    ratio_sample = (yields_evol[k][idx_1]/yields_evol[k][idx_2])*\
                                   (x2_at_nb / x1_at_nb)
                    ratio_std = x1_sol / x2_sol
                    x.append( ((ratio_sample/ratio_std) - 1.0) * 1000.0)
                ratio_sample = (yields_evol[k][idy_1]/yields_evol[k][idy_2])*\
                               (y2_at_nb / y1_at_nb)
                ratio_std = y1_sol / y2_sol
                y.append( ((ratio_sample/ratio_std) - 1.0) * 1000.0)

        # If simple mass ratio ..
        else:
            for k in range(0,len(yields_evol)):
                if xaxis_ratio:
                    x.append(yields_evol[k][idx_1]/yields_evol[k][idx_2])
                y.append(yields_evol[k][idy_1]/yields_evol[k][idy_2])

        # Make sure the length of array are the same when xaxis = '[X/Y]'
        too_much = len(y)-len(x)
        y = y[too_much:]

        #Reserved for plotting
        if not return_x_y:
            shape,marker,color=self.__msc(source,shape,marker,color)

        #Reserved for plotting
        if not return_x_y:
           plt.figure(fig, figsize=(fsize[0],fsize[1]))
               #if xaxis == 'age':
               #plt.xscale('log')
           #plt.yscale('log')
           if grain_notation:
               plt.ylabel("$\delta$($^{"+str(int(y1_at_nb))+"}$"+y1_elem+"/$^{"+\
                           str(int(y2_at_nb))+"}$"+y2_elem+")")
           else:
               plt.ylabel(str(int(y1_at_nb))+"-"+y1_elem+"/"+\
                           str(int(y2_at_nb))+"-"+y2_elem)
               plt.yscale('log')
           if xaxis == 'age':
               plt.xlabel('Age [yr]')
           elif xaxis_ratio:
               plt.xlabel("$\delta$($^{"+str(int(x1_at_nb))+"}$"+x1_elem+"/$^{"+\
                           str(int(x2_at_nb))+"}$"+x2_elem+")")
           else:
               plt.xlabel(xaxis)

        #If x and y need to be returned ...
        if return_x_y:
            return x, y

        else:
            plt.plot(x,y,label=label,linestyle=shape,marker=marker,\
                     color=color,markevery=markevery)
            plt.legend()
            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,\
                  rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)
            #plt.xlim(self.history.dt,self.history.tend)
            #self.save_data(header=['Iso mass ratio',yaxis],data=[x,y])


    ##############################################
    #               Read Solar Iso               #
    ##############################################
    def __read_solar_iso(self, file_path, xx1, xx2, yy1, yy2):

        # Initialisation of the variable to be returned
        x1_rsi = -1
        x2_rsi = -1
        y1_rsi = -1
        y2_rsi = -1

        # Open the data file
        with open(os.path.join(nupy_path, file_path), 'r') as data_file:

            # For every line (for each isotope) ...
            for line_1_str in data_file:

                # Split the line
                split = [str(x) for x in line_1_str.split()]

                # Copy the number fraction of the isotopes (if found)
                if split[0] == xx1:
                    x1_rsi = float(split[1])
                if split[0] == xx2:
                    x2_rsi = float(split[1])
                if split[0] == yy1:
                    y1_rsi = float(split[1])
                if split[0] == yy2:
                    y2_rsi = float(split[1])

        # Close the file
        data_file.close()

        # Return the number fractions
        return x1_rsi, x2_rsi, y1_rsi, y2_rsi


    ##############################################
    #                  Plot MDF                  #
    ##############################################
    def plot_mdf(self, fig=19, return_x_y=False, axis_mdf = '[Fe/H]',\
        dx = 0.05, solar_ab='yield_tables/iniabu/iniab2.0E-02GN93.ppn',\
        sigma_gauss=0.0, nb_sigma = 3.0,\
        marker='',shape='', norm=True,\
        color='',label='',fsize=[10,4.5],fontsize=14,rspace=0.6,\
        bspace=0.15,labelsize=15,legend_fontsize=14):

        '''
        This function plots the stellar metallicity distribution function (MDF),
        or the distribution function of any other abundance ratio.

        Parameters
        ----------

        return_x_y : boolean
             If False, show the plot.  If True, return two arrays containing
             the X and Y axis data, respectively.
        axis_mdf : string
             Abundance ratio for the distribution such as '[Fe/H]' or '[Mg/H]'.
        dx : float
             Bin resolution of the distribution.
        solar_ab : string
             Path to the solar abundances used to normalize abundance ratios.
        sigma_gauss : float
             Each point in the MDF can be associated with a gaussian. This implies
             that in reality, the metallicity should have a certain dispersion
             when stars form at each timestep (instead of using only a single
             average value).  The sigma_guass parameter sets the sigma value eac
             gaussian function.
        nb_sigma : float
             When sigma_gauss is greater than zero, nb_sigma is the number of
             sigma by which the X axis range is expanded.
        norm : boolean
             Normalize the MDF is True
             Default : True
        fig : string, float
             Figure name.
        marker : string
             Figure marker.
        shape : string
             Line style.
        color : string
             Line color.
        label : string
             Figure label.
        fsize : 2D float array
             Figure dimension/size.
        fontsize : integer
             Font size of the numbers on the X and Y axis.
        rspace : float
             Extra space on the right for the legend.
        bspace : float
             Extra space at the bottom for the Y axis label.
        labelsize : integer
             Font size of the X and Y axis labels.
        legend_fontsize : integer
                Font size of the legend.

        Examples
        ----------

        >>> o1.plot_mdf(axis_mdf='[Fe/H]', sigma_gauss=0.2)

        '''

        import matplotlib
        import matplotlib.pyplot as plt

        # Define variable only used for plot visual
        source='all'
        markevery = 10000

        # Get the [X/Y] values
        x_min, x_max, x_all = self.__get_XY(axis_mdf, solar_ab)

        # Construct the x axis for the MDF (lower boundary of the bin)
        mdf_x = []
        mdf_y = []
        x_i = x_min
        while x_i <= x_max:
            mdf_x.append(x_i)
            mdf_y.append(0.0)
            x_i += dx
        mdf_x.append(x_i)
        mdf_y.append(0.0)

        # For each step in the simulation ...
        # len(x_all)-1 because we move by [low,up] using i_sim and i_sim+1
        for i_sim in range(0, len(x_all)-1):

          # Check if the values are ok
          if np.isfinite(x_all[i_sim]) and np.isfinite(x_all[i_sim+1]):

            # Calculate delta X / Mlocked
            f_mlocked = self.history.m_locked[i_sim] / \
              (x_all[i_sim+1] - x_all[i_sim])

            # For each MDF x bin ...
            # len(mdf_x)-1 because mdf_x[-1] = last upper lim
            for i in range(0, len(mdf_x)-1):

              # If there is an overlap, add the corresponding mass in the bin
              # Bin is overlaping the lower-boundary of the simulation
              if mdf_x[i] <= x_all[i_sim] and \
                 x_all[i_sim] < mdf_x[i+1] and \
                 mdf_x[i+1] < x_all[i_sim+1]:
                  mdf_y[i] += (mdf_x[i+1] - x_all[i_sim]) * f_mlocked

              # Bin is totaly within the boundarie of the simulation
              elif x_all[i_sim] <= mdf_x[i] and \
                mdf_x[i+1] < x_all[i_sim+1]:
                  mdf_y[i] += (mdf_x[i+1] - mdf_x[i]) * f_mlocked

              # Bin is overlaping the uper-boundary of the simulation
              elif x_all[i_sim] <= mdf_x[i] and \
                mdf_x[i] < x_all[i_sim+1] and \
                x_all[i_sim+1] < mdf_x[i+1]:
                  mdf_y[i] += (x_all[i_sim+1] - mdf_x[i]) * f_mlocked

              # Bin is totaly overlaping the simulation
              elif mdf_x[i] <= x_all[i_sim] and \
                x_all[i_sim+1] < mdf_x[i+1]:
                  mdf_y[i] += (x_all[i_sim+1] - x_all[i_sim]) * f_mlocked

        # Correct the x axis to use the middle value of each bin
        x_cor = dx/2.0
        for i in range(0, len(mdf_x)):
            mdf_x[i] += x_cor

        # Normalisation to 1.0 (max value)
        if norm:
            cte = 1.0 / max(mdf_y)
            for i in range(0, len(mdf_y)):
                mdf_y[i] = mdf_y[i] * cte

        # If the gaussian approximation is wanted ...
        if sigma_gauss > 0.0:
            mdf_x, mdf_y = \
                self.__gauss_approx(mdf_x, mdf_y, sigma_gauss, dx, nb_sigma, norm)

        # Reserved for plotting
        if not return_x_y:
            shape,marker,color=self.__msc(source,shape,marker,color)
            plt.figure(fig, figsize=(fsize[0],fsize[1]))
            plt.xlabel(axis_mdf)
            plt.ylabel('$N_\star$')
            plt.plot(mdf_x,mdf_y,label=label,linestyle=shape,marker=marker,\
                     color=color,markevery=markevery)
            plt.legend()
            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,\
                  rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)

        # If the MDF should be returned ...
        else :
            return mdf_x, mdf_y


    ##############################################
    #                 Get [X/Y]                  #
    ##############################################
    def __get_XY(self, axis_mdf = '[Fe/H]', \
             solar_ab=os.path.join('yield_tables', 'iniabu',\
             'iniab2.0E-02GN93.ppn')):

        # Access solar abundance
        iniabu = ry.iniabu(os.path.join(nupy_path, solar_ab))
        x_ini_iso = iniabu.iso_abundance(self.history.isotopes)

        # Access
        elements = self.history.elements
        x_ini = self._iso_abu_to_elem(x_ini_iso)

        yields_evol = self.history.ism_elem_yield

        # Isolate the X and Y in [X/Y]
        xaxis_elem1 = axis_mdf.split('/')[0][1:]
        xaxis_elem2 = axis_mdf.split('/')[1][:-1]

        # Get the solar values
        x_elem1_ini = x_ini[elements.index(xaxis_elem1)]
        x_elem2_ini = x_ini[elements.index(xaxis_elem2)]

        # Get the simulation values
        elem_idx1 = self.history.elements.index(xaxis_elem1)
        elem_idx2 = self.history.elements.index(xaxis_elem2)

        x = []
        x_no_inf = []
        # Calculate [X/Y] at each timestep
        for k in range(0,len(yields_evol)):
            x1 = yields_evol[k][elem_idx1] / np.sum(yields_evol[k])
            x2 = yields_evol[k][elem_idx2] / np.sum(yields_evol[k])
            spec = np.log10(x1/x2) - np.log10(x_elem1_ini/x_elem2_ini)
            if np.isfinite(spec):
                x_no_inf.append(spec)
            x.append(spec)

        # Return [X/Y]_min, [X/Y]_max, and the [X/Y] array
        x_no_inf = np.sort(x_no_inf)
        return x_no_inf[0], x_no_inf[-1], x


    ##############################################
    #               Gauss Approx.                #
    ##############################################
    def __gauss_approx(self, mdf_x, mdf_y, sigma_gauss, dx, nb_sigma, norm):

        # Declaration of the MDF after the gaussian approximation
        mdf_y_ga = []

        # Number of bins
        nb_bin_ga = len(mdf_x)

        # Constant in the gaussian
        cte_ga = 1 / (2 * sigma_gauss**2)

        # For each bin on the x axis ...
        for i_ga in range(0,nb_bin_ga):

            # Add the contribution of all the other bins
            mdf_y_ga.append(0.0)
            for j_ga in range(0,nb_bin_ga):
                mdf_y_ga[i_ga] += mdf_y[j_ga] * \
                    np.exp((-1) * cte_ga * (mdf_x[i_ga]-mdf_x[j_ga])**2)

        # Add a few sigma at higher Z ...
        x_ga = mdf_x[-1] + dx
        x_ga_max = x_ga + nb_sigma * sigma_gauss
        i_ga = nb_bin_ga
        while x_ga < x_ga_max:
            mdf_x.append(x_ga)
            mdf_y_ga.append(0.0)
            for j_ga in range(0,nb_bin_ga):
                mdf_y_ga[i_ga] += mdf_y[j_ga] * \
                    np.exp((-1) * cte_ga * (mdf_x[i_ga]-mdf_x[j_ga])**2)
            i_ga += 1
            x_ga += dx

        # Renormalization to one (max value)
        if norm:
            cte = 1.0 / max(mdf_y_ga)
            for i in range(0, len(mdf_y_ga)):
                mdf_y_ga[i] = mdf_y_ga[i] * cte

        return mdf_x, mdf_y_ga


    ##############################################
    #                 Plot Abun                  #
    ##############################################
        '''
        def plot_abun(self, fig=20, age=9.2e9,\
        solar_norm=False, iso_on=False,\
        list_elem=[], list_iso=[],return_x_y=False, over_plot_solar=False,\
        solar_ab_m='yield_tables/iniabu/iniab2.0E-02GN93.ppn', species_labels=True,f_y_annotate=0.9,\
        marker='o',marker_s='^',shape='',shape_s='-', color='b', color_s='r', label='Prediction',label_s='solar',fsize=[10,4.5],\
        fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,\
        legend_fontsize=14, markersize=5):

        '''
        '''
        This function plots the abundance distribution in mass
        fraction at any given point in time or at any given metallicity.

        Parameters
        ----------
        age : float
             Time passed since the beginning of the simulation at which abundance distribution is taken.
             Default: Time until the formation of the pre-solar cloud.
        solar_norm: boolean
             If True, the abundances will be divide by the solar abundances.
             If False, there is no normalization.
             Default: False
        iso_on : boolean
             If True, the abundances will be in terms of isotopes.
             If False, the abundances will be in terms of elements.
        list_elem : array of strings
             Contain the list of elements included in the abundances distribution.
             Default: [], all elements will be included.
             Example: list_elem=['H','He','C','N','O','Fe']
        list_iso : array of strings
             Contain the list of isotopes included in the abundances distribution.
             Default: [], all isotopes will be included.
             Example: list_iso=['H-1','C-12','C-13','N-14','N-15']
             Note: list_iso dominates over list_elem is both are used.
        return_x_y : boolean
             If False, show the plot.
             If True, return two arrays containing the X and Y axis data, respectively.
             If True and iso_list=True, return a multi-D array [i][j][:] where
             j=0 -> list of mass numbers A for the isotopes associated with the element i,
             j=1 -> list of abundances
        over_plot_solar : boolean
             If True, plot the solar abundances distribution along with the predictions
             Note: This option is desactivated if solar_norm=True
        solar_ab_m : string
             Path of the solar normalization containing the mass fraction of each isotope
             Default: solar_ab_m='yield_tables/iniabu/iniab2.0E-02GN93.ppn'
        f_y_annotate : float (between 0 and 1)
             Fraction of the Y axis where the name of the element will be shown
             Default: f_y_annotate=0.9
        fig : string, float
             Figure name.
        marker : string
             Figure marker.
        marker_s : string
             Marker of the solar abundance distribution.
        shape : string
             Line style.
        shape_s : string
             Line style of the solar abundance distribution.
        color : string
             Line color.
        color_s : string
             Line color of the solar abundances pattern.
        label : string
             Figure label.
        label_s : string
            Label of the solar abundance distribution.
        fsize : 2D float array
             Figure dimension/size.
        fontsize : integer
             Font size of the numbers on the X and Y axis.
        rspace : float
             Extra space on the right for the legend.
        bspace : float
             Extra space at the bottom for the Y axis label.
        labelsize : integer
             Font size of the X and Y axis labels.
        legend_fontsize : integer
              Font size of the legend.
        markersize : float
             Size of the markers
             Default: markersize=5

        '''
        '''

        import matplotlib
        import matplotlib.pyplot as plt

        # Copy the list of selected isotopes
        if len(list_iso) > 0:
            iso_list_raw = list_iso
        else:
            iso_list_raw = self.history.isotopes

        # Make sure to remove Pb, as it is not in the right order in the yields..
        # This is causing problems.
        iso_list = []
        for i in range(0,len(iso_list_raw)):
            if not iso_list_raw[i].split('-')[0] == 'Pb':
                iso_list.append(iso_list_raw[i])

        # Read the solar normalization
        solar_file = ry.iniabu(os.path.join(nupy_path, solar_ab_m))

        # Make sure the wanted isotopes are available
        for i_iso in range(0, len(iso_list)):
            if iso_list[i_iso] not in self.history.isotopes:
                print (iso_list[i_iso], ' is not available..')
                return

        # Recover the solar isotopes mass fractions
        iso_sol = np.array(solar_file.iso_abundance(iso_list))

        # Copy the name of selected elements and isotopes
        iso_names_lower = []
        for i_el in range(0,len(iso_list)):
            iso_names_lower.append(iso_list[i_el].lower())

        # Copy the list of selected elements
        if len(list_elem) > 0 and len(list_iso) == 0:
             print ('The list_elem option is not implemented yet..')
        #    if elements are selected
        if True:  # Shoul be else once the list_elem is implemented..
            elem_sol_temp = self._iso_abu_to_elem(iso_sol, iso_list=iso_list)
            elem_names = elem_sol_temp[0]  # Name of element
            elem_sol = elem_sol_temp[1]    # Solar elemental abundances

        # Define the charges (Z)
        Z_numbers = []

        # Define the iso array [element][0 -> A; 1 -> abundance]
        iso_array_sol = []

        # For all isotopes found in the solar file
        i_Z_index = -1
        for i_iso in range(0,len(solar_file.names)):

            # Extract the name (in lower cases) and the A number
            name_temp = ''
            A_temp = ''
            for i_el in range(len(solar_file.names[i_iso])):
                if solar_file.names[i_iso][i_el].isdigit():
                    A_temp += solar_file.names[i_iso][i_el]
                elif not solar_file.names[i_iso][i_el] == ' ':
                    name_temp += solar_file.names[i_iso][i_el]
            A_temp = int(A_temp)
            iso_temp = name_temp + '-' + str(A_temp)
            name_temp_upper = name_temp.title()

            # If the isotope or element is considered ..
            considered = False
            #if len(list_elem) > 0 and len(list_iso) == 0:
            #    if name_temp_upper in elem_sol:
            #        considered = True
            if iso_temp in iso_names_lower:
                considered = True
            if considered:

                # Add the selected Z and A numbers
                if not solar_file.z[i_iso] in Z_numbers:
                    Z_numbers.append(solar_file.z[i_iso])
                    # Should take the solar element abundances here..
                    i_Z_index += 1
                    iso_array_sol.append([[],[],[],[]])
                i_iso_index = iso_names_lower.index(iso_temp)
                iso_array_sol[i_Z_index][0].append(A_temp)
                iso_array_sol[i_Z_index][1].append(iso_sol[i_iso_index])
                iso_name_index = iso_names_lower.index(iso_temp)
                iso_array_sol[i_Z_index][2].append(iso_list[iso_name_index])
                iso_array_sol[i_Z_index][3].append(name_temp_upper)

        # Get the index for given age
        ages = self.history.age
        timesteps=self.history.timesteps
        i_sim = min(range(len(ages)), key=lambda i: abs(ages[i]-age))

        print ('Extract abundance from closest available time ','{:.3E}'.format(ages[i_sim]),' yrs')
        print (self.history.metallicity[i_sim])


        # Get the mass of elements and isotopes at the desired time
        ymgal_el_sim  = self.history.ism_elem_yield[i_sim]
        ymgal_iso_sim = self.history.ism_iso_yield[i_sim]
        ism_mass_sim = np.sum(ymgal_el_sim)

        # Define the predicted element and isotope mass fractions
        elem_m_frac_sim = []
        iso_m_frac_sim = []
        for i_Z_index in range(0,len(iso_array_sol)):
            iso_m_frac_sim.append([])

        # For each selected element and isotope ..
        for i_Z_index in range(0,len(iso_array_sol)):

            # Get the mass of the current element
            i_Z_sim = self.history.elements.index(iso_array_sol[i_Z_index][3][0])
            elem_m_frac_sim.append(ymgal_el_sim[i_Z_sim]/ism_mass_sim)

            # For isotope having the same Z number ..
            for i_iso_index in range(0,len(iso_array_sol[i_Z_index][0])):
                i_iso_sim = self.history.isotopes.index(iso_array_sol[i_Z_index][2][i_iso_index])
                iso_m_frac_sim[i_Z_index].append(ymgal_iso_sim[i_iso_sim]/ism_mass_sim)

        # Normalize the predictions if needed ..
        if solar_norm:
            for i_Z in range(len(iso_array_sol)):
                elem_m_frac_sim[i_Z] = elem_m_frac_sim[i_Z] / elem_sol[i_Z]
                for i_iso in range(len(iso_array_sol[i_Z][1])):
                    iso_m_frac_sim[i_Z][i_iso] = iso_m_frac_sim[i_Z][i_iso] / iso_array_sol[i_Z][1][i_iso]

        # Calculate the min and max value of the A number
        A_nb_min = 1000
        A_nb_max = -1000
        for i_Z in range(len(iso_array_sol)):
            if max(iso_array_sol[i_Z][0]) > A_nb_max:
                A_nb_max = max(iso_array_sol[i_Z][0])
            if min(iso_array_sol[i_Z][0]) < A_nb_min:
                A_nb_min = min(iso_array_sol[i_Z][0])

        # If the data needs to be returned ..
        if return_x_y:
            if iso_on:
                return iso_m_frac_sim
            else:
                return Z_numbers, elem_m_frac_sim

        # If a figure is generated ..
        else:

            # Plot the frame
            plt.figure(fig, figsize=(fsize[0],fsize[1]))
            source = 'all'
            shape,marker,color=self.__msc(source,shape,marker,color)

            # If isotopes abundances ..
            if iso_on:

                # Plot the isotopes of each elements
                plt.xlabel('mass number A')
                if solar_norm:
                    plt.ylabel('X/X$_\odot$')
                    plt.plot([A_nb_min,A_nb_max+10],[1,1], '--k')
                else:
                    plt.ylabel('mass number A')
                    if over_plot_solar:
                        for i_Z in range(len(iso_array_sol)):
                            plt.plot(iso_array_sol[i_Z][0], iso_array_sol[i_Z][1], linestyle='--',\
                                 color=color_s,marker=marker,markersize=markersize)
                        # Get the legend
                        plt.plot(iso_array_sol[i_Z][0], iso_array_sol[i_Z][1], linestyle='--',\
                             color=color_s,label='Solar',marker=marker,markersize=markersize)
                for i_Z in range(len(iso_array_sol)):
                    plt.plot(iso_array_sol[i_Z][0], iso_m_frac_sim[i_Z], linestyle=shape,\
                         color=color,marker=marker,markersize=markersize)
                # Get the legend
                plt.plot(iso_array_sol[i_Z][0], iso_m_frac_sim[i_Z], linestyle=shape,\
                     color=color,label=label,marker=marker,markersize=markersize)


                if species_labels:
                     for i_Z in range(len(iso_array_sol)):
                          #print ('iso ',iso_array_sol[i_Z][2])
                          #for h in range(len(iso_array_sol[i_Z][2])):
                          plt.annotate(iso_array_sol[i_Z][2][0].split('-')[0],(iso_array_sol[i_Z][0][0], iso_m_frac_sim[i_Z][0]), \
                                xytext=(-2,0),textcoords='offset points',horizontalalignment='right', verticalalignment='top')

                # Annotate the element
                #ylimm = np.array(plt.gca().get_ylim())
                #if ylimm[0] == 0.0:
                #    ylimm[0] = 1.0e-15
                #print (ylimm)
                #yrange = np.log10(ylimm[1]) - np.log10(ylimm[0])
                #y_an = 10**(ylimm[1] - yrange * (1 - f_y_annotate))
                #for i_Z in range(len(iso_array_sol)):
                #    plt.annotate(iso_array_sol[i_Z][3][0], xy=(iso_array_sol[i_Z][0][0],y_an))
                #    print (iso_array_sol[i_Z][3][0], iso_array_sol[i_Z][0][0],y_an)

            # If elemental abundances ..
            else:

                # Plot the isotopes of each elements
                plt.xlabel('charge number Z')
                if solar_norm:
                    plt.ylabel('X/X$_\odot$')
                    plt.plot([Z_numbers[0],Z_numbers[-1]],[1,1], '--k')
                else:
                    plt.ylabel('X')
                    if over_plot_solar:
                        plt.plot(Z_numbers, elem_sol, linestyle=shape_s,\
                             color=color_s,label=label_s,marker=marker_s,markersize=markersize)
                plt.plot(Z_numbers, elem_m_frac_sim, linestyle=shape,\
                         color=color,label=label,marker=marker,markersize=markersize)

            # Plot visual aspect
            plt.yscale('log')
            if len(label)>0:
                    plt.legend()
            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,\
               rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize, markersize=markersize)
        '''


    ##############################################
    #             Create Log Folder              #
    ##############################################
    def create_log_folder(self, fsize=[10,4.5],fontsize=12,rspace=0.6,bspace=0.15,\
            labelsize=15,legend_fontsize=10):

        '''
        This function calculates the basic galaxy properties and put the
        relevant information in a folder

        '''

        import matplotlib
        import matplotlib.pyplot as plt

        # Create the folder
        log_path = self.inst_name + '_log/'
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        # Save the age-metallicity relationship
        xy = self.plot_spectro(return_x_y=True)
        fig = plt.figure()
        plt.plot(xy[0],xy[1])
        plt.xlabel('Age [yr]', fontsize=fontsize)
        plt.ylabel('[Fe/H]', fontsize=fontsize)
        plt.ylim(max(xy[1])-5, max(xy[1])+0.5)
        plt.savefig(log_path+'age_vs_FeH.pdf')
        plt.close(fig)

        # Save the log_age-metallicity relationship
        xy = self.plot_spectro(return_x_y=True)
        fig = plt.figure()
        plt.plot(xy[0],xy[1])
        plt.xlabel('Age [yr]', fontsize=fontsize)
        plt.ylabel('[Fe/H]', fontsize=fontsize)
        plt.ylim(max(xy[1])-5, max(xy[1])+0.5)
        plt.xlim(1e7, 2*self.history.tend)
        plt.xscale('log')
        plt.savefig(log_path+'log10_age_vs_FeH.pdf')
        plt.close(fig)

        # Save the star formation history
        fig = plt.figure()
        plt.plot(self.history.age,self.history.sfr_abs)
        plt.xlabel('Age [yr]', fontsize=fontsize)
        plt.ylabel('Star formation rate [Msun/yr]', fontsize=fontsize)
        plt.savefig(log_path+'star_formation_history.pdf')
        plt.close(fig)

        # Save the SN rates
        fig = plt.figure()
        y_cc = 100.0*np.array(self.history.sn2_numbers[1:])/ \
                 np.array(self.history.timesteps)
        plt.plot(self.history.age[1:],y_cc,linestyle='-',label='CC SNe')
        plt.plot(self.history.age[1:], \
                 100.0*np.array(self.history.sn1a_numbers[1:])/ \
                 np.array(self.history.timesteps),linestyle='--',\
                 label='SNe Ia')
        plt.xlabel('Age [yr]', fontsize=fontsize)
        plt.ylabel('Supernova rate [per century]', fontsize=fontsize)
        plt.yscale('log')
        plt.legend(fontsize=legend_fontsize)
        plt.ylim(max(y_cc)/1.0e4, 5.0*max(y_cc))
        plt.savefig(log_path+'sne_rate.pdf')
        plt.close(fig)

        # Open the output file
        f = open(log_path+'z_info.txt', 'w')

        # Write yields info
        f.write('Yields (see YIELDS_LIBRARY.txt for references)\n')
        f.write('==============================================\n')
        f.write('AGB and massive: '+self.table+'\n')
        f.write('SNe Ia: '+self.sn1a_table+'\n')
        f.write('Neutron star merger: '+self.nsmerger_table+'\n')

        # Write IMF info
        f.write('\nInitial mass function\n')
        f.write('=====================')
        f.write('Type: '+self.imf_type+'\n')
        f.write('Mass boundary: '+str(self.imf_bdys)+' [Msun]\n')
        f.write('Yields mass range: '+str(self.imf_yields_range)+' [Msun]\n')

        # Write galaxy properties
        f.write('\nGalaxy properties\n')
        f.write('=================\n')
        f.write('Initial mass of gas: '+str("%.4g" % np.sum(self.ymgal[0]))+' [Msun]\n')
        f.write('Final mass of gas: '+str("%.4g" % np.sum(self.ymgal[-1]))+' [Msun]\n')
        f.write('Final CC/Ia SN ratio: '+\
                 str("%.4g" % (self.history.sn2_numbers[-1]/\
                 self.history.sn1a_numbers[-1])+'\n'))
        f.write('Final metallicity (Z): ' + str(self.history.metallicity[-1]) + '\n')

        # Close the output file
        f.close()
