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

#standard packages
import matplotlib.pyplot as plt
import copy
import math
import random

# Import the class inherited by SYGMA
import sygma
from chem_evol import *


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
                 DM_evolution=False, Z_trans=1e-20, f_dyn=0.1, sfe=0.1, \
                 outflow_rate=-1.0, inflow_rate=-1.0, rand_sfh=0.0, cte_sfr=1.0, \
                 m_DM_0=1.0e11, mass_loading=1.0, t_star=-1.0, sfh_file='none', \
                 in_out_ratio=1.0, stellar_mass_0=-1.0, \
                 z_dependent=True, exp_ml=2.0, \
                 imf_type='kroupa', alphaimf=2.35, imf_bdys=[0.1,100], \
                 sn1a_rate='power_law', iniZ=0.0, dt=1e6, special_timesteps=30, \
                 tend=13e9, mgal=-1, transitionmass=8, iolevel=0, \
                 ini_alpha=True, \
                 table='yield_tables/isotope_yield_table_MESA_only.txt', \
                 hardsetZ=-1, sn1a_on=True,\
                 sn1a_table='yield_tables/sn1a_t86.txt',\
                 ns_merger_on=True, f_binary=1.0, f_merger=0.0028335,\
                 nsmerger_table = 'yield_tables/r_process.txt', iniabu_table='', \
                 extra_source_on=False, \
                 extra_source_table='yield_tables/mhdjet_NTT_delayed.txt', \
                 pop3_table='yield_tables/popIII_heger10.txt', \
                 imf_bdys_pop3=[0.1,100], imf_yields_range_pop3=[10,30], \
                 starbursts=[], beta_pow=-1.0, gauss_dtd=[1e9,6.6e8],exp_dtd=2e9,\
                 nb_1a_per_m=1.0e-3, f_arfo=1, t_merge=-1.0,\
                 imf_yields_range=[1,30],exclude_masses=[], \
		 netyields_on=False,wiersmamod=False,skip_zero=False,\
                 redshift_f=0.0,print_off=False,long_range_ref=False,\
                 f_s_enhance=1.0,m_gas_f=1.0e10,cl_SF_law=False,\
                 external_control=False, calc_SSP_ej=False,\
                 input_yields=False, popIII_on=True,\
                 sfh_array=np.array([]),ism_ini=np.array([]),\
                 mdot_ini=np.array([]), mdot_ini_t=np.array([]),\
                 ytables_in=np.array([]), zm_lifetime_grid_nugrid_in=np.array([]),\
                 isotopes_in=np.array([]), ytables_pop3_in=np.array([]),\
                 zm_lifetime_grid_pop3_in=np.array([]), ytables_1a_in=np.array([]),\
		 ytables_nsmerger_in=np.array([]), \
                 dt_in=np.array([]), dt_split_info=np.array([]),\
                 ej_massive=np.array([]), ej_agb=np.array([]),\
                 ej_sn1a=np.array([]), ej_massive_coef=np.array([]),\
                 ej_agb_coef=np.array([]), ej_sn1a_coef=np.array([]),\
                 dt_ssp=np.array([]),yield_interp='lin'):

        # Announce the beginning of the simulation 
        print 'OMEGA run in progress..'
        start_time = t_module.time()
	self.start_time = start_time

        # Call the init function of the class inherited by SYGMA
        chem_evol.__init__(self, imf_type=imf_type, alphaimf=alphaimf, \
                 imf_bdys=imf_bdys, sn1a_rate=sn1a_rate, iniZ=iniZ, dt=dt, \
                 special_timesteps=special_timesteps, tend=tend, mgal=mgal, \
                 transitionmass=transitionmass, iolevel=iolevel, \
                 ini_alpha=ini_alpha, table=table, hardsetZ=hardsetZ, \
                 sn1a_on=sn1a_on, sn1a_table=sn1a_table, \
		 ns_merger_on=ns_merger_on, f_binary=f_binary, f_merger=f_merger,\
                 nsmerger_table=nsmerger_table, \
                 iniabu_table=iniabu_table, extra_source_on=extra_source_on, \
                 extra_source_table=extra_source_table, pop3_table=pop3_table, \
                 imf_bdys_pop3=imf_bdys_pop3, \
                 imf_yields_range_pop3=imf_yields_range_pop3, \
                 starbursts=starbursts, beta_pow=beta_pow, \
                 gauss_dtd = gauss_dtd, exp_dtd = exp_dtd, \
                 nb_1a_per_m=nb_1a_per_m, Z_trans=Z_trans, f_arfo=f_arfo, \
                 imf_yields_range=imf_yields_range,exclude_masses=exclude_masses, \
                 netyields_on=netyields_on,wiersmamod=wiersmamod, \
                 input_yields=input_yields,\
                 t_merge=t_merge,popIII_on=popIII_on,\
                 ism_ini=ism_ini,ytables_in=ytables_in,\
                 zm_lifetime_grid_nugrid_in=zm_lifetime_grid_nugrid_in,\
                 isotopes_in=isotopes_in,ytables_pop3_in=ytables_pop3_in,\
                 zm_lifetime_grid_pop3_in=zm_lifetime_grid_pop3_in,\
                 ytables_1a_in=ytables_1a_in, \
		 ytables_nsmerger_in=ytables_nsmerger_in, dt_in=dt_in,\
                 dt_split_info=dt_split_info,ej_massive=ej_massive,\
                 ej_agb=ej_agb,ej_sn1a=ej_sn1a,\
                 ej_massive_coef=ej_massive_coef,ej_agb_coef=ej_agb_coef,\
                 ej_sn1a_coef=ej_sn1a_coef,dt_ssp=dt_ssp,\
		 yield_interp=yield_interp)

        if self.need_to_quit:
            return

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
        self.print_off = print_off
        self.long_range_ref = long_range_ref
        self.sfh_array = sfh_array
        self.mdot_ini = mdot_ini
        self.mdot_ini_t = mdot_ini_t
        self.m_gas_f = m_gas_f
        self.cl_SF_law = cl_SF_law
        self.external_control = external_control

        # Set cosmological parameters - Dunkley et al. (2009)
        self.omega_0   = 0.257   # Current mass density parameter
        self.omega_b_0 = 0.044   # Current baryonic mass density parameter
        self.lambda_0  = 0.742   # Current dark energy density parameter
        self.H_0       = 71.9    # Hubble constant [km s^-1 Mpc^-1]

        # Look for errors in the input parameters
        self.__check_inputs_omega()

        # Define whether the open box scenario is used or not
        if self.in_out_control or self.SF_law or self.DM_evolution:
            self.open_box = True
        else:
            self.open_box = False

        # Check if the timesteps need to be refined
        if self.SF_law or self.DM_evolution:
            self.t_SF_t = []
            self.redshift_t = []
            for k in range(self.nb_timesteps):
                self.t_SF_t.append(0.0)
                self.redshift_t.append(0.0)
            self.t_SF_t.append(0.0)
            self.redshift_t.append(0.0)
            self.__calculate_redshift_t()
            self.__calculate_t_SF_t()
            need_t_raf = False
            for i_raf in range(self.nb_timesteps):
                if self.history.timesteps[-1] > self.t_SF_t[i_raf] / self.sfe:
                    need_t_raf = True
                    break
            if need_t_raf:
              if self.long_range_ref:
                self.__rafine_steps_lr()
              else:
                self.__rafine_steps()

            # Re-Create entries for the mass-loss rate of massive stars
            self.massive_ej_rate = []
            for k in range(self.nb_timesteps + 1):
                self.massive_ej_rate.append(0.0)

        # Declare arrays used to follow the evolution of the galaxy
        self.__declare_evol_arrays()

        # If the mass fraction ejected by SSPs needs to be calculated ...
        # Need to be before self.__initialize_gal_prop()!!
        self.mass_frac_SSP = -1.0
        if calc_SSP_ej:

            # Run SYGMA with five different metallicities
            Z = [0.02, 0.01, 0.006, 0.001, 0.0001]
            s_inst = []
            self.mass_frac_SSP = 0.0
            for i_Z_SSP in range(0,len(Z)):
                s_inst = sygma.sygma(imf_type=imf_type, alphaimf=alphaimf,\
                 imf_bdys=imf_bdys, sn1a_rate=sn1a_rate, iniZ=Z[i_Z_SSP], dt=dt, \
                 special_timesteps=special_timesteps, tend=tend, mgal=1.0, \
                 transitionmass=transitionmass, iolevel=iolevel, \
                 ini_alpha=ini_alpha, table=table, hardsetZ=hardsetZ, \
                 sn1a_on=sn1a_on, sn1a_table=sn1a_table, \
                 iniabu_table=iniabu_table, extra_source_on=extra_source_on, \
                 extra_source_table=extra_source_table, pop3_table=pop3_table, \
                 imf_bdys_pop3=imf_bdys_pop3, \
                 imf_yields_range_pop3=imf_yields_range_pop3, \
                 starbursts=starbursts, beta_pow=beta_pow, \
                 gauss_dtd = gauss_dtd, exp_dtd = exp_dtd, \
                 nb_1a_per_m=nb_1a_per_m, Z_trans=Z_trans, f_arfo=f_arfo, \
                 imf_yields_range=imf_yields_range,exclude_masses=exclude_masses,\
                 netyields_on=netyields_on,wiersmamod=wiersmamod)
                self.mass_frac_SSP += sum(s_inst.ymgal[-1])

            # Calculate the average mass fraction returned
            self.mass_frac_SSP = self.mass_frac_SSP / len(Z)
            print 'Average SSP mass fraction returned = ',self.mass_frac_SSP

        # Set the general properties of the selected galaxy
        self.__initialize_gal_prop()

        # Fill arrays used to follow the evolution
        self.__fill_evol_arrays()

        # Read the primordial composition of the inflow gas
        if self.in_out_control or self.SF_law or self.DM_evolution:
            prim_comp_table = 'yield_tables/bb_walker91.txt'
	    self.prim_comp = ry.read_yield_sn1a_tables(global_path + \
                prim_comp_table, self.history.isotopes)

        # Assume the baryonic ratio for the initial gas reservoir, if needed
        if len(self.ism_ini) == 0 and not self.SF_law and not self.DM_evolution:
          if self.bar_ratio and not self.cl_SF_law:
            scale_m_tot = self.m_DM_0 * self.omega_b_0 / \
            (self.omega_0*sum(self.ymgal[0]))
            for k_cm in range(len(self.ymgal[0])):
                self.ymgal[0][k_cm] = self.ymgal[0][k_cm] * scale_m_tot

        # Add the stellar ejecta coming from external galaxies that just merged
        if len(self.mdot_ini) > 0:
            self.__add_ext_mdot()

        # If the timestep are not control by an external program ...
        if not self.external_control:

            # Run the simulation
            self.__run_simulation()


    ##############################################
    #             Check Inputs OMEGA             #
    ##############################################
    def __check_inputs_omega(self):

        '''
        This function checks for incompatible input entries, and stops
        the simulation if needed.

        '''

        # Input galaxy
        if not self.galaxy in ['none', 'milky_way', 'milky_way_cte', \
                               'sculptor', 'fornax', 'carina']:
            print 'Error - Selected galaxy not available.'
            return

        # Random SFH
        if self.rand_sfh > 0.0 and self.stellar_mass_0 < 0.0:
            print 'Error - You need to choose a current stellar mass.'
            return

        # Inflow control when non-available
        if self.in_out_control and (self.SF_law or self.DM_evolution):
            print 'Error - Cannot control inflows and outflows when SF_law or'\
                  'DM_evolution is equal to True.'
            return

        # Defined initial dark matter halo mass when non-available
        #if self.m_DM_ini > 0.0 and not self.DM_evolution:
        #    print 'Warning - Can\'t control m_DM_ini when the mass of', \
        #          'the dark matter halo is not evolving.'

        # Inflow and outflow control when the dark matter mass if evolving
        if (self.outflow_rate >= 0.0 or self.inflow_rate >= 0.0) and \
            self.DM_evolution:
            print 'Error - Cannot fix inflow and outflow rates when the mass'\
                  'of the dark matter halo is evolving.'
            return


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
            print '..Time refinement..'
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

        # Update/redeclare all the arrays
        ymgal = self._get_iniabu()
        self.len_ymgal = len(ymgal)
        self.mdot, self.ymgal, self.ymgal_massive, self.ymgal_agb, \
        self.mgal_1a, self.ymgal_nsm, self.mdot_massive, self.mdot_agb, self.mdot_1a, \
        self.mdot_nsm, self.sn1a_numbers, self.sn2_numbers, self.nsm_numbers, \
 	self.imf_mass_ranges, \
        self.imf_mass_ranges_contribution, self.imf_mass_ranges_mtot = \
        self._get_storing_arrays(ymgal)

        # Initialisation of the composition of the gas reservoir
        if len(self.ism_ini) > 0:
            for i_ini in range(0,self.len_ymgal):
                self.ymgal[0][i_ini] = self.ism_ini[i_ini]


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
            print '..Time refinement (long range)..'
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

        # Update/redeclare all the arrays
        ymgal = self._get_iniabu()
        self.len_ymgal = len(ymgal)
        self.mdot, self.ymgal, self.ymgal_massive, self.ymgal_agb, \
        self.ymgal_1a, self.ymgal_nsm, self.mdot_massive, self.mdot_agb, self.mdot_1a, \
        self.mdot_nsm, self.sn1a_numbers, self.sn2_numbers, self.nsm_numbers, \
	self.imf_mass_ranges, self.imf_mass_ranges_contribution, self.imf_mass_ranges_mtot = \
        self._get_storing_arrays(ymgal)

        # Initialisation of the composition of the gas reservoir
        if len(self.ism_ini) > 0:
            for i_ini in range(0,self.len_ymgal):
                self.ymgal[0][i_ini] = self.ism_ini[i_ini]


    ##############################################
    #            Declare Evol Arrays             #
    ##############################################
    def __declare_evol_arrays(self):

        '''
        This function declares the arrays used to follow the evolution of the
        galaxy regarding its growth and the exchange of gas with its surrounding.

        '''

        # Arrays with specific values at every timestep
        self.sfr_input = []     # Star formation rate [Mo yr^-1]
        self.m_DM_t = []        # Mass of the dark matter halo
        self.m_tot_ISM_t = []   # Mass of the ISM in gas at every timestep
        self.m_outflow_t = []   # Mass of the outflow at every timestep
        self.eta_outflow_t = [] # Mass-loading factor == M_outflow / SFR
        self.t_SF_t = []        # Star formation timescale at every timestep
        self.redshift_t = []    # Redshift associated to every timestep
        self.m_inflow_t = []    # Mass of the inflow at every timestep

        # Extends the arrays to cover all timestep
        for k in range(self.nb_timesteps):
            self.sfr_input.append(0.0)
            self.m_DM_t.append(0.0)
            self.m_tot_ISM_t.append(0.0)
            self.m_outflow_t.append(0.0)
            self.eta_outflow_t.append(0.0)
            self.t_SF_t.append(0.0)
            self.redshift_t.append(0.0)
            self.m_inflow_t.append(0.0)

        # Add one additional slot for t = tend when needed
        self.sfr_input.append(0.0)
        self.m_DM_t.append(0.0)
        self.m_tot_ISM_t.append(0.0)
        self.t_SF_t.append(0.0)
        self.redshift_t.append(0.0)


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
            if not self.sfh_file == 'none':
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
                self.__copy_sfr_input('milky_way_data/sfh_mw_cmr01.txt')

            # Read constant SFH
            else:
                self.__copy_sfr_input('milky_way_data/sfh_cte.txt')

        # Sculptor dwarf galaxy ...
        elif self.galaxy == 'sculptor':

            # Set the current dark and stellar masses (corrected for mass loss)
            self.m_DM_0 = 1.5e9
            self.stellar_mass_0 = 7.8e6
            self.stellar_mass_0 = self.stellar_mass_0 * 0.5

            # Read deBoer et al. (2012) SFH
            self.__copy_sfr_input('sculptor_data/sfh_deBoer12.txt')

        # Fornax dwarf galaxy ...
        elif self.galaxy == 'fornax':

            # Set the current dark and stellar masses (corrected for mass loss)
            self.m_DM_0 = 7.08e8
            self.stellar_mass_0 = 4.3e7
            self.stellar_mass_0 = self.stellar_mass_0 * 0.5

            # Read deBoer et al. (2012) SFH
            self.__copy_sfr_input('fornax_data/sfh_fornax_deboer_et_al_2012.txt')

        # Carina dwarf galaxy ...
        elif self.galaxy == 'carina':

            # Set the current dark and stellar masses (corrected for mass loss)
            self.m_DM_0 = 3.4e6
            self.stellar_mass_0 = 1.07e6
            self.stellar_mass_0 = self.stellar_mass_0 * 0.5

            # Read deBoer et al. (2014) SFH
            self.__copy_sfr_input('carina_data/sfh_deBoer14.txt')

        # Keep the SFH in memory
        self.history.sfr_abs = self.sfr_input


    ##############################################
    ##             Copy SFR Array               ##
    ##############################################
    def __copy_sfr_array(self):

        '''
        See copy_sfr_input() for more info.

        '''
   
        # Variable to keep track of the SYGMA's time step
        i_dt_csa = 0
        t_csa = self.history.timesteps[0]
        nb_dt_csa = len(self.history.timesteps) + 1

        # Variable to keep track of the total stellar mass from the input SFH
        m_stel_sfr_in = 0.0

        # For every timestep given in the array (starting at the second step)
        for i_csa in range(1,len(self.sfh_array)):

            # While we stay in the same time bin ...
            while t_csa <= self.sfh_array[i_csa][0]:

                # Keep the same SFR INSIDE this time bin (it's why i_csa-1)
                self.sfr_input[i_dt_csa] = self.sfh_array[i_csa-1][1]

                # Go to the next time step
                i_dt_csa += 1

                # Exit the loop if the array is full
                if i_dt_csa >= nb_dt_csa:
                    break

                # Calculate the new time
                t_csa += self.history.timesteps[i_dt_csa]

            # Exit the loop if the array is full
            if i_dt_csa >= nb_dt_csa:
                break

            # Calculate the time spent in the previous array time bin
            dt_prev_array = self.history.timesteps[i_dt_csa] - \
                 t_csa + self.sfh_array[i_csa][0]
 
            # Calculate the mass formed during the OMEGA's timestep
            m_stel_omega = dt_prev_array * self.sfh_array[i_csa-1][1] + \
                (t_csa - self.sfh_array[i_csa][0])*self.sfh_array[i_csa][1]

            # Assign the average SFR
            self.sfr_input[i_dt_csa] = m_stel_omega / \
                 self.history.timesteps[i_dt_csa]

            # Go to the next time step
            i_dt_csa += 1

            # Exit the loop if the array is full
            if i_dt_csa >= nb_dt_csa:
                break

            # Calculate the new time
            t_csa += self.history.timesteps[i_dt_csa]

        # If the array has been read completely, but the sfr_input array is
        # not full, fil the rest of the array with the last read value
        while i_dt_csa < nb_dt_csa:
            self.sfr_input[i_dt_csa] = self.sfr_input[i_dt_csa-1]
            i_dt_csa += 1


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

        # Calculate the initial mass of gas
        m_gas_ini = self.m_gas_f + self.stellar_mass_0

        # Scale the initial mass of all isotopes
        scale_m_tot = m_gas_ini / self.mgal
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
        with open(global_path+path_sfh_in, 'r') as sfr_file:

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
            norm_sfr_in = self.stellar_mass_0 / (0.5 * m_stel_sfr_in)
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
        norm_sfr_in = self.stellar_mass_0 / (0.5 * m_stel_sfr_in)
        for i_csi in range(0,len(timesteps)+1):
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
            self.__calculate_redshift_t()

            # Calculate the mass of the dark matter halo at every timestep
            self.__calculate_m_DM_t()

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
        temp_var = math.sqrt((self.lambda_0/self.omega_0)/(1+z_gttfz)**3)
        x_var = math.log( temp_var + math.sqrt( temp_var**2 + 1 ) )
        return 2 / ( 3 * self.H_0 * math.sqrt(self.lambda_0)) * \
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
                math.sinh(temp_var)**0.66666667 - 1


    ##############################################
    #           Calculate redshift(t)            #
    ##############################################
    def __calculate_redshift_t(self):

        '''
        This function calculates the redshift associated to every timestep
        assuming that 'tend' represents redshift zero.

        '''
 
        # Execute the function only if needed
        if self.DM_evolution:

            # Calculate the current age of the Universe (LambdaCDM - z = 0)
            current_age_czt = self.__get_t_from_z(self.redshift_f)

            # Calculate the age of the Universe when the galaxy forms
            age_formation_czt = current_age_czt - self.history.tend

            # Initiate the age of the galaxy
            t_czt = 0.0

            #For each timestep
            for i_czt in range(0, self.nb_timesteps+1):

                #Calculate the age of the Universe at that time [yr]
                age_universe_czt = age_formation_czt + t_czt

                #Calculate the redshift at that time
                self.redshift_t[i_czt] = self.__get_z_from_t(age_universe_czt)

                #Udpate the age of the galaxy [yr]
                if i_czt < self.nb_timesteps:
                    t_czt += self.history.timesteps[i_czt]

            #Correction for last digit error (e.g. z = -2.124325345e-8)
            if self.redshift_t[-1] < 0.0:
                self.redshift_t[-1] = 0.0


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

            # Use the current value for every timestep
            for i_cmdt in range(0, self.nb_timesteps+1):
                self.m_DM_t[i_cmdt] = self.m_DM_0

        # If the mass of the dark matter halo evolves with time ...
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
        with open(global_path+"m_dm_evolution/poly3_fits.txt", 'r') as m_dm_file:

            # Read the first line
            line_str = m_dm_file.readline()
            parts_1 = [float(x) for x in line_str.split()]

            # If the input dark matter mass is higher than the ones provided
            # by the fits ...
            if math.log10(self.m_DM_0) > parts_1[3]:

                # Use the highest dark matter mass available
                parts_2 = copy.copy(parts_1)
                print 'Warning - Current dark matter mass too high for' \
                      'the available fits.'

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
                            self.redshift_t[i_ctst])**(-1.5) / self.H_0 * \
                                9.7759839e11

                    # If the dark matter mass is not evolving ...
                    else:
                        self.t_SF_t[i_ctst] = self.f_dyn * 0.1 / self.H_0 * \
                                9.7759839e11


    ##############################################
    #           Calculate M_tot ISM(t)           #
    ##############################################
    def __calculate_m_tot_ISM_t(self):

        '''
        This function calculates the mass of the gas reservoir at every 
        timestep using a classical star formation law.

        '''

        # Execute this function only if needed
        if self.SF_law or self.DM_evolution:

            # For each timestep ...
            for i_cm in range(0, self.nb_timesteps+1):

                # If it's the last timestep ... use the previous sfr_input
                if i_cm == self.nb_timesteps:

                    # Calculate the total mass of the ISM using the previous SFR
                    self.m_tot_ISM_t[i_cm] = self.sfr_input[i_cm-1] * \
                        self.t_SF_t[i_cm] / self.sfe

                # If it's not the last timestep ...
                else:

                    # Calculate the total mass of the ISM using the current SFR
                    self.m_tot_ISM_t[i_cm] = self.sfr_input[i_cm] * \
                        self.t_SF_t[i_cm] / self.sfe

            # Scale the initial gas reservoir that was already set
            scale_m_tot = self.m_tot_ISM_t[0] / sum(self.ymgal[0])
            for k_cm in range(len(self.ymgal[0])):
                self.ymgal[0][k_cm] = self.ymgal[0][k_cm] * scale_m_tot


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
                    for i_iso in range(0,self.nb_isotopes):
                        self.mdot[i_cur][i_iso] += \
                            self.mdot_ini[i_merg][i_ext][i_iso] * f_dt

                    # Move to the next external bin
                    i_ext += 1
                    if i_ext == len_mdot_ini_i_merg:
                        break
                    t_ext_prev = t_ext
                    t_ext = self.mdot_ini_t[i_merg][i_ext+1]

                # Quit the loop if all external bins have been considered
                if i_ext == len_mdot_ini_i_merg:
                    break

                # While we need to change the current time bin ...
                while t_cur < t_ext:

                    # Calculate the overlap time between ext. and cur. bins
                    dt_trans = t_cur - max([t_ext_prev, t_cur_prev]) 

                    # Calculate the mass fraction that needs to be transfered
                    f_dt = dt_trans / (t_ext - t_ext_prev)

                    # Transfer all isotopes in the current mdot array
                    for i_iso in range(0,self.nb_isotopes):
                        self.mdot[i_cur][i_iso] += \
                            self.mdot_ini[i_merg][i_ext][i_iso] * f_dt

                    # Move to the next current bin
                    i_cur += 1
                    if i_cur == self.nb_timesteps:
                        break
                    t_cur_prev = t_cur
                    t_cur += self.history.timesteps[i_cur]


    ##############################################
    #                Run Simulation              #
    ##############################################
    def __run_simulation(self):

        '''
        This function calculates the evolution of the chemical abundances of a
        galaxy as a function of time.
         
        '''

        # For every timestep i considered in the simulation ...
        for i in range(1, self.nb_timesteps+1):

            # Run a timestep using the input SFR
            self.run_step(i, self.sfr_input[i-1])
            
        # Calculate the last SFR at the end point of the simulation
        if self.cl_SF_law and not self.open_box:
            self.history.sfr_abs[-1] = self.sfe_gcs * sum(self.ymgal[i])


    ##############################################
    #                   Run Step                 #
    ##############################################
    def run_step(self, i, sfr_rs, m_added = np.array([]), m_lost = 0.0):

        '''
        This function calculates the evolution of one single step in the
        chemical evolution.

        Argument
        ========

          i : Index of the timestep.
          sfr_rs : Input star formation rate [Mo/yr] for the step i.
          m_added : Mass (and composition) added for the step i.
          m_lost : Mass lost for the step i.
         
        '''

        # Make sure that the the number of timestep is not exceeded
        if not i == (self.nb_timesteps+1):

            #test
            if i == 1:
                self.sfr_test = sfr_rs

            # Calculate the current mass fraction of gas converted into stars
            self.__cal_m_frac_stars(i, sfr_rs)

            # Run the timestep i (!need to be right after __cal_m_frac_stars!)
            self._evol_stars(i)

            # Add the incoming gas (if any)
            len_m_added = len(m_added)
            for k_op in range(0, len_m_added):
                self.ymgal[i][k_op] += m_added[k_op]

            # If gas needs to be removed ...
            if m_lost > 0.0:

                # Calculate the gas fraction lost
                f_lost = m_lost / sum(self.ymgal[i])
                if f_lost > 1.0:
                    f_lost = 1.0
                    print '!!Warning -- Remove more mass than available!!'
                
                # Remove the mass for each isotope
                f_lost_2 = (1.0 - f_lost)
                for k_op in range(0, self.nb_isotopes):
                    self.ymgal[i][k_op] = f_lost_2 * self.ymgal[i][k_op]
                    self.ymgal_agb[i][k_op] = f_lost_2 * self.ymgal_agb[i][k_op]
                    self.ymgal_1a[i][k_op] = f_lost_2 * self.ymgal_1a[i][k_op]
		    self.ymgal_nsm[i][k_op] = f_lost_2 * self.ymgal_nsm[i][k_op]
                    self.ymgal_massive[i][k_op] = f_lost_2*self.ymgal_massive[i][k_op]

            # If the open box scenario is used ...
            if self.open_box:

                # Calculate the total mass of the gas reservoir at timstep i
                # after the star formation and the stellar ejecta
                m_tot_current = 0.0
                for k_op in range(0, self.nb_isotopes):
                    m_tot_current += self.ymgal[i][k_op]

                # Get the current mass of inflow
                m_inflow_current = self.__get_m_inflow(i, m_tot_current)

                # Add primordial gas coming with the inflow
                if m_inflow_current > 0.0:
                    ym_inflow = self.prim_comp.get(quantity='Yields') * \
                                m_inflow_current
                    for k_op in range(0, self.nb_isotopes):
                        self.ymgal[i][k_op] += ym_inflow[k_op]
                
                #Calculate the fraction of gas removed by the outflow
                if not (m_tot_current + m_inflow_current) == 0.0:
                    frac_rem = self.m_outflow_t[i-1] / \
                        (m_tot_current + m_inflow_current)
                else:
                    frac_rem = 0.0

                # Limit the outflow mass to the amount of available gas
                if frac_rem > 1.0:
                    frac_rem = 1.0
                    self.m_outflow_t[i-1] = m_tot_current + m_inflow_current
                    if not self.print_off:
                        print 'Warning - '\
                          'Outflows eject more mass than available.  ' \
                          'It has been reduced to the amount of available gas.'

                # Remove mass from the ISM because of the outflow
                for k_op in range(0, self.nb_isotopes):
                    self.ymgal[i][k_op] = (1.0 - frac_rem) * \
                        self.ymgal[i][k_op]
                    self.ymgal_agb[i][k_op] = (1.0 - frac_rem) * \
                        self.ymgal_agb[i][k_op]
                    self.ymgal_1a[i][k_op] = (1.0 - frac_rem) * \
                        self.ymgal_1a[i][k_op]
		    self.ymgal_nsm[i][k_op] = (1.0 - frac_rem) * \
			self.ymgal_nsm[i][k_op]
                    self.ymgal_massive[i][k_op] = (1.0 - frac_rem) * \
                        self.ymgal_massive[i][k_op]

            # Get the new metallicity of the gas
            self.zmetal = self._getmetallicity(i)

            # Update the history class
            self._update_history(i)

            # If this is the last step ...
            if i == self.nb_timesteps:

                # Do the final update of the history class
                self._update_history_final()

                # Add the evolution arrays to the history class
                self.history.m_DM_t = self.m_DM_t
                self.history.m_tot_ISM_t = self.m_tot_ISM_t
                self.history.m_outflow_t = self.m_outflow_t
                self.history.m_inflow_t = self.m_inflow_t
                self.history.eta_outflow_t = self.eta_outflow_t
                self.history.t_SF_t = self.t_SF_t
                self.history.redshift_t = self.redshift_t

                # If external control ...
                if self.external_control:
                    self.history.sfr_abs[i] = self.history.sfr_abs[i-1]

                # Announce the end of the simulation
                print '   OMEGA run completed -',self._gettime()

        # Error message
        else:
            print 'The simulation is already over.'


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
            self.history.sfr_abs[i-1] = self.sfe_gcs * sum(self.ymgal[i-1])
            self.sfrin = self.history.sfr_abs[i-1] * self.history.timesteps[i-1]

        else:

            # Calculate the total mass of stars formed during this timestep
            self.sfrin = sfr_rs * self.history.timesteps[i-1]
            self.history.sfr_abs[i-1] = sfr_rs

        # Calculate the mass fraction of gas converted into stars
        mgal_tot = 0.0
        for k_ml in range(0, self.nb_isotopes):
            mgal_tot += self.ymgal[i-1][k_ml]
        self.sfrin = self.sfrin / mgal_tot

        # Modify the history of SFR if there is not enough gas
        if self.sfrin > 1.0:
           self.history.sfr_abs[i-1] = mgal_tot / self.history.timesteps[i-1]


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

        # If the constant inflow rate is kept constant ...
        if self.inflow_rate >= 0.0:

            # Use the input rate to calculate the inflow mass
            # Note : i-1 --> current timestep, see __copy_sfr_input()
            m_inflow_current = self.inflow_rate * self.history.timesteps[i-1]

        # If the inflow rate follows the outflow rate ...
        elif self.in_out_control:

            # Use the input scale factor to calculate the inflow mass
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
                            print 'Warning - Negative inflow.  ' \
                              'The outflow rate has been increased.', i

                    # Assume no inflow
                    m_inflow_current = 0.0

        # Keep the mass of inflow in memory
        self.m_inflow_t[i-1] = float(m_inflow_current)

        return m_inflow_current


###############################################################################################
######################## Here start the analysis methods ######################################
###############################################################################################



    def plot_mass(self,fig=0,specie='C',source='all',norm=False,label='',shape='',marker='',color='',markevery=20,multiplot=False,return_x_y=False,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14):
    
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
 	       
        Examples
	----------

	>>> s.plot('C-12')

        '''
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
            print 'Isotope or element not available'
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
            plt.plot(x,y,label=label,linestyle=shape,marker=marker,color=color,markevery=markevery)
            plt.legend()
            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)
	    plt.xlim(self.history.dt,self.history.tend)	
	    #return x,y
	    self.save_data(header=['Age[yrs]',specie],data=[x,y])


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
	           x.append(yields_evol[k][iso_idx]/sum(yields_evol[k]))
	       if norm=='ini':
	           x.append(yields_evol[k][iso_idx]/sum(yields_evol[k])/yields_evol[0][iso_idx])
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
                    x.append(yields_evol[k][iso_idx]/sum(yields_evol[k]))
                if norm=='ini':
                    x.append(yields_evol[k][iso_idx]/sum(yields_evol[k])/yields_evol[0][iso_idx])
                    print yields_evol[0][iso_idx]

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
                if sum(yields_evol[k]) ==0:
                    continue
                if norm=='no':
                    y.append(yields_evol[k][iso_idx]/sum(yields_evol[k]))
                if norm=='ini':
                    y.append(yields_evol[k][iso_idx]/sum(yields_evol[k])/yields_evol[0][iso_idx])

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
                if sum(yields_evol[k]) ==0:
                    continue
                if norm=='no':
                    y.append(yields_evol[k][iso_idx]/sum(yields_evol[k]))
                if norm=='ini':
                    y.append(yields_evol[k][iso_idx]/sum(yields_evol[k])/yields_evol[0][iso_idx])

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

    def plot_spectro(self,fig=3,xaxis='age',yaxis='[Fe/H]',source='all',label='',shape='-',marker='o',color='k',markevery=100,show_data=False,show_sculptor=False,show_legend=True,return_x_y=False,sub_plot=False,linewidth=3,sub=1,plot_data=False,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14,only_one_iso=False,solar_ab=''):
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

        #Error message if there is the "subplot" has not been provided
        if sub_plot and sub == 1:
            print '!! Error - You need to use the \'sub\' parameter and provide the frame for the plot !!'
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
            iniabu=ry.iniabu(global_path+solar_ab)
        else:
            iniabu=ry.iniabu(global_path+'yield_tables/iniabu/iniab2.0E-02GN93.ppn')

        x_ini_iso=iniabu.iso_abundance(self.history.isotopes)
        elements,x_ini=self._iso_abu_to_elem(x_ini_iso)
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

            #print xaxis_elem1, xaxis_elem2

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
                if sum(yields_evol[k]) ==0:
                    continue
                #in case no contribution during timestep
                x1=yields_evol[k][elem_idx1]/sum(yields_evol[k])
                x2=yields_evol[k][elem_idx2]/sum(yields_evol[k])
                if x1 <= 0.0 or x2 <= 0.0:
                    spec = -30.0
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
        #print 'Fe_sol = ',x_elem1_ini,' , H_sol = ',x_elem2_ini

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
            if sum(yields_evol[k]) ==0:
                continue
            #in case no contribution during timestep
            x1=yields_evol[k][elem_idx1]/sum(yields_evol[k])
            x2=yields_evol[k][elem_idx2]/sum(yields_evol[k])
            if x1 <= 0.0 or x2 <= 0.0:
                spec = -30.0
            else:
                #print 'Fe_sim = ',x1, ' , H_sim = ',x2
                spec=np.log10(x1/x2) - np.log10(x_elem1_ini/x_elem2_ini)
            y.append(spec)
            if xaxis=='age':
                x.append(x_age[k])
        if len(y)==0:
            print 'Y values all zero'
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
	    self.save_data(header=[xaxis,yaxis],data=[x,y])


    def plot_totmasses(self,fig=4,mass='gas',source='all',norm='no',label='',shape='',marker='',color='',markevery=20,log=True,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14):
        '''
	Plots either gas or star mass in fraction of total mass
	vs time.
        
        Parameters
        ----------

        mass : string
            either 'gas' for ISM gas mass
            or 'stars' for gas locked away in stars (totalgas - ISM gas)

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
	        gas_evol.append(sum(yields_evol[k]))

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
	self.save_data(header=['age','mass'],data=[x,y])


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
	#For Wiersma09
	Hubble_0=73.
	Omega_lambda=0.762
	Omega_m=0.238

        figure=plt.figure(fig, figsize=(fsize[0],fsize[1]))
        age=self.history.age
        sn1anumbers=self.history.sn1a_numbers#[:-1]
        sn2numbers=self.history.sn2_numbers
	if xaxis=='redshift':
		print 'this features is not tested yet.'
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
                self.save_data(header=['age',label],data=[x,y])


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

	marker : string
		marker type
	shape : string
		line shape type
	fig: figure id
	
        Examples
        ----------
        >>> s.star_formation_rate()
	
	'''

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
		print 'age',len(mean_age)
		print 'weights',len(sfr)
		print 'bdy',len(age_bdy)
		print sfr[:5]
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
            plt.plot(age,sfr_plot,label=label,marker=marker,color=color,linestyle=shape)

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
		print len(age),len(sfr)
		plt.xlabel('Age [yrs]')
		plt.plot(age,sfr,label=label,marker=marker,linestyle=shape)
		plt.ylabel('Fraction of current gas mass into stars')	
        	self.save_data(header=['age','SFR'],data=[age,sfr])

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

	#print 'Total mass transformed in stars, total mass transformed in AGBs, total mass transformed in massive stars:'
	#print sum(self.history.m_locked),sum(self.history.m_locked_agb),sum(self.history.m_locked_massive)



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
			time_new.append(time[k]/1e9)
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
            self.save_data(header=['age','mass-loading'],data=[age,mass_lo])

            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,\
                rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)

        else:
            print 'Not available with a closed box.'


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
                outflow_plot.append(self.history.m_outflow_t[i_op] / \
                    self.history.timesteps[i_op])
            outflow_plot[0] = 0.0

            #Plot data
            plt.plot(age,outflow_plot,label=label,marker=marker,\
                     color=color,linestyle=shape)

            #Save plot
            self.save_data(header=['age','outflow rate'],data=[age,outflow_plot])

            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,\
                rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)

        else:
            print 'Not available with a closed box.'


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
                inflow_plot.append(self.history.m_inflow_t[i_op] / \
                    self.history.timesteps[i_op])

            #Plot data
            plt.plot(age,inflow_plot,label=label,marker=marker,\
                     color=color,linestyle=shape)

            #Save plot
            self.save_data(header=['age','inflow rate'],data=[age,inflow_plot])

            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,\
                rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)

        else:
            print 'Not available with a closed box.'


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
            DM_plot = self.history.m_DM_t

            #Plot data
            plt.plot(age,DM_plot,label=label,marker=marker,\
                     color=color,linestyle=shape)

            #Save plot
            self.save_data(header=['age','dark matter halo mass'],\
                           data=[age,DM_plot])

            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,\
                rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)

        else:
            print 'Not available with a closed box.'


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

            #print len(self.history.age)
            #print len(self.history.t_SF_t)

            #Copy data for x and y axis
            age = self.history.age
            t_SF_plot = self.history.t_SF_t

            #Plot data
            plt.plot(age,t_SF_plot,label=label,marker=marker,\
                     color=color,linestyle=shape)

            #Save plot
            self.save_data(header=['age','star formation timescale'],\
                                   data=[age,t_SF_plot])

            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,\
                rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)

        else:
            print 'Not available with a closed box.'


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
            redshift_plot = self.history.redshift_t

            #Plot data
            plt.plot(age,redshift_plot,label=label,marker=marker,\
                     color=color,linestyle=shape)

            #Save plot
            self.save_data(header=['age','redshift'],data=[age,redshift_plot])

            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,\
                rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)

        else:
            print 'Not available with a closed box.'


    ###################################################
    #                  Plot Iso Ratio                 #
    ###################################################
    def plot_iso_ratio(self,return_x_y=False,
        xaxis='age',yaxis='C-12/C-13',\
        solar_ab='yield_tables/iniabu/iniab2.0E-02GN93.ppn',\
        solar_iso='solar_normalization/Asplund_et_al_2009_iso.txt',\
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
	     If False, show the plot.  If True, return two arrays containing
	     the X and Y axis data, respectively.
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
	    SN Ia ('sn1a'), or only massive stars
	    ('massive')
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

        # Marker on the plots
        markevery=500

        # Declaration of the array to plot
        x = []
        y = []

        #Access solar abundance
        iniabu=ry.iniabu(global_path+solar_ab)
        x_ini_iso=iniabu.iso_abundance(self.history.isotopes)
        elements,x_ini=self._iso_abu_to_elem(x_ini_iso)

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
            print '!! Source not valid !!'
            return

        # Verify the X-axis
        xaxis_ratio = False
        if xaxis == 'age':
            x = self.history.age
        elif xaxis[0] == '[' and xaxis[-1] == ']':
            xaxis_elem1 =xaxis.split('/')[0][1:]
            xaxis_elem2 =xaxis.split('/')[1][:-1]
            if not xaxis_elem1 in self.history.elements and \
               not xaxis_elem2 in self.history.elements:
                 print '!! Elements in xaxis are not valid !!'
                 return

            #X-axis ini values
            x_elem1_ini=x_ini[elements.index(xaxis_elem1)]
            x_elem2_ini=x_ini[elements.index(xaxis_elem2)]

            #X-axis gce values
            elem_idx1=self.history.elements.index(xaxis_elem1)
            elem_idx2=self.history.elements.index(xaxis_elem2)

            for k in range(0,len(yields_evol_el)):
                if sum(yields_evol_el[k]) == 0:
                    continue
                #in case no contribution during timestep
                x1=yields_evol_el[k][elem_idx1]/sum(yields_evol_el[k])
                x2=yields_evol_el[k][elem_idx2]/sum(yields_evol_el[k])
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
                print '!! xaxis not valid.  Need to be \'age\' or a ratio !!'
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
            print '!! yaxis not valid.  Need to be a ratio !!'
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
            print '!! Isotopes in xaxis are not valid !!'
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
            print '!! Isotopes in yaxis are not valid !!'
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

        # Get the solar values
        if xaxis_ratio:
            x1_sol, x2_sol, y1_sol, y2_sol = \
              self.__read_solar_iso(solar_iso, x_1, x_2, y_1, y_2)
        else:
            x1_sol, x2_sol, y1_sol, y2_sol = \
                self.__read_solar_iso(solar_iso, '', '', y_1, y_2)

        # Calculate the isotope ratios (delta notation)
        for k in range(0,len(yields_evol)):
            if xaxis_ratio:
                ratio_sample = (yields_evol[k][idx_1]/yields_evol[k][idx_2])*\
                               (x2_at_nb / x1_at_nb)
                ratio_std = x1_sol / x2_sol
                x.append( ((ratio_sample/ratio_std) - 1) * 1000)
            ratio_sample = (yields_evol[k][idy_1]/yields_evol[k][idy_2])*\
                           (y2_at_nb / y1_at_nb)
            ratio_std = y1_sol / y2_sol
            y.append( ((ratio_sample/ratio_std) - 1) * 1000)
  
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
           plt.ylabel("$\delta$($^{"+str(int(y1_at_nb))+"}$"+y1_elem+"/$^{"+\
                           str(int(y2_at_nb))+"}$"+y2_elem+")")
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
	    self.save_data(header=['Iso mass ratio',yaxis],data=[x,y])


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
        with open(global_path+file_path, 'r') as data_file:

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
        marker='',shape='',\
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
        cte = 1.0 / max(mdf_y)
        for i in range(0, len(mdf_y)):
            mdf_y[i] = mdf_y[i] * cte

        # If the gaussian approximation is wanted ...
        if sigma_gauss > 0.0:
            mdf_x, mdf_y = \
                self.__gauss_approx(mdf_x, mdf_y, sigma_gauss, dx, nb_sigma)

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
             solar_ab='yield_tables/iniabu/iniab2.0E-02GN93.ppn'):

        # Access solar abundance
        iniabu = ry.iniabu(global_path+solar_ab)
        x_ini_iso = iniabu.iso_abundance(self.history.isotopes)

        # Access 
        elements, x_ini = self._iso_abu_to_elem(x_ini_iso)
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
            x1 = yields_evol[k][elem_idx1] / sum(yields_evol[k])
            x2 = yields_evol[k][elem_idx2] / sum(yields_evol[k])
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
    def __gauss_approx(self, mdf_x, mdf_y, sigma_gauss, dx, nb_sigma):

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
        cte = 1.0 / max(mdf_y_ga)
        for i in range(0, len(mdf_y_ga)):
            mdf_y_ga[i] = mdf_y_ga[i] * cte       

        return mdf_x, mdf_y_ga
