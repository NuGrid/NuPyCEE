from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

'''

GCE SYGMA (Stellar Yields for Galaxy Modelling Applications) module

Functionality
=============

This tool allows the modeling of simple stellar populations.  Creating a SYGMA
instance runs the simulation while extensive analysis can be done with the plot_*
functions found in the chem_evol_plot module.  See the DOC directory for a detailed
documentation.


Made by
=======

v0.1 NOV2013: C. Fryer, C. Ritter

v0.2 JAN2014: C. Ritter

v0.3 APR2014: C. Ritter, J. F. Navarro, F. Herwig, C. Fryer, E. Starkenburg,
              M. Pignatari, S. Jones, K. Venn, P. A. Denissenkov & the
              NuGrid collaboration

v0.4 FEB2015: C. Ritter, B. Cote

v0.5 MAY2016: C. Ritter; Note: See from now the official releases via github.

Stop keeking track of version from now on.

MARCH2018: B. Cote
- Switched to Python 3
- Capability to include radioactive isotopes


Usage
=====

Simple run example:

>>> import sygma as s

Get help with

>>> help s

Now start a calculation by providing the initial metal fraction
iniZ, the final evolution time tend, and the total mass of the SSP:

>>> s1=s.sygma(iniZ=0.0001, tend=5e9, mgal=1e5)

For more information regarding the input parameter see below or
try

>>> s.sygma?

For plotting utilize the plotting functions (plot_*) as described further
below, for example:

>>> s1.plot_totmasses()

You can write out the information about the composition of the total
ejecta of a SSP via

>>> s.write_evol_table(elements=['H','C','O'])

Yield tables are available in the NUPYCEE subdirectory
yield/textunderscore tables. Add your yield tables to
this directory and SYGMA will be able to read the table
if you have specified the $table$ variable. Only
for tables with Z=0 is the variable $pop3/textunderscore table$ used.
Both tables need yields specified in the SYGMA (and OMEGA)
yield input format. See the default table for the structure.
It is important to provide an initial abundance
file which must match the number of species provided in the yield tables.
Provide the file in the iniAbu directory inside the directory yield/extunderscore tables.
The input variable with which the table file can be specified is $iniabu/textunderscore table$.
For the necessary structure see again the default choice of that variable.

For example with artificial yields of only H-1, you can try

>>> s2 = s.sygma(iniZ=0.0001,dt=1e8,tend=1.5e10, mgal=1e11,table='yield_tables/agb_and_massive_stars_h1.txt',
    sn1a_table='yield_tables/sn1a_h1.txt',iniabu_table='yield_tables/iniab1.0E-04GN93_alpha_h1.ppn.txt')

'''

# Import standard python packages
import os

# Define where is the working directory
# This is where the NuPyCEE code will be extracted
nupy_path = os.path.dirname(os.path.realpath(__file__))

# Import NuPyCEE codes
from NuPyCEE.chem_evol import *

class sygma( chem_evol ):

    '''
    Input parameters (SYGMA)
    ================

    sfr : string
        Description of the star formation, usually an instantaneous burst.

        Choices :

                'input' - read and use the sfr_input file to set the percentage
                            of gas that is converted into stars at each timestep.

                'schmidt' - use an adapted Schmidt law (see Timmes95)

        Default value : 'input'

    ================
    '''
    # Combine docstrings from chem_evol with sygma docstring
    __doc__ = __doc__+chem_evol.__doc__

    ##############################################
    #                Constructor                 #
    ##############################################
    def __init__(self, sfr='input', \
                 imf_type='kroupa', alphaimf=2.35, imf_bdys=[0.1,100], \
                 sn1a_rate='power_law', iniZ=0.02, dt=1e6, special_timesteps=30, \
                 nsmerger_bdys=[8, 100], tend=13e9, mgal=1e4, transitionmass=8.0, iolevel=0, \
                 ini_alpha=True, table='yield_tables/agb_and_massive_stars_nugrid_MESAonly_fryer12delay.txt', \
                 table_radio='', decay_file='', sn1a_table_radio='',\
                 bhnsmerger_table_radio='', nsmerger_table_radio='',\
                 hardsetZ=-1, sn1a_on=True, sn1a_table='yield_tables/sn1a_i99_W7.txt',sn1a_energy=1e51,\
                 ns_merger_on=False, bhns_merger_on=False, f_binary=1.0, f_merger=0.0008, \
                 t_merger_max=1.3e10, m_ej_nsm = 2.5e-02, nsm_dtd_power=[],\
                 m_ej_bhnsm=2.5e-02, \
                 bhnsmerger_table = 'yield_tables/r_process_arnould_2007.txt', \
                 nsmerger_table = 'yield_tables/r_process_arnould_2007.txt', iniabu_table='', \
                 extra_source_on=False, nb_nsm_per_m=-1.0, t_nsm_coal=-1.0, \
                 extra_source_table=['yield_tables/extra_source.txt'], \
                 f_extra_source=[1.0], pre_calculate_SSPs=False, \
                 extra_source_mass_range=[[8,30]], \
                 extra_source_exclude_Z=[[]], \
                 total_ejecta_interp=True, yield_tables_dir='',\
                 high_mass_extrapolation='copy',\
                 radio_refinement=100, use_decay_module=False,\
                 f_network='isotopes_modified.prn', f_format=1,\
                 pop3_table='yield_tables/popIII_heger10.txt', \
                 imf_bdys_pop3=[0.1,100], imf_yields_range_pop3=[10,30], \
                 imf_pop3_char_mass=40.0, \
                 starbursts=[], beta_pow=-1.0,gauss_dtd=[1e9,6.6e8],exp_dtd=2e9,\
                 nb_1a_per_m=1.0e-3,direct_norm_1a=-1, Z_trans=0.0, \
                 f_arfo=1.0, imf_yields_range=[1,30],exclude_masses=[], \
                 netyields_on=False,wiersmamod=False,yield_interp='lin', \
                 stellar_param_on=False, t_dtd_poly_split=-1.0, \
                 delayed_extra_yields_log_int=False, \
                 stellar_param_table='yield_tables/stellar_feedback_nugrid_MESAonly.txt',
                 tau_ferrini=False, delayed_extra_log=False, dt_in=np.array([]),\
                 nsmerger_dtd_array=np.array([]), bhnsmerger_dtd_array=np.array([]),\
                 ytables_in=np.array([]), zm_lifetime_grid_nugrid_in=np.array([]),\
                 isotopes_in=np.array([]), ytables_pop3_in=np.array([]),\
                 zm_lifetime_grid_pop3_in=np.array([]), ytables_1a_in=np.array([]), \
                 mass_sampled=np.array([]), scale_cor=np.array([]), \
                 poly_fit_dtd_5th=np.array([]), poly_fit_range=np.array([]),\
                 ytables_nsmerger_in=np.array([]), dt_split_info=np.array([]),\
                 delayed_extra_dtd=np.array([]), delayed_extra_dtd_norm=np.array([]), \
                 delayed_extra_yields=np.array([]), delayed_extra_yields_norm=np.array([]), \
                 delayed_extra_yields_radio=np.array([]), ism_ini_radio=np.array([]), \
                 delayed_extra_yields_norm_radio=np.array([]), \
                 ytables_radio_in=np.array([]), radio_iso_in=np.array([]), \
                 ytables_1a_radio_in=np.array([]), ytables_nsmerger_radio_in=np.array([])):

        # Call the init function of the class inherited by SYGMA
        chem_evol.__init__(self, imf_type=imf_type, alphaimf=alphaimf, \
                 imf_bdys=imf_bdys, sn1a_rate=sn1a_rate, iniZ=iniZ, dt=dt, \
                 special_timesteps=special_timesteps, tend=tend, mgal=mgal, \
                 nsmerger_bdys=nsmerger_bdys, transitionmass=transitionmass, iolevel=iolevel, \
                 ini_alpha=ini_alpha, table=table, hardsetZ=hardsetZ, \
                 sn1a_on=sn1a_on, sn1a_table=sn1a_table,sn1a_energy=sn1a_energy,\
                 table_radio=table_radio, decay_file=decay_file,\
                 sn1a_table_radio=sn1a_table_radio, bhnsmerger_table_radio=bhnsmerger_table_radio,\
                 nsmerger_table_radio=nsmerger_table_radio,\
                 ns_merger_on=ns_merger_on, nsmerger_table=nsmerger_table, \
                 f_binary=f_binary, f_merger=f_merger, \
                 bhns_merger_on=bhns_merger_on,
                 m_ej_bhnsm=m_ej_bhnsm, bhnsmerger_table=bhnsmerger_table, \
                 nsm_dtd_power=nsm_dtd_power, yield_tables_dir=yield_tables_dir, \
                 total_ejecta_interp=total_ejecta_interp, \
                 t_merger_max=t_merger_max, m_ej_nsm = m_ej_nsm, \
                 iniabu_table=iniabu_table, extra_source_on=extra_source_on, \
                 extra_source_table=extra_source_table,f_extra_source=f_extra_source, \
                 extra_source_mass_range=extra_source_mass_range, \
                 extra_source_exclude_Z=extra_source_exclude_Z,
                 pop3_table=pop3_table, pre_calculate_SSPs=pre_calculate_SSPs, \
                 nb_nsm_per_m=nb_nsm_per_m, t_nsm_coal=t_nsm_coal, \
                 imf_bdys_pop3=imf_bdys_pop3, \
                 imf_pop3_char_mass=imf_pop3_char_mass, \
                 imf_yields_range_pop3=imf_yields_range_pop3, \
                 starbursts=starbursts, beta_pow=beta_pow, \
                 gauss_dtd=gauss_dtd,exp_dtd=exp_dtd,\
                 nb_1a_per_m=nb_1a_per_m,direct_norm_1a=direct_norm_1a, \
                 Z_trans=Z_trans, f_arfo=f_arfo, t_dtd_poly_split=t_dtd_poly_split, \
                 imf_yields_range=imf_yields_range,exclude_masses=exclude_masses,\
                 netyields_on=netyields_on,wiersmamod=wiersmamod,\
                 yield_interp=yield_interp, tau_ferrini=tau_ferrini,\
                 delayed_extra_log=delayed_extra_log,\
                 ytables_in=ytables_in, nsmerger_dtd_array=nsmerger_dtd_array, \
                 bhnsmerger_dtd_array=bhnsmerger_dtd_array, \
                 zm_lifetime_grid_nugrid_in=zm_lifetime_grid_nugrid_in,\
                 isotopes_in=isotopes_in,ytables_pop3_in=ytables_pop3_in,\
                 zm_lifetime_grid_pop3_in=zm_lifetime_grid_pop3_in,\
                 ytables_1a_in=ytables_1a_in, ytables_nsmerger_in=ytables_nsmerger_in, \
                 dt_in=dt_in,stellar_param_on=stellar_param_on,\
                 stellar_param_table=stellar_param_table,\
                 poly_fit_dtd_5th=poly_fit_dtd_5th, \
                 poly_fit_range=poly_fit_range, dt_split_info=dt_split_info, \
                 delayed_extra_dtd=delayed_extra_dtd,\
                 delayed_extra_dtd_norm=delayed_extra_dtd_norm,\
                 delayed_extra_yields=delayed_extra_yields,\
                 delayed_extra_yields_norm=delayed_extra_yields_norm,\
                 delayed_extra_yields_radio=delayed_extra_yields_radio,\
                 delayed_extra_yields_norm_radio=delayed_extra_yields_norm_radio,\
                 ytables_radio_in=ytables_radio_in, radio_iso_in=radio_iso_in,\
                 ytables_1a_radio_in=ytables_1a_radio_in, ism_ini_radio=ism_ini_radio,\
                 ytables_nsmerger_radio_in=ytables_nsmerger_radio_in,\
                 radio_refinement=radio_refinement,\
                 use_decay_module=use_decay_module,\
                 f_network=f_network, f_format=f_format,\
                 high_mass_extrapolation=high_mass_extrapolation)

        if self.need_to_quit:
            return

        # Announce the beginning of the simulation
        print ('SYGMA run in progress..')
        start_time = t_module.time()
        self.start_time = start_time

        # Attribute the input parameter to the current object
        self.sfr = sfr
        self.mass_sampled = mass_sampled
        self.scale_cor = scale_cor

        # Get the SFR of every timestep
        self.sfrin_i = self.__sfr()

        # Run the simulation
        self.__run_simulation()

        # Do the final update of the history class
        self._update_history_final()

        # Announce the end of the simulation
        print ('   SYGMA run completed -',self._gettime())


    ##############################################
    #                Run Simulation              #
    ##############################################
    def __run_simulation(self):

        '''
        This function calculates the evolution of the ejecta released by simple
        stellar populations as a function of time.

        '''

        # For every timestep i considered in the simulation ...
        for i in range(1, self.nb_timesteps+1):

            # Get the current fraction of gas that turns into stars
            self.sfrin = self.sfrin_i[i-1]

            # Run the timestep i
            self._evol_stars(i, 0.0, self.mass_sampled, self.scale_cor)

#            if i == 1:
#                self.ymgal_radio[i][2] = 1.0

            # Decay radioactive isotopes
            if self.len_decay_file > 0:
                if self.use_decay_module:
                    self._decay_radio_with_module(i)
                else:
                    self._decay_radio(i)

            # Get the new metallicity of the gas
            self.zmetal = self._getmetallicity(i)

            # Update the history class
            self._update_history(i)


    ##############################################
    #                    SFR                     #
    ##############################################
    def __sfr(self):

        '''
        This function calculates the percentage of gas mass which transforms into
        stars at every timestep, and then returns the result in an array.

        '''

        # Declaration of the array containing the mass fraction converted
        # into stars at every timestep i.
        if self.sfr == 'input':
            sfr_i = []
        else:
            sfr_i = np.zeros(self.nb_timesteps+1)

        # Output information
        if self.iolevel >= 3:
            print ('Entering sfr routine')

        # For every timestep i considered in the simulation ...
        for i in range(1, self.nb_timesteps+1):

            # If an array is used to generate starbursts ...
            if len(self.starbursts) > 0:
                if len(self.starbursts) >= i:

                    # Use the input value
                    sfr_i[i] = self.starbursts[i-1]
                    self.history.sfr.append(sfr_i[i-1])

            # If an input file is read for the SFR ...
            elif self.sfr == 'input':

                # Open the input file, read all lines, and close the file
                f1 = open(os.path.join(nupy_path, 'sfr_input'))
                lines = f1.readlines()
                f1.close()

                # The number of lines needs to be at least equal to the
                # number of timesteps
                if self.nb_timesteps > (len(lines)):
                    print ('Error - SFR input file does not' \
                          'provide enough timesteps')
                    return

                # Copy the SFR (mass fraction) of every timestep
                for k in range(len(lines)):
                    if k == (i-1):
                        sfr_i.append(float(lines[k]))
                        self.history.sfr.append(sfr_i[i-1])
                        break

            # If the Schmidt law is used (see Timmes98) ...
            elif self.sfr == 'schmidt':

                # Calculate the mass of available gas
                mgas = sum(ymgal[i-1])

                # Calculate the SFR according to the current gas fraction
                B = 2.8 * self.mgal * (mgas / self.mgal)**2    # [Mo/Gyr]
                sfr_i[i] = (B/mgas) * (timesteps[i-1] / 1.e9) # mass fraction
                self.history.sfr.append(sfr_i[i-1])

        # Return the SFR (mass fraction) of every timestep
        return sfr_i

###############################################################################################
######################## Here start the analysis methods ######################################
###############################################################################################


    def write_stellar_param_table(self,table_name='gce_stellar_param_table.txt', path="evol_tables",interact=False):

        '''
        Writes out evolution of stellar parameter such as luminosity and kinetic energy.
        Stellar parameter quantities are available via <sygma instance>.stellar_param_attrs.

        Table structure:

        &Age &Quantity1 &Quantity2 ...

        &0.000E+00 &0.000E+00 &0.000E+00

        &0.000E+00 &0.000E+00 &0.000E+00

        Parameters
        ----------

        table_name : string,optional
          Name of table. If you use a notebook version, setting a name
          is not necessary.

        path : string
                directory where to save table.

        interact: bool
                If true, saves file in current directory (notebook dir) and creates HTML link useful in ipython notebook environment

        Examples
        ----------

        >>> s.write_evol_table(table_name='testoutput.txt')


        '''

        time_evol=self.history.age

        #get available quantities

        parameter=self.stellar_param_attrs
        parameter_values=self.stellar_param

        metal_evol=self.history.metallicity
        #header
        out='&Age [yr]  '
        for i in range(len(parameter)):
            out+= ('&'+parameter[i]+((20-len(parameter[i]))*' '))
        out = out + '\n'
        out+=('&'+'{:.3E}'.format(time_evol[0]))
        for i in range(len(parameter_values)):
                out+= ( ' &'+ '{:.3E}'.format(0.))
        out = out + '\n'
        #data
        for t in range(len(parameter_values[0])):
            out+=('&'+'{:.3E}'.format(time_evol[t+1]))
            for i in range(len(parameter_values)):
                out+= ( ' &'+ '{:.3E}'.format(parameter_values[i][t]))
            out+='\n'

        if interact==True:
                import random
                randnum=random.randrange(10000,99999)
                name=table_name+str(randnum)+'.txt'
                #f1=open(os.path.join(nupy_path, 'evol_tables', name),'w')
                f1=open(name,'w')
                f1.write(out)
                f1.close()
                print ('Created table '+name+'.')
                print ('Download the table using the following link:')
                #from IPython.display import HTML
                #from IPython import display
                from IPython.core.display import HTML
                import IPython.display as display
                #return HTML("""<a href="evol_tables/download.php?file="""+name+"""">Download</a>""")
                #test=
                #return display.FileLink('../../nugrid/SYGMA/SYGMA_online/SYGMA_dev/evol_table/'+name)
                #if interact==False:
                #return HTML("""<a href="""+nupy_path+"""/evol_tables/"""+name+""">Download</a>""")
                return HTML("""<a href="""+name+""">Download</a>""")
                #else:
                #        return name
        else:
                print ('file '+table_name+' saved in subdirectory evol_tables.')
                f1=open(path+'/'+table_name,'w')
                f1.write(out)
                f1.close()



    def plot_stellar_param(self,fig=8,quantity='Ekindot_wind',label='',marker='o',color='r',shape='-',fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14,markevery=1):

        '''

        Plots the evolution of stellar parameter as provided as input in stellar parameter table (stellar_param_table variable).

        Parameters
        ----------

        quantity: string

                Name of stellar parameter of interest. Check for available parameter via <sygma instance>.stellar_param_attrs

        Examples
        ----------

        >>> s.plot_stellar_param(quantity='Ekin_wind')


        '''

        import matplotlib.pyplot as plt
        #in case stellar parameter are not used.
        if self.stellar_param_on==False:
                print ('Set stellar_param_on to true to use this function.')
                return

        if not quantity in self.stellar_param_attrs:
                print ('Quantity ',quantity,' not provided in yield table')
                return
        idx=self.stellar_param_attrs.index(quantity)
        quantity_evol=self.stellar_param[idx]
        age=self.history.age[1:]

        plt.figure(fig)

        plt.plot(age,quantity_evol,label=label,marker=marker,color=color,linestyle=shape,markevery=markevery)

        ax=plt.gca()
        self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)
        plt.ylabel('log-scaled '+quantity)
        plt.xlabel('log-scaled age [yr]')
        plt.yscale('log')
        plt.xscale('log')

    def plot_metallicity(self,source='all',label='',marker='o',color='r',shape='-',fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14):

        '''
        Plots the metal fraction defined as the sum of all isotopes except
        H1, H2, H3, He3, He4, Li6, Li7.

        Parameters
        ----------

        source : string
             Specifies if yields come from

             all sources ('all'), including
             AGB+SN1a, massive stars. Or from

             distinctive sources:

             only agb stars ('agb'),

             only SN1a ('SN1a')

             only massive stars ('massive')

        Examples
        ----------

        >>> s.plot_metallicity(source='all')

        '''

        import matplotlib.pyplot as plt
        iso_non_metal=['H-1','H-2','H-3','He-3','He-4','Li-6','Li-7']
        idx_iso_non_metal=[]
        for h in range(len(iso_non_metal)):
                if iso_non_metal[h] in self.history.isotopes:
                        idx_iso_non_metal.append(self.history.isotopes.index(iso_non_metal[h]))

        x=self.history.age

        if source == 'all':
              yields_evol=self.history.ism_iso_yield
        elif source =='agb':
              yields_evol=self.history.ism_iso_yield_agb
        elif source == 'sn1a':
              yields_evol=self.history.ism_iso_yield_1a
        elif source == 'massive':
             yields_evol=self.history.ism_iso_yield_massive

        Z_evol=[]
        for k in range(len(yields_evol)):
                isotopes=self.history.isotopes
                nonmetals=0
                for h in range(len(idx_iso_non_metal)):
                        nonmetals=nonmetals + yields_evol[k][idx_iso_non_metal[h]]
                Z_step=(sum(yields_evol[k]) - nonmetals)/sum(yields_evol[k])
                Z_evol.append(Z_step)

        plt.plot(x,Z_evol,label=label)

        plt.xscale('log')
        plt.xlabel('age [yr]')
        plt.ylabel('metal fraction Z')
        #self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)


    def plot_table_param(self,fig=8,ax='',xaxis='mini',quantity='Lifetime',iniZ=0.02,masses=[],label='',marker='o',color='r',shape='-',table='yield_tables/isotope_yield_table.txt',fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14):

        '''
        Plots the yield table quantities such as lifetimes versus initial mass as given in yield input tables.

        Parameters
        ----------

        xaxis : string
             if 'mini': use initial mass
             if 'time': use lifetime
        iniZ  : float
              Metallicity of interest
        masses: list
              List of initial masses to be plotted

        table: string
              Yield table

        Examples
        --------

        >>> s.plot_table_param(quantity='Lifetime')

        '''

        import re
        import matplotlib.pyplot as plt
        y_table=ry.read_nugrid_yields(os.path.join(nupy_path, table))

        plt.figure(fig)

        # find all available masses
        if len(masses)==0:
                allheader=y_table.table_mz
                for k in range(len(allheader)):
                        if str(iniZ) in allheader[k]:
                                mfound=float(allheader[k].split(',')[0].split('=')[1])
                                masses.append(mfound)

        if xaxis=='mini':
                x=masses
        elif xaxis=='time':
                x=[]
                for k in range(len(masses)):
                        x.append(y_table.get(Z=iniZ, M=masses[k], quantity='Lifetime'))

        param=[]
        for k in range(len(masses)):
                param.append(y_table.get(Z=iniZ, M=masses[k], quantity=quantity))

        if type(ax)==str:
                ax=plt.gca()
        ax.plot(x,param,label=label,marker=marker,color=color,linestyle=shape)
        ax=plt.gca()
        self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)
        plt.ylabel(quantity)
        if xaxis=='mini':
                plt.xlabel('initial mass [M$_{\odot}$]')
        elif xaxis=='time':
                plt.xlabel('lifetime [yr]')
        plt.yscale('log')


    def plot_table_remnant(self,fig=8,xaxis='mini',iniZ=0.02,masses=[],label='',marker='o',color='r',shape='-',table='yield_tables/isotope_yield_table.txt',fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14):

        '''

        Plots the remnant masses versus initial mass given in yield tables.

        Parameters
        ----------

        xaxis : string
             if 'mini': use initial mass; if of the form [specie1/specie2] use spec. notation of
        yaxis : string

        iniZ : float
             Metallicity to choose.

        Examples
        ----------

        >>> s.plot_table_remnant(iniZ=0.02)


        '''

        import re
        import matplotlib.pyplot as plt
        y_table=ry.read_nugrid_yields(os.path.join(nupy_path, table))
        plt.figure(fig)
        # find all available masses
        if len(masses)==0:
                allheader=y_table.table_mz
                for k in range(len(allheader)):
                        if str(iniZ) in allheader[k]:
                                mfound=float(allheader[k].split(',')[0].split('=')[1])
                                masses.append(mfound)

        mfinals=[]
        for k in range(len(masses)):
                mfinals.append(y_table.get(Z=iniZ, M=masses[k], quantity='Mfinal'))

        plt.plot(masses,mfinals,label=label,marker=marker,color=color,linestyle=shape)
        ax=plt.gca()
        self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)
        plt.ylabel('remnant mass [M$_{\odot}$]')
        plt.xlabel('initial mass [M$_{\odot}$]')
        plt.minorticks_on()

    def plot_yield_mtot(self,fig=8,plot_imf_mass_ranges=True,fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14):

        '''

        Plots total mass ejected by stars (!). To distinguish between the total mass of yields from the table and fitted total mass

        Parameters
        ----------

        plot_imf_mass_ranges : boolean
                If true plots the initial mass ranges for which yields are consider.

        Examples
        ----------

        >>> s.plot_yield_mtot()


        '''

        import matplotlib.pyplot as plt
        plt.figure(8)
        yall=[]
        for k in range(len(self.yields)):
            yall.append(sum(self.yields[k]))
        mall=self.m_stars
        x=[]
        ms=[]
        for m in np.arange(self.imf_bdys[0],self.imf_bdys[-1],1):
            x.append(self.func_total_ejecta(m))
            ms.append(m)
        plt.plot(ms,x,linestyle=':',label='fit')
        plt.plot(mall,yall,marker='x',color='k',linestyle='',label='input yield grid')
        plt.xlabel('initial mass [M$_{\odot}$]')
        plt.ylabel('total yields [M$_{\odot}$]')
        plt.legend()

        if plot_imf_mass_ranges==True:
                ranges=self.imf_mass_ranges
                for k in range(len(ranges)):
                    plt.vlines(ranges[k][0],0,100,linestyle='--')
                plt.vlines(ranges[-1][1],0,100,linestyle='--')

        ax=plt.gca()
        self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)


    def plot_table_yield_mass(self,fig=8,xaxis='mini',yaxis='C-12',iniZ=0.0001,netyields=False,masses=[],label='',marker='o',color='r',shape='-',table='yield_tables/isotope_yield_table.txt',fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14):

        '''
        Plots yields for isotopes given in yield tables.

        Parameters
        ----------

        xaxis : string
                if 'mini' use initial mass on x axis

        yaxis : string
        isotope to plot.

        Examples
        ----------

        >>> s.plot_iso_ratio(yaxis='C-12')

        '''

        import matplotlib.pyplot as plt
        # find all available masses
        if len(masses)==0:
                allheader=y_table.table_mz
                for k in range(len(allheader)):
                        if str(iniZ) in allheader[k]:
                                mfound=float(allheader[k].split(',')[0].split('=')[1])
                                masses.append(mfound)
        if True:
                x=[]
                y=[]
                for mini in masses:
                                x.append(mini)
                                plt.xlabel('initial mass [M$_{\odot}$]')
                                headerx='Mini/Msun'

                                if netyields==True:
                                        y_ini=ini_isos_frac[ini_isos.index(yaxis)]
                                        miniadd=(y_ini*(mini-mfinal))
                                        y.append(y_delay.get(M=mini,Z=Z,specie=yaxis) + miniadd)
                                else:
                                        y.append(y_delay.get(M=mini,Z=Z,specie=yaxis))
                                plt.ylabel('yield [M$_{\odot}$]')
                                headery='Yield [Msun]'

                if len(label)==0:
                        plt.plot(x,y,label='Z='+str(Z),marker=marker,color=color,linestyle=shape)
                else:
                        plt.plot(x,y,label=label,marker=marker,color=color,linestyle=shape)


        ax=plt.gca()
        self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)
        #return x,y
        self.__save_data(header=[headerx,headery],data=[x,y])


    def plot_net_yields(self,fig=91,species='[C-12/Fe-56]',netyields_iniabu='yield_tables/iniabu/iniab_solar_Wiersma.ppn'):

        '''
                Plots net yields as calculated in the code when using netyields_on=True.

                To be used only with net yield input.

        Parameters
        ----------

        species : string
                Isotope ratio in spectroscopic notation.


        Examples
        ----------

        >>> s.plot_net_yields(species='[C-12/Fe-56]')


        '''

        import matplotlib.pyplot as plt
        iniabu=ry.iniabu(os.path.join(nupy_path, netyields_iniabu))
        isonames=iniabu.names
        specie1=species.split('/')[0][1:]
        specie2=species.split('/')[1][:-1]
        for k in range(len(isonames)):
                elem=re.split('(\d+)',isonames[k])[0].strip().capitalize()
                A=int(re.split('(\d+)',isonames[k])[1])
                if specie1 == elem+'-'+str(A):
                        x1sol=iniabu.iso_abundance(elem+'-'+str(A))
                if specie2 == elem+'-'+str(A):
                        x2sol=iniabu.iso_abundance(elem+'-'+str(A))

        specie1=species.split('/')[0][1:]
        specie2=species.split('/')[1][:-1]
        idx1=self.history.isotopes.index(specie1)
        idx2=self.history.isotopes.index(specie2)
        y=[]
        x=range(len(self.history.netyields))
        x=self.history.age
        x=self.history.netyields_masses
        for k in range(len(self.history.netyields)):
            x1=self.history.netyields[k][idx1]/sum(self.history.netyields[k])
            x2=self.history.netyields[k][idx2]/sum(self.history.netyields[k])
            y.append(np.log10(x1/x2 / (x1sol/x2sol)))
        print ('create figure')
        plt.figure(fig)
        plt.plot(x,y,marker='o')
        plt.ylabel(species)
        plt.xlabel('initial mass [M$_{\odot}$]')
        #plt.xscale('log')

    def plot_table_yield(self,fig=8,xaxis='mini',yaxis='C-12',iniZ=0.0001,netyields=False,masses=[],label='',marker='o',color='r',shape='-',table='yield_tables/isotope_yield_table.txt',fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14,solar_abu='',netyields_iniabu=''):

        '''

        Plots the yields of the yield input grid versus initial mass. Yields can be plotted in solar masses or in spectroscopic notation.


        Parameters
        ----------

        xaxis : string
             if 'mini': use initial mass; if of the form [specie1/specie2] use spec. notation
        yaxis : string
             specifies isotopes or elements with 'C-12' or 'C': plot yield of isotope;
             if chosen spectros: use form [specie3/specie4]
        iniZ : float
             specifies the metallicity to be plotted
        masses : list
             if empty plot all available masses for metallicity iniZ; else choose only masses in list masses
        table : string
             table to plot data from; default sygma input table
        solar_abu : string
             solar abundance for spectroscopic notation
             default: yield_tables/iniabu/iniab2.0E-02GN93.ppn (if empty string)
        netyields : bool
             if true assume net yields in table and add corresponding initial contribution to get total yields
        netyields_iniabu : string
             initial abundance, only used in conjuction with net yields

        Examples
        ----------

        >>> s.plot_iso_ratio(yaxis='C-12')
        >>> s.plot_iso_ratio(yaxis='C/Fe')
        >>> s.plot_iso_ratio(yaxis='[C/Fe]')
        >>> s.plot_iso_ratio(xaxis='[Fe/H]',yaxis='[C/Fe]')


        '''
        import re
        import matplotlib.pyplot as plt

        y_table=ry.read_nugrid_yields(os.path.join(nupy_path, table))
        plt.figure(fig, figsize=(fsize[0],fsize[1]))


        #spectroscopic notation?
        specx=False
        specy=False
        if ('[' in yaxis):
                specy=True
        if ('[' in xaxis):
                specx=True


        ####Get solar metallicity elements, if necessary
        if specx or specy:
                if len(solar_abu) ==0:
                     iniabu_sol=ry.iniabu(os.path.join(nupy_path, 'yield_tables',\
                             'iniabu', 'iniab2.0E-02GN93.ppn'))
                else:
                     iniabu_sol=ry.iniabu(os.path.join(nupy_path, solar_abu))
                isonames=iniabu_sol.names

                ini_elems_frac_sol=[]
                ini_elems_sol=[]
                for k in range(len(isonames)):
                                elem=re.split('(\d+)',isonames[k])[0].strip().capitalize()
                                A=int(re.split('(\d+)',isonames[k])[1])
                                if elem not in ini_elems_sol:
                                        ini_elems_sol.append(elem)
                                        ini_elems_frac_sol.append(iniabu_sol.iso_abundance(elem+'-'+str(A)))
                                else:
                                        ini_elems_frac_sol[ini_elems_sol.index(elem)]+= iniabu_sol.iso_abundance(elem+'-'+str(A))

                ini_isos_frac_sol=[]
                ini_isos_sol=[]
                for k in range(len(isonames)):
                                        elem=re.split('(\d+)',isonames[k])[0].strip().capitalize()
                                        A=int(re.split('(\d+)',isonames[k])[1])
                                        newname=elem+'-'+str(A)
                                        ini_isos_sol.append(newname)
                                        ini_isos_frac_sol.append(iniabu_sol.iso_abundance(elem+'-'+str(A)))

        # for net yields need initial abundance of elements or isotopes
        if netyields:

                iniabu=ry.iniabu(os.path.join(nupy_path, netyields_iniabu))
                isonames=iniabu.names
                #get initial elements
                if True:
                        ini_elems=[]
                        ini_elems_frac=[]
                        for k in range(len(isonames)):
                                elem=re.split('(\d+)',isonames[k])[0].strip().capitalize()
                                A=int(re.split('(\d+)',isonames[k])[1])
                                if elem not in ini_elems:
                                        ini_elems.append(elem)
                                        ini_elems_frac.append(iniabu.iso_abundance(elem+'-'+str(A)))
                                else:
                                        ini_elems_frac[ini_elems.index(elem)]+=iniabu.iso_abundance(elem+'-'+str(A))
                #get isotopes
                if True:
                        ini_isos=[]
                        ini_isos_frac=[]
                        for k in range(len(isonames)):
                                elem=re.split('(\d+)',isonames[k])[0].strip().capitalize()
                                A=int(re.split('(\d+)',isonames[k])[1])
                                newname=elem+'-'+str(A)
                                ini_isos.append(newname)
                                ini_isos_frac.append(iniabu.iso_abundance(elem+'-'+str(A)))

        #find all available masses
        if len(masses)==0:
                allheader=y_table.table_mz
                for k in range(len(allheader)):
                        if str(iniZ) in allheader[k]:
                                mfound=float(allheader[k].split(',')[0].split('=')[1])
                                masses.append(mfound)

        # calculate and collect values to plot
        idx1=-1
        Z=iniZ
        y_delay=y_table
        if True:
                x=[]
                y=[]
                for mini in masses:

                        idx1+=1
                        totmass=sum(y_delay.get(M=mini,Z=Z,quantity='Yields'))
                        if netyields:
                                mfinal = y_delay.get(Z=Z, M=mini, quantity='Mfinal')

                        #if xaxis spectro notation
                        if (not 'mini' in xaxis):
                                if specx:
                                        x1=xaxis.split('/')[0][1:]
                                        x2=xaxis.split('/')[1][:-1]
                                else:
                                        x1=xaxis
                                #if isotope
                                if '-' in xaxis:
                                        if specx:
                                                yx2=y_delay.get(M=mini,Z=Z,specie=x2)
                                        yx1=y_delay.get(M=mini,Z=Z,specie=x1)
                                        if netyields:
                                                ini_species=ini_isos
                                                ini_species_frac=ini_isos_frac
                                        if specx:
                                                ini_species_frac_sol=ini_isos_frac_sol
                                                ini_species_sol = ini_isos_sol

                                #if element
                                else:
                                        yx2=0
                                        yx1=0
                                        #sum up isotopes to get elements
                                        isoavail=y_delay.get(M=mini,Z=Z,quantity='Isotopes')
                                        for k in range(len(isoavail)):
                                                if x1 == isoavail[k].split('-')[0]:
                                                        yx1+=y_delay.get(M=mini,Z=Z,specie=isoavail[k])
                                                if specx:
                                                        if x2 == isoavail[k].split('-')[0]:
                                                                yx2+=y_delay.get(M=mini,Z=Z,specie=isoavail[k])
                                        if netyields:
                                                ini_species=ini_elems
                                                ini_species_frac=ini_elems_frac
                                        if specx:
                                                ini_species_frac_sol=ini_elems_frac_sol
                                                ini_species_sol = ini_elems_sol
                                if specy:
                                        x1_ini_sol=ini_species_frac_sol[ini_species_sol.index(x1)]
                                        x2_ini_sol=ini_species_frac_sol[ini_species_sol.index(x2)]

                                if netyields:
                                        x1_ini=ini_species_frac[ini_species.index(x1)]
                                        if specx:
                                                x2_ini=ini_species_frac[ini_species.index(x2)]
                                                miniadd=(x2_ini*(m-mfinal))
                                                yx2_frac=( yx2+miniadd  )/totmass
                                        miniadd=(x1_ini*(mini-mfinal))
                                        yx1_frac=( yx1+miniadd  )/totmass
                                else:
                                        if specx:
                                                yx2_frac=yx2 #/totmass
                                        yx1_frac=yx1 #/totmass
                                if specx:
                                        if yx2_frac==0:
                                                print ('mini: ',mini,x2,' 0 value: yx2_frac',yx2_frac)
                                        else:
                                                x.append( np.log10( yx1_frac/yx2_frac * x2_ini_sol/x1_ini_sol) )
                                        plt.xlabel(xaxis)
                                        headerx=xaxis
                                else:
                                        x.append(yx1)
                                        plt.xlabel('yield [M$_{\odot}$]')
                                        headerx='yields/Msun'
                                        plt.xscale('log')
                        else:  #'mini' == xaxis:
                                x.append(mini)
                                plt.xlabel('initial mass [M$_{\odot}$]')
                                headerx='Mini/Msun'
                        #else:
                        #       return 'wrong input'
                        #       x.append(y_delay.get(M=mini,Z=Z,specie=xaxis))

                        ### yaxis

                        if True: #(not 'mini' in xaxis):
                                if specy:
                                        y1=yaxis.split('/')[0][1:]
                                        y2=yaxis.split('/')[1][:-1]
                                else:
                                        y1=yaxis
                                #if isotope
                                if '-' in yaxis:
                                        if specy:
                                                yy2=y_delay.get(M=mini,Z=Z,specie=y2)
                                        yy1=y_delay.get(M=mini,Z=Z,specie=y1)
                                        if netyields:
                                                ini_species=ini_isos
                                                ini_species_frac=ini_isos_frac
                                        if specy:
                                                ini_species_frac_sol=ini_isos_frac_sol
                                                ini_species_sol = ini_isos_sol
                                #if element
                                else:
                                        yy2=0
                                        yy1=0
                                        #sum up isotopes to get elements
                                        isoavail=y_delay.get(M=mini,Z=Z,quantity='Isotopes')
                                        for k in range(len(isoavail)):
                                                if y1 == isoavail[k].split('-')[0]:
                                                        yy1+=y_delay.get(M=mini,Z=Z,specie=isoavail[k])
                                                if specy:
                                                        if y2 == isoavail[k].split('-')[0]:
                                                                yy2+=y_delay.get(M=mini,Z=Z,specie=isoavail[k])
                                        if netyields:
                                                ini_species=ini_elems
                                                ini_species_frac=ini_elems_frac
                                        if specy:
                                                ini_species_frac_sol=ini_elems_frac_sol
                                                ini_species_sol = ini_elems_sol
                                if specy:
                                        y1_ini_sol=ini_species_frac_sol[ini_species_sol.index(y1)]
                                        y2_ini_sol=ini_species_frac_sol[ini_species_sol.index(y2)]

                                if netyields:
                                        y1_ini=ini_species_frac[ini_species.index(y1)]
                                        if specy:
                                                y2_ini=ini_species_frac[ini_species.index(y2)]
                                                miniadd=(y2_ini*(m-mfinal))
                                                yy2_frac=( yy2+miniadd  )/totmass
                                        miniadd=(y1_ini*(mini-mfinal))
                                        yy1_frac=( yy1+miniadd  )/totmass
                                else:
                                        if specy:
                                                yy2_frac=yy2 #/totmass
                                        yy1_frac=yy1 #/totmass
                                if specy:
                                        if yy2_frac==0:
                                                print ('mini: ',mini,y2,' 0 value: yy2_frac',yy2_frac)
                                        else:
                                                y.append( np.log10( yy1_frac/yy2_frac * y2_ini_sol/y1_ini_sol) )
                                        plt.ylabel(yaxis)
                                        headery=yaxis
                                else:
                                        y.append(yy1)
                                        plt.ylabel('yield [M$_{\odot}$]')
                                        headery='yields/Msun'
                                        plt.yscale('log')
                #plot results for specific mass
                if len(label)==0:
                        plt.plot(x,y,label='Z='+str(Z),marker=marker,color=color,linestyle=shape)
                else:
                        plt.plot(x,y,label=label,marker=marker,color=color,linestyle=shape)

        if specx and specy:
                for k in range(len(x)):
                        plt.annotate(str(masses[k]), xy = (x[k], y[k]),xytext = (0, 0), textcoords = 'offset points')

        ax=plt.gca()
        self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)
        #return x,y
        #self.__save_data(header=[headerx,headery],data=[x,y])


    def plot_mass_ratio(self,fig=0,xaxis='age',species_ratio='C/N',source='all',label='',shape='',marker='',color='',markevery=20,multiplot=False,return_x_y=False,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14,logy=True):

        '''
        Mass ratio of two species indicated by species_ratio over time.
        Choice can either be elemental ratio or isotopic ratios.
        Masses of species are in solar masses.
        Note: Similar to plot_mass but with ratios of masses.

        Parameters
        ----------


        specie : string
             ratio of element or isotope, e.g. 'C/O', 'C-12/O-12'
        xaxis  : string
             if 'age' : time evolution
             if '[Fe/H]' : use [Fe/H]
        source : string
             Specifies if yields come from
             all sources ('all'), including
             AGB+SN1a, massive stars. Or from
             distinctive sources:
             only agb stars ('agb'), only
             SN1a ('SN1a'), or only massive stars ('massive')
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
        logy : bool
             if yes, choose yaxis in log scale

        Examples
        ----------

        >>> s.plot_mass_ratio('C-12')

        '''
        import matplotlib.pyplot as plt

        if len(label)<1:
                if source=='agb':
                        label=species_ratio+', AGB'
                if source=='massive':
                        label=species_ratio+', Massive'
                if source=='sn1a':
                        label=species_ratio+', SNIa'

        #Reserved for plotting
        if not return_x_y:
            shape,marker,color=self.__msc(source,shape,marker,color)

        specie1=species_ratio.split('/')[0]
        specie2=species_ratio.split('/')[1]

        norm = 'no'
        x,y1=self.plot_mass(fig=0,specie=specie1,source=source,norm=norm,label=label,shape=shape,marker=marker,color=color,markevery=20,multiplot=False,return_x_y=True,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14)

        x,y2=self.plot_mass(fig=0,specie=specie2,source=source,norm=norm,label=label,shape=shape,marker=marker,color=color,markevery=20,multiplot=False,return_x_y=True,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14)

        y_temp=[]
        for k in range(len(y1)):
            if y2[k]==0.:
               y_temp.append(0.)
            else:
               y_temp.append(y1[k]/y2[k])
        y=y_temp

        if xaxis == '[Fe/H]':
            age,fe_h=self.plot_spectro(return_x_y=True,xaxis='age',yaxis='[Fe/H]')
            #match ages in x and age_dum
            y_temp=[]
            for k in range(len(age)):
                 idx=x.index(age[k])
                 y_temp.append(y[idx] )
            y = y_temp
            x = fe_h
        #Reserved for plotting
        if not return_x_y:
           plt.figure(fig, figsize=(fsize[0],fsize[1]))
           if xaxis=='age':
              plt.xlabel('age [yr]')
              plt.xscale('log')
           elif xaxis=='[Fe/H]':
              plt.xlabel('[Fe/H]')
           plt.ylabel('mass ratio X$_i$/X$_j$')
           if logy==True:
            plt.yscale('log')
        self.y=y
        #If x and y need to be returned ...
        if return_x_y:
            return x, y

        else:
            if len(label)==0:
                label=specie1+'/'+specie2
            plt.plot(x,y,label=label,linestyle=shape,marker=marker,color=color,markevery=markevery)
            plt.legend()
            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)
            if xaxis=='age':
                plt.xlim(self.history.dt,self.history.tend)
            #self.__save_data(header=['Age[yr]',specie],data=[x,y])


    def plot_mass(self,fig=0,specie='C',source='all',norm='no',label='',shape='',marker='',color='',markevery=20,multiplot=False,return_x_y=False,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14,linewidth=2):

        '''
        mass evolution (in Msun) of an element or isotope vs time.
        Note: Used in WENDI.

        Parameters
        ----------


        specie : string
             1) isotope or element name, in the form 'C' or 'C-12'
        source : string
             Specifies if yields come from
             all sources ('all'), including
             AGB+SN1a, massive stars. Or from
             distinctive sources:
             only agb stars ('agb'), only
             SN1a ('SN1a'), or only massive stars
             ('massive')
        norm : string
            if 'no', no normalization
            If 'current', normalize to current total amount of specie

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

        >>> s.plot_mass(specie='C')

        '''

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
            if norm == 'no':
                y.append(yields_evol[k][idx])
            elif norm == 'current':
                y.append( yields_evol[k][idx]/yields_evol_all[k][idx])
            else:
                print ('wrong specification of norm parameter')

        x=x[1:]
        y=y[1:]
        if multiplot==True:
                return x,y

        #Reserved for plotting
        if not return_x_y:
           plt.figure(fig, figsize=(fsize[0],fsize[1]))
           plt.xscale('log')
           plt.xlabel('log-scaled age [yr]')
           if norm == 'no':
               plt.ylabel('log-scaled integrated ejecta [M$_{\odot}$]')
               plt.yscale('log')
           else:
               plt.ylabel('log-scaled integrated fraction of ejecta')
        self.y=y

        #If x and y need to be returned ...
        if return_x_y:
            return x, y

        else:
            plt.plot(x,y,label=label,linestyle=shape,marker=marker,color=color,markevery=markevery,linewidth=linewidth)
            plt.legend()
            ax=plt.gca()
            self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)
            plt.xlim(self.history.dt,self.history.tend)
            #return x,y
            self.__save_data(header=['Age[yr]',specie],data=[x,y])

    def __plot_mass_multi(self,fig=1,specie=['C'],ylims=[],source='all',norm=False,label=[],shape=['-'],marker=['o'],color=['r'],markevery=20,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14):

        '''
        Use the function plot_mass multiple times
        Mass evolution (in Msun) of an element or isotope vs time.


        Parameters
        ----------


        yaxis : array
             isotopes or element names, in the form 'C' or 'C-12'
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

        '''
        import matplotlib.pyplot as plt
        #fig=plt.figure(fig)
        #nplots=1#len(specie)
        #f, ax_plots = plt.subplots(nplots, sharex=True, sharey=False)
        limits=[]
        ax=plt.gca()
        props = dict(boxstyle='square', facecolor='w', alpha=1)
        for k in range(len(specie)):
                #if len(ylims)>0:
                        #ylims1=ylims[k]
                        #ax_plots[k].set_ylim(ylims[k][0],ylims[k][1])
                if len(label)==0:
                        label1=specie[k]
                else:
                        label1=label[k]
                x,y=self.plot_mass(fig=fig,specie=specie[k],source=source,norm=norm,label=label1,shape=shape[k],marker=marker[k],color=color[k],markevery=20,multiplot=True)
                #x=np.log10(np.array(x))
                y=np.log10(np.array(y))
                #ax_plots[k]
                plt.plot(x,y,label=label1,linestyle=shape[k],marker=marker[k],color=color[k],markevery=markevery)
                #ax_plots[k].set_ylim(min(y),max(y))
                limits.append([min(y),max(y)])
                #ax_plots[k].set_xlim(min(x),max(x))
                #ax_plots[k].
                plt.xscale('log')
                #ax_plots[k].set_yscale('log')
                plt.xlabel('log-scaled age [yr]')
                #ax_plots[k].locator_params(axis = 'y', nbins = 2)
                #if norm == False:
                    #ax_plots[k].set_ylabel('yield [Msun]')
                    #ax_plots[k].set_yscale('log')
                #else:
                #    ax_plots[k].set_ylabel('(IMF-weighted) fraction of ejecta')
                #plt.legend()
                self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize,lwtickboth=[3,1],lwtickmajor=[5,1])
                #ax_plots[k].
                #plt.legend().set_visible(False)
                #x.legend(loc='center right', bbox_to_anchor=(1, 0.5),markerscale=0.8,fontsize=fontsize).set_visible(False)
                #ax_plots[k]
                #.text(0.90, 0.80, label1, transform=ax_plots[k].transAxes, fontsize=18,verticalalignment='top', bbox=props)
                plt.xlim(self.history.dt,self.history.tend)
        if norm == False:
                fig=plt.gcf()
                fig.text(0.002, 0.5, 'Log (Yield [M$_{\odot}$])', ha='center', va='center', rotation='vertical')
        else:
                fig=plt.gcf()
                fig.text(0.01, 0.5, '(IMF-weighted) fraction of ejecta', ha='center', va='center', rotation='vertical')

        #set lim here
        #for k in range(len(ax_plots)):
        #        ax_plots[k].set_ylim(limits[k][0],limits[k][1])

        #f.subplots_adjust(hspace=0.35)#0)
        #plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        return


    def plot_massfrac(self,fig=2,xaxis='age',yaxis='O-16',source='all',norm='no',label='',shape='',marker='',color='',markevery=20,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14):

        '''
        Plots mass fraction of isotope or element
        vs time.

        Parameters
        ----------
        xaxis : string
            either 'age' for time
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
            plt.xlabel('X('+xaxis+')')
            plt.xscale('log')
        elif 'age' == xaxis:
            x=self.history.age#[1:]
            plt.xscale('log')
            plt.xlabel('age [yr]')
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
                    print (yields_evol[0][iso_idx])

            plt.xlabel('X('+xaxis+')')
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
        self.__save_data(header=[xaxis,yaxis],data=[x,y])

    def plot_spectro(self,fig=3,xaxis='age',yaxis='[Fe/H]',source='all',label='',shape='-',marker='o',color='k',markevery=100,show_data=False,show_sculptor=False,show_legend=True,return_x_y=False,sub_plot=False,linewidth=3,sub=1,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14,solar_ab=''):
        '''
        Plots elements in spectroscopic notation.
        Note: Used in WENDI.

        Parameters
        ----------

        xaxis : string
            Spectroscopic notation of elements e.g. [Fe/H]
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

        x_ini_iso=iniabu.iso_abundance(self.history.isotopes)
        elements = self.history.elements
        x_ini=self._iso_abu_to_elem(x_ini_iso)
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



        ######################################


        if 'age' == xaxis:
            x=self.history.age

            #Operations associated with plot visual aspects
            if not return_x_y and not sub_plot:
                plt.xscale('log')
                plt.xlabel('log-scaled age [yr]')

            self.x=x
        else:


            xaxis_elem1=xaxis.split('/')[0][1:]
            xaxis_elem2=xaxis.split('/')[1][:-1]

            #X-axis ini values
            x_elem1_ini=x_ini[elements.index(xaxis_elem1)]
            x_elem2_ini=x_ini[elements.index(xaxis_elem2)]

            #X-axis gce values
            elem_idx1=self.history.elements.index(xaxis_elem1)
            elem_idx2=self.history.elements.index(xaxis_elem2)

            x=[]
            for k in range(0,len(yields_evol)):
                if sum(yields_evol[k]) ==0:
                    continue
                #in case no contribution during timestep
                x1=yields_evol[k][elem_idx1]/sum(yields_evol[k])
                x2=yields_evol[k][elem_idx2]/sum(yields_evol[k])
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

        #Y-axis gce values
        elem_idx1=self.history.elements.index(yaxis_elem1)
        elem_idx2=self.history.elements.index(yaxis_elem2)

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
        #while self.history.sfr_abs[i_rem] == 0.0:
        #    del y[-1]
        #    del x[-1]
        #    i_rem -= 1

        #If this function is supposed to return the x, y arrays only ...
        if return_x_y:

            return x, y

        #If this is a sub-figure managed by an external module
        elif sub_plot:

            if self.galaxy == 'none':
                if show_legend:
                    sub.plot(x,y,linestyle=shape,label=label,marker=marker,color=color,markevery=markevery)
                else:
                    sub.plot(x,y,linestyle=shape,marker=marker,color=color,markevery=markevery)
            else:
                if show_legend:
                    sub.plot(x,y,linestyle=shape,label=label,marker=marker,color=color,markevery=markevery,linewidth=linewidth)
                else:
                    sub.plot(x,y,linestyle=shape,marker=marker,color=color,markevery=markevery,linewidth=linewidth)

        #If this function is supposed to plot ...
        else:

            #Plot a thicker line for specific galaxies, since they must be visible with all the obs. data
            if True:
                if show_legend:
                    plt.plot(x,y,linestyle=shape,label=label,marker=marker,color=color,markevery=markevery)
                else:
                    plt.plot(x,y,linestyle=shape,marker=marker,color=color,markevery=markevery)
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
            self.__save_data(header=[xaxis,yaxis],data=[x,y])


    def __plot_abu_distr(self,fig=0,t=-1,x_axis='A',solar_norm=True,marker1=2,linest=0,y_range=[],label='CHEM  module',fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14):

        '''
        EXPERIMENTAL: DO NOT USE
        Plots abundance distribution of stable isotopes
        of certain time t (X/X_sol vs A).:

        Parameters
        ----------

        t : float
            default end of the simulation tend
        x_axis :  string
            only 'A' setting, mass number, possible
        solar_norm : boolean
            if true, normalize to solar value
        linest : string
            linestyle
        makre1 : string
            for marker
        y_range : list
            contains min and max

        Examples
        ----------
        >>> s.plot_abu_distr()

        '''
        import matplotlib.pyplot as plt

        color=['r','k','b','g']
        marker_type=['o','+','s','D']
        line_style=['--','-','-.',':']


        x=[]
        y=[]
        fig=plt.figure(fig,figsize=(10,8))
        ax = fig.add_subplot(1,1,1)
        if len(y_range)>0:
            plt.ylim(y_range[0],y_range[1])
        elem_array=[]
        y_array=[]
        iso_name='H-1'
        ele='H'
        i=0
        colormap = plt.cm.prism#flag #gist_ncar
        plt.gca().set_color_cycle([colormap(j) for j in np.linspace(0, 0.9, 100)])


        if t==-1:
            t=self.history.tend
        idx=int(t/self.history.dt) - 1
        isotopes1=self.history.isotopes
        yields_iso1=self.history.ism_iso_yield[idx]
        isotopes=[]
        yields_iso=[]

        #Check for stable isotopes
        from nugrid_set import is_stable
        for k in range(len(isotopes1)):
            if is_stable(isotopes1[k]):
                isotopes.append(isotopes1[k])
                yields_iso.append(yields_iso1[k])


        iniabu=ry.iniabu('yield_tables/iniabu/iniab2.0E-02GN93.ppn')
        x_ini_iso=iniabu.iso_abundance(isotopes)
        #elements,x_ini=self.iso_abu_to_elem(self.isotopes,x_ini_iso)

        for k in range(len(isotopes)):
            elem_name=isotopes[k].split('-')[0]
            if x_axis=='A':
                # Xi / Xsun
                yields_norm=(yields_iso[k]/sum(yields_iso))/x_ini_iso[isotopes.index(isotopes[k])]

                #in the case of isotope of same element as before
                if elem_name==ele:
                    elem_array.append(isotopes[k].split('-')[1] )
                    y_array.append(yields_norm )
                    print ('new iso',isotopes[k])
                    #i+=1
                    if not k==(len(isotopes)-1):
                        continue
                print (k)
                plt.plot(elem_array,y_array,marker=marker_type[marker1],markersize=8,linestyle=line_style[linest])
                print (elem_array)
                print (y_array[0])
                i+=1
                if (i%2) ==0:
                    high=6
                else:
                    high=4
                print ('i',i)
                print ('high',high)
                y_pos=(max(y_array))
                x_pos=  elem_array[y_array.index(max(y_array))]
                ax.annotate(iso_name, xy=(x_pos, y_pos), xytext=(elem_array[0],high))

                ele=elem_name
                iso_name=isotopes[k]
                y_array=[]
                y_array.append(yields_norm)
                elem_array=[]
                elem_array.append(isotopes[k].split('-')[1] )
                #x.append(line_1[1])
                plt.plot([0,72],[1,1],linestyle='--')
                plt.plot([0,72],[2,2],linestyle='--')
                plt.plot([0,72],[0.5,0.5],linestyle='--')





            #if x_axis=='Z':
            #       x.append(line_1[0])

        plt.xlabel("mass number A")
            #plt.plot(x,y,marker=marker_type[0],markersize=10,linestyle='None',label=self.label)
        ax.set_yscale('log')
        if len(y_range)>0:
            plt.ylim(y_range[0],y_range[1])
        plt.xlim(0,72)
        simArtist = plt.Line2D((0,1),(0,0), mfc='none',marker=marker_type[marker1], linestyle='w')
        ax=plt.gca()
        plt.legend([simArtist],[label])
        ax=plt.gca()
        self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)


    def plot_totmasses(self,fig=4,mass='gas',source='all',norm='no',label='',shape='',marker='',color='',markevery=20,log=True,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14):
        '''
        Plots either gas or star mass as fraction of total mass
        vs time.
        Note: Used in WENDI.

        Parameters
        ----------

        mass : string
            either 'gas' for ISM gas mass
            or 'stars' for gas locked away in stars (total gas - ISM gas)

        norm : string
            normalization, either 'no' for no normalization (total gas mass in solar masses),

            for normalization to the initial gas mass (mgal) with 'ini',

            for normalization to the current total gas mass 'current'.
            The latter case makes sense when comparing different
            sources (see below)

        source : string
            specifies if yields come from
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
            plt.xlabel('log-scaled age [yr]')
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
            if norm=='ini':
                ism_gasm.append(gas_evol[k]/self.history.mgal)
                star_m.append((self.history.mgal-gas_evol[k])/self.history.mgal)
                x.append(x_all[k])
            if norm == 'current':
                if not self.history.gas_mass[k] ==0.:
                      ism_gasm.append(gas_evol[k]/gas_mass[k])
                      star_m.append((self.history.mgal-gas_evol[k])/gas_mass[k])
                      x.append(x_all[k])
            #else:
             #   ism_gasm.append(0.)
             #   star_m.append(0.)
            elif norm == 'no':
                ism_gasm.append(gas_evol[k])
                star_m.append(self.history.mgal-gas_evol[k])
                x.append(x_all[k])
        if mass == 'gas':
            y=ism_gasm
        if mass == 'stars':
            y=star_m
        plt.plot(x,y,linestyle=shape,marker=marker,markevery=markevery,color=color,label=label)
        if len(label)>0:
                plt.legend()
        if norm=='current':
            plt.ylim(0,1)
        if not norm=='no':
            if mass=='gas':
                plt.ylabel('mass fraction')
                plt.title('Gas mass as a fraction of total gas mass')
            else:
                plt.ylabel('mass fraction')
                plt.title('Star mass as a fraction of total star mass')
        else:
            if mass=='gas':
                plt.ylabel('log-scaled integrated ejected mass [M$_{\odot}$]')
            else:
                plt.ylabel('log-scaled integrated mass locked in stars [M$_{\odot}$]')

            if mass=='gas':
                plt.ylabel('log-scaled integrated ejected mass [M$_{\odot}$]')
            else:
                plt.ylabel('log-scaled integrated mass locked in stars [M$_{\odot}$]')

        if log==True:
            plt.yscale('log')
            if not norm=='no':
                plt.ylim(1e-4,1.2)
        ax=plt.gca()
        self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)
        plt.xlim(self.history.dt,self.history.tend)
        self.__save_data(header=['age','mass'],data=[x,y])

    def plot_sn_distr(self,fig=5,rate=True,rate_only='',xaxis='time',fraction=False,label1='SNIa',label2='SN2',shape1=':',shape2='--',marker1='o',marker2='s',color1='k',color2='b',markevery=20,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14):

        '''
        Plots the SN1a distribution:

        The evolution of the number of SN1a and SN2


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
                plt.xlabel('redshift z')
                timesteps=self.history.timesteps[idx-1:]
                sn2numbers=sn2numbers[idx:]
                sn1anumbers=sn1anumbers[idx:]
        else:
                plt.xlabel('age [yr]')
                plt.xscale('log')
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
                plt.xscale('log')
                plt.ylabel('number of SNIa going off per WD born')
                label='SNIafractionperWD';label='sn1a '+label
                x=age1
                y=ratio
                self.__save_data(header=['age',label],data=[x,y])
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
                self.__save_data(header=['age','SNIa '+label,'age','CCSN '+label],data=[x[0],y[0],x[1],y[1]])
        else:
                if rate_only=='sn1a':
                        label='sn1a '+label
                else:
                        label='ccsn '+label
                self.__save_data(header=['age',label],data=[x,y])


    ##############################################
    #          Plot Star Formation Rate          #
    ##############################################
    def plot_star_formation_rate(self,fig=6,fraction=True,source='all',marker='',shape='',color='',label='',abs_unit=True,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14):
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
        import matplotlib.pyplot as plt
        if (len(marker)==0 and len(shape)==0) and len(color)==0:
                shape,marker,color=self.__msc(source,shape,marker,color)
        plt.figure(fig, figsize=(fsize[0],fsize[1]))
        #maybe a histogram for display the SFR?
        if False:
                age=self.history.age
                #age=[0.1]+self.history.age[1:-1]
                sfr=self.history.sfr
                plt.xlabel('age [yr]')
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

        #Plot the mass fraction of gas available converted into stars
        if fraction:
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
                plt.xlabel('age [yr]')
                plt.plot(age,sfr,label=label,marker=marker,linestyle=shape)
                plt.ylabel('fraction of current gas mass into stars')
                self.__save_data(header=['age','SFR'],data=[age,sfr])

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
                plt.xlabel('age [yr]')
                plt.ylabel('ISM mass transformed into stars')
                plt.xscale('log');plt.yscale('log')
                plt.legend()

        ax=plt.gca()
        self.__fig_standard(ax=ax,fontsize=fontsize,labelsize=labelsize,rspace=rspace, bspace=bspace,legend_fontsize=legend_fontsize)

        #print ('Total mass transformed in stars, total mass transformed in AGBs, total mass transformed in massive stars:')
        #print (sum(self.history.m_locked),sum(self.history.m_locked_agb),sum(self.history.m_locked_massive))

    def plot_mass_range_contributions(self,fig=7,specie='C',prodfac=False,rebin=1,time=-1,label='',shape='-',marker='o',color='r',markevery=20,extralabel=False,log=False,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14, histtype="stepfilled"):


        '''
	Plots yield contribution (Msun) of a certain mass range
	versus initial mass. Each stellar ejecta in one mass range
	is represented by the same yields, yields from certain stellar simulation.
	Be aware that a larger mass range means also a larger amount
	of yield for that range.
	Note: Used in WENDI.

        Parameters
        ----------
        specie : string
            Element or isotope name in the form 'C' or 'C-12'
	prodfac : boolean
	    if true, calculate stellar production factor for each mass range.
	    It is the final stellar yield divided by the initial (IMF-weighted)
	    total mass (ISM+stars) in that region.
	rebin : change the bin size to uniform mass intervals of size 'rebin'
	        default 0, bin size is defined by ranges of (stellar) yield inputs	
	log : boolean
	    if true, use log yaxis
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
        >>> s.plot_mass_range_contribution(element='C')

        '''

        import matplotlib.pyplot as plt
        figure=plt.figure(fig, figsize=(fsize[0],fsize[1]))


        #e.g. for testing: ratio
        #ratio, e.g. C/Fe
        if '/' in specie:
                specie1=specie.split('/')[0]
                mean_val,bin_bdys,y1,color,label=self.__plot_mass_range_contributions_single(fig,specie1,prodfac,rebin,time,label,shape,marker,color,markevery,extralabel,log,fsize,fontsize,rspace,bspace,labelsize,legend_fontsize)
                specie2=specie.split('/')[1]
                mean_val,bin_bdys,y2,color,label=self.__plot_mass_range_contributions_single(fig,specie2,prodfac,rebin,time,label,shape,marker,color,markevery,extralabel,log,fsize,fontsize,rspace,bspace,labelsize,legend_fontsize)

                y=np.array(y1)/np.array(y2)
                label=specie
        #to get the total mass
        if 'all' in specie:
                eles=self.history.elements
                for k in range(len(eles)):
                        specie=eles[k]
                        ytmp=0
                        mean_val,bin_bdys,ytmp,color,labeltmp=self.__plot_mass_range_contributions_single(fig,specie,prodfac,rebin,time,label,shape,marker,color,markevery,extralabel,log,fsize,fontsize,rspace,bspace,labelsize,legend_fontsize)
                        if k==0:
                                y=np.array(ytmp)
                        else:
                                y= y+np.array(ytmp)
        #default, mass, C, C-12
        else:
                mean_val,bin_bdys,y,color,label=self.__plot_mass_range_contributions_single(fig,specie,prodfac,rebin,time,label,shape,marker,color,markevery,extralabel,log,fsize,fontsize,rspace,bspace,labelsize,legend_fontsize)



        if prodfac==True:
                p1 =plt.hist(mean_val, bins=bin_bdys,weights=y,facecolor=color,color=color,alpha=0.5,label=label,ec='black', histtype=histtype)
        else:
                p1 =plt.hist(mean_val, bins=bin_bdys,weights=y,facecolor=color,color=color,alpha=0.5,bottom=0.001,label=label,ec='black', histtype=histtype)
        #'''
        if len(label)>0:
                plt.legend()
        ax1=plt.gca()
        ax1.set_xlabel('initial mass [M$_{\odot}$]')
        if prodfac==False:
                ax1.set_ylabel('IMF-weighted yield [M$_{\odot}$]')
        else:
                ax1.set_ylabel('production factor')
        if log==True:
                ax1.set_yscale('log')
        lwtickboth=[6,2]
        lwtickmajor=[10,3]
        plt.xlim(min(bin_bdys),max(bin_bdys))
        plt.legend(loc=2,prop={'size':legend_fontsize})
        plt.rcParams.update({'font.size': fontsize})
        ax1.yaxis.label.set_size(labelsize)
        ax1.xaxis.label.set_size(labelsize)
        #ax.xaxis.set_tick_params(width=2)
        #ax.yaxis.set_tick_params(width=2)
        ax1.tick_params(length=lwtickboth[0],width=lwtickboth[1],which='both')
        ax1.tick_params(length=lwtickmajor[0],width=lwtickmajor[1],which='major')
        #Add that line below at some point
        #ax.xaxis.set_tick_params(width=2)
        #ax.yaxis.set_tick_params(width=2)
        ax1.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),markerscale=0.8,fontsize=legend_fontsize)
        #self.__save_data(header=['Mean mass','mass bdys (bins)','Yield'],data=[mean_val,bin_bdys,y])
        plt.subplots_adjust(right=rspace)
        plt.subplots_adjust(bottom=bspace)

        #print [mean_val,bin_bdys,y]
        return

    def __plot_mass_range_contributions_single(self,fig=7,specie='C',prodfac=False,rebin=1,time=-1,label='',shape='-',marker='o',color='r',markevery=20,extralabel=False,log=False,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14):

        '''
        Internal plotting function for function plot_mass_range_contributions
        '''
        import matplotlib.pyplot as plt

        if len(label)==0:
            label=specie


        contribution=[]
        #find starburst corresponding to time
        if time>0:
                age1=self.history.age
                if time in age1:
                        age_idx=age1.index(time)
                else:
                        age_idx=min(range(len(age1)), key=lambda i: abs(age1[i]-time))
                        print ('Age not found, choose closest age',age1[age_idx])
                mass_ranges=self.history.imf_mass_ranges[age_idx]
                contribution=self.history.imf_mass_ranges_contribution[age_idx]
                mtots=self.history.imf_mass_ranges_mtot[age_idx]
        #take sum of all star bursts
        else:

                firstTime=True
                for k in range(len(self.history.imf_mass_ranges)):
                        mass_ranges1=self.history.imf_mass_ranges[k]
                        if len(mass_ranges1)>0 and firstTime==True:
                                mass_ranges=self.history.imf_mass_ranges[k]
                                contribution=list(self.history.imf_mass_ranges_contribution[k])
                                mtots=self.history.imf_mass_ranges_mtot[k]
                                firstTime=False
                        elif len(mass_ranges1)>0 and (not len(self.history.imf_mass_ranges[k]) == len(mass_ranges)):
                                print ('Error: Different mass range intervalls used: cannot combine them')
                                print ('Choose a specific staburst via time')
                                return 0
                        elif len(mass_ranges1)>0:
                                mtots=np.array(mtots)+np.array(self.history.imf_mass_ranges_mtot[k])
                                #carries all isotopes, hence an array: self.history.imf_mass_ranges_contribution[k][h]
                                for h in range(len(self.history.imf_mass_ranges_contribution[k])):
                                        contribution[h]= np.array(contribution[h])+np.array(self.history.imf_mass_ranges_contribution[k][h])


        isotopes=self.history.isotopes
        iso_idx_array=[]
        x_iso_ism_ini=np.array(self.history.ism_iso_yield[0])/self.history.mgal
        #get the element specific indices
        for k in range(len(isotopes)):
            if not '-' in specie:
                specie=specie+'-'
            if specie in isotopes[k]:
                iso_idx_array.append(isotopes.index(isotopes[k]))
        #get the sum of yields for each isotope and mass range
        y=[0]*len(contribution)
        for k in range(len(contribution)):
            yields=0 #[0]*len(contribution[k])
            x_ini=0
            for iso_idx in iso_idx_array:
                yields+= np.array(contribution[k][iso_idx])
                x_ini+=x_iso_ism_ini[iso_idx]
            if prodfac==True:
                if x_ini ==0:
                        print ('Initial abundance not available!')
                        print ('Cannot plot production factor.')
                        return 0
                y[k]=np.array(yields)/(mtots[k]*x_ini)
            else:
                y[k]=yields

        #masses1=[]
        #for m in range(len(mass_ranges)):
        #    masses1.append( (mass_ranges[m][0]+mass_ranges[m][1])/2.)
        #masses_idx= sorted(range(len(masses1)),key=lambda x:masses1[x])

        #for idx in masses_idx:
        #    masses.append(masses1[idx])
        #    yields.append(yields1[idx])
        #'''
        #for histo
        bin_bdys=[]
        mean_val=[]
        #bin_bdys.append(1)
        bin_values=[]
        #rebin the plot using the bin size rebin
        if rebin >0:
                #_imf(self,mmin,mmax,inte,mass=0,iolevel=0)
                mass=mass_ranges[0][0]
                mmax=mass_ranges[-1][1]
                bin_bdys1=[]
                while True:
                        bin_min=round(mass,5)
                        mass+=rebin
                        bin_max=round(mass,5)
                        if (mmax==0):
                                break
                        if bin_max>mmax:
                                bin_max=mmax
                                mass=mmax
                                mmax=0
                        bin_bdys1.append([bin_min,bin_max])
                        bin_values.append(0)
                        for k in range(len(mass_ranges)):
                                if (mass_ranges[k][1]<=bin_min) or (mass_ranges[k][0]>=bin_max):
                                        continue
                                #if mass range inside bin
                                elif (mass_ranges[k][0]>=bin_min) and (mass_ranges[k][1]<=bin_max):
                                        bin_values[-1]+=y[k]
                                        #print ('bin includes mass range',y[k])
                                        continue
                                #if mass range includes bin:
                                elif (mass_ranges[k][0]<=bin_min) and (mass_ranges[k][1]>=bin_max):
                                        #normalization to bin mass
                                        h=y[k]/self._imf(mass_ranges[k][0],mass_ranges[k][1],inte=2)
                                        y1=h*self._imf(bin_min,bin_max,inte=2)
                                        bin_values[-1]+=y1
                                        #print ('mass range inlcudes bin ',bin_min,bin_max,y1)

                                #if upper part of bin is not in mass range
                                elif (mass_ranges[k][1]<bin_max):
                                        #normalization to bin mass
                                        h=y[k]/self._imf(mass_ranges[k][0],mass_ranges[k][1],inte=2)
                                        y1=h*self._imf(bin_min,mass_ranges[k][1],inte=2)
                                        bin_values[-1]+=y1
                                        #print ('add upper half from ',bin_min,mass_ranges[k][1],y1)
                                #if lower part of bin is not in mass range
                                elif mass_ranges[k][0]>bin_min:
                                        #normalization to bin mass
                                        h=y[k]/self._imf(mass_ranges[k][0],mass_ranges[k][1],inte=2)
                                        y1=h*self._imf(mass_ranges[k][0],bin_max,inte=2)
                                        #print ('add lower half from ',mass_ranges[k][0],bin_max,y1)
                                        bin_values[-1]+=y1
                        #if bin_values[-1]==0:
                        #        print ('Note that no values are found in bin range:',bin_min,'-',bin_max)
                                #return 0
                        if mmax==bin_max:
                                break
        #        print (bin_bdys1)
#                print (bin_values)
                mass_ranges=bin_bdys1
                y=bin_values


        for k in range(len(mass_ranges)):
                #yields1.append(yields[k]) #
                if k==0:
                        bin_bdys.append(mass_ranges[k][0])
                bin_bdys.append(mass_ranges[k][1])
                mean_val.append( (mass_ranges[k][0] + mass_ranges[k][1])/2.)

        return mean_val,bin_bdys,y,color,label


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

    def __save_data(self,header=[],data=[],filename='plot_data.txt'):
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


    def write_evol_table(self,elements=['H'],isotopes=['H-1'],table_name='gce_table.txt', path="",interact=False):

        '''
        Writes out evolution of the SSP accummulated ejecta in fraction of total mass of the SSP in the following format
        (each timestep in a line):

        &Age       &H-1       &H-2  ....

        &0.000E+00 &1.000E+00 &7.600E+08 & ...

        &1.000E+07 &1.000E+00 &7.600E+08 & ...

        ....

        This method has a notebook and normal version. If you are
        using a python notebook the function will
        open a link to a page containing the table.


        Parameters
        ----------
        table_name : string,optional
          Name of table. If you use a notebook version, setting a name
          is not necessary.
        elements : array
                Containing the elements with the name scheme 'H','C'
        isotopes : array
                If elements list empty, ignore elements input and use isotopes input; Containing the isotopes with the name scheme 'H-1', 'C-12'
        interact: bool
                If true, saves file in current directory (notebook dir) and creates HTML link useful in ipython notebook environment

        Examples
        ----------

        >>> s.write_evol_table(elements=['H','C','O'],table_name='testoutput.txt')

        '''
        if path == "":
            path = os.path.join(nupy_path, 'evol_tables')

        yields_evol=self.history.ism_iso_yield
        metal_evol=self.history.metallicity
        time_evol=self.history.age
        idx=[]
        if len(elements)>0:
                elements_tot=self.history.elements
                for k in range(len(elements_tot)):
                        if elements_tot[k] in elements:
                                idx.append(elements_tot.index(elements_tot[k]))
                yields_specie=self.history.ism_elem_yield
                specie=elements
        elif len(isotopes)>0:
                iso_tot=self.history.isotopes
                for k in range(len(iso_tot)):
                        if iso_tot[k] in isotopes:
                                idx.append(iso_tot.index(iso_tot[k]))
                yields_specie=self.history.ism_iso_yield
                specie=isotopes

        else:
                print ('Specify either isotopes or elements')
                return
        if len(idx)==0:
                print ('Please choose the right isotope names')
                return 0


        frac_yields=[]
        for h in range(len(yields_specie)):
                frac_yields.append([])
                for k in range(len(idx)):
                        frac_yields[-1].append( np.array(yields_specie[h][idx[k]])/self.history.mgal)

        mtot_gas=self.history.gas_mass


        metal_evol=self.history.metallicity
        #header
        out='&Age [yr]  '
        for i in range(len(specie)):
            out+= ('&'+specie[i]+((10-len(specie[i]))*' '))
        out+='M_tot \n'
        #data
        for t in range(len(frac_yields)):
            out+=('&'+'{:.3E}'.format(time_evol[t]))
            #out+=(' &'+'{:.3E}'.format(frac_yields[t]))
            for i in range(len(specie)):
                out+= ( ' &'+ '{:.3E}'.format(frac_yields[t][i]))
            out+=( ' &'+ '{:.3E}'.format(mtot_gas[t]))
            out+='\n'

        if interact==True:
                import random
                randnum=random.randrange(10000,99999)
                name=table_name+str(randnum)+'.txt'
                #f1=open(os.path.join(nupy_path, 'evol_tables', name),'w')
                f1=open(name,'w')
                f1.write(out)
                f1.close()
                print ('Created table '+name+'.')
                print ('Download the table using the following link:')
                #from IPython.display import HTML
                #from IPython import display
                from IPython.core.display import HTML
                import IPython.display as display
                #return HTML("""<a href="evol_tables/download.php?file="""+name+"""">Download</a>""")
                #test=
                #return display.FileLink('../../nugrid/SYGMA/SYGMA_online/SYGMA_dev/evol_table/'+name)
                #if interact==False:
                #return HTML("""<a href="""+nupy_path+"""/evol_tables/"""+name+""">Download</a>""")
                return HTML("""<a href="""+name+""">Download</a>""")
                #else:
                #        return name
        else:
                print ('file '+table_name+' saved in subdirectory evol_tables.')
                f1=open(os.path.join(path, table_name),'w')
                f1.write(out)
                f1.close()


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


