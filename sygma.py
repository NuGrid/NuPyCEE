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
              M. Pignatari, S. Jones, K. Venn1, P. A. Denissenkov & the 
              NuGrid collaboration

v0.4 FEB2015: C. Ritter, B. Cote


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

>>> s2 = s.sygma(iniZ=0.0001,dt=1e8,tend=1.5e10, mgal=1e11,table='yield_tables/isotope_yield_table_h1.txt',
    sn1a_table='yield_tables/sn1a_h1.txt',iniabu_table='yield_tables/iniab1.0E-04GN93_alpha_h1.ppn.txt')

'''

# Standard package
import matplotlib.pyplot as plt

# Import the class inherited by SYGMA
from chem_evol import *
from plotting import *


class sygma( chem_evol, plotting ):

    '''
    Input parameters (SYGMA)
    ================

    sfr : string
	Description of the star formation, usually an instantaneous burst.
        Choices : 'input' - read and use the sfr_input file to set the percentage
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
                 sn1a_rate='power_law', iniZ=0.0, dt=1e6, special_timesteps=30, \
                 nsmerger_bdys=[8, 100], tend=13e9, mgal=1e4, transitionmass=8, iolevel=0, \
                 ini_alpha=True, table='yield_tables/isotope_yield_table.txt', \
                 hardsetZ=-1, sn1a_on=True, sn1a_table='yield_tables/sn1a_t86.txt',\
		 ns_merger_on=True, f_binary=1.0, f_merger=0.0028335, \
                 nsmerger_table = 'yield_tables/r_process_rosswog_2014.txt', iniabu_table='', \
                 extra_source_on=False, \
                 extra_source_table='yield_tables/mhdjet_NTT_delayed.txt', \
                 pop3_table='yield_tables/popIII_heger10.txt', \
                 imf_bdys_pop3=[0.1,100], imf_yields_range_pop3=[10,30], \
                 starbursts=[], beta_pow=-1.0,gauss_dtd=[1e9,6.6e8],exp_dtd=2e9,\
                 nb_1a_per_m=1.0e-3,direct_norm_1a=-1, Z_trans=0.0, \
                 f_arfo=1.0, imf_yields_range=[1,30],exclude_masses=[], \
                 netyields_on=False,wiersmamod=False,yield_interp='lin', \
                 dt_in=np.array([]),\
                 ytables_in=np.array([]), zm_lifetime_grid_nugrid_in=np.array([]),\
                 isotopes_in=np.array([]), ytables_pop3_in=np.array([]),\
                 zm_lifetime_grid_pop3_in=np.array([]), ytables_1a_in=np.array([]), \
		 ytables_nsmerger_in=np.array([])):

        # Call the init function of the class inherited by SYGMA
        chem_evol.__init__(self, imf_type=imf_type, alphaimf=alphaimf, \
                 imf_bdys=imf_bdys, sn1a_rate=sn1a_rate, iniZ=iniZ, dt=dt, \
                 special_timesteps=special_timesteps, tend=tend, mgal=mgal, \
                 nsmerger_bdys=nsmerger_bdys, transitionmass=transitionmass, iolevel=iolevel, \
                 ini_alpha=ini_alpha, table=table, hardsetZ=hardsetZ, \
                 sn1a_on=sn1a_on, sn1a_table=sn1a_table, \
		 ns_merger_on=ns_merger_on, nsmerger_table=nsmerger_table, \
		 f_binary=f_binary, f_merger=f_merger, \
                 iniabu_table=iniabu_table, extra_source_on=extra_source_on, \
                 extra_source_table=extra_source_table, pop3_table=pop3_table, \
                 imf_bdys_pop3=imf_bdys_pop3, \
                 imf_yields_range_pop3=imf_yields_range_pop3, \
                 starbursts=starbursts, beta_pow=beta_pow, \
                 gauss_dtd=gauss_dtd,exp_dtd=exp_dtd,\
                 nb_1a_per_m=nb_1a_per_m,direct_norm_1a=direct_norm_1a, \
                 Z_trans=Z_trans, f_arfo=f_arfo, \
                 imf_yields_range=imf_yields_range,exclude_masses=exclude_masses,\
                 netyields_on=netyields_on,wiersmamod=wiersmamod,\
                 yield_interp=yield_interp, \
                 ytables_in=ytables_in, \
                 zm_lifetime_grid_nugrid_in=zm_lifetime_grid_nugrid_in,\
                 isotopes_in=isotopes_in,ytables_pop3_in=ytables_pop3_in,\
                 zm_lifetime_grid_pop3_in=zm_lifetime_grid_pop3_in,\
		 ytables_1a_in=ytables_1a_in, ytables_nsmerger_in=ytables_nsmerger_in, \
		 dt_in=dt_in)

        if self.need_to_quit:
            return

        # Announce the beginning of the simulation 
        print 'SYGMA run in progress..'
        start_time = t_module.time()
	self.start_time = start_time

        # Attribute the input parameter to the current object
        self.sfr = sfr

        # Get the SFR of every timestep
        self.sfrin_i = self.__sfr()

        # Run the simulation
        self.__run_simulation()

        # Do the final update of the history class
        self._update_history_final()

        # Announce the end of the simulation
        print '   SYGMA run completed -',self._gettime()


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
            self._evol_stars(i)

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
        sfr_i = []

        # Output information
        if self.iolevel >= 3:
            print 'Entering sfr routine'

        # For every timestep i considered in the simulation ...
        for i in range(1, self.nb_timesteps+1):

            # If an array is used to generate starbursts ...
            if len(self.starbursts) > 0:
                if len(self.starbursts) >= i:

                    # Use the input value
                    sfr_i.append(self.starbursts[i-1])
                    self.history.sfr.append(sfr_i[i-1])

            # If an input file is read for the SFR ...
            if self.sfr == 'input':

                # Open the input file, read all lines, and close the file
                f1 = open(global_path+'sfr_input')
                lines = f1.readlines()
                f1.close()

                # The number of lines needs to be at least equal to the
                # number of timesteps
                if self.nb_timesteps > (len(lines)):
                    print 'Error - SFR input file does not' \
                          'provide enough timesteps'
                    return

                # Copy the SFR (mass fraction) of every timestep
                for k in range(len(lines)):
                    if k == (i-1):
                        sfr_i.append(float(lines[k]))
                        self.history.sfr.append(sfr_i[i-1])
                        break

            # If the Schmidt law is used (see Timmes98) ... 
            if self.sfr == 'schmidt':

                # Calculate the mass of available gas
                mgas = sum(ymgal[i-1])

                # Calculate the SFR according to the current gas fraction
                B = 2.8 * self.mgal * (mgas / self.mgal)**2    # [Mo/Gyr]
                sfr_i.append(B/mgas) * (timesteps[i-1] / 1.e9) # mass fraction
                self.history.sfr.append(sfr_i[i-1])

        # Return the SFR (mass fraction) of every timestep
        return sfr_i


 
###############################################################################################
######################## Here start the analysis methods ######################################
###############################################################################################

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
		plt.xlabel('Log-scaled age [yrs]')
		#ax_plots[k].locator_params(axis = 'y', nbins = 2)
		#if norm == False:
		    #ax_plots[k].set_ylabel('Log-scaled yields [Msun]')
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
		fig.text(0.002, 0.5, 'Log (Yields [Msun])', ha='center', va='center', rotation='vertical')
	else:
		fig=plt.gcf()
		fig.text(0.01, 0.5, '(IMF-weighted) fraction of ejecta', ha='center', va='center', rotation='vertical')

	#set lim here
	#print limits
	#for k in range(len(ax_plots)):
	#	ax_plots[k].set_ylim(limits[k][0],limits[k][1])

	#f.subplots_adjust(hspace=0.35)#0)
	#plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
	return 

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
        idx=int(t/self.history.dt) -1
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
                    print 'new iso',isotopes[k]
                    #i+=1
                    if not k==(len(isotopes)-1):
                        continue
                print k
                plt.plot(elem_array,y_array,marker=marker_type[marker1],markersize=8,linestyle=line_style[linest])
                print elem_array
                print y_array[0]
                i+=1
                if (i%2) ==0:
                    high=6
                else:
                    high=4
                print 'i',i
                print 'high',high
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

        plt.xlabel("Mass number A")
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

    def __plot_mass_range_contributions_single(self,fig=7,specie='C',prodfac=False,rebin=1,time=-1,label='',shape='-',marker='o',color='r',markevery=20,extralabel=False,log=False,fsize=[10,4.5],fontsize=14,rspace=0.6,bspace=0.15,labelsize=15,legend_fontsize=14):

        '''
	Internal plotting function for function plot_mass_range_contributions
        '''
    

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
                        print 'Age not found, choose closest age',age1[age_idx]
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
                                print 'Error: Different mass range intervalls used: cannot combine them'
                                print 'Choose a specific staburst via time'
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
			print 'Initial abundance not available!'
			print 'Cannot plot production factor.'
			return 0
		y[k]=np.array(yields)/(mtots[k]*x_ini)
	    else:
	 	y[k]=yields
		
	#print mass_ranges
	#print 'contribution'
	#print y

        #masses1=[]
        #for m in range(len(mass_ranges)):
        #    masses1.append( (mass_ranges[m][0]+mass_ranges[m][1])/2.)
        #masses_idx= sorted(range(len(masses1)),key=lambda x:masses1[x])

        #print 'len same: ',len(mass_ranges),len(masses)
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
	#print mass_ranges
	if rebin >0:
		#print mass_ranges
		#print y
		#nprint '-------------------'
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
			#print 'Bin :',bin_bdys1[-1]
			for k in range(len(mass_ranges)):
				if (mass_ranges[k][1]<=bin_min) or (mass_ranges[k][0]>=bin_max):
					continue
				#print 'interval ',mass_ranges[k]
				#if mass range inside bin
				elif (mass_ranges[k][0]>=bin_min) and (mass_ranges[k][1]<=bin_max):
					bin_values[-1]+=y[k]
					#print 'bin includes mass range',y[k]
					continue
				#if mass range includes bin:
				elif (mass_ranges[k][0]<=bin_min) and (mass_ranges[k][1]>=bin_max):
                                        #normalization to bin mass
                                        h=y[k]/self._imf(mass_ranges[k][0],mass_ranges[k][1],inte=2)
                                        y1=h*self._imf(bin_min,bin_max,inte=2)
                                        bin_values[-1]+=y1
                                        #print 'mass range inlcudes bin ',bin_min,bin_max,y1
					
				#if upper part of bin is not in mass range
				elif (mass_ranges[k][1]<bin_max):
					#normalization to bin mass
					h=y[k]/self._imf(mass_ranges[k][0],mass_ranges[k][1],inte=2)
					y1=h*self._imf(bin_min,mass_ranges[k][1],inte=2)
					bin_values[-1]+=y1
					#print 'add upper half from ',bin_min,mass_ranges[k][1],y1
                                #if lower part of bin is not in mass range
                                elif mass_ranges[k][0]>bin_min:					
					#normalization to bin mass
					h=y[k]/self._imf(mass_ranges[k][0],mass_ranges[k][1],inte=2)
					y1=h*self._imf(mass_ranges[k][0],bin_max,inte=2)
					#print 'add lower half from ',mass_ranges[k][0],bin_max,y1
					bin_values[-1]+=y1
			#if bin_values[-1]==0:
			#	print 'Note that no values are found in bin range:',bin_min,'-',bin_max
				#return 0
			if mmax==bin_max:
				break
	#	print bin_bdys1
#		print bin_values
		mass_ranges=bin_bdys1
		y=bin_values


	for k in range(len(mass_ranges)):
                #yields1.append(yields[k]) #
		if k==0:
                	bin_bdys.append(mass_ranges[k][0])
                bin_bdys.append(mass_ranges[k][1])
		mean_val.append( (mass_ranges[k][0] + mass_ranges[k][1])/2.)
	#print 'test'		
	#print yields
	#print bin_bdys
	#print mean_val

	return mean_val,bin_bdys,y,color,label

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


    def write_evol_table(self,elements=['H'],isotopes=['H-1'],table_name='gce_table.txt', path="",interact=False):

        '''
	Writes out evolution of time, metallicity
	and each isotope in the following format
	(each timestep in a line):

	&Age       &H-1       &H-2  ....

	&0.000E+00 &1.000E+00 &7.600E+08 & ...

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
            path = global_path+'evol_tables/'

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
		print 'Specify either isotopes or elements'
		return
	if len(idx)==0:
		print 'Please choose the right isotope names'
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
		#f1=open(global_path+'evol_tables/'+name,'w')
		f1=open(name,'w')
		f1.write(out)
		f1.close()
		print 'Created table '+name+'.'
		print 'Download the table using the following link:'
		#from IPython.display import HTML
		#from IPython import display
		from IPython.core.display import HTML
		import IPython.display as display
		#print help(HTML)
		#return HTML("""<a href="evol_tables/download.php?file="""+name+"""">Download</a>""")
		#test=
		#return display.FileLink('../../nugrid/SYGMA/SYGMA_online/SYGMA_dev/evol_table/'+name)
                #if interact==False:
                #return HTML("""<a href="""+global_path+"""/evol_tables/"""+name+""">Download</a>""")
		return HTML("""<a href="""+name+""">Download</a>""")
                #else:
                #        return name
	else:
		print 'file '+table_name+' saved in subdirectory evol_tables.'
		f1=open(path+table_name,'w')
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
			time_new.append(time[k]/1e9)
			if firstidx:
				index=k
				firstidx=False	
	return tz.t_to_z(time_new,Hubble_0,Omega_lambda,Omega_m),index


