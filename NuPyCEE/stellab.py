from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

'''

Stellab (Stellar Abundances)


Fonctionality
=============

This class plots observational stellar abundances data for different galaxies.


Made by
=======

JUNE2015: B. Cote


Usage
=====

Import the module:

>>> import stellab

Load the data:

>>> s = stellab.stellab()

Plot [Mg/Fe] vs [Fe/H] for the Fornax dwarf spheoridal galaxy:

>>> s.plot_spectro(xaxis='[Fe/H]', yaxis='[Mg/Fe]', galaxy='fornax')

See the Sphinx documentation for more options

'''


# Import standard packages
import matplotlib
import matplotlib.pyplot as plt
import os

# Define where is the working directory
# This is where the NuPyCEE code will be extracted
nupy_path = os.path.dirname(os.path.realpath(__file__))

# Import NuPyCEE codes
import NuPyCEE.read_yields as ry

class stellab():

    '''
    See the plot_spectro() function to plot the data.

    '''

    # Input paramters
    def __init__(self):


        # read abundance data library via
        # index file abundance_data_library.txt
        # creates self.paths, self.paths_s, self.cs, self.leg
        self.read_abundance_data_library(os.path.join(nupy_path, "stellab_data",\
                "abundance_data_library.txt"))

        # Declaration of the name of available galaxies
        #self.galaxy_name = []
        #self.galaxy_name.append('milky_way')
        #self.galaxy_name.append('sculptor')
        #self.galaxy_name.append('fornax')
        #self.galaxy_name.append('carina')
        #self.galaxy_name.append('lmc')

        # Declaration of arrays containing information on every data set
        self.elem_list = []  # Available elements
        self.elem      = []  # Nominator and denominator of each abundance ratio
        self.ab        = []  # Stellar abundances
        #self.paths     = []  # Path to the stellar abundances file
        #self.paths_s   = []  # Path to the solar values
        self.solar     = []  # Solar values used
        #self.cs        = []  # Color and symbol
        #self.leg       = []  # Legend

        ## List of all the solar values used in the data sets
        #for i_path in range(0,len(self.paths)):
        #    self.paths_s.append(self.paths[i_path]+'_s')

        # For every data set ...
        for i_entry in range(0,len(self.paths)):

            # Copy the stellar abundaces
            self.__read_data_file(self.paths[i_entry], i_entry)

            # Copy the solar values used for the normalization
            self.__read_solar_values(self.paths_s[i_entry], i_entry)

            # Copy the list of available elements
            self.elem_list.append([])
            for i_elem in range(0,len(self.solar[i_entry])):
                self.elem_list[i_entry].append(self.solar[i_entry][i_elem][0])

        # Declaration of arrays containing information for every solar reference
        self.paths_norm = []  # Path of the files
        self.sol_norm   = []  # Solar abundance "eps_x = log(n_x/n_H) + 12"

        # List of all the reference solar normalization (path to file)
        self.paths_norm.append('Anders_Grevesse_1989')
        self.paths_norm.append('Grevesse_Noels_1993')
        self.paths_norm.append('Grevesse_Sauval_1998')
        self.paths_norm.append('Asplund_et_al_2009')
        self.paths_norm.append('Asplund_et_al_2005')
        self.paths_norm.append('Lodders_et_al_2009')

        # For every solar reference ...
        for i_sol in range(0,len(self.paths_norm)):

            # Copy the solar values
            self.__read_solar_norm(self.paths_norm[i_sol], i_sol)


    ##############################################
    #        Read abundance data library         #
    ##############################################
    def read_abundance_data_library(self,filename):

        # open index file
        f=open(filename)
        lines=f.readlines()
        f.close()

        # Declaration of the name of available galaxies

        # name of galaxy
        self.galaxy_name = []
        # color + symbol
        self.cs=[]
        # legend
        self.leg=[]
        # path to the stellar abundances file
        self.paths = []
        for k in range(len(lines)):
            line = lines[k]
            #ignore header
            if k == 0:
                continue
            #if commented line
            if line[0] == '#':
                continue
            # Declaration of the name of available galaxies
            if line[0] == 'H':
                galaxy=line[1:].strip()
                self.galaxy_name.append(galaxy)
                #print ('found galaxy ',galaxy)
            # read data set
            else:
                #print (line)
                path = 'stellab_data/'+line.split('&&')[0].strip()
                leg = line.split('&&')[1].strip()
                cs = line.split('&&')[2].strip()
                self.paths.append(path)
                self.leg.append(leg)
                self.cs.append(cs)
                #print ('path : ',path)
                #print ('leg : ',leg)

        self.paths_s   = []  # Path to the solar values

        # List of all the solar values used in the data sets
        for i_path in range(0,len(self.paths)):
            self.paths_s.append(self.paths[i_path]+'_s')


    ##############################################
    #               Read Data File               #
    ##############################################
    def __read_data_file(self, file_path, i_entry):

        # Create an entry
        self.elem_list.append([])
        self.elem.append([])
        self.ab.append([])

        # Open the data file
        with open(os.path.join(nupy_path, file_path + ".txt"), "r") as data_file:

            # Read and split the first line (header)
            line_1_str = data_file.readline()
            ratios_f = [str(x) for x in line_1_str.split()]

            # Extract the X and Y in the [X/Y] ratio
            err_available = self.__extract_x_y(ratios_f, i_entry)

            # Get the number of ratios
            nb_ratios = len(self.elem[i_entry])

            # For every remaining line (for each star) ...
            i_star = 0
            for line_2_str in data_file:

                # Split the line
                ab_f = [float(x) for x in line_2_str.split()]

                # Create a star entry
                self.ab[i_entry].append([])

                # For every abundance ratio ...
                i_ab = 0
                for i_ratio in range(0,nb_ratios):

                    # Copy the ratio (stellar abundance)
                    self.ab[i_entry][i_star].append([ab_f[i_ab],0.0])
                    i_ab += 1

                    # Add the error bars if any
                    if err_available:
                      if ab_f[i_ab] == -30.0:
                        self.ab[i_entry][i_star][i_ratio][1] = 0.0
                      else:
                        self.ab[i_entry][i_star][i_ratio][1] = ab_f[i_ab]
                      i_ab += 1

                # Move to the next line (next star)
                i_star += 1

        # Close the file
        data_file.close()


    ##############################################
    #             Read Solar Values              #
    ##############################################
    def __read_solar_values(self, file_path, i_entry):

        # Create an entry
        self.solar.append([])

        # Open the data file
        with open(os.path.join(nupy_path, file_path + ".txt"), "r") as data_file:

            # For every line (for each element) ...
            i_elem = 0
            for line_1_str in data_file:

                # Create an entry
                self.solar[i_entry].append([])

                # Split the line
                split = [str(x) for x in line_1_str.split()]

                # Copy the element and the solar value
                self.solar[i_entry][i_elem].append(str(split[0]))
                self.solar[i_entry][i_elem].append(float(split[1]))

                # Go to the next element
                i_elem += 1

        # Close the file
        data_file.close()


    ##############################################
    #              Read Solar Norm               #
    ##############################################
    def __read_solar_norm(self, file_path, i_sol):

        # Create an entry
        self.sol_norm.append([])

        # Open the data file
        with open(os.path.join(nupy_path, "stellab_data", "solar_normalization",\
            file_path + ".txt"), "r") as data_file:

            # For every line (for each element) ...
            i_elem = 0
            for line_1_str in data_file:

                # Create an entry
                self.sol_norm[i_sol].append([])

                # Split the line
                split = [str(x) for x in line_1_str.split()]

                # Copy the element and the solar value
                self.sol_norm[i_sol][i_elem].append(str(split[1]))
                self.sol_norm[i_sol][i_elem].append(float(split[2]))

                # Go to the next element
                i_elem += 1

        # Close the file
        data_file.close()


    ##############################################
    #                Extract X Y                 #
    ##############################################
    def __extract_x_y(self, ratios_f, i_entry):

        # Variable that indicates if the file contains error bars
        err_available = False

        # For each available ratios [X/Y] ...
        for i_ratio in range(0,len(ratios_f)):

            # Look if the ratio is actually an error bar
            if ratios_f[i_ratio] == 'err':
                err_available = True

            # If it is not an error bar ...
            else:

                # Get X and Y
                x_extxt, y_extxt = self.__get_x_y(ratios_f[i_ratio])

                # Copy the extracted X and Y strings
                self.elem[i_entry].append([x_extxt,y_extxt])

        # Return wether or not error bars were present in the input string
        return err_available


    ##############################################
    #                  Get X Y                   #
    ##############################################
    def __get_x_y(self, str_ratio):

        # Initialisation of the numerator X and the denominator Y
        x_in = ''
        y_in = ''
        is_num = True

        for i_str in range(0,len(str_ratio)):
            if str_ratio[i_str] == '/':
                is_num = False
            elif (not str_ratio[i_str] == '[') and \
                 (not str_ratio[i_str] == ']'):
                if is_num:
                    x_in += str_ratio[i_str]
                else:
                    y_in += str_ratio[i_str]

        # Return the X and Y of the [X/Y] ratio
        return x_in, y_in


    ##############################################
    #                Plot Spectro                #
    ##############################################
    def plot_spectro(self,fig=-1, galaxy='', xaxis='[Fe/H]', yaxis='[Mg/Fe]', \
                   fsize=[10,4.5], fontsize=14, rspace=0.6, bspace=0.15,\
                   labelsize=15, legend_fontsize=14, ms=6.0, norm='', obs='',\
                   overplot=False, return_xy=False, show_err=False, \
                   show_mean_err=False, stat=False, flat=False,abundistr=False,show_legend=True, \
                   sub=1, sub_plot=False, alpha=1.0, lw=1.0):

        '''
        This function plots observational data with the spectroscopic notation:
        [X/Y] = log10(n_X/n_Y) - log10(n_X/n_Y)_solar where 'n_X' and 'n_Y' are
        the number densities of elements X and Y.

        Parameters
        ---------

        galaxy : string

            Name of the target galaxy.  The code then automatically selects
            the corresponding data sets (when available).

            Choices : 'milky_way', 'sculptor', 'carina', 'fornax'

            Default value : 'milky_way'

        xaxis : string

            Elements on the x axis in the form of '[X/Y]'.

            Default value : '[Fe/H]'

        yaxis : string

            Elements on the y axis in the form of '[X/Y]'.

            Default value : '[Mg/Fe]'

        norm : string

            Common solar normalization used to scale all the data.  Use the
            list_solar_norm() function for a list of available normalizations.
            When not specified, each data uses the solar normalization of the
            reference paper.

            Example : norm='Anders_Grevesse_1989'

        obs : string array

            Personal selection of observational data.  Use the list_ref_papers()
            function for a list of availble data sets.  When not specified, all
            the available data for the selected galaxy will be plotted.

            Example : obs=['milky_way_data/Venn_et_al_2004_stellab',
            'milky_way_data/Hinkel_et_al_2014_stellab']

        show_err : boolean

            If True, show error bars when available in the code.

            Default value : False

        show_mean_err : boolean

            If True, print the mean X and Y errors when error bars are available
            in the code.

            Default value : False

        return_xy : boolean

            If True, return the X and Y axis arrays instead of plotting the data.

            Default value = False

            Example : x, y = stellab.plot_spectro(return_xy=True)

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

        >>> stellab.plot_spectro(yaxis='[Ti/H]',xaxis='[Mg/H]',galaxy='sculptor',norm='Anders_Grevesse_1989',show_err=True)

        '''

        # Copy the marker size
        ms_copy = ms
        lw_copy = lw

        # Extract the X and Y of the input [X/Y]
        xx, yx = self.__get_x_y(xaxis)
        xy, yy = self.__get_x_y(yaxis)
        elem_in = ([xx,yx], [xy,yy])

        # Initialization of the variables used to calculate the average error
        sum_x = 0.0
        sum_y = 0.0
        sum_count = 0

        # Initialization of the variables used for statistical plot
        if stat:
            xy_plot_all = []
            xy_plot_all.append([])
            xy_plot_all.append([])

        # Show the frame of the plot
        if not overplot and not return_xy and not sub_plot:
            if fig>=0:
                plt.figure(fig,figsize=(fsize[0],fsize[1]))
            else:
                plt.figure(fig,figsize=(fsize[0],fsize[1]))

        # If data need to be re-normalized ...
        re_norm = False
        if len(norm) > 0:

            # Get the array index associated with the solar normalization
            i_re_norm = self.__get_i_re_norm(norm)

            # Look if the normalization reference is valid ...
            if not i_re_norm == -1:
                re_norm = True

            # Warning message if the reference is not valid
            else:
                print ('!! Warning - The solar normalization is not valid !!')

        # Return the list of index of the wanted data set
        # If a specif set of references is choosen ...
        if len(obs) > 0:

            # Get the indexes for the wanted references
            i_obs = self.__get_i_data_set(obs)

        # If a specific galaxy is choosen ...
        elif len(galaxy) > 0:

            # Get the indexes of the wanted galaxy
            i_obs = self.__get_i_data_galaxy(galaxy)

        # If the default mode is used ...
        else:

            # Use the Milky Way
            i_obs = self.__get_i_data_galaxy('milky_way')
        # Keep the number of indexes in memory
        len_i_obs = len(i_obs)

        # Prepare the returning array if the option is choosen
        if return_xy:
            ret_x = []
            ret_y = []
            ret_x_err = []
            ret_y_err = []
            #store index of stars of all data sets
            ret_star_i = []

        # For every data set ...
        for i_obs_index in range(0,len_i_obs):

          # Copy the data set index
          i_ds = i_obs[i_obs_index]

          #store temporary index of stars for current data set
          ret_star_i_tmp = []

          # If the current data set contains the input elements ...
          if elem_in[0][0] in self.elem_list[i_ds] and \
             elem_in[0][1] in self.elem_list[i_ds] and \
             elem_in[1][0] in self.elem_list[i_ds] and \
             elem_in[1][1] in self.elem_list[i_ds]:

            # Variable that tells if the ratio can be plotted
            ok_for_plot = True

            # Local variables
            xy_plot  = [[0.0 for ii  in range(len(self.ab[i_ds]))] \
                             for ijk in range(2)]
            err_plot = [[0.0 for ii  in range(len(self.ab[i_ds]))] \
                             for ijk in range(2)]

            # Common loop for the x-axis and the y-axis ...
            for i_x_y in range(0,2):

              # Find the corresponding entry in the data set for the numinator
              i_num, den_ok = self.__find_num(i_ds, i_x_y, elem_in)

              # If a perfect match was found ...
              if den_ok:

                  # Copy the current data set for the current input element ratio
                  for i_star in range(0,len(self.ab[i_ds])):
                      xy_plot[i_x_y][i_star]  = self.ab[i_ds][i_star][i_num][0]
                      err_plot[i_x_y][i_star] = self.ab[i_ds][i_star][i_num][1]
                      ret_star_i_tmp.append(i_star)
              # If the data need to be manipulated to recover the wanted ratio ...
              else:

                  # Find the corresponding entry in the data for the denominator
                  i_denom, div = self.__find_denom(i_ds, i_x_y, i_num, elem_in)

                  # Verify if the ratio can be plotted
                  if i_denom == -1:
                      ok_for_plot = False

                  else:

                      # If a division is needed to recover the wanted ratio ...
                      # Want [A/C] with nominator [A/B] and denominator [C/B]
                      if div:

                          # Calculate the ratio for each star in the data set ...
                          for i_star in range(0,len(self.ab[i_ds])):
                              if self.ab[i_ds][i_star][i_num][0] == -30.0 or \
                                 self.ab[i_ds][i_star][i_denom][0] == -30.0 :
                                  xy_plot[i_x_y][i_star] = -30.0
                                  ret_star_i_tmp.append(i_star)
                              else:
                                  xy_plot[i_x_y][i_star] = \
                                    self.ab[i_ds][i_star][i_num][0] - \
                                      self.ab[i_ds][i_star][i_denom][0]
                                  ret_star_i_tmp.append(i_star)

                      # If a multiplication is needed to recover the input ratio ..
                      # Want [A/C] with nominator [A/B] and denominator [B/C]
                      else:

                          # Calculate the ratio for each star in the data set ...
                          for i_star in range(0,len(self.ab[i_ds])):
                              xy_plot[i_x_y][i_star] = \
                                self.ab[i_ds][i_star][i_num][0] + \
                                  self.ab[i_ds][i_star][i_denom][0]
                              ret_star_i_tmp.append(i_star)

              # If a re-normalization is needed ...
              if re_norm and ok_for_plot:

                  # Calculate the wanted solar normalization
                  # eps = log(n_X/n_Y) + 12
                  eps_x = self.__get_eps(i_re_norm,elem_in[i_x_y][0],True)
                  eps_y = self.__get_eps(i_re_norm,elem_in[i_x_y][1],True)
                  sol_wanted = eps_x - eps_y

                  # Calculate the solar normalization used in data
                  eps_x = self.__get_eps(i_ds,elem_in[i_x_y][0],False)
                  eps_y = self.__get_eps(i_ds,elem_in[i_x_y][1],False)
                  sol_data = eps_x - eps_y

                  # If the solar values where available in the ref. paper ...
                  if not eps_x == -30.0 and not eps_y == -30.0:

                      # Indication of the success of re-normalization
                      sol_ab_found = True

                      # For every star ...
                      for i_star in range(0,len(self.ab[i_ds])):

                          # Correct the normalization
                          xy_plot[i_x_y][i_star] = \
                              xy_plot[i_x_y][i_star] + sol_data - sol_wanted

                  # Warning message if the re-normalization cannot be made
                  else:
                      sol_ab_found = False
                      if i_x_y == 0:
                          warn_ratio = xaxis
                      else:
                          warn_ratio = yaxis
                      if eps_x == -30.0 and eps_y == -30.0:
                          warn_el = elem_in[i_x_y][0]+' and '+elem_in[i_x_y][1]
                          print ('Solar values for '+warn_el+' not found in ' + \
                          self.leg[i_ds]+'.  '+warn_ratio+ ' was not modified.')
                      else:
                          if eps_x == -30.0:
                              warn_el = elem_in[i_x_y][0]
                          else:
                              warn_el = elem_in[i_x_y][1]
                          print ('Solar value for '+warn_el+' not found in ' + \
                          self.leg[i_ds]+'.  '+warn_ratio + ' was not modified.')
                      self.leg[i_ds]

            #####################################################

            # If ratio is available
            if ok_for_plot:

                # If a plot is generated
                if not return_xy:

                    # Reduce the size of APOGEE data
                    if self.paths[i_ds] == 'milky_way_data/APOGEE_stellab':
                        ms = 1
                        lw = 1
                    else:
                        ms = ms_copy
                        lw = lw_copy

                    if re_norm and not sol_ab_found:
                        leg_temp = self.leg[i_ds]+' **'
                    else:
                        leg_temp = self.leg[i_ds]
                    if stat:
                        xy_plot_all[0].append(xy_plot[0])
                        xy_plot_all[1].append(xy_plot[1])
                    else:
                      if show_err:
                        if show_legend:
                          if sub_plot:
                            sub.errorbar(xy_plot[0], xy_plot[1], \
                            xerr=err_plot[0], yerr=err_plot[1],
                            fmt=self.cs[i_ds][0], ecolor=self.cs[i_ds][1], \
                            color=self.cs[i_ds][1], label=leg_temp, markersize=ms, \
                            alpha=alpha, linewidth=lw)
                          else:
                            plt.errorbar(xy_plot[0], xy_plot[1], \
                            xerr=err_plot[0], yerr=err_plot[1],
                            fmt=self.cs[i_ds][0], ecolor=self.cs[i_ds][1], \
                            color=self.cs[i_ds][1], label=leg_temp, markersize=ms, \
                            alpha=alpha, linewidth=lw)
                        else:
                          if sub_plot:
                            sub.errorbar(xy_plot[0], xy_plot[1], \
                            xerr=err_plot[0], yerr=err_plot[1],
                            fmt=self.cs[i_ds][0], ecolor=self.cs[i_ds][1], \
                            color=self.cs[i_ds][1], markersize=ms, alpha=alpha, \
                            linewidth=lw)
                          else:
                            plt.errorbar(xy_plot[0], xy_plot[1], \
                            xerr=err_plot[0], yerr=err_plot[1],\
                            fmt=self.cs[i_ds][0], ecolor=self.cs[i_ds][1], \
                            color=self.cs[i_ds][1], markersize=ms, alpha=alpha, \
                            linewidth=lw)
                      else:
                        if show_legend:
                          if sub_plot:
                            sub.plot(xy_plot[0],xy_plot[1],self.cs[i_ds],\
                            label=leg_temp, markersize=ms, alpha=alpha, \
                            linewidth=lw)
                          else:
                            plt.plot(xy_plot[0],xy_plot[1],self.cs[i_ds],\
                            label=leg_temp, markersize=ms, alpha=alpha, \
                            linewidth=lw)
                        else:
                          if sub_plot:
                            sub.plot(xy_plot[0],xy_plot[1],self.cs[i_ds], \
                                  markersize=ms, alpha=alpha, linewidth=lw)
                          else:
                            plt.plot(xy_plot[0],xy_plot[1],self.cs[i_ds], \
                                  markersize=ms, alpha=alpha, linewidth=lw)
                      if show_legend:
                        if sub_plot:
                          sub.legend()
                        else:
                          plt.legend()

                # If the data need to be returned ...
                else:

                    # Add the data set to the returning arrays
                    for i_ret in range(0,len(xy_plot[0])):
                        if xy_plot[0][i_ret] < 5 and xy_plot[0][i_ret] > -5 and \
                           xy_plot[1][i_ret] < 5 and xy_plot[1][i_ret] > -5:
                            ret_x.append(xy_plot[0][i_ret])
                            ret_y.append(xy_plot[1][i_ret])
                            ret_x_err.append(err_plot[0][i_ret])
                            ret_y_err.append(err_plot[1][i_ret])
                            ret_star_i.append(ret_star_i_tmp[i_ret])

            # If the average error need to be calculated...
            if show_mean_err:

                # For every data point ...
                for i_dp in range(0,len(self.ab[i_ds])):

                    # Add the error to the sum
                    if not xy_plot[0][i_dp] == -30.0 and \
                       not xy_plot[1][i_dp] == -30.0 and \
                       not err_plot[0][i_dp] == 0.0 and \
                       not err_plot[1][i_dp] == 0.0:
                         sum_x += err_plot[0][i_dp]
                         sum_y += err_plot[1][i_dp]
                         sum_count += 1

        # Calculate the average error
        if show_mean_err and not return_xy:
            if sum_count > 0:
                print ('Mean',xaxis,'error =',sum_x/sum_count)
                print ('Mean',yaxis,'error =',sum_y/sum_count)

        # Provide a standard plot
        if not return_xy:

            # If plot median, 68%, 95%,
            if stat:

                # Build the x_bin
                x_bin = []
                y_bin = []
                dx_bin = 0.3
                x_max = 1.0
                xx = -5.0
                while xx <= x_max:
                    x_bin.append(xx)
                    y_bin.append([])
                    xx += dx_bin

                # Put the y data in the right bin
                for xi in range(0,len(xy_plot_all[0])):
                    for xxi in range(0,len(xy_plot_all[0][xi])):
                        for xb in range(len(x_bin)-1):
                            if x_bin[xb] <= xy_plot_all[0][xi][xxi] < x_bin[xb+1]:
                              if xy_plot_all[1][xi][xxi] > -10.0:
                                y_bin[xb].append(xy_plot_all[1][xi][xxi])

                y_stat = []
                for i in range(0,7):
                    y_stat.append([])
                for xb in range(len(x_bin)-1):
                    x_bin[xb] += 0.5 * dx_bin
                    y_temp = sorted(y_bin[xb])
                    if len(y_temp) > 0:
                        i_med = int(len(y_temp)/2)
                        y_stat[0].append(y_temp[0])
                        y_stat[1].append(y_temp[i_med - int(len(y_temp)*0.475)])
                        y_stat[2].append(y_temp[i_med - int(len(y_temp)*0.340)])
                        y_stat[3].append(y_temp[i_med])
                        y_stat[4].append(y_temp[i_med + int(len(y_temp)*0.340)])
                        y_stat[5].append(y_temp[i_med + int(len(y_temp)*0.475)])
                        y_stat[6].append(y_temp[-1])
                    else:
                        y_stat[0].append(-30.0)
                        y_stat[1].append(-30.0)
                        y_stat[2].append(-30.0)
                        y_stat[3].append(-30.0)
                        y_stat[4].append(-30.0)
                        y_stat[5].append(-30.0)
                        y_stat[6].append(-30.0)

                if flat:
                    for i in range(0,len(y_stat[0])):
                        y_stat[0][i] -= y_stat[3][i]
                        y_stat[1][i] -= y_stat[3][i]
                        y_stat[2][i] -= y_stat[3][i]
                        y_stat[4][i] -= y_stat[3][i]
                        y_stat[5][i] -= y_stat[3][i]
                        y_stat[6][i] -= y_stat[3][i]
                        y_stat[3][i] = 0.0

                del x_bin[-1]
                plt.fill_between(x_bin,y_stat[0],y_stat[-1],color='0.95')
                plt.plot(x_bin,y_stat[1],linestyle='-', color='w',linewidth=3)
                plt.plot(x_bin,y_stat[1],linestyle=':', color='k',linewidth=1.5)
                plt.plot(x_bin,y_stat[2],linestyle='-', color='w',linewidth=3)
                plt.plot(x_bin,y_stat[2],linestyle='--',color='k',linewidth=1)
                plt.plot(x_bin,y_stat[3],               color='w',linewidth=3)
                plt.plot(x_bin,y_stat[3],               color='k',linewidth=1.5)
                plt.plot(x_bin,y_stat[4],linestyle='-' ,color='w',linewidth=3)
                plt.plot(x_bin,y_stat[4],linestyle='--',color='k',linewidth=1)
                plt.plot(x_bin,y_stat[5],linestyle='-', color='w',linewidth=3)
                plt.plot(x_bin,y_stat[5],linestyle=':', color='k',linewidth=1.5)

            if not sub_plot:
                ax = plt.gca()
                plt.xlabel(xaxis)
                plt.ylabel(yaxis)
                matplotlib.rcParams.update({'font.size': 14})
                self.__fig_standard(ax=ax, fontsize=fontsize, labelsize=labelsize, \
                rspace=rspace, bspace=bspace, legend_fontsize=legend_fontsize)

        # Return the data if the option is choosen
        else:
            if show_mean_err and show_err:
              if sum_count > 0:
                return ret_x, ret_y, ret_x_err, ret_y_err, \
                       sum_x/sum_count, sum_y/sum_count
              else:
                return ret_x, ret_y, ret_x_err, ret_y_err, sum_count, sum_count
            if abundistr:
                return ret_x, ret_y, ret_x_err, ret_y_err,ret_star_i

            return ret_x, ret_y


    ##############################################
    #               Get i Data Set               #
    ##############################################
    def __get_i_data_set(self, obs):

        # Declaration of the list of index to be returned
        i_return = []

        # For every wanted data set ...
        for i_gids in range(0,len(obs)):

            # Initialization of the index
            i_search = -1

            # Look every available data set
            for i_look in range(0,len(self.paths)):

                # Add the index if the data set is found
                if self.paths[i_look] == obs[i_gids]:
                    i_search = i_look
                    i_return.append(i_look)

            # Warning message if the data set is not available
            if i_search == -1:
                print ('!! Warning - '+obs[i_gids]+' not available !!')

        # Return a bad index if the wanted normalization is not available
        return i_return


    ##############################################
    #              Get i Data Galaxy             #
    ##############################################
    def __get_i_data_galaxy(self, galaxy):

        # Declaration of the list of index to be returned
        i_return = []

        # Verify is the galaxy is available
        if not galaxy in self.galaxy_name:
            print ('!! Warning - '+galaxy+' not available !!')
        else:

            # Keep the number of characters in the name of the galaxy
            nb_char = len(galaxy)

            # Look every available data set
            for i_look in range(0,len(self.paths)):

                # Extract the nb_char first characters of the data set name
                ds_name = ''
                if len(self.paths[i_look]) > nb_char:
                    for i_extr in range(0,nb_char):
                        ds_name += self.paths[i_look][i_extr+13]

                # Add the index if it's for the right galaxy
                if ds_name == galaxy:
                    i_return.append(i_look)

        # Return a bad index if the wanted normalization is not available
        return i_return


    ##############################################
    #                Get i Re-Norm               #
    ##############################################
    def __get_i_re_norm(self, norm):

        # For every available standard solar normalization ...
        for i_irn in range(0,len(self.paths_norm)):

            # Return the index if the file is found
            if self.paths_norm[i_irn] == norm:
                return i_irn

        # Return a bad index if the wanted normalization is not available
        return -1


    ##############################################
    #                  Find Num                  #
    ##############################################
    def __find_num(self, i_ds, i_x_y, elem_in):

        # Arbitraty value for the index, just in case nothing is found
        i_num = -1

        # For each element ratio in the data set ...
        for i_elem in range(0,len(self.elem[i_ds])):

            # Copy the index if the right numinator is found
            if self.elem[i_ds][i_elem][0] == elem_in[i_x_y][0]:
                i_num = i_elem

                # Look for a perfect match
                if self.elem[i_ds][i_elem][1]==elem_in[i_x_y][1]:

                    # Return the entry index (the exact wanted ratio is found)
                    return i_num, True

        # Return the entry index (the exact wanted ratio is not found)
        return i_num, False


    ##############################################
    #                  Get Eps                   #
    ##############################################
    def __get_eps(self, i_eps, elem_eps, is_solar_ref):

        # Select the right array
        if is_solar_ref:
            temp_solar = self.sol_norm[i_eps]
        else:
            temp_solar = self.solar[i_eps]

        # Find the index associated with the wanted element
        for i_get_eps in range(0,len(temp_solar)):
            if temp_solar[i_get_eps][0] == elem_eps:
                return temp_solar[i_get_eps][1]

        # Print a warning message if the element is not found
        print ('Error - Element not found in __get_eps function.')
        return -30.0


    ##############################################
    #                 Find Denom                 #
    ##############################################
    def __find_denom(self, i_ds, i_x_y, i_num, elem_in):

        # For each element ratio in the data set ...
        for i_elem in range(0,len(self.elem[i_ds])):

            # If the denominator is found (in the division form) ...
            if self.elem[i_ds][i_elem][0] == elem_in[i_x_y][1] and \
               self.elem[i_ds][i_elem][1] == self.elem[i_ds][i_num][1]:

                # Return the entry index (a division is needed)
                return i_elem, True

            # If the denominator is found (in the multiplication form) ...
            elif self.elem[i_ds][i_elem][1] == elem_in[i_x_y][1] and \
               self.elem[i_ds][i_elem][0] == self.elem[i_ds][i_num][1]:

                # Return the entry index (a multiplication is needed)
                return i_elem, False

        # Return arbitrary value if nothing is found
        return -1, False


    ##############################################
    #                Fig Standard                #
    ##############################################
    def __fig_standard(self, ax, fontsize=8, labelsize=8, lwtickboth=[6,2], \
                       lwtickmajor=[10,3], rspace=0.6, bspace=0.15, \
                       legend_fontsize=14):

        '''
        Internal function in order to get standardized figure font sizes.
        It is used in the plotting functions.

        '''

        plt.legend(loc=2,prop={'size':legend_fontsize})
        plt.rcParams.update({'font.size': fontsize})
        ax.yaxis.label.set_size(labelsize)
        ax.xaxis.label.set_size(labelsize)
        ax.tick_params(length=lwtickboth[0],width=lwtickboth[1],which='both')
        ax.tick_params(length=lwtickmajor[0],width=lwtickmajor[1],which='major')
#        if len(ax.lines)>0:
#            for h in range(len(ax.lines)):
#                ax.lines[h].set_markersize(markersize)
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), \
             markerscale=0.8,fontsize=legend_fontsize)
        plt.subplots_adjust(right=rspace)
        plt.subplots_adjust(bottom=bspace)


    ##############################################
    #              List Solar Norm               #
    ##############################################
    def list_solar_norm(self):

        '''
        This function plots the list of solar normalizations available.

        '''

        # Print every available solar abundances
        for i_lsn in range(0,len(self.paths_norm)):
            print (self.paths_norm[i_lsn])


    ##############################################
    #              List Ref Papers               #
    ##############################################
    def list_ref_papers(self,galaxy=''):

        '''
        This function prints lists of available data sets.

        Parameters
        ---------

        galaxy : string

            Name of the target galaxy for which data sets will be displayed.
            If empty includes data sets of all available galaxies.

            Choices : 'milky_way', 'sculptor', 'carina', 'fornax'

            Default value : ''

        '''

        if len(galaxy)==0:
           # Print every available observational reference papers
           for i_lsn in range(0,len(self.paths)):
              print (self.paths[i_lsn])
        else:
           i_obs = self.__get_i_data_galaxy(galaxy)
           for i_ref in i_obs:
               print (self.paths[i_ref])

    ##############################################
    #                 Get star id                #
    ##############################################

    def get_star_id(self,find_elements=['Sc'],find_Fe_H_range=[],obs='milky_way_data/Venn_et_al_2004_stellab'):

        '''
        Gets index of stars with certain properties such as available elements or Fe/H range.

        Parameters
        ---------

        find_elements : array
            Find index of all stars with elements
        obs : string
            Star data set
        '''

        overplot=False

        # get the elements available

        # Get the indexes for the wanted references
        i_obs = self.__get_i_data_set([obs])
        i_ds=i_obs[0]
        elements = self.elem_list[i_ds]
        print ('Number of elements available in dataset: ',elements)
        # get [X/Fe] for each element
        abunds_y=[]
        abunds_y_err=[]
        eles_found=[]
        num_stars=0
        star_ids=[]
        star_fe_h=[]
        for k in range(len(find_elements)):
            yaxis='['+find_elements[k]+'/'+'Fe]'
            ret_x, ret_y, ret_x_err, ret_y_err,ret_star_i=self.plot_spectro(fig=-1, galaxy='', xaxis='[Fe/H]', yaxis=yaxis, \
                   fsize=[10,4.5], fontsize=14, rspace=0.6, bspace=0.15,\
                   labelsize=15, legend_fontsize=14, ms=6.0, norm='', obs=[obs],\
                   overplot=False, return_xy=True, show_err=True, \
                   show_mean_err=False, stat=False, flat=False, show_legend=True, \
                   sub=1, sub_plot=False, alpha=1.0, lw=1.0,abundistr=True)


            if k==0:
                star_ids=ret_star_i
                star_fe_h=ret_x
            else:
                star_ids_tmp=[]
                star_fe_h_tmp=[]
                for h in range(len(star_ids)):
                    if star_ids[h] in ret_star_i:
                        star_ids_tmp.append(star_ids[h])
                        star_fe_h_tmp.append(star_fe_h[h])
                star_ids=star_ids_tmp
                star_fe_h=star_fe_h_tmp

        # get [Fe/H] values
        if len(find_Fe_H_range)>0:
            fe_h_max=find_Fe_H_range[1]
            if fe_h_max==0:
                fe_h_max=max(star_fe_h)
            fe_h_min=find_Fe_H_range[0]
            if fe_h_min==0:
                fe_h_min=min(star_fe_h)
            star_ids_tmp=[]
            print ('adjust to given metallicity range from ',fe_h_min,' to ',fe_h_max)
            for k in range(len(star_ids)):
                if (star_fe_h[k]>fe_h_min) and (star_fe_h[k]<fe_h_max):
                    star_ids_tmp.append(star_ids[k])
            star_ids=star_ids_tmp
        return star_ids


    ##############################################
    #                 Plot Abun                  #
    ##############################################

    def plot_abun(self,fig=-1,obs='milky_way_data/Venn_et_al_2004_stellab',\
                   star_idx=0,find_elements=[],fsize=[10,4.5],label='Star 1',elem_label_on=True,\
                   fontsize=14,shape='-',color='k', rspace=0.6, bspace=0.15,\
                   labelsize=15, legend_fontsize=14, ms=6.0, norm='',\
                   marker='o',markersize=3,return_xy=False,show_err=True, \
                   show_mean_err=False, stat=False, flat=False, show_legend=True, \
                   sub=1, sub_plot=False, alpha=1.0, lw=1.0,iolevel=0):


        '''
        Plots abundance distribution [Elements/Fe] vs Z

        Parameters
        ---------

        elements : array
        find_elements : array
            Plot only
        '''

        overplot=False

        # get the elements available

        # Get the indexes for the wanted references
        i_obs = self.__get_i_data_set([obs])
        i_ds=i_obs[0]
        elements = self.elem_list[i_ds]
        if iolevel>0:
            print ('Number of elements available in dataset: ',elements)
        # get [X/Fe] for each element
        abunds_y=[]
        abunds_y_err=[]
        eles_found=[]
        num_stars=0
        for k in range(len(elements)):
            yaxis='['+elements[k]+'/'+'Fe]'
            ret_x, ret_y, ret_x_err, ret_y_err,ret_star_i=self.plot_spectro(fig=-1, galaxy='', xaxis='[Fe/H]', yaxis=yaxis, \
                   fsize=[10,4.5], fontsize=14, rspace=0.6, bspace=0.15,\
                   labelsize=15, legend_fontsize=14, ms=6.0, norm='', obs=[obs],\
                   overplot=False, return_xy=True, show_err=True, \
                   show_mean_err=False, stat=False, flat=False, show_legend=True, \
                   sub=1, sub_plot=False, alpha=1.0, lw=1.0,abundistr=True)

            if star_idx in ret_star_i:
               idx=ret_star_i.index(star_idx)
               #ret_x[idx]
               abunds_y.append(ret_y[idx])
               #ret_x_err[idx]
               abunds_y_err.append(ret_y_err[idx])
               eles_found.append(elements[k])
               #print ('get value: ',ret_y[idx])
            if len(ret_star_i)>num_stars:
               num_stars=len(ret_star_i)
        if iolevel>0:
            print ('Number of stars available in dataset: ',num_stars)
        err_on = show_err
        self.__plot_distr(fig,eles_found,abunds_y,abunds_y_err,err_on,elem_label_on,shape,color,label,marker,markersize,fsize)


    def __plot_distr(self,fig, elements,abunds,abunds_err,err_on,label_on,shape,color,label,marker,markersize,fsize):

        '''
        Helping function to plot the abundance distribution [Elements/Fe] vs Z

        Parameters
        ---------

        elements : array
            list of elements to be plotted
        abunds : array
            abundances as [X/Fe]
        label_on : boolean
            if true plots the element label for each element

        '''
        plt.figure(fig,figsize=(fsize[0],fsize[1]))

        Z_numbers=[]
        for i_ele in range(len(elements)):
          # get name of element
          Z=ry.get_z_from_el(elements[i_ele])
          Z_numbers.append(Z)

        Z_numbers_sort=[]
        abunds_sort=[]
        idx_sorted=sorted(range(len(Z_numbers)),key=lambda x:Z_numbers[x])
        for k in range(len(idx_sorted)):
           Z_numbers_sort.append(Z_numbers[idx_sorted[k]])
           abunds_sort.append(abunds[idx_sorted[k]])

        plt.ylabel('[X/Fe]')
        plt.xlabel('charge number Z')
        plt.plot(Z_numbers_sort, abunds_sort, linestyle=shape,\
                         color=color,label=label,marker=marker,markersize=markersize)

        ## plot element labels
        if label_on:
            for i_ele in range(len(elements)):
                plt.annotate(elements[i_ele],(Z_numbers[i_ele],abunds[i_ele]), \
                    xytext=(-2,0),textcoords='offset points',horizontalalignment='right', verticalalignment='top')


        #err on
        if err_on:
            for i_ele in range(len(elements)):
                plt.errorbar(Z_numbers[i_ele],abunds[i_ele], \
                    xerr=0, yerr=abunds_err[i_ele],
                    color=color,marker=marker, ecolor=color, \
                    label=label, markersize=markersize, alpha=1)


    def write_data_table(self,data,attr,filename):

        '''
        Write stellar data table for the stellab database. Format as for
        files found in stellab_data and read in by stellab.

        Parameters
        ---------

        data : list
            List of stellar data such as abundances. Each list element
            contains list entries for a star.

        attr : list
            List of types of stellar data such as [Fe/H] for each star.
        filename : string
            Name of data file to be written
        '''
        out=''

        #create header with types
        for k in range(len(attr)):
            sp = len(attr[k])
            out = out + attr[k]+(10-sp)*' '
        out = out + '\n'

        #loop over stars
        for k in range(len(data)):
            for h in range(len(data[k])):
                out = out + '{:.3E}'.format(data[k][h]) + ' '
            out = out + '\n'

        f1=open(filename,'w')
        f1.write(out)
        f1.close()
        print ('file ',filename,' created.')


