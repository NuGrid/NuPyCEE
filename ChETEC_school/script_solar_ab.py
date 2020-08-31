import numpy as np

# Function to retrun the list of isotope index for an input element
def get_list_iso_index(specie, inst):
    specie_list = []
    for i_gl in range(0,len(inst.history.isotopes)):
        if (specie+'-') in inst.history.isotopes[i_gl]:
            specie_list.append(i_gl)
    return specie_list

# List of charge numbers (NEED to be as in the yields table)
Z_charge = []
for i in range(1,84):
    # Exclude elements not considered in NuGrid yields
    if (not i == 43) and (not i == 61):
        Z_charge.append(i)
        
# List of elements (NuGrid yields)
elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', \
            'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', \
            'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', \
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', \
            'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', \
            'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', \
            'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']
nb_elements = len(elements)

# List of atomic weigth to plot the mass fractions
atom_weigth = [1.0079, 4.0026, 6.941, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180, \
               22.990, 24.305, 26.982, 28.086, 30.974, 32.065, 35.453, 39.453, 39.098, \
               40.078, 44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, \
               63.546, 65.390, 69.723, 72.610, 74.922, 78.96, 79.904, 83.80, 85.468, \
               87.62, 88.906, 91.224, 92.906, 95.94, 98.0, 101.07, 102.91, 106.42, 107.87, \
               112.41, 114.82, 118.71, 121.76, 127.60, 126.90, 131.29, 132.91, 137.33, \
               138.91, 140.12, 140.91, 144.24, 145.0, 150.36, 151.96, 157.25, 158.93, 162.50, \
               164.93, 167.26, 168.93, 173.04, 174.97, 178.49, 180.95, 183.84, 186.21, \
               190.23, 192.22, 195.08, 196.97, 200.59, 204.38, 207.2, 208.98]

# Read solar abundances (number densities)
solar_Z = []
solar_ab = []
solar_ab_path = 'Lodders_et_al_2009.txt'
with open("Lodders_et_al_2009.txt", 'r') as f:
    not_finished = True
    for line in f:
        split_line = [str(x) for x in line.split()]
        if not_finished:
            solar_Z.append(int(split_line[0]))
            solar_ab.append(10**(float(split_line[2])-12))
            if split_line[1] == 'Bi':
                not_finished = False
f.close()

# Convert number of atoms into masses
for i in range(len(solar_ab)):
    solar_ab[i] *= atom_weigth[i]
    
# Normalize to 1.0
norm = 1.0/sum(solar_ab)
for i in range(len(solar_ab)):
    solar_ab[i] *= norm

# Function to extract the contribution of individual sources
def get_individual_sources(inst, i_step_sol):
    
    # Declare the abundances arrays
    m_el_all = np.zeros(nb_elements)
    m_el_agb = np.zeros(nb_elements)
    m_el_massive = np.zeros(nb_elements)
    m_el_sn1a = np.zeros(nb_elements)
    m_el_nsm = np.zeros(nb_elements)
    
    # Get the mass distrubution of individual sources
    for i_el in range(0,nb_elements):
        specie_list = get_list_iso_index(elements[i_el], inst)
        for i_iso in range(0,len(specie_list)):
            m_el_all[i_el] += inst.ymgal[i_step_sol][specie_list[i_iso]]
            m_el_agb[i_el] += inst.ymgal_agb[i_step_sol][specie_list[i_iso]]
            m_el_massive[i_el] += inst.ymgal_massive[i_step_sol][specie_list[i_iso]]
            m_el_sn1a[i_el] += inst.ymgal_1a[i_step_sol][specie_list[i_iso]]
            m_el_nsm[i_el] += inst.ymgal_nsm[i_step_sol][specie_list[i_iso]]
    
    # Normalize each sources
    norm_all_for_all = 1.0 / sum(m_el_all)
    for i_el in range(0,nb_elements):
        m_el_all[i_el] *= norm_all_for_all
        m_el_agb[i_el] *= norm_all_for_all
        m_el_massive[i_el] *= norm_all_for_all
        m_el_sn1a[i_el] *= norm_all_for_all
        m_el_nsm[i_el] *= norm_all_for_all
    
    # Return abundances patterns
    return m_el_all, m_el_agb, m_el_massive, m_el_sn1a, m_el_nsm

# Get the position of the element labels
yy = np.zeros(nb_elements)
for i in range(0,nb_elements):
    yy[i] = 10**(np.log10(solar_ab[i])+1.0)


# Milky Way OMEGA+ parameters
#kwargs = {"special_timesteps":150, "t_star":1.0, "mgal":1.0,
#          "m_DM_0":1.0e12, "sfe":3.0e-10, "mass_loading":0.5,
#          "imf_yields_range":[1,30],
#          "table":'yield_tables/agb_and_massive_stars_K10_K06_0.5HNe.txt',
#          "exp_infall":[[100/2.2, 0.0, 0.68e9], [13.0/2.2, 1.0e9, 7.0e9]],
#          "nsmerger_table":'yield_tables/r_process_arnould_2007.txt',
#          "ns_merger_on":True, "nsm_dtd_power":[1e7, 10e9, -1]}

# Common parameters to all yields table
kwargs = dict()
kwargs["special_timesteps"] = 150
kwargs["t_star"] = 1.0
kwargs["mgal"] = 1.0
kwargs["m_DM_0"] = 1.0e12
kwargs["nsmerger_table"] = 'yield_tables/r_process_arnould_2007.txt'
kwargs["ns_merger_on"] = True
kwargs["nsm_dtd_power"] = [1e7, 10e9, -1]
kwargs["m_ej_nsm"] = 2.5e-2

# C15 LC18
kwargs["exp_infall"] = [[50.0, 0.0, 0.68e9], [7.0, 1.0e9, 7.0e9]]
kwargs["imf_yields_range"] = [1.0, 100.0]
kwargs["sfe"] = 2.2e-10
kwargs["mass_loading"] = 0.9
kwargs["transitionmass"] = 8.0
kwargs["Z_trans"] = -1
kwargs["table"]= "yield_tables/agb_and_massive_stars_C15_LC18_R_mix.txt"


# Timestep index where the Sun should aproximately form.
# The index is only valid with "special_timesteps=120".
# Do not modify.
i_t_Sun = 143


