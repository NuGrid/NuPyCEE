# TODO make sure this can run in that directory ..

# Import python packages
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import copy
import glob

# Import various NuPyCEE codes
from NuPyCEE import sygma
from NuPyCEE import chem_evol
from NuPyCEE import read_yields
from NuPyCEE import yields_combiner

# Reload codes
import imp
imp.reload(read_yields)
imp.reload(yields_combiner)
yc = yields_combiner.yields_combiner()

# Launch a SYGMA run to access NuGrid's list of isotopes
ss = sygma.sygma(iniZ=0.02)

# Create an instance of the yields combiner code
yc = yields_combiner.yields_combiner()

# Function to get the yields directly from the table
def get_raw_yields(table):
    
    # Declare the yields dictionary
    yields = dict()
    i_yields = None
    iso_list = []
    
    # For each line in the file ..
    with open(table) as f:
        for line in f.readlines():
            
            # Start a new stellar model if needed ..
            if "Table:" in line:
                model = get_model_label(line)
                yields[model] = dict()
                yields[model]["Yields"] = dict()
                yields[model]["X0"] = dict()
                
            # Get the array order for Yields and X0
            if i_yields == None:
                if "&Isotopes" in line:
                    i_yields, i_X0 = get_col_index(line)
                
            # Read isotopes
            if line[0] == "&" and "&Isotopes" not in line:
                split = line.split()
                iso = split[0][1:]
                yields[model]["Yields"][iso] = float(split[i_yields][1:])
                yields[model]["X0"][iso] = float(split[i_X0][1:])
                if iso not in iso_list:
                    iso_list.append(iso)

    # Return the raw yields
    return yields, iso_list

# Function to return the model label (M=?,Z=?)
def get_model_label(line):
    no_parenthesis = line.split("(")[1].split(")")[0]
    return "("+no_parenthesis+")"

# Function to extract where are the yields and X0 values
def get_col_index(line):
    split = line.split()
    return split.index("&Yields"), split.index("&X0")

# Function to combine the list of isotopes between yields
def combine_iso_lists(iso_lists):
    iso_combine = []
    for iso_list in iso_lists:
        for iso in iso_list:
            if iso not in iso_combine:
                iso_combine.append(iso)
    return iso_combine

# Load paths for AGB and massive star yields tables
y_path_root = "./NuPyCEE/yield_tables/"
agb_path_list = glob.glob("%sagb_stars*" %y_path_root)
massive_path_list = glob.glob("%smassive_stars*" %y_path_root)

# For each combination of AGB and massive stars ..
for agb_path in agb_path_list:
    for massive_path in massive_path_list:
        
        # Print progress
        print("Checking",agb_path.split("agb_stars_")[1].split(".txt")[0],\
              "+", massive_path.split("massive_stars_")[1].split(".txt")[0])
        is_good = True
        
        # Extract the yields directly from the text file
        agb_yields, agb_iso_list = get_raw_yields(agb_path)
        massive_yields, massive_iso_list = get_raw_yields(massive_path)
        iso_list = combine_iso_lists([agb_iso_list,massive_iso_list])
        
        # Get the yields combiner object
        path_list = [agb_path.split("NuPyCEE/")[1],
                     massive_path.split("NuPyCEE/")[1]]
        ytable_comb = yc.combine_tables(path_list, iso_list)
    
        # For individual yields table ..
        yields_list = [agb_yields, massive_yields]
        raw_iso_list = [agb_iso_list, massive_iso_list]
        for i_y in range(len(yields_list)):
            raw_yields = yields_list[i_y]
            raw_iso = raw_iso_list[i_y]
                         
            # For eac model in that table ..
            for model in raw_yields.keys():
                
                # Extract M and Z
                M = float(model.split("M=")[1].split(",")[0])
                Z = float(model.split("Z=")[1].split(")")[0])
                
                # Copy the X0 values if available
                if ytable_comb.net_yields_available:
                    y_comb_full = ytable_comb.get(M=M, Z=Z, quantity="X0")
                
                # For each isotope in that table ..
                for iso in raw_iso:
                
                    # Print error is combined yields differ from text file
                    y_text_file = raw_yields[model]["Yields"][iso]
                    y_comb = ytable_comb.get(M=M, Z=Z, quantity=iso)
                    if not y_text_file == y_comb:
                        print("  Problem in yields")
                        is_good = False
                        
                    # Print error is combined X0 differ from text file
                    if ytable_comb.net_yields_available:
                        i_iso = ytable_comb.isotopes.index(iso)
                        y_text_file = raw_yields[model]["X0"][iso]
                        y_comb = y_comb_full[i_iso]
                        if not y_text_file == y_comb:
                            print("  Problem in X0")
                            is_good = False
                        
        # Print success
        if is_good:
            print("  All good.")
