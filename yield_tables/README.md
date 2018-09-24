# Stellar Yields Library


### Available Yields Tables to Run with NuPyCEE

- - - - - 
**agb_and_massive_stars_nugrid_MESAonly_xxxx.txt**

- AGB and massive: NuGrid Collaboration, Ritter et al. (2018, R18), <http://adsabs.harvard.edu/abs/2018MNRAS.480..538R>

|   Z points | 0.0001 | 0.001 | 0.006 | 0.01 | 0.02 |
|-----------:|-------:|------:|------:|-----:|-----:|
| R18        | 0.0001 | 0.001 | 0.006 | 0.01 | 0.02 |
| R18        | 0.0001 | 0.001 | 0.006 | 0.01 | 0.02 |

**Note** 

- fryer12delay and fryer12rapid at the end of the file name refers to the remnant mass prescription use for the core-collapse explosion of massive stars <http://adsabs.harvard.edu/abs/2012ApJ...749...91F>
- fryer12mix at the end of the file name represents a mixture of 50% delay and 50% rapid.
	
- - - - - 

**agb_and_massive_stars_K10_K06.txt**

- AGB: Karakas (2010, K10), <http://adsabs.harvard.edu/abs/2010MNRAS.403.1413K>
- Massive: Kobayashi et al. (2006, K06), <http://adsabs.harvard.edu/abs/2006ApJ...653.1145K>

|   Z points | 0.0001 | 0.001 | 0.004 | 0.008 | 0.02 |
|-----------:|-------:|------:|------:|------:|-----:|
| K10        | 0.0001 | `interp` | 0.004 | 0.008 | 0.02 |
| K06        | 0.001  |  0.001 | 0.004 |  `interp`  | 0.02 |

**Note** 

- We excluded the 6.5 Msun model at Z = 0.02 in K10, since this model was not available for the other metallicities.
- X.XHNe at the end of the file name refers to the fraction of all massive stars from 20 to 40 Msun that explode as hypernovae (HNe).
	- 0.0 --> 0% 
	- 0.5 --> 50% 
	- 1.0 --> 100%

- - - - - - 
**agb_and_massive_stars_K10_LC18.txt**

- AGB: Karakas (2010, K10), <http://adsabs.harvard.edu/abs/2010MNRAS.403.1413K>
- Massive: Limongi & Chieffi (2018, LC18), <http://adsabs.harvard.edu/abs/2018arXiv180509640L>
	- taken from <http://orfeo.iaps.inaf.it/>	 

|   Z points |3.236e-5|1.0e-4  |3.236e-4|3.236e-3| 0.004  | 0.008  |1.345e-2|  0.02  |
|-----------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| K10        |1.0e-4  |1.0e-4  |`interp`|`interp`| 0.004  | 0.008  |`interp`|  0.02  |
| LC18         |3.236e-5|`interp`|3.236e-4|3.236e-3|`interp`|`interp`|1.345e-2|1.345e-2|

**Note** 

- We excluded the 6.5 Msun model at Z = 0.02 in K10, since this model was not available for the other metallicities.
- Xyyy at the end of the file name refers to the Set X with rotation velocity of yyy km/s.
	- e.g., abg_and_massive_stars_K10_LCprep_F150.txt means Set F with V = 150 km/s.
	- yyy = avg represents a mixture of rotation velocities as seen in Figure 4 of Prantzos et al. (2018), <http://adsabs.harvard.edu/doi/10.1093/mnras/sty316>

- - - - - - 
**agb_and_massive_stars_portinari98_marigo01.txt**

- AGB: Marigo (2001, M01), <http://adsabs.harvard.edu/abs/2001A%26A...370..194M>
- Massive: Portinari et al. (1998, P98), <http://adsabs.harvard.edu/abs/1998A%26A...334..505P>

|   Z points |0.004  | 0.008  |   0.02 |
|-----------:|-------:|-------:|-------:|
| M01        |0.004  | 0.008  | 0.02 |
| P98         |0.004  | 0.008  | 0.02 |

**Note** 

- The net yields of M01 has been converted into total yields using the initial compositions used to calculate NuGrid's stellar model.
- More metallicities are available in the original P98 paper.

- - - - - - 
**agb_and_massive_stars_portinari98_marigo01_net_yields.txt**

- AGB: Marigo (2001, M01), <http://adsabs.harvard.edu/abs/2001A%26A...370..194M>
- Massive: Portinari et al. (1998, P98), <http://adsabs.harvard.edu/abs/1998A%26A...334..505P>

|   Z points |0.004  | 0.008  |   0.02 |
|-----------:|-------:|-------:|-------:|
| M01        |0.004  | 0.008  | 0.02 |
| P98         |0.004  | 0.008  | 0.02 |

**Note** 

- Same as *agb_and_massive_stars_portinari98_marigo01.txt*, but in net yields form.
- The total yields of P98 has been converted into net yields using the initial compositions used to calculate NuGrid's stellar model.

- - - - - - 


**Old Content. This will be soon updated.**

Contains yield tables and tables with stellar feedback.
The iniabu directory contains the initial abundances to be
used with the iniabu_table code variable.
The others directory contains temporary files not relevant
for the user. 


#### Yield table formats

There are two formats for yield tables. Tables with
agb_and_massive_****.txt contain the AGB and massive star yields
and have a specific format to be used with the 'table' code parameter.
The second format is used for all other yield tables such as
SNIa tables with sn1a_****.txt to be used with different code parameter.


#### File naming scheme

agb_and_massive_stars_*** : Containing AGB and massive star yields to be used with 'table' code parameter.
iniab*** : Initial abundance files (in iniabu). iniabu_bb*** contains BigBang abundances.
sn1a*** : Files for SNIa yields to be used with 'sn1a_table' code parameter.
pop3_table** : PoP III abundances (Z=0) to be used with the 'pop3_table' code parameter.
stellar_feedback_** : Contains stellar parameter derived from stellar models. Used with 'stellar_param_table' code variable.

If the table does not match in above categories other names can be chosen such as  
mhdjet_NTT_delayed.txt, ndw_wind_expand.001.txt, r_process_arnould_2007.txt
The three tables can be used with the 'extra_source_table' code variable.

#### Name scheme for yield table containing yields AGB stars and massive stars (with agb_and_massive_stars_*****.txt)

agb_and_massive_stars.txt : default table file; should be MESA; agb_and_massive_stars_nugrid_MESAonly_fryer12delay.txt

fryer12 : with Fryer 2012 mass cut prescription
yemcut  : mass cut at ye jump; classical approach

MESA_only : tables only with MESA models

#### For Wiersma09 comparison, combined yields from Portinari 1998 and Marigo 2001:

agb_and_massive_stars_portinari98_marigo01.txt : C,Mg,Fe were modified according to W09.

agb_and_massive_stars_portinari98_marigo01_nomod.txt : same as 
                agb_and_massive_stars_portinari98_marigo01.txt, but without C,Mg,Fe modification
		(also with all original initial masses)

agb_and_massive_stars_portinari98_marigo01_nomod_gce_standard.txt : chosen to match NuGrid's initial masses
								 e.g. 1.672Msun = 1.65Msun

agb_and_massive_stars_portinari98_marigo01_nomod_gce_addZ: contains additional metallicity
	Z=0.0004 compared to agb_and_massive_stars_portinari98_marigo01_nomod_gce_standard.txt



#### Notes and naming scheme for the SYGMA Widget in WENDI:

1) analytic prescription (Fryer12)
    a) delay  (agb_and_massive_stars_nugrid_MESAonly_fryer12delay.txt)
    b) rapid  (agb_and_massive_stars_nugrid_MESAonly_fryer12rapid.txt)
    c) mix    (agb_and_massive_stars_nugrid_MESAonly_fryer12mix.txt)
	Mix of 50% - 50% mix of delay and rapid yields
2) Ye=0.4982 (Young06)
    a) fallback at Ye (agb_and_massive_stars_nugrid_MESAonly_ye.txt)




#### for testing :

agb_and_massive_stars_cnoni.txt
agb_and_massive_stars_h1.txt
iniab_h1.ppn
iniab_cnoni.ppn


The SN1a abundances of Seitenzahl13 are either
provided divided in stable and unstable ones
or in a mix of both types.



#### Tables containing stellar feedback (with stellar_feedback_*****)

Stellar feedback can be followed by setting stellar_param_on=True
and defining the table containing data via the variable  stellar_param_table variable.

stellar_feedback_nugrid_MESAonly.txt: Contains the stellar feedback derived from NuGrid MESA models.
 


