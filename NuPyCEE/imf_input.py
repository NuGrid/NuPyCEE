
#File to define a custom IMF
#Define your IMF in custom_imf
#so that the return value represents
#the chosen IMF value for the input mass

def custom_imf(mass):
	
	#Salpeter IMF
	#return mass**-1.5
	return mass**-2.35

