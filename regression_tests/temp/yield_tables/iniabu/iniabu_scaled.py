f1=open('iniab1.0E-05GN93_alpha.ppn')
lines=f1.readlines()
f1.close()

write_out=''
out=''
for k in range(len(lines)):
	line=lines[k].split()
	iso=line[1]
	a=line[2]
	for iso1 in ['he','c', 'o', 'mg', 'ca', 'ti', 'fe', 'co','zn','PROT','h','n']:
		if iso == iso1:
			out+= ('"'+iso.capitalize()+'-'+a+'",')
			write_out+=lines[k]
print out
print write_out
f2=open('iniab1.0E-05GN93_alpha_scaled.ppn','w')
f2.write(write_out)
f2.close()
		
