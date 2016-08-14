import matplotlib
matplotlib.use('agg')
import unittest

class TestModuleImports(unittest.TestCase):
     '''
	Import tests.
     '''
     def test_import_sygma(self):
          import SYGMA   
	
     def test_import_omega(self):
          import OMEGA

     def test_import_stellab(self):
          import STELLAB


class TestDefaults(unittest.TestCase):
     '''
     Test simulations with default variables.    
     '''

     def run_sygma(self):
          import sygma as s
	  s1 = s.sygma()
     def run_omega(self):
	  import omega as o
	  o1 = o.omega() 
     def run_stellab(self):
	  import stellab as st
	  st1 = st.stellab()


