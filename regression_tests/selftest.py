import matplotlib
matplotlib.use('agg')
import unittest

class TestModuleImports(unittest.TestCase):
     '''
	Import tests.
     '''
     def test_import_sygma(self):
          from NuPyCEE import sygma
	
     def test_import_omega(self):
          from NuPyCEE import omega

     def test_import_stellab(self):
          from NuPyCEE import stellab


class TestDefaults(unittest.TestCase):
     '''
     Test simulations with default variables.    
     '''

     def run_sygma(self):
          from NuPyCEE import sygma as s
          s1 = s.sygma()
     def run_omega(self):
          from NuPyCEE import omega as o
          o1 = o.omega() 
     def run_stellab(self):
          from NuPyCEE import stellab as st
          st1 = st.stellab()


