!**********************************************
!*  Declarataion of the Communication Module  *
!**********************************************
module comm
    parameter (max_entry=10000)
    integer i_entry(0:max_entry)
    integer i_entry_max
end module comm

!*******************************************
!*     Declarataion of the Path module     *
!*******************************************
module path_module
    character*256 files_path
    character*10  re_import_test
end module path_module


!*****************************************
!*     Declarataion of the ISO Module    *
!*****************************************
module iso
   implicit none
   character*256 init_file, input_file_network, input_file_abundance, output_file_abundance
   integer max_number_isotopes, number_isotopes, current_isotope, abundance_file_format
   character*20  element_names(0:200)
   integer number_reaction_types, max_number_reactions, max_number_fissile_isotopes
   double precision max_hwz ! maximum half life - above that, isotope will be considered stable
   parameter (number_reaction_types=22)
   parameter (max_number_isotopes=10000)  ! maximum number of species
   parameter (max_number_reactions=100)  ! maximum number of reactions per species
   parameter (max_number_fissile_isotopes=200) ! maximum number of species which undergo fission
   parameter (max_hwz=1e20)  ! maximum half life - above that, isotope will be considered stable - ~3000 Gyr
   integer z(max_number_isotopes), n(max_number_isotopes)  ! Number of Protons and Neutrons of given species
   integer reactions(max_number_isotopes,-1:max_number_reactions)  ! 0: Number of reactions 1..n reaction type of given species, -1 index in fission vector
   double precision decay_constant(max_number_isotopes,0:max_number_reactions)  ! decay rates of reactions of given species in 1/s  0 ... total, 1..n partial (later normalized to 1)
   double precision decay(max_number_isotopes)  ! sum of decays per time step of given species
   double precision production(max_number_isotopes)  ! sum of production per time step of given species
   double precision level(max_number_isotopes)  ! level in keV above ground state of given species (allows isomers...)
   double precision spin(max_number_isotopes)  ! spin of given species (needed in case of fission)
   double precision level_product(max_number_isotopes,max_number_reactions)  ! level in keV above ground state of product species (allows isomers...)
   integer product_isomer(max_number_isotopes,max_number_reactions)  ! species produced by reaction on (isomer,reaction)
   double precision abundance(max_number_isotopes)  ! abundance of given species 
   double precision initial_abundance(max_number_isotopes)  ! abundance of given species 
   character*3 reaction_types(number_reaction_types)
   integer reaction_vector(2,number_reaction_types) 
!  gives change in Z,N for desired reaction - EC is (-1,1), B- is (+1,-1) etc.
   data reaction_types  /'B-',   'EC',   'N',     'P',    'A',   'BN' , 'EP',   &          ! 7
                       'BA',   'EA',                                          &          ! 9
                       '2N',   '2P',   '2A',    'B2A',                        &          ! 13
                       'B2N',  'B3N',  'B4N', 'E2P', 'BNA',   'EPA', 'IT' , '12C',   &   ! 21
                       'SF'/  ! 22
   data reaction_vector /1,-1,   -1,1,   0,-1,   -1,0,   -2,-2,  1,-2,  -2,1,   &
                       -1,-3,  -3,-1,                                         &  ! in the current file, all BA is in fact B-A
                       +0,-2,  -2,0,   -4,-4,   -3,-5,                        &
                       +1,-3,  +1,-4,  +1,-5,   -3,1,   -1,-4,   -4,1,  0,0,   -6,-6,  &
                       +0,0/  
! spontanous fission - requires special treatment
   double precision s_fission_vector(0:max_number_fissile_isotopes,max_number_isotopes)  ! fission yields per fissile isotope
   double precision time_decay, dt
   double precision kt, mb_distribution(1000)  ! Maxwell-Boltzmann Distribution, need for fission/scission
   parameter (kt=1.8d0)  ! temperature of neutron before scission in MeV
   integer index_p, index_n, index_a, index_12c, time_steps, short_lived_counter
   logical short_lived(max_number_isotopes)

end module iso


!*****************************************
!*     Declarataion of the ISO Module    *
!*****************************************
! Taken from GEFSUB.FOR
module GEFSUB_FOR

!     '    Copyright 2009,2010,2011,2012,2013,2014,2015: 
!     '       Dr. Karl-Heinz Schmidt,Rheinstrasse 4,64390 Erzhausen,Germany 
!     '       and 
!     '       Dr. Beatriz Jurado,Centre d'Etudes Nucleaires de Bordeaux-Gradignan, 
!     '       Chemin du Solarium,Le Haut Vigneau,BP 120,33175 Gradignan,Cedex, 
!     '       France 
!     ' 
!     '    This program is free software: you can redistribute it and/or modify 
!     '    it under the terms of the GNU General Public License as published by 
!     '    the Free Software Foundation,either version 3 of the License,or 
!     '    (at your option) any later version. 
!     ' 
!     '    This program is distributed in the hope that it will be useful, 
!     '    but WITHOUT ANY WARRANTY; without even the implied warranty of 
!     '    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
!     '    GNU General Public License for more details. 
!     ' 
!     '    You should have received a copy of the GNU General Public License 
!     '    along with this program.  If not,see <http://www.gnu.org/licenses/>. 
!     ' 
!     ' 
!     /' Documentation: '/ 
!     /' (1) K.-H. Schmidt and B. Jurado,Contribution to '/ 
!     /'     ESNT Workshop "The scission process",Saclay (France),April 12-16,2010 '/ 
!     /' (2) B. Jurado and K.-H. Schmidt,Contribution to '/ 
!     /'     Seminar an fission,Gent (Belgium),May 17-20,2010 '/ 
!     /' (3) K.-H. Schmidt and B. Jurado,Contribution to '/ 
!     /'     Seminar on fission,Gent (Belgium),May 17-20,2010 '/ 
!     /' (4) B. Jurado and K.-H. Schmidt,Contribution to '/ 
!     /'     EFNUDAT Workshop,Paris (France),May 25-27,2010 '/ 
!     /' (5) K.-H. Schmidt and B. Jurado,Contribution to '/ 
!     /'     EFNUDAT Workshop,Paris (France),May 25-27,2010 '/ 
!     /' (6) K.-H. Schmidt and B. Jurado,'/ 
!     /'     Final Report to EFNUDAT,October,2010 '/ 
!     /' (7) K.-H. Schmidt and B. Jurado,Phys. Rev. Lett. 104 (2010) 21250 '/ 
!     /' (8) K.-H. Schmidt and B. Jurado,Phys. Rev. C 82 (2011) 014607 '/ 
!     /' (9) K.-H. Schmidt and B. Jurado,Phys. Rev. C 83 (2011) 061601 '/ 
!     /' (10) K.-H. Schmidt and B. Jurado,arXiv:1007.0741v1[nucl-th] (2010) '/ 
!     /' (11) K.-H. Schmidt and B. Jurado,JEF/DOC 1423,NEA of OECD,2012 '/ 
!     /' (12) K.-H. Schmidt and B. Jurado,Phys. Rev. C 86 (2012) 044322 '/ 
!     /' (13) K.-H. Schmidt,B. Jurado,Ch. Amouroux,JEFF-Report 24,NEA of OECD,2014 '/ 
!     ' 
!     ' 
!     /' Further documentation and the newest version of the GEF code are '/ 
!     /' available from                                                   '/ 
!     /' http://www.cenbg.in2p3.fr/GEF and http://www.khs-erzhausen.de/ . '/ 
!     ' 
!     ' 
!     '    The development of the GEF code has been supported by the European Union, 
!     '    EURATOM 6 in the Framework Program "European Facilities for Nuclear Data 
!     '    Measurements" (EFNUDAT),contract number FP6-036434,the Framework 
!     '    Program "European Research Infrastructure for Nuclear Data Applications 
!     '    (ERINDA),contract number FP7-269499,and by the OECD Nuclear Energy Agency. 

contains

!*****************************************
!*               U Valid                 *
!*****************************************
INTEGER*4 function U_Valid(I_Z,I_A)
      IMPLICIT NONE
      INTEGER*4 I_Z
      INTEGER*4 I_A
      INTEGER*4  Ivalid
      Ivalid = 1 
!     '   If I_A / I_Z < 210.E0/90.E0 
      IF (  I_A / I_Z .LT. 172.E0 / 80.E0 .OR. I_A / I_Z .GT. 250.E0/90.E0  ) THEN 
      Ivalid = 0 
      End If 
      IF (  I_Z .LT. 70 .OR. I_Z .GT. 120  ) THEN 
      Ivalid = 0 
      End If 
!     ' Ivalid = 1 
      U_Valid = Ivalid 

end


!*****************************************
!*              U Delta S0               *
!***************************************** 
REAL*4 function get_U_Delta_S0(I_Z,I_A)

      IMPLICIT NONE
      INTEGER*4 I_Z
      INTEGER*4 I_A
!     ' I_Z and I_A refer to the fissioning nucleus90 22 
      REAL*4  Delta
      Delta = 0.3 
      IF (  I_Z .EQ. 90 .AND. I_A .EQ. 228  )  Delta = 0.70 
!     'N 
      IF (  I_Z .EQ. 90 .AND. I_A .EQ. 230  )  Delta = 0.6 
!     'N 
      IF (  I_Z .EQ. 90 .AND. I_A .EQ. 233  )  Delta = 0.3 
!     ' 
      IF (  I_Z .EQ. 91 .AND. I_A .EQ. 228  )  Delta = 0.65 
!     ' 
      IF (  I_Z .EQ. 92 .AND. I_A .EQ. 233  )  Delta = 0.5 
!     'N 
      IF (  I_Z .EQ. 92 .AND. I_A .EQ. 234  )  Delta = 0.6 
!     'N 
      IF (  I_Z .EQ. 92 .AND. I_A .EQ. 235  )  Delta = 0.3 
      IF (  I_Z .EQ. 92 .AND. I_A .EQ. 236  )  Delta = 0.3 
!     'N 
      IF (  I_Z .EQ. 92 .AND. I_A .EQ. 237  )  Delta = 0.3 
      IF (  I_Z .EQ. 92 .AND. I_A .EQ. 238  )  Delta = 0.3 
      IF (  I_Z .EQ. 92 .AND. I_A .EQ. 239  )  Delta = 0.1 
!     ' 
      IF (  I_Z .EQ. 93 .AND. I_A .EQ. 238  )  Delta = -0.1 
!     'N 
!     ' 
      IF (  I_Z .EQ. 94 .AND. I_A .EQ. 240  )  Delta = -0.1 
!     'N 
      IF (  I_Z .EQ. 94 .AND. I_A .EQ. 241  )  Delta = -0.5 
!     'N 
      IF (  I_Z .EQ. 94 .AND. I_A .EQ. 242  )  Delta = -0.15 
!     'N 
      IF (  I_Z .EQ. 94 .AND. I_A .EQ. 243  )  Delta = -0.45 
!     'N 
      IF (  I_Z .EQ. 94  )  Delta = 0.25 
!     ' 
      IF (  I_Z .EQ. 95 .AND. I_A .EQ. 242  )  Delta = -0.35 
!     'N 
!     ' 
      IF (  I_Z .EQ. 95 .AND. I_A .EQ. 243  )  Delta = -0.1 
!     'N 
!     ' 
      IF (  I_Z .EQ. 95 .AND. I_A .EQ. 244  )  Delta = -0.1 
!     ' 
      IF (  I_Z .EQ. 96 .AND. I_A .EQ. 244  )  Delta = 0.0
!     'N 
      IF (  I_Z .EQ. 96 .AND. I_A .EQ. 246  )  Delta = -0.2 
!     'N 
      get_U_Delta_S0 = Delta 

end


!*****************************************
!*             Get Getyield              *
!***************************************** 
REAL*4 function get_Getyield(E_rel,E_ref,T_low,T_high)

      IMPLICIT NONE
      REAL*4 E_rel
      REAL*4 E_ref
      REAL*4 T_low
      REAL*4 T_high
!     /' Erel: Energy relative to the barrier '/ 
!     /' T_low: Effective temperature below barrier '/ 
!     /' T_high: Effective temperature above barrier '/ 
      REAL*4  Exp1
      REAL*4  Yield
!     ' 
      Exp1 = E_rel/T_low - E_ref/0.4 
!     ' energy far below barrier 
!     ' Subtraction of E_ref/0.4 to prevent numerical problems. 
      IF (  Exp1 .LT. -50  ) THEN 
      Yield = 0 
      Else 
      Yield = Exp(E_rel / T_high - E_ref/0.4) * 1.E0 / (1.E0 + exp(-E_rel/ (T_high*T_low/(T_high-T_low) ) ) ) 
      End If 

      get_Getyield = Max(Yield,0.0) 

end
 

!*****************************************
!*               Get F1                  *
!***************************************** 
REAL*4 function get_F1(Z_S_A)
      IMPLICIT NONE
      REAL*4 Z_S_A
!     /' Fit to the lower part of the data '/ 
      REAL*4  Result
      Result = exp(-9.05E0 + 4.58E0 * Log(Z_S_A/2.3E0)) 
      get_F1 = Result 

end


!*****************************************
!*               Get F2                  *
!*****************************************
REAL*4 function get_F2(Z_S_A)

      IMPLICIT NONE
      REAL*4 Z_S_A
!     /' Fit to the upper part of the data '/ 
      REAL*4  Result
      Result = exp(12.08E0 - 3.27E0 * Log(Z_S_A/2.3E0)) 
      get_F2 = Result 

end


!*****************************************
!*             Get Masscurv              *
!*****************************************
REAL*4 function get_Masscurv(Z,A,RL,kappa)

      IMPLICIT NONE
      REAL*4 Z
      REAL*4 A
      REAL*4 RL
      REAL*4 kappa
!     /'  Fit to  Data of Fig. 7 of                                             '/ 
!     /'  "Shell effect in the symmetric-modal fission of pre-actinide nuclei"  '/ 
!     /'  S. I. Mulgin,K.-H. Schmidt,A. Grewe,S. V. Zhdanov                  '/ 
!     /'  Nucl. Phys. A 640 (1998) 375 
!     /' (From fit of the width of the mass distributions.) '/                                         '/ 
      REAL*4  RI,Result1,Result2,Result
      REAL*4  Z_square_over_A
      REAL*4  ZsqrA
      REAL*4  c_rot
      DATA c_rot/600.0/
      REAL*4 F1 
      REAL*4 F2 
!     ' 
      Z_square_over_A = Z**2/A 
      RI = (A - 2*Z)/A 
      ZsqrA = Z_square_over_A * (1.E0 - kappa * RI**2) / &
          (1.E0 - kappa * ((226.E0 - 2.E0*91.E0)/226.E0)**2) + &
           c_rot * RL**2 / A**(7.0/3.0) 
!     ' Hasse & Myers 
!     '      + 0.0017 * RL^2 
!     ' 
      Result1 = get_F1(ZsqrA) 
      Result2 = get_F2(ZsqrA) 
      Result = Min(Result1,Result2) 
      get_Masscurv = Result 
!     ' 

end


!*****************************************
!*             Get Masscurv1             *
!*****************************************
REAL*4 function get_Masscurv1(Z,A,RL,kappa)

      IMPLICIT NONE
      REAL*4 Z
      REAL*4 A
      REAL*4 RL
      REAL*4 kappa
!     /'  Fit to  Data of Fig. 7 of                                             '/ 
!     /'  "Shell effect in the symmetric-modal fission of pre-actinide nuclei"  '/ 
!     /'  S. I. Mulgin,K.-H. Schmidt,A. Grewe,S. V. Zhdanov                  '/ 
!     /'  Nucl. Phys. A 640 (1998) 375 
!     /' (The left part assumed to be valid for the yields of the fission channels.) '/                                         '/ 
      REAL*4  RI,Result1,Result2,Result
!     '    Dim As Single A,A_central,Z 
      REAL*4  Z_square_over_A
      REAL*4  ZsqrA
      REAL*4  c_rot
      DATA c_rot/600.0/
      REAL*4 F1 
      REAL*4 F2 
!     ' 
!     'A_central = -28.8156 + Z * 2.86587  ' Stability line for heavy nuclei 
!     ' 
      Z_square_over_A = Z**2/A 
      RI = (A - 2*Z)/A 
      ZsqrA = Z_square_over_A * (1.E0 - kappa * RI**2) / &
          (1.E0 - kappa * ((226.E0 - 2.E0*91.E0)/226.E0)**2) + &
           c_rot * RL**2 / A**(7.0/3.0) 
!     ' Hasse & Myers 
!     '      + 0.0017 * RL^2 
!     ' 
      IF (  ZsqrA .LT. 36.0  ) THEN 
!     ' adjusted to Y(S2) in light nuclei (80<Z<92) 
      ZsqrA = ZsqrA + 0.9 * (36.0 - ZsqrA) 
      End If 
!     ' 
      Result1 = get_F1(ZsqrA) 
!     '  Result2 = get_F2(ZsqrA) 
!     '  Result = Min(Result1,Result2) 
      get_Masscurv1 = Result1 
!     ' 

end


!*****************************************
!         Get De Saddle Scission         *
!***************************************** 
REAL*4 function get_De_Saddle_Scission(Z_square_over_Athird,ESHIFTSASCI)

      IMPLICIT NONE
      REAL*4 Z_square_over_Athird
      REAL*4 ESHIFTSASCI
!     /' Energy release between saddle and scission '/ 
!     /' M. Asghar,R. W. Hasse,J. Physique C 6 (1984) 455 '/ 
      REAL*4  Result
      Result = (31.E0 - 11.E0) / (1550.E0 - 1300.E0) * &
          (Z_square_over_Athird - 1300.E0 + ESHIFTSASCI) + 11.E0 
!     ' This formula with ESHIFTSASCI = 0 is the parameterisation of the results 
!     ' of Ashgar and Hasse,JPC 6 (1984) 455,see 
!     ' F. Rejmund,A. V. Ignatyuk,A. R. Junghans,K.-H. Schmidt 
!     ' Nucl. Phys. A 678 (2000) 215 
      Result = max(Result,0.0) 
      get_De_Saddle_Scission = Result 

end


!*****************************************
!*             Get TEgidy                *
!***************************************** 
REAL*4 function get_TEgidy(A,DU,Fred)

      IMPLICIT NONE
      REAL*4 A
      REAL*4 DU
      REAL*4 Fred
!     /' Temperature parameter of the constant-temperature formula for the 
!     nuclear level density. 
!     Input parameters: A = Mass number of nucleus 
!     DU = Shell effect (corrected for pairing:P=0 for odd-A nuclei) 
!     From "Correlations between the nuclear level density parameters" 
!     Dorel Bucurescu,Till von Egidy 
!     Phys. Rev. C 72 (2005) 067304    and 
!     "Systematics of nuclear level density parameters" 
!     Dorel Bucurescu,Till von Egidy 
!     J. Phys. G: Nucl. Part. Phys. 31 (2005) S1675 and 
!     "Systematics of nuclear level density parameters" 
!     Till von Egidy,Dorel Bucurescu 
!     Phys. Rev. C 72 (2005) 044311 '/ 
      REAL*4  Temp_smooth,Temp,T_Fac
!     ' Temp_smooth = 17.45E0 / (A^0.666667E0) 
!     ' Temp = (17.45E0 - 0.51E0 * DU + 0.051 * DU^2) / (A^0.666667E0) 
      Temp_smooth = 1.0 / (0.0570 * A**0.6666667) 
      Temp = 1.0 / ( (0.0570 + 0.00193*DU) * A**0.6666667) 
!     ' from  PRC 80 (2009) 054310 
      T_Fac = Temp / Temp_smooth 
      Temp = Temp * Fred 
!     /' (For influence of deformation) '/ 
      get_TEgidy = Temp 

end


!*****************************************
!*             Get TRusanov              *
!***************************************** 
REAL*4 function get_TRusanov(E,A)

      IMPLICIT NONE
      REAL*4 E
      REAL*4 A
!     /' Fermi-gas level density,parameterisation of Rusanov et al. '/ 
      IF (  E >0  ) THEN 
      get_TRusanov = SQRT(E / (0.094E0 * A) ) 
      Else 
      get_TRusanov = 0.0 
      End If 

end


!*****************************************
!*              Get LyMass               *
!*****************************************
REAL*4 function get_LyMass(Z,A,beta)

      IMPLICIT NONE
      REAL*4 Z
      REAL*4 A
      REAL*4 beta
!     ' 
!     /' liquid-drop mass,Myers & Swiatecki,Lysekil,1967  '/ 
!     /' pure liquid drop,without pairing and shell effects '/ 
!     ' 
!     /' On input:    Z     nuclear charge of nucleus        '/ 
!     /'              N     number of neutrons in nucleus    '/ 
!     /'              beta  deformation of nucleus           '/ 
!     /' On output:   binding energy of nucleus              '/ 
!     ' 
      REAL*4  pi
      PARAMETER (pi=3.14159)
      REAL*4  N
      REAL*4  alpha
      REAL*4  XCOM,XVS,XE,EL
!     ' 
      N = A - Z 
      alpha = SQRT(5.E0/(4.E0*pi)) * beta 
      XCOM = 1.E0 - 1.7826E0 * ((A - 2.E0*Z)/A)**2 
!     /' factor for asymmetry dependence of surface and volume term '/ 
      XVS = - XCOM * (15.4941E0*A - 17.9439E0*A**(2.E0/3.E0)*(1.E0+0.4E0*Alpha**2)) 
!     /' sum of volume and surface energy '/ 
      XE = Z**2 * (0.7053E0/A**(1.E0/3.E0)*(1.E0-0.2E0*Alpha**2) - 1.1529E0/A) 
      EL = XVS + XE 
!     /'   EL = EL + LyPair(Z,A); '/ 
      get_LyMass = EL 

end


!*****************************************
!*             Get LyPair                *
!*****************************************
REAL*4 function get_LyPair(Z,A)

      IMPLICIT NONE
      INTEGER*4 Z
      INTEGER*4 A
!     /' Calculates pairing energy '/ 
!     /' odd-odd nucleus:   get_LyPair = 0 '/ 
!     /' even-odd nucleus:  get_LyPair = -12/sqr(A) '/ 
!     /' even-even nucleus: get_LyPair = -2*12/sqr(A) '/ 
      REAL*4  E_PAIR
!     ' 
      E_PAIR = - 12.E0 / SQRT(REAL(A)) * ( MOD((Z+1) , 2) + MOD((A-Z+1), 2)) 
!     ' 
      get_LyPair = E_PAIR 

end

!*****************************************
!*                TFPair                 *
!***************************************** 
REAL*4 function TFPair(Z,A)

      IMPLICIT NONE
      INTEGER*4 Z
      INTEGER*4 A
!     /' Pairing energy from Thomas-Fermi model of Myers and Swiatecki '/ 
!     /' Shifted that TFPair is zero for odd-odd nuclei '/ 
      INTEGER*4  N
      REAL*4  E_Pair
      N = A - Z 
      IF (   MOD(Z,2)  .EQ. 0 .AND.  MOD(N,2)  .EQ. 0  ) THEN 
!     /' even-even '/ 
      E_Pair = - 4.8E0 / Z**0.333333E0 - 4.8E0 / N**0.333333E0 + 6.6E0 / A**0.666666E0 
      END IF 
      IF (   MOD(Z,2)  .EQ. 0 .AND.  MOD(N,2)  .EQ. 1  ) THEN 
!     /' even Z,odd N '/ 
      E_Pair = - 4.8E0 / Z**0.333333E0 + 6.6E0 / A**0.666666E0 
      END IF 
      IF (   MOD(Z,2)  .EQ. 1 .AND.  MOD(N,2)  .EQ. 0  ) THEN 
!     /' odd Z,even N '/ 
      E_Pair = - 4.8E0 / N**0.333333E0 + 6.6E0 / A**0.666666E0 
      END IF 
      IF (   MOD(Z,2)  .EQ. 1 .AND.  MOD(N,2)  .EQ. 1  ) THEN 
!     /' odd N,odd N '/ 
      E_Pair = 0.0 
      END IF 
      TFPair = E_Pair 

end


!*****************************************
!*                 Pmass                 *
!*****************************************
REAL*4 function Pmass(Z,A,beta)

      IMPLICIT NONE
      REAL*4 Z
      REAL*4 A
      REAL*4 beta
!     /' Liquid-drop model of Pearson,2001 '/ 
      REAL*4  N,EA,BE
      REAL*4  avol
      DATA avol/-15.65/
      REAL*4  asf
      DATA asf/17.63/
      REAL*4  r0
      DATA r0/1.233/
      REAL*4  asym
      DATA asym/27.72/
      REAL*4  ass
      DATA ass/-25.60/
      REAL*4  alpha
      REAL*4  pi
      PARAMETER (pi=3.14159)
!     ' 
      N = A - Z 
      alpha = SQRT(5.E0/(4.E0*pi)) * beta 
      EA = avol + asf * A**(-0.333333)*(1.E0+0.4E0*Alpha**2) + &
          0.6E0 * 1.44E0 * Z**2 / (A**1.333333 * r0 )*(1.E0-0.2E0*Alpha**2) + &
          (asym + ass * A**(-0.333333)) * (N-Z)**2 / A**2 
      BE = EA * A 
      Pmass = BE 

end


!*****************************************
!*              Get FEDEFOP              *
!***************************************** 
REAL*4 function get_FEDEFOP(Z,A,beta)

      IMPLICIT NONE
      REAL*4 Z
      REAL*4 A
      REAL*4 beta
!     /' According to liquid-drop model of Pearson 2001 '/ 
      REAL*4  asf
      DATA asf/17.63/
      REAL*4  r0
      DATA r0/1.233/
      REAL*4  N,Alpha
      REAL*4  pi
      PARAMETER (pi=3.14159)
!     ' 
      N = A - Z 
      alpha = SQRT(5.E0/(4.E0*pi)) * beta 
      get_FEDEFOP = asf * A**(0.666667)*(0.4E0*Alpha**2) - &
          0.6E0 * 1.44E0 * Z**2 / (A**0.333333 * r0 )*(0.2E0*Alpha**2) 

end


!*****************************************
!*           Get FEDEFOLys               *
!*****************************************
REAL*4 function get_FEDEFOLys(Z,A,beta)

      IMPLICIT NONE
      REAL*4 Z
      REAL*4 A
      REAL*4 beta
      REAL*4 LYMASS 
      get_FEDEFOLys = get_LyMass(Z,A,beta) - get_LyMass(Z,A,0.0) 

end


!*****************************************
!*             Get LDMass                *
!*****************************************
REAL*4 function get_LDMass(Z,A,beta)

      IMPLICIT NONE
      REAL*4 Z
      REAL*4 A
      REAL*4 beta
      REAL*4  N,BEtab
      REAL*4 LYMASS 
      REAL*4 FEDEFOLYS 
      REAL*4 BEldmTF 
      REAL*4 BEexp 
      N = A - Z 
      BEtab = get_BEldmTF(NINT(N),NINT(Z)) + 2.0 * 12.0 / SQRT(REAL(A)) - 0.00001433*Z**2.39 
!     ' The values in BEtab are the negative binding energies! 
!     ' Pairing in Thomas Fermi masses is zero for Z,N even ! 
      IF (  BEtab .EQ. 0.0  ) THEN 
      BEtab = get_LyMass(Z,A,0.0) 
!     '         Print "Warning: Binding energy of Z=";Z;",A=";A;" not in mass table,";                         " replaced by LYMASS" 
!     '         Print "I_Mode = ";I_Mode 
      End If 
      get_LDMass = BEtab + get_FEDEFOLys(Z,A,beta) 

end


!*****************************************
!*               Get BEexp               *
!*****************************************
! Taken from BEexp.FOR file
REAL*4 function get_BEexp(N,Z)

    IMPLICIT NONE
    SAVE
    INTEGER*4 N
    INTEGER*4 Z
    INTEGER*4 I
    INTEGER*4 J
    INTEGER*4 NI
    INTEGER*4 ZI
    INTEGER*4 AI
    REAL*4 R
    INTEGER*4 IFIRST
    DATA IFIRST /1/
    REAL*4, DIMENSION(0:203,0:136) :: BEexptab
    If (IFIRST.EQ.1) THEN
      print*,'get_BEexp(N,Z), was here  ONCE', BEexptab(N,Z)
      DO I = 0, 203, 1
        DO J = 0, 136, 1
          BEexptab(I,J) = -1.E11
        END DO
      END DO          
      OPEN (UNIT = 1, FILE = 'BEexp.dat', STATUS = 'OLD', ACTION = 'READ')
      DO I = 1, 3352, 1      
         READ (1,*)  ZI, AI, R
         NI = AI - ZI
         BEexptab(NI,ZI) = R
      END DO   
      CLOSE (UNIT = 1)
      IFIRST = 0
    END IF 
    get_BEexp = BEexptab(N,Z)  

end


!*****************************************
!*              Get BELdmFT              *
!*****************************************
! Taken from BELdmFT.FOR file
    REAL*4 function get_BEldmTF(N,Z)
    IMPLICIT NONE
    SAVE
    INTEGER*4 N
    INTEGER*4 Z
    INTEGER*4 I
    INTEGER*4 NI
    INTEGER*4 ZI
    REAL*4 R
    INTEGER*4 IFIRST
    DATA IFIRST /1/
    REAL*4, DIMENSION(0:203,0:136) :: BEldmTFtab
    If (IFIRST.EQ.1) THEN
      print*,'get_BEldmTF(N,Z), was here  ONCE', BEldmTFtab(N,Z)
      OPEN (UNIT = 1, FILE = 'BEldmTF.dat', STATUS = 'OLD', ACTION = 'READ')
      DO I = 1, 8293, 1      
         READ (1,*)  NI, ZI, R
         BEldmTFtab(NI,ZI) = R
      END DO   
      CLOSE (UNIT = 1)
      IFIRST = 0
    END IF 
    get_BEldmTF = BEldmTFtab(N,Z)  

end


!*****************************************
!*            Get AME2012                *
!*****************************************
REAL*4 function get_AME2012(IZ,IA)

      IMPLICIT NONE
      INTEGER*4 IZ
      INTEGER*4 IA
!     ' Masses from the 2003 mass evaluation,complemented by TF masses 
!     ' and Lysekil masses. 
      REAL*4  BEexpval
      REAL*4  Z,A,N
      INTEGER*4  INeu
      REAL*4 LYPAIR 
      REAL*4 U_SHELL 
      REAL*4 LDMASS 
      REAL*4 BEexp 
      INeu = IA - IZ 
      A = REAL(IA) 
      Z = REAL(IZ) 
      N = A - Z 
      BEexpval = get_BEexp(INeu,IZ) 
      IF (  BEexpval .GT. -1.E10  ) THEN 
      get_AME2012 = BEexpval 
      Else 
      get_AME2012 = get_LDMass(Z,A,0.0) + get_U_SHELL(IZ,IA) + get_LyPair(IZ,IA) 
      End If 

end


!*****************************************
!*             Get U Shell               *
!*****************************************
REAL*4 function get_U_SHELL(Z,A)

      IMPLICIT NONE
      INTEGER*4 Z
      INTEGER*4 A
      INTEGER*4  N
      REAL*4  Res
      REAL*4 ShellMO 
      N = A - Z 
      Res = get_ShellMO(N,Z) 
      IF (  Res .GT. 0.0  )  Res = 0.3 * Res 
!     ' KHS (12. Feb. 2012) 
!     '      ' The positive shell effects for deformed nuclei seem to be too positive 
!     ' This gives too many high-energetic prompt neutrons. 
      get_U_SHELL = Res 

end


!*****************************************
!*           Get U Shell Exp             *
!*****************************************
REAL*4 function get_U_SHELL_exp(IZ,IA)

      IMPLICIT NONE
      INTEGER*4 IZ
      INTEGER*4 IA
      REAL*4  Res
      REAL*4  Z,A
      REAL*4 LDMASS 
      REAL*4 LYPAIR 
      REAL*4 AME2012 
      Z = REAL(IZ) 
      A = REAL(IA) 

      Res = 0.5 * ( get_AME2012(IZ,IA) - get_LyPair(IZ,IA) - get_LDMass(Z,A,0.0) ) + &
          0.125 * ( get_AME2012(IZ,IA-1) - get_LyPair(IZ,IA-1) - get_LDMass(Z,A-1.0,0.0) ) + &
          0.125 * ( get_AME2012(IZ,IA+1) - get_LyPair(IZ,IA+1) - get_LDMass(Z,A+1.0,0.0) ) + &
          0.125 * ( get_AME2012(IZ+1,IA+1) - get_LyPair(IZ+1,IA+1) - &
          get_LDMass(Z+1.0,A+1.0,0.0) ) + 0.125 * ( get_AME2012(IZ-1,IA-1) - &
          get_LyPair(IZ-1,IA-1) - get_LDMass(Z-1.0,A-1.0,0.0) ) 
      get_U_SHELL_exp = Res 

end


!*****************************************
!*          Get U Shell EO Exp           *
!*****************************************
REAL*4 function get_U_SHELL_EO_exp(IZ,IA)

      IMPLICIT NONE
      INTEGER*4 IZ
      INTEGER*4 IA
!     ' Returns experimental shell and even-odd staggering 
      REAL*4  Res
      REAL*4  Z,A
      REAL*4 LDMASS 
      REAL*4 LYPAIR 
      REAL*4 AME2012 
      Z = REAL(IZ) 
      A = REAL(IA) 
      Res = get_AME2012(IZ,IA) - get_LDMass(Z,A,0.0) 
      get_U_SHELL_EO_exp = Res 

end


!*****************************************
!*             Get U MASS                *
!*****************************************
REAL*4 function get_U_MASS(Z,A)

      IMPLICIT NONE
      REAL*4 Z
      REAL*4 A
!     /' LD + congruence energy + shell (no pairing) '/ 
      REAL*4  BE
      REAL*4 U_SHELL 
      REAL*4 LDMASS 
!      IF (  Z .LT. 0 .OR. A .LT. 0  ) THEN 
!     '       Print "U_Mass: Z,A",Z,A 
!      End If 
      BE = get_LDMass(Z,A,0.0) + get_U_SHELL(NINT(Z),NINT(A))  
      get_U_MASS = BE 

end


!*****************************************
!*              Get ECOUL                *
!***************************************** 
REAL*4 function get_ECOUL(Z1,A1,beta1,Z2,A2,beta2,d)

      IMPLICIT NONE
      REAL*4 Z1
      REAL*4 A1
      REAL*4 beta1
      REAL*4 Z2
      REAL*4 A2
      REAL*4 beta2
      REAL*4 d
!     ' 
!     /' Coulomb potential between two nuclei                    '/ 
!     /' surfaces are in a distance of d                         '/ 
!     /' in a tip to tip configuration                           '/ 
!     ' 
!     /' approximate formulation                                 '/ 
!     /' On input: Z1      nuclear charge of first nucleus       '/ 
!     /'           A1      mass number of irst nucleus   '/ 
!     /'           beta1   deformation of first nucleus          '/ 
!     /'           Z2      nuclear charge of second nucleus      '/ 
!     /'           A2      mass number of second nucleus  '/ 
!     /'           beta2   deformation of second nucleus         '/ 
!     /'           d       distance of surfaces of the nuclei    '/ 
!     ' 
      REAL*4  N1,N2,recoul
      REAL*4  dtot
      REAL*4  r0
      DATA r0/1.16/
!     ' 
      N1 = A1 - Z1 
      N2 = A2 - Z2 
      dtot = r0 *( (Z1+N1)**0.3333333E0 * (1.E0+0.6666667E0*beta1) + &
          (Z2+N2)**0.3333333E0 * (1.E0+0.6666667E0*beta2) ) + d 
      REcoul = Z1 * Z2 * 1.44E0 / dtot 
!     ' 
      get_ECOUL = REcoul 

end


!*****************************************
!*              Beta Light               *
!*****************************************
REAL*4 function get_beta_light(Z,betaL0,betaL1)

      IMPLICIT NONE
      INTEGER*4 Z
      REAL*4 betaL0
      REAL*4 betaL1
!     /' Deformation of light fission fragment for S1 and S2 '/ 
!     /' Systematic correlation Z vs. beta for deformed shells '/ 
!     /' Z of fission fragment '/ 
      REAL*4  beta
      beta = (Z - betaL0) * betaL1/20.E0 
      get_beta_light = beta 

end


!*****************************************
!*              Beta Heavy               *
!*****************************************
REAL*4 function get_beta_heavy(Z,betaH0,betaH1)

      IMPLICIT NONE
      INTEGER*4 Z
      REAL*4 betaH0
      REAL*4 betaH1
!     /' Deformation of heavy fission fragment for S2 '/ 
!     /' Systematic correlation Z vs. beta for deformed shells '/ 
!     /' Z of fission fragment '/ 
      REAL*4  beta
      beta = (Z - betaH0) * betaH1/20.E0 
      get_beta_heavy = beta 

end


!*****************************************
!*                Z Equi                 *
!***************************************** 
REAL*4 function get_Z_equi(ZCN,A1,A2,beta1,beta2,d,Imode,POLARadd,POLARfac)

      IMPLICIT NONE
      INTEGER*4 ZCN
      INTEGER*4 A1
      INTEGER*4 A2
      REAL*4 beta1
      REAL*4 beta2
      REAL*4 d
      INTEGER*4 Imode
      REAL*4 POLARadd
      REAL*4 POLARfac

!     /' Determines the minimum potential of the scission-point configuration 
!     represented by two deformed nuclei divided by a tip distance d. 
!     A1,A2,beta1,beta2,d are fixed,Z1 is searched for and returned on output.  '/ 
!     ' 
!     /' ZCN: Z of fissioning nucleus '/ 
!     /' A1: A of first fission fragment '/ 
!     /' A2: A of second fission fragment '/ 
!     /' beta1: deformation of first fission fragment '/ 
!     /' beta2: deformation of second fission fragment '/ 
!     /' d: tip distance '/ 
!     ' 
      REAL*4  RZ_equi
      REAL*4  RA1,RA2,RZCN,RACN
      REAL*4  Z1UCD,Z2UCD
      REAL*4  re1,re2,re3,eps1,eps2,DZ_Pol
!     /' help variables '/ 
      REAL*4 ECOUL 
      REAL*4 LYMASS 
!     ' 
      RA1 = REAL(A1) 
      RA2 = REAL(A2) 
      RZCN = REAL(ZCN) 
      RACN = RA1 + RA2 
      Z1UCD = RA1 / (RA1 + RA2) * RZCN 
      Z2UCD = RZCN - Z1UCD 
      re1 = get_LyMass( Z1UCD-1.E0,RA1,beta1 ) + &
          get_LyMass( Z2UCD+1.E0,RA2,beta2 ) + &
          get_ECOUL( Z1UCD-1.E0,RA1,beta1,Z2UCD+1.E0,RA2,beta2,d ) 
      re2 = get_LyMass( Z1UCD,RA1,beta1) + get_LyMass( Z2UCD,RA2,beta2) + &
          get_ECOUL( Z1UCD,RA1,beta1,Z2UCD,RA2,beta2,d ) 
      re3 = get_LyMass( Z1UCD+1.E0,RA1,beta1 ) + &
            get_LyMass( Z2UCD-1.E0,RA2,beta2 ) + &
            get_ECOUL( Z1UCD+1.E0,RA1,beta1,Z2UCD-1.E0,RA2,beta2,d ) 
      eps2 = ( re1 - 2.E0*re2 + re3 ) / 2.E0 
      eps1 = ( re3 - re1 ) / 2.E0 
      DZ_Pol = -eps1 / ( 2.E0 * eps2 ) 
!     ' 
      IF (  DZ_Pol .GT. 2 .OR. DZ_Pol .LT. -2  )  DZ_Pol = 0 
!     ' 
      IF (  Imode .GT. 0  ) THEN 
!     /' Purely empirical enhancement of charge polarization '/ 
      DZ_POL = DZ_POL * POLARfac + POLARadd 
      End If 
!     ' 
      RZ_equi = Z1UCD + DZ_POL 
      get_Z_equi = RZ_equi 

end


!*****************************************
!*            Beta Opt Light             *
!*****************************************
subroutine Beta_opt_light(A1,A2,Z1,Z2,d,beta2_imposed,beta1_opt)

      IMPLICIT NONE
      REAL*4 A1
      REAL*4 A2
      REAL*4 Z1
      REAL*4 Z2
      REAL*4 d
      REAL*4 beta2_imposed
      REAL*4 beta1_opt
!     /' Determines the optimum deformation of the light fragment when the deformation of the 
!     heavy fragment is imposed. '/ 
!     ' 
      REAL*4  beta1,dbeta1,beta1_prev,beta1_next
      REAL*4  Uguess,Uplus,Uminus,Uprev,Unext
      INTEGER*4  I
      REAL*4 ECOUL 
      REAL*4 LYMASS 
!     ' 
!     /' List('Beta_opt_light called with '); 
!     List(A1,A2,Z1,Z2,d,beta2_imposed,beta1_opt); 
!     DCL Byes Bit(1) aligned; 
!     Call GPYES('Continue',Byes); '/ 
      beta1 = 0.5 
      dbeta1 = 0.01 
      Uguess = get_LyMass(Z1,A1,beta1) + get_LyMass(Z2,A2,beta2_imposed) + &
          get_ECOUL(Z1,A1,beta1,Z2,A2,beta2_imposed,d) 
      Uplus = get_LyMass(Z1,A1,beta1 + dbeta1) + get_LyMass(Z2,A2,beta2_imposed) + &
          get_ECOUL(Z1,A1,beta1 + dbeta1,Z2,A2,beta2_imposed,d) 
      Uminus = get_LyMass(Z1,A1,beta1 - dbeta1) + get_LyMass(Z2,A2,beta2_imposed) + &
          get_ECOUL(Z1,A1,beta1 - dbeta1,Z2,A2,beta2_imposed,d) 
      IF (  Uplus .GT. Uguess .AND. Uminus .GT. Uguess  ) THEN 
      beta1_opt = beta1 
      Else 
      IF (  Uplus .LT. Uguess  )  dbeta1 = 0.01 
      IF (  Uminus .LT. Uguess  )  dbeta1 = -0.01 
      Unext = Uguess 
      beta1_next = beta1 
      DO I = 1 , 10000 
      beta1_prev = beta1_next 
      Uprev = Unext 
      beta1_next = beta1_prev + dbeta1 
      Unext = get_LyMass(Z1,A1,beta1_next) + get_LyMass(Z2,A2,beta2_imposed) + &
          get_ECOUL(Z1,A1,beta1_next,Z2,A2,beta2_imposed,d) 
      IF (  Unext .GE. Uprev  )  Exit 
      END DO 
      beta1_opt = beta1_prev 
      END IF 

end
      

!*****************************************
!*              Beta Equi                *
!*****************************************
subroutine Beta_Equi(A1,A2,Z1,Z2,d,beta1prev,beta2prev,beta1opt,beta2opt)

      IMPLICIT NONE
      REAL*4 A1
      REAL*4 A2
      REAL*4 Z1
      REAL*4 Z2
      REAL*4 d
      REAL*4 beta1prev
      REAL*4 beta2prev
      REAL*4 beta1opt
      REAL*4 beta2opt
!     /' Determines the minimum potential of the scission-point configuration 
!     represented by two deformed nuclei,divided by a tip distance d. 
!     A1,A2,Z1,Z2,d are fixed,beta1 and beta2 are searched for and returned on output '/ 
!     ' 
      INTEGER*4  B_analytical
      DATA B_analytical/0/
!     ' Switch to use the analytical approximation 
!     ' that replaces the long numerical calculation. 
      REAL*4  x,y,xcoul
      REAL*4  xcoul236U
      DATA xcoul236U/1369.64/
!     ' 
      REAL*4  beta1,beta2
!     ' 
!     '      Dim As Double U,Uprev,Ulast,Ubest,Uopt 
      REAL*4  U,Uprev,Ulast,Ubest,Uopt
!     ' 
!     '      Dim As Double sbeta1,sbeta2 
      REAL*4  sbeta1,sbeta2
!     ' 
      INTEGER*4  N,N1,N2,Nopt
!     ' 
!     '      Dim As Double eps = 5.E-4 
      REAL*4  eps
      DATA eps/5.E-4/
!     ' 
      INTEGER*4  I
      REAL*4 LYMASS 
      REAL*4 ECOUL 
!     ' 
      IF (  B_analytical .EQ. 0  ) THEN 
!     ' Numerical algorithm 
!     ' 
      beta1 = beta1prev 
      beta2 = beta2prev 
      Uprev = get_LyMass(Z1,A1,beta1) + get_LyMass(Z2,A2,beta2) + &
          get_ECOUL(Z1,A1,beta1,Z2,A2,beta2,d) 
      Uopt = Uprev 
!     ' 
!     /' Test slope of variation of U '/ 
      beta1 = beta1prev + eps 
      U = 1.E30 
!     ' 
      beta2 = beta2prev 
!     '     For beta2 = beta2prev to 0 Step -eps 
      DO I = 1 , Int(beta2prev/eps) 
      beta2 = beta2 - eps 
      Ulast = U 
      U = get_LyMass(Z1,A1,beta1) + get_LyMass(Z2,A2,beta2) + &
          get_ECOUL(Z1,A1,beta1,Z2,A2,beta2,d) 
      IF (  U .GT. Ulast  ) THEN 
      Exit 
      Else 
      Ubest = U 
      END IF 
      END DO 
      IF (  Ubest .LT. Uopt  ) THEN 
      Uopt = Ubest 
      sbeta1 = eps 
      sbeta2 = -eps 
      END IF 
!     ' 
      U = 1.E30 
      beta2 = beta2prev 
!     '   For beta2 = beta2prev To 1 Step eps 
      DO I = 1 , Int((1 - beta2prev)/eps) 
      beta2 = beta2 + eps 
      Ulast = U 
      U = get_LyMass(Z1,A1,beta1) + get_LyMass(Z2,A2,beta2) + &
          get_ECOUL(Z1,A1,beta1,Z2,A2,beta2,d) 
      IF (  U .GT. Ulast  ) THEN 
      Exit 
      Else 
      Ubest = U 
      END IF 
      END DO 
      IF (  Ubest .LT. Uopt  ) THEN 
      Uopt = Ubest 
      sbeta1 = eps 
      sbeta2 = eps 
      End If 
!     ' 
      beta1 = beta1prev - eps 
      U = 1.E30 
      beta2 = beta2prev 
!     '   For beta2 = beta2prev To 0 Step -eps 
      DO I = 1 , Int(beta2prev/eps) 
      beta2 = beta2 - eps 
      Ulast = U 
      U = get_LyMass(Z1,A1,beta1) + get_LyMass(Z2,A2,beta2) + &
          get_ECOUL(Z1,A1,beta1,Z2,A2,beta2,d) 
      IF (  U .GT. Ulast  ) THEN 
      Exit 
      Else 
      Ubest = U 
      End If 
      END DO 
      IF (  Ubest .LT. Uopt  ) THEN 
      Uopt = Ubest 
      sbeta1 = -eps 
      sbeta2 = -eps 
      END IF 
!     ' 
      U = 1.E30 
      beta2 = beta2prev 
!     '   For beta2 = beta2prev To 1 Step eps 
      DO I = 1 , Int((1-beta2prev)/eps) 
      beta2 = beta2 + eps 
      Ulast = U 
      U = get_LyMass(Z1,A1,beta1) + get_LyMass(Z2,A2,beta2) + &
          get_ECOUL(Z1,A1,beta1,Z2,A2,beta2,d) 
      IF (  U .GT. Ulast  ) THEN 
      Exit 
      Else 
      Ubest = U 
      END IF 
      END DO 
      IF (  Ubest .LT. Uopt  ) THEN 
      Uopt = Ubest 
      sbeta1 = -eps 
      sbeta2 = eps 
      END IF 
!     ' 
!     ' 
      Ubest = get_LyMass(Z1,A1,beta1prev) + get_LyMass(Z2,A2,beta2prev) + &
          get_ECOUL(Z1,A1,beta1prev,Z2,A2,beta2prev,d) 
      U = get_LyMass(Z1,A1,beta1prev+REAL(sbeta1)) + &
          get_LyMass(Z2,A2,beta2prev+REAL(sbeta2)) + &
          get_ECOUL(Z1,A1,beta1prev+sbeta1,Z2,A2,beta2prev+REAL(sbeta2),d) 
!     ' 
!     '   L1: 
      DO N = 1 , 1000 
!     ' 
!     '   L2: 
      DO N1 = 1 , N 
      N2 = N-N1 
      beta1 = beta1prev + sbeta1*N1 
      beta2 = beta2prev + sbeta2*N2 
      U = get_LyMass(Z1,A1,beta1) + get_LyMass(Z2,A2,beta2) + &
          get_ECOUL(Z1,A1,beta1,Z2,A2,beta2,d) 
      IF (  U .LT. Ubest  ) THEN 
      Ubest = U 
      beta1opt = beta1 
      beta2opt = beta2 
      Nopt = N 
      END IF 
      END DO 
      IF (  N-Nopt .GT. 2  )  Exit 
      END DO 
!     ' 
!     ' 
      Else 
!     ' Analytical approximation 
!     ' Must be adapted if the relevant parameters of GEF are modified! 
      xcoul = (Z1 + Z2)**2 / (A1 + A2)**(1.0/3.0) 
      x = (Z1 / (Z1 + Z2))**(xcoul/xcoul236U) 
      y = 1.2512E-4 + 0.00122851*x - 0.00267707*x**2 + 0.00372901*x**3 - 0.00219903*x**4 
      beta1opt = y * xcoul 
!     ' 
      x = (Z2 / (Z1 + Z2))**(xcoul/xcoul236U) 
      y = 1.2512E-4 + 0.00122851*x - 0.00267707*x**2 + 0.00372901*x**3 - 0.00219903*x**4 
      beta2opt = y * xcoul 
!     ' 
      End If 
!     ' 

end


!*****************************************
!*             Get U Ired                *
!***************************************** 
REAL*4 function get_U_Ired(Z,A)

      IMPLICIT NONE
      REAL*4 Z
      REAL*4 A
!     ' Effective moment of inertia by pairing with correction for excitation energy 
      REAL*4  I_rigid_spher,IfragEff
!     ' 
      REAL*4 U_SHELL 
!     ' 
      I_rigid_spher = 1.16E0**2 * A**1.6667E0 / 103.8415E0 
!     '   IfragEff = I_rigid_spher + 0.003 * A^(4.0/3.0) * U_shell(Cint(Z),Cint(A)) 
!     '   IfragEff = I_rigid_spher + 0.005 * A^(4.0/3.0) * U_shell(Cint(Z),Cint(A)) 
!     ' reduction due to shell (Deleplanque et al. PRC 69 (2004) 044309) 
      IfragEff = 0.45 * I_rigid_spher 
!     ' Effect of superfluidity 
!     '   IfragEff = 0.65 * IfragEff   ' Average effect of superfluidity and deformation 
!     ' 
      get_U_Ired = IfragEff 

end


!*****************************************
!*                U IredFF               *
!***************************************** 
REAL*4 function U_IredFF(Z,A)

      IMPLICIT NONE
      REAL*4 Z
      REAL*4 A
!     ' Effective moment of inertia by pairing with correction for excitation energy 
!     ' of final fission fragments 
!     ' 
      REAL*4 U_Ired 
      REAL*4 U_I_Shell 
!     ' 
      U_IredFF = get_U_Ired(Z,A) * get_U_I_Shell(Z,A) 

end


!*****************************************
!*            Get U I Shell              *
!*****************************************
REAL*4 function get_U_I_Shell(Z,A)

      IMPLICIT NONE
      REAL*4 Z
      REAL*4 A
      INTEGER*4  N_shells(6)
!     ' Shell effect on the effective moment of inertia 
      INTEGER*4  I
      REAL*4  dNmin,dZmin,dNsubmin
      REAL*4  Inv_add
      DATA Inv_add/0/
      REAL*4  I_inv_add_Z
      DATA I_inv_add_Z/0/
      REAL*4  I_inv_add_N
      DATA I_inv_add_N/0/
      REAL*4  I_inv_add_Nsub
      DATA I_inv_add_Nsub/0/
      N_shells(1) = 20 
      N_shells(2) = 28 
      N_shells(3) = 50 
      N_shells(4) = 82 
      N_shells(5) = 126 
      N_shells(6) = 56 
      dNmin = 100 
      dZmin = 100 
      dNsubmin = 100 
      DO I = 1 , 5 
      dZmin = Min(dZmin,Abs(N_shells(I) - Z)) 
      END DO 
!     ' 
      DO I = 1 , 5 
      dNmin = Min(dNmin,Abs(N_shells(I) - (A-Z))) 
      END DO 
!     ' 
      dNsubmin = Abs(N_shells(6) - (A-Z)) 
!     ' 
!     ' Effect of shells: 
      IF (  dZmin .LT. 10.0  ) THEN 
!     '        I_inv_add_Z = 0.33 * (6.0 * sqr(A/140.) - dZmin) * sqr(140./A) 
      I_inv_add_Z = 0.33 * (6.0 * SQRT(A/140.0) - dZmin) * (140.0/A)**1.5 
!     ' A^(-1/3) dependence: "A simple phenomenology for 2gamma+ states", 
!     ' N. V. Zamfir,D. Bucurescu,R. F. Casten,M. Ivascu, 
!     ' Phys. Lett. B 241 (1990) 463 
      I_inv_add_Z = Max(I_inv_add_Z,0.0) 
      End If 
      IF (  dNmin .LT. 10.0  ) THEN 
!     '        I_inv_add_N = 0.42 * (8.0 * sqr(A/140.) - dNmin) * sqr(140./A) 
      I_inv_add_N = 0.42 * (8.0 * SQRT(A/140.0) - dNmin) * (140.0/A)**1.5 
      I_inv_add_N = Max(I_inv_add_N,0.0) 
      End If 
      IF (  DNsubmin .LT. 6.0  ) THEN 
!     '    I_inv_add_Nsub = 1.7 * (4.0 - dNsubmin) * (1.0 - 0.32 * Abs(40.0-Z)) 
      I_inv_add_Nsub = 1.7 * (4.0 - dNsubmin) * (1.0 - 0.18 * Abs(40.0-Z)) 
!     ' N = 56 subshell only around Z = 40 
      I_inv_add_Nsub = Max(I_inv_add_Nsub,0.0) 
      End If 
      get_U_I_Shell = 1.0 / (1.0 + Max(I_inv_add_N,I_inv_add_Nsub) + I_inv_add_Z) 

end


!*****************************************
!*            Get U Alev Ld              *
!*****************************************
REAL*4 function get_U_alev_ld(Z,A)

      IMPLICIT NONE
      REAL*4 Z
      REAL*4 A
!     '  get_U_alev_ld = 0.073 * A + 0.095 * A^0.666667  'Ignatyuk (1970's) 
      get_U_alev_ld = 0.078 * A + 0.115 * A**0.6666667 
!     ' Ignatyuk (Bologna 2000) 
!     '  get_U_alev_ld = 0.089 * A    ' only volume term 

end


!*****************************************
!*              Get U Temp               *
!*****************************************
REAL*4 function get_U_Temp(Z,A,E,Ishell,Ipair,Tscale,Econd)

      IMPLICIT NONE
      REAL*4 Z
      REAL*4 A
      REAL*4 E
      INTEGER*4 Ishell
      INTEGER*4 Ipair
      REAL*4 Tscale
      REAL*4 Econd
!     ' Temperature (modified Gilbert-Cameron composite level density) 
!     ' KHS (10. 2. 2012) 
      REAL*4  alev
      REAL*4  Eeff0,Eeff1,Rho0,Rho1,TCT,TFG
      REAL*4  fgamma
      DATA fgamma/0.055/
      REAL*4  RShell,RPair,Res
      REAL*4 U_ALEV_LD 
      REAL*4 U_SHELL 
      REAL*4 LYPAIR 
      REAL*4 TEGIDY 
!     ' Used global parameters: Tscale 
!     '   alev = get_U_alev_ld(Z,A) * 1.1   ' Factor adjusted to high-energy prompt neutrons in U235(nth,f) 
!     '  alev = get_U_alev_ld(Z,A) * 0.8  ' " with the correction for non-constant T (FG range) 
      alev = get_U_alev_ld(Z,A) 
!     ' 
      IF (  Ishell .EQ. 1  ) THEN 
      RShell = get_U_SHELL(NINT(Z),NINT(A)) 
      Else 
      RShell = 0.0 
      End If 
      TCT = get_TEgidy(A,RShell,Tscale) 
!     ' 
      IF (  Ipair .EQ. 1  ) THEN 
      RPair = get_LyPair(NINT(Z),NINT(A)) 
      Else 
      Rpair = 0.0 
      End If 
      Eeff0 = E - Econd + RPair + Rshell*(1.0 - exp(-fgamma * E)) 
!     ' 
      IF (  Eeff0 .GT. 0.5  ) THEN 
      Eeff1 = Eeff0 + 0.1 
      Rho0 = 1.E0/Eeff0**1.25 * exp(2.E0 * SQRT(alev * Eeff0)) 
      Rho1 = 1.E0/Eeff1**1.25 * exp(2.E0 * SQRT(alev * Eeff1)) 
!     '         Rho0 = 1.E0/Eeff0 * exp(2.E0 * sqr(alev * Eeff0)) 
!     '         Rho1 = 1.E0/Eeff1 * exp(2.E0 * sqr(alev * Eeff1)) 
      TFG = 0.1E0 / (log(Rho1) - log(Rho0)) 
      Else 
      TFG = 0.0 
      End If 
      Res = TCT 
      IF (  TFG .GT. Res  )  Res = TFG 
!     ' 
!     ' If Res > 1.4 Then Res = 1.4 
!     ' 
      get_U_Temp = Res 

end


!*****************************************
!*           Get U Even Odd              *
!*****************************************
REAL*4 function get_U_Even_Odd(I_Channel,PEO)

      IMPLICIT NONE
      INTEGER*4 I_Channel
      REAL*4 PEO
!     ' Creates even-odd fluctuations 
      REAL*4  R
      IF (   MOD(I_Channel,2)  .EQ. 0  ) THEN 
      R = 1.0 + PEO 
      Else 
      R = 1.0 - PEO 
      End If 
      get_U_Even_Odd = R 

end


!*****************************************
!*              Get BFTF                 *
!***************************************** 
REAL*4 function get_BFTF(RZ,RA,I_Switch)

      IMPLICIT NONE
      REAL*4 RZ
      REAL*4 RA
      INTEGER*4 I_Switch
!     /' Fission barriers from Myers and Swiatecki,Thomas-Fermi model '/ 
!     /'  I_Switch: 0: liquid-drop; 1: with shells and pairing, 
!     2: averaged over pairing,3: with shell and pairing + pairing gap at barrier '/ 
!     ' 4: liquid-drop + g.s. shell,no Z correction 
      REAL*4  RN,RI,Rkappa,RS,RF,RX
      REAL*4  RX0
      DATA RX0/48.5428/
      REAL*4  RX1
      DATA RX1/34.15/
      REAL*4  RB
      INTEGER*4  IZ,IA
      REAL*4 U_SHELL 
      REAL*4 U_SHELL_EXP 
      REAL*4 U_SHELL_EO_EXP 
      REAL*4 LYPAIR 
!     ' 
      IZ = NINT(RZ) 
      IA = NINT(RA) 
      RN = RA - RZ 
      RI = (RN-RZ) / RA 
      Rkappa = 1.9E0 + (RZ - 80.E0) / 75.E0 
      RS = RA**0.666667E0 * (1.E0 - Rkappa * RI**2) 
      RX = RZ**2 / (RA * (1.E0 - Rkappa * RI**2)) 
      IF (  RX .LT. 30  ) THEN 
!     /' out of range '/ 
      RF = 1.E10 
      End If 
      IF (  RX .GT. RX0  ) THEN 
!     /' out of range '/ 
      RF = 0.0 
      End If 
      IF (  RX .LT. RX1 .AND. RX .GT. 30  ) THEN 
      RF = 0.595553E0 - 0.124136E0 * (RX - RX1) 
      End If 
      IF (  RX .GE. RX1 .AND. RX .LE. RX0  ) THEN 
      RF = 0.000199749 * (RX0 - RX)**3 
      End If 
      RB = RF * RS 
!     ' 
      Select CASE( I_Switch) 
      CASE( 0) 
      get_BFTF = RB 
      CASE( 1) 
!     ' including even-odd staggering due to increased pairing strength at barrier 
!     ' Tentative modification from comparison with experimental fission barriers 
!     ' (shell correction at the barrier?) 
      IF (  RZ .GT. 86.5  )  RB = RB - 0.15 * (RZ - 86.5) 
!     '    If RZ > 90 Then RB = RB + 0.3 * (RZ - 90.0) 
!     '    If RZ > 98 Then RB = RB - 0.15 * (RZ - 98.0) 
      IF (  RZ .GT. 90  )  RB = RB + 0.35 * (RZ - 90.0) 
      IF (  RZ .GT. 93  )  RB = RB + 0.15 * (RZ - 93.0) 
      IF (  RZ .GT. 95  )  RB = RB - 0.25 * (RZ - 95.0) 
!     '    get_BFTF = RB - get_U_SHELL(IZ,IA) 
!     '    get_BFTF = RB - get_U_SHELL_exp(IZ,IA) 
      get_BFTF = RB - get_U_SHELL_EO_exp(IZ,IA) + get_LyPair(IZ,IA) * 14.0/12.0 
      CASE( 2) 
!     ' averaged over even-odd staggering 
      IF (  RZ .GT. 86.5  )  RB = RB - 0.15 * (RZ - 86.5) 
      IF (  RZ .GT. 90  )  RB = RB + 0.35 * (RZ - 90.0) 
      IF (  RZ .GT. 93  )  RB = RB + 0.15 * (RZ - 93.0) 
      IF (  RZ .GT. 95  )  RB = RB - 0.25 * (RZ - 95.0) 
      get_BFTF = RB - get_U_SHELL_exp(IZ,IA) 
      CASE( 3) 
!     ' like Case 1 + pairing gap at barrier 
      IF (  RZ .GT. 86.5  )  RB = RB - 0.15 * (RZ - 86.5) 
      IF (  RZ .GT. 90  )  RB = RB + 0.35 * (RZ - 90.0) 
      IF (  RZ .GT. 93  )  RB = RB + 0.15 * (RZ - 93.0) 
      IF (  RZ .GT. 95  )  RB = RB - 0.25 * (RZ - 95.0) 
      get_BFTF = RB - get_U_SHELL_EO_exp(IZ,IA) 
      CASE( 4) 
!     ' like case 3 but without Z correction 
!     ' This is the direct description from the topographic theorem. 
      get_BFTF = RB - get_U_SHELL_exp(IZ,IA) 
      Case DEFAULT 
!     '         Print "Undefined option in BFTF" 
!     '         Sleep 
      End Select 

end


!*****************************************
!*              Get BFTFA                *
!*****************************************
REAL*4 function get_BFTFA(RZ,RA,I_Switch)

      IMPLICIT NONE
      REAL*4 RZ
      REAL*4 RA
      INTEGER*4 I_Switch
!     /' inner barrier height '/ 
      REAL*4  EA,BF0,Z4A,Z3A,DB
      REAL*4  coeff
      DATA coeff/0.5/
      REAL*4 BFTF 
      BF0 = get_BFTF(RZ,RA,I_Switch) 
!     ' Z4A = RZ^4 / RA 
!     '  EB - EA from fit to Smirenkin barriers: 
!     '  V. M. Kupriyanov,K. K. Istekov,B. I. Fursov,G. N. Smirenkin 
!     '  Sov. J. Nucl. Phys. 32 (1980) 184 
!     '  DB = -10.3517 + 1.6027E-5 * Z4A + 5.4945E-11 * Z4A^2  ' EA - EB 
!     ' 
!     '  EB - EA from fit to data from Dahlinger et al. (KHS,21. Dec. 2012) 
      Z3A = RZ**3 / RA 
      DB = -(5.40101 - 0.00666175*Z3A + 1.52531E-6*Z3A**2) 
      IF (  DB .GT. 0.0  ) THEN 
      EA = BF0 - DB 
      Else 
      EA = BF0 
      End If 
      get_BFTFA = EA 

end


!*****************************************
!*             Get BFTFB                 *
!*****************************************
REAL*4 function get_BFTFB(RZ,RA,I_Switch)

      IMPLICIT NONE
      REAL*4 RZ
      REAL*4 RA
      INTEGER*4 I_Switch
!     /' outer barrier height '/ 
      REAL*4  EB,BF0,Z4A,Z3A,DB
      REAL*4  coeff
      DATA coeff/0.5/
      REAL*4 BFTF 
      BF0 = get_BFTF(RZ,RA,I_Switch) 
!     ' Z4A = RZ^4 / RA 
!     '  EB - EA from fit to Smirenkin barriers: 
!     '  V. M. Kupriyanov,K. K. Istekov,B. I. Fursov,G. N. Smirenkin 
!     '  Sov. J. Nucl. Phys. 32 (1980) 184 
!     '   DB = -10.3517 + 1.6027E-5 * Z4A + 5.4945E-11 * Z4A^2  ' EA - EB 
!     ' 
!     '  EB - EA from fit to data from Dahlinger et al. (KHS,21. Dec. 2012) 
      Z3A = RZ**3 / RA 
      DB = -(5.40101 - 0.00666175*Z3A + 1.52531E-6*Z3A**2) 
      IF (  DB .LT. 0.0  ) THEN 
      EB = BF0 + DB 
      Else 
      EB = BF0 
      End If 
      get_BFTFB = EB 

end


!*****************************************
!*           Get Gaussintegral           *
!*****************************************
REAL*4 function get_Gaussintegral(R_x,R_sigma)

      IMPLICIT NONE
      REAL*4 R_x
      REAL*4 R_sigma
!     /' Smoothed step function. Grows from 0 to 1 around R_x 
!     with a Gauss-integral function with given sigma'/ 
      REAL*4  R_ret
!     ' Note: The variable R_sigma = standard deviation / sqr(2) ! 
      REAL*4 ERF 
      R_ret = 0.5E0 + 0.5E0 * Erf(R_x / R_sigma) 
      get_Gaussintegral = R_ret 

end


!*****************************************
!*              Get U Box                *
!*****************************************
REAL*4 function get_U_Box(x,sigma,length)

      IMPLICIT NONE
      REAL*4 x
      REAL*4 sigma
      REAL*4 length
      REAL*4  y
!     ' Note: The variable sigma = standard deviation / sqr(2) ! 
      REAL*4 GAUSSINTEGRAL 
      y = get_Gaussintegral(x+0.5*length,sigma) - get_Gaussintegral(x-0.5*length,sigma) 
      get_U_Box = y/length 

end


!*****************************************
!*              Get U Box2               *
!*****************************************
REAL*4 function get_U_Box2(x,sigma1,sigma2,length)

      IMPLICIT NONE
      REAL*4 x
      REAL*4 sigma1
      REAL*4 sigma2
      REAL*4 length
      REAL*4  y
!     ' Note: The variable sigma = standard deviation / sqr(2) ! 
      REAL*4 GAUSSINTEGRAL 
      y = get_Gaussintegral(x+0.5*length,sigma2) - get_Gaussintegral(x-0.5*length,sigma1) 
      get_U_Box2 = y/length 

end


!*****************************************
!*               U Gauss                 *
!*****************************************
REAL*4 function U_Gauss(x,sigma)

      IMPLICIT NONE
      REAL*4 x
      REAL*4 sigma
      REAL*4  y
      REAL*4  pi
      PARAMETER (pi=3.14159)
!     ' 
      y = 1.0 / (SQRT(2.0 * pi) * sigma) * exp(-x**2/ ( 2.0 * sigma**2 ) ) 
      U_Gauss = y 

end


!*****************************************
!*            Get U Gauss Mod            *
!*****************************************
REAL*4 function get_U_Gauss_mod(x,sigma)

      IMPLICIT NONE
      REAL*4 x
      REAL*4 sigma
!     ' Gaussian with Sheppard correction 
      REAL*4  y
      REAL*4  sigma_mod
      REAL*4  pi
      PARAMETER (pi=3.14159)
      sigma_mod = SQRT(sigma**2 + 1./12.) 
!     ' 
      y = 1.0 / (SQRT(2.0 * pi) * sigma_mod) * exp(-x**2/ ( 2.0 * sigma_mod**2 ) ) 
      get_U_Gauss_mod = y 

end


!*****************************************
!*            Get U LinGauss             *
!*****************************************
REAL*4 function get_U_LinGauss(x,R_Sigma)

      IMPLICIT NONE
      REAL*4 x
      REAL*4 R_Sigma
!     /' Gaussian times a linear function '/ 
!     /' Not normalized! '/ 
      REAL*4  R_Res
      IF (  R_Sigma .GT. 0.0  ) THEN 
      R_Res = x * exp(-x**2/(2.0 * R_Sigma**2)) 
      Else 
      R_Res = 0.0 
      End If 
      get_U_LinGauss = R_Res 

end


!*****************************************
!*             Get ShellMO               *
!*****************************************
! Taken from ShellMO.FOR
REAL*4 function get_ShellMO(N,Z)

      IMPLICIT NONE
      SAVE
      INTEGER*4 N
      INTEGER*4 Z
      INTEGER*4 I
      INTEGER*4 NI
      INTEGER*4 ZI
      REAL*4 R
      INTEGER*4 IFIRST
      DATA IFIRST /1/
      REAL*4, DIMENSION(0:203,0:136) :: ShellMOtab
      If (IFIRST.EQ.1) THEN
        print*,'get_ShellMO(N,Z), was here  ONCE', SHellMOtab(N,Z)
        OPEN (UNIT = 1, FILE = 'BEldmTF.dat', STATUS = 'OLD', ACTION = 'READ')
        DO I = 1, 8270, 1      
           READ (1,*)  NI, ZI, R
           ShellMOtab(NI,ZI) = R
        END DO   
        CLOSE (UNIT = 1)
        IFIRST = 0
      END IF 
      get_ShellMO = SHellMOtab(N,Z)  

end


end module GEFSUB_FOR
!===========================================


!*******************************************
!*  Declarataion of the GEFSUBdcl2 Module  *
!*******************************************
! Taken from GEFSUBdcl2.FOR
module GEFSUBdcl2

!     Output of FBtoFO from GEFSUB.BAS
      REAL*4 Getyield
      REAL*4 Masscurv
      REAL*4 Masscurv1
!     ' 
      REAL*4 De_Saddle_Scission
!     ' 
      REAL*4 TEgidy
!     ' 
      REAL*4 TRusanov
!     ' 
      REAL*4 LyMass
!     ' 
      REAL*4 LyPair
!     ' 
      REAL*4 TFPair
!     ' 
      REAL*4 Pmass
!     ' 
      REAL*4 FEDEFOLys
!     ' 
      REAL*4 FEDEFOP
!     ' 
      REAL*4 LDMass
!     ' 
      REAL*4 AME2012
!     ' 
      REAL*4 U_SHELL
!     ' 
      REAL*4 U_SHELL_exp
!     ' 
      REAL*4 U_SHELL_EO_exp
!     ' 
      REAL*4 U_MASS
!     ' 
      REAL*4 ECOUL
!     ' 
      REAL*4 beta_light
!     ' 
      REAL*4 beta_heavy
!     ' 
      REAL*4 Z_equi
!     ' 
!     ' 
!     ' 
      REAL*4 U_Ired
!     ' 
      REAL*4 U_IredFF
!     ' 
      REAL*4 U_I_Shell
!     ' 
      REAL*4 U_alev_ld
!     ' 
      REAL*4 U_Temp
!     ' 
      REAL*4 U_Even_Odd
!     ' 
      REAL*4 BFTF
      REAL*4 BFTFA
      REAL*4 BFTFB
!     ' 
      REAL*4 Gaussintegral
!     ' 
!     /' Utility functions '/ 
!     ' 
!     ' 
      REAL*4 U_Box
      REAL*4 U_Box2
      REAL*4 U_Gauss
      REAL*4 U_Gauss_mod
      REAL*4 U_LinGauss
!     ' 
      INTEGER*4 U_Valid
!     ' 
!     ' 
      REAL*4 U_Delta_S0
!     ' 
!     ' 
!     ' 
!     /' Internal variables '/ 
      REAL*4  pi
      PARAMETER (pi=3.14159)
      INTEGER*4  I_N_CN
!     /' Neutron number of fissioning nucleus '/ 
      INTEGER*4  I,J,K
      REAL*4  T_coll_Mode_1,T_coll_Mode_2,T_coll_Mode_3,T_coll_Mode_4
      REAL*4  T_asym_Mode_1,T_asym_Mode_2,T_asym_Mode_3,T_asym_Mode_4,T_asym_Mode_0
      REAL*4  Sigpol_Mode_1,Sigpol_Mode_2,Sigpol_Mode_3,Sigpol_Mode_4
      REAL*4  R_Z_Curv_S0,R_Z_Curv1_S0,R_A_Curv1_S0
      REAL*4  ZC_Mode_0,ZC_Mode_1,ZC_Mode_2,ZC_Mode_3,ZC_Mode_4,ZC_Mode_4L
      REAL*4  SigZ_Mode_0,SigZ_Mode_1,SigZ_Mode_2,SigZ_Mode_3,SigZ_Mode_4
      REAL*4  SN
      REAL*4  E_exc_S0_prov,E_exc_S1_prov,E_exc_S2_prov,E_exc_S3_prov,E_exc_S4_prov
      REAL*4  E_exc_S11_prov,E_exc_S22_prov
      REAL*4  E_exc_Barr
      REAL*4  E_LD_S1,E_LD_S2,E_LD_S3,E_LD_S4
      REAL*4  R_Shell_S1_eff,R_Shell_S2_eff,R_Shell_S3_eff,R_Shell_S4_eff
      REAL*4  Yield_Norm
      REAL*4  R_E_exc_eff
      REAL*4  R_Z_Heavy,R_Z_Light
      INTEGER*4  I_Mode
      REAL*4  T_Pol_Mode_0,T_Pol_Mode_1,T_Pol_Mode_2,T_Pol_Mode_3,T_Pol_Mode_4
      REAL*4  E_Min_Barr
      REAL*4  RI
      REAL*4  rbeta,beta1,beta2
      REAL*4  rbeta_ld,rbeta_shell
      REAL*4  ZUCD
      REAL*4  Z
      REAL*4  E_tunn
      REAL*4  beta1_opt,beta2_opt,beta1_prev,beta2_prev
      REAL*4  Z1,Z2
      INTEGER*4  IZ1,IN1,IZ2,IN2
      REAL*4  A1,A2
      INTEGER*4  IA1,IA2
      REAL*4  E_defo
      REAL*4  R_Pol_Curv_S0,R_Pol_Curv_S1,R_Pol_Curv_S2,R_Pol_Curv_S3,R_Pol_Curv_S4
      REAL*4  RA,RZ
      REAL*4  SigA_Mode_0,SigA_Mode_1,SigA_Mode_2,SigA_Mode_3,SigA_Mode_4
      REAL*4  AC_Mode_0,AC_Mode_1,AC_Mode_2,AC_Mode_3,AC_Mode_4
      REAL*4  R_A_heavy,R_A_light
      REAL*4  RZpol
      REAL*4  T_intr_Mode_0,T_intr_Mode_1_heavy,T_intr_Mode_1_light
      REAL*4  T_intr_Mode_2_heavy,T_intr_Mode_2_light
      REAL*4  T_intr_Mode_3_heavy,T_intr_Mode_3_light
      REAL*4  T_intr_Mode_4_heavy,T_intr_Mode_4_light
      REAL*4  T
      REAL*4  DU0,DU1,DU2,DU3,DU4
      REAL*4  T_low_S1_used
      REAL*4  SigA_Mode_11,SigA_Mode_22
      INTEGER*4  Ngtot
      DATA Ngtot/0/
      INTEGER*4  Nglight
      DATA Nglight/0/
      INTEGER*4  Ngheavy
      DATA Ngheavy/0/
      REAL*4  Egtot1000
      DATA Egtot1000/0/
      REAL*4  S1_enhance,S2_enhance
      REAL*4  DZ_S2_lowE
      DATA DZ_S2_lowE/0/
      INTEGER*4  I_A_CN,I_Z_CN
      REAL*4  P_I_rms_CN
!     /' rms initial angular momentum '/ 
      DATA P_I_rms_CN/0/
!     ' 
!     ' Model parameters of GEF 
!     ' 
      REAL*4  Emax_valid
!     /' Maximum allowed excitation energy '/ 
      DATA Emax_valid/100/
      REAL*4  xP_DZ_Mean_S1
      DATA xP_DZ_Mean_S1/0.05/
      REAL*4  xP_DZ_Mean_S2
      DATA xP_DZ_Mean_S2/-1.0/
      REAL*4  xP_DZ_Mean_S3
!     /' Shift of mean Z of Mode 3 '/ 
      DATA xP_DZ_Mean_S3/0/
      REAL*4  xP_DZ_Mean_S4
!     /' Shell for structure at A around 190 '/ 
      DATA xP_DZ_Mean_S4/0/
      REAL*4  xP_Z_Curv_S1
      DATA xP_Z_Curv_S1/0.30/
      REAL*4  P_Z_Curvmod_S1
!     /' Scales energy-dependent shift '/ 
      DATA P_Z_Curvmod_S1/1.75/
      REAL*4  xP_Z_CurV_S2
      DATA xP_Z_CurV_S2/0.095/
      REAL*4  S2leftmod
!     /' Asymmetry in diffuseness of S2 mass peak '/ 
      DATA S2leftmod/0.55/
      REAL*4  S2leftmod_global
      DATA S2leftmod_global/0.6/
      REAL*4  P_Z_Curvmod_S2
!     /' Scales energy-dependent shift '/ 
      DATA P_Z_Curvmod_S2/10/
      REAL*4  xP_A_Width_S2
!     /' A width of Mode 2 (box) '/ 
      DATA xP_A_Width_S2/14.5/
!     '  Dim Shared As Single _P_Z_Curv_S3 = 0.076 
      REAL*4  xP_Z_Curv_S3
      DATA xP_Z_Curv_S3/0.068/
      REAL*4  P_Z_Curvmod_S3
!     /' Scales energy-dependent shift '/ 
      DATA P_Z_Curvmod_S3/10/
      REAL*4  P_Z_Curv_SL4
      DATA P_Z_Curv_SL4/0.28/
!     '  Dim Shared As Single _P_Z_Curv_S4 = 0.025  /' Curvature in Z of Mode 4 '/ 
      REAL*4  xP_Z_Curv_S4
      DATA xP_Z_Curv_S4/0.05/
      REAL*4  P_Z_Curvmod_S4
!     /' Scales energy-dependent shift '/ 
      DATA P_Z_Curvmod_S4/10/
      REAL*4  xDelta_S0
!     /' Shell effect for SL,for individual systems '/ 
      DATA xDelta_S0/0/
      REAL*4  xP_Shell_S1
!     /' Shell effect for Mode 1 '/ 
      DATA xP_Shell_S1/-1.85/
      REAL*4  xP_Shell_S2
!     /' Shell effect for Mode 2 '/ 
      DATA xP_Shell_S2/-4.0/
      REAL*4  xP_Shell_S3
!     /' Shell effect for Mode 3 '/ 
      DATA xP_Shell_S3/-6.0/
      REAL*4  P_Shell_SL4
!     /' Shell enhancing S1 '/ 
      DATA P_Shell_SL4/-1.3/
      REAL*4  xP_Shell_S4
!     /' Shell effect for Mode 4 '/ 
      DATA xP_Shell_S4/-1.0/
      REAL*4  PZ_S3_olap_pos
!     /' Pos. of S3 shell in light fragment (in Z) '/ 
      DATA PZ_S3_olap_pos/39.7/
      REAL*4  PZ_S3_olap_curv
!     /' for width of S3 shell in light fragment '/ 
      DATA PZ_S3_olap_curv/0.008/
      REAL*4  Level_S11
!     /' Level for mode S11 '/ 
      DATA Level_S11/-1.3/
      REAL*4  Shell_fading
!     /' fading of shell effect with E* '/ 
      DATA Shell_fading/50/
      REAL*4  xT_low_S1
      DATA xT_low_S1/0.342/
      REAL*4  xT_low_S2
!     /' Slope parameter for tunneling '/ 
      DATA xT_low_S2/0.31/
      REAL*4  xT_low_S3
!     /' Slope parameter for tunneling '/ 
      DATA xT_low_S3/0.31/
      REAL*4  xT_low_S4
!     /' Slope parameter for tunneling '/ 
      DATA xT_low_S4/0.31/
      REAL*4  xT_low_SL
!     /' Slope parameter for tunneling '/ 
      DATA xT_low_SL/0.31/
      REAL*4  T_low_S11
!     /' Slope parameter for tunneling '/ 
      DATA T_low_S11/0.36/
      REAL*4  xP_att_pol
!     /' Attenuation of 132Sn shell '/ 
      DATA xP_att_pol/4.5/
      REAL*4  dE_Defo_S1
!     /' Deformation energy expense for Mode 1 '/ 
      DATA dE_Defo_S1/-2.8/
      REAL*4  dE_Defo_S2
!     /' Deformation energy expense for Mode 2 '/ 
      DATA dE_Defo_S2/0/
      REAL*4  dE_Defo_S3
!     /' Deformation energy expense for Mode 3 '/ 
      DATA dE_Defo_S3/0/
      REAL*4  dE_Defo_S4
      DATA dE_Defo_S4/0/
!     /' Deformation energy expense for Mode 4 '/ 
!     '    Dim Shared As Single betaL0 = 24.5 
!     '    Dim Shared As Single betaL1 = 0.65 
!     '    Dim Shared As Single betaL0 = 26.7 
!     '    Dim Shared As Single betaL1 = 0.71 
      REAL*4  betaL0
      DATA betaL0/25.4/
      REAL*4  betaL1
      DATA betaL1/0.69/
      REAL*4  betaH0
!     /' Offset for deformation of heavy fragment '/ 
      DATA betaH0/48.0/
      REAL*4  betaH1
      DATA betaH1/0.55/
      REAL*4  kappa
!     /' N/Z dedendence of A-asym. potential '/ 
      DATA kappa/0/
      REAL*4  TCOLLFRAC
!     /' Tcoll per energy gain from saddle to scission '/ 
      DATA TCOLLFRAC/0.04/
      REAL*4  ECOLLFRAC
      DATA ECOLLFRAC/0.055/
!     'Dim Shared As Single ECOLLFRAC = 0.1 
      REAL*4  TFCOLL
      DATA TFCOLL/0.034/
      REAL*4  TCOLLMIN
      DATA TCOLLMIN/0.12/
      REAL*4  ESHIFTSASCI_intr
!     /' Shift of saddle-scission energy '/ 
      DATA ESHIFTSASCI_intr/-58/
      REAL*4  ESHIFTSASCI_coll
!     /' Shift of saddle-scission energy '/ 
      DATA ESHIFTSASCI_coll/-90/
      REAL*4  ESHIFTSASCI_coll_global
      DATA ESHIFTSASCI_coll_global/-90/
      REAL*4  EDISSFRAC
      DATA EDISSFRAC/0.35/
      REAL*4  SIGDEFO
      DATA SIGDEFO/0.165/
      REAL*4  SIGDEFO_0
      DATA SIGDEFO_0/0.165/
      REAL*4  SIGDEFO_slope
      DATA SIGDEFO_slope/0/
      REAL*4  EexcSIGrel
      DATA EexcSIGrel/0.7/
      REAL*4  DNECK
!     /' Tip distance at scission / fm '/ 
      DATA DNECK/1/
      REAL*4  FTRUNC50
!     /' Truncation near Z=50 '/ 
      DATA FTRUNC50/1/
      REAL*4  ZTRUNC50
!     /' Z value for truncation '/ 
      DATA ZTRUNC50/50/
      REAL*4  FTRUNC28
!     /' Truncation near Z=28 '/ 
      DATA FTRUNC28/0.56/
      REAL*4  ZTRUNC28
!     /' Z value for truncation '/ 
      DATA ZTRUNC28/30.5/
      REAL*4  ZMAX_S2
!     /' Maximum Z of S2 channel in light fragment '/ 
      DATA ZMAX_S2/60/
      REAL*4  NTRANSFEREO
!     /' Steps for E sorting for even-odd effect '/ 
      DATA NTRANSFEREO/6/
      REAL*4  NTRANSFERE
!     /' Steps for E sorting for energy division '/ 
      DATA NTRANSFERE/12/
      REAL*4  Csort
!     /' Smoothing of energy sorting '/ 
      DATA Csort/0.1/
      REAL*4  PZ_EO_symm
!     /' Even-odd effect in Z at symmetry '/ 
      DATA PZ_EO_symm/2.25/
      REAL*4  PN_EO_Symm
!     /' Even-odd effect in N at symmetry '/ 
      DATA PN_EO_Symm/0.5/
      REAL*4  R_EO_THRESH
!     /' Threshold for asymmetry-driven even-odd effect'/ 
      DATA R_EO_THRESH/0.04/
      REAL*4  R_EO_SIGMA
      DATA R_EO_SIGMA/0.35/
      REAL*4  R_EO_MAX
!     /' Maximum even-odd effect '/ 
      DATA R_EO_MAX/0.40/
      REAL*4  xPOLARadd
!     /' Offset for enhanced polarization '/ 
      DATA xPOLARadd/0.32/
      REAL*4  POLARfac
!     /' Enhancement of polarization of ligu. drop '/ 
      DATA POLARfac/1/
      REAL*4  T_POL_RED
!     /' Reduction of temperature for sigma(Z) '/ 
      DATA T_POL_RED/0.01/
      REAL*4  xHOMPOL
!     /' hbar omega of polarization oscillation '/ 
      DATA xHOMPOL/2.0/
      REAL*4  ZPOL1
!     /' Extra charge polarization of S1 '/ 
      DATA ZPOL1/0/
      REAL*4  P_n_x
!     /' Enhanced inverse neutron x section '/ 
      DATA P_n_x/0/
      REAL*4  Tscale
      DATA Tscale/1/
      REAL*4  EOscale
!     /' Scaling factor for even-odd structure in yields '/ 
      DATA EOscale/1.0/
      REAL*4  Econd
      DATA Econd/2/
      INTEGER*4  Emode
!     /' 0: E over BF_B; 1: E over gs; 2: E_neutron '/ 
      DATA Emode/1/
      REAL*4  T_orbital
!     /' From orbital ang. momentum '/ 
      DATA T_orbital/0/
      REAL*4  Jscaling
!     /' General scaling of fragment angular momenta '/ 
      DATA Jscaling/1.0/
      REAL*4  Spin_odd
!     /' RMS Spin enhancement for odd Z '/ 
      DATA Spin_odd/0.4/
!     ' 
!     /' I. Properties of nuclide distributions '/ 
!     ' 
      REAL*4 , DIMENSION(0:6,2,150) :: Beta
!     ' 
      REAL*4 , DIMENSION(0:4,2,150) :: Edefo
!     ' 
      REAL*4 , DIMENSION(0:4,2,350) :: Zmean
!     ' 
      REAL*4 , DIMENSION(0:4,2,350) :: Zshift
!     ' 
      REAL*4 , DIMENSION(0:4,2,350) :: Temp
!     ' 
      REAL*4 , DIMENSION(0:4,2,350) :: TempFF
!     ' 
      REAL*4 , DIMENSION(0:4,2,350) :: Eshell
!     ' 
      REAL*4 , DIMENSION(0:6,2,350) :: PEOZ
!     ' 
      REAL*4 , DIMENSION(0:6,2,350) :: PEON
!     ' pre-neutron evaporation 
!     ' 
      REAL*4 , DIMENSION(0:6,2,350) :: EPART
!     ' 
      REAL*4 , DIMENSION(0:6,2,1:200,1:150) :: SpinRMSNZ
!     ' 
!     ' 
!     /' Masses etc. '/ 
!     ' 
      REAL*4 , DIMENSION(0:203,0:136) :: BEldmTF
!     ' 
      REAL*4 , DIMENSION(0:203,0:136) :: BEexp
!     ' 
      REAL*4 , DIMENSION(0:203,0:136) :: DEFOtab
!     ' 
      REAL*4 , DIMENSION(0:203,0:136) :: ShellMO
!     ' 
      REAL*4 , DIMENSION(0:203,0:136) :: EVOD
!     ' 
!     ' 
      REAL*4 , DIMENSION(0:200,0:150) :: NZPRE
!     ' 
      REAL*4 , DIMENSION(0:6,0:200,0:150) :: NZMPRE
!     /' Internal parameters for error analysis: '/ 
      REAL*4  P_DZ_Mean_S1
      REAL*4  P_DZ_Mean_S2
      REAL*4  P_DZ_Mean_S3
      REAL*4  P_DZ_Mean_S4
      REAL*4  P_Z_Curv_S1
      REAL*4  P_Z_Curv_S2
      REAL*4  P_A_Width_S2
      REAL*4  P_Z_Curv_S3
      REAL*4  P_Z_Curv_S4
      REAL*4  Delta_S0
      REAL*4  P_Shell_S1
      REAL*4  P_Shell_S2
      REAL*4  P_Shell_S3
      REAL*4  P_Shell_S4
      REAL*4  T_low_S1
      REAL*4  T_low_S2
      REAL*4  T_low_S3
      REAL*4  T_low_S4
      REAL*4  T_low_SL
      REAL*4  P_att_pol
      REAL*4  HOMPOL
      REAL*4  POLARadd
!     ' 
!     ' 
!     /' Control parameters: '/ 
      REAL*4  B_F
!     /' Fission barrier '/ 
      DATA B_F/0/
      REAL*4  B_F_ld
!     /' Fission barrier,liquid drop '/ 
      DATA B_F_ld/0/
      REAL*4  E_B
!     /' Outer fission barrier '/ 
      DATA E_B/0/
      REAL*4  E_B_ld
!     /' Outer fission barrier,liquid drop '/ 
      DATA E_B_ld/0/
      REAL*4  R_E_exc_Eb
!     /' Energy above outer barrier '/ 
      DATA R_E_exc_Eb/0/
      REAL*4  R_E_exc_GS
!     /' Energy above ground state '/ 
      DATA R_E_exc_GS/0/
      REAL*4  P_Z_Mean_S0
!     /' Mean Z of Mode 1 '/ 
      DATA P_Z_Mean_S0/0/
      REAL*4  P_Z_Mean_S1
!     /' Mean Z of Mode 1 '/ 
      DATA P_Z_Mean_S1/52.8/
      REAL*4  P_Z_Mean_S2
!     /' Mean Z of Mode 2 '/ 
      DATA P_Z_Mean_S2/55/
      REAL*4  P_Z_Mean_S3
!     /' Mean Z of Mode 3 '/ 
      DATA P_Z_Mean_S3/65/
      REAL*4  P_Z_Mean_S4
!     /' Mean Z of Mode 4 '/ 
      DATA P_Z_Mean_S4/42.05/
      REAL*4  NC_Mode_0
!     /' Mean N of symm. Mode '/ 
      DATA NC_Mode_0/0/
      REAL*4  NC_Mode_1
!     /' Mean N of Mode 1 '/ 
      DATA NC_Mode_1/0/
      REAL*4  NC_Mode_2
!     /' Mean N of Mode 2 '/ 
      DATA NC_Mode_2/0/
      REAL*4  NC_Mode_3
!     /' Mean N of Mode 3 '/ 
      DATA NC_Mode_3/0/
      REAL*4  NC_Mode_4
      DATA NC_Mode_4/0/
      REAL*4  B_S1
!     /' Barrier S1,relative to SL '/ 
      DATA B_S1/0/
      REAL*4  B_S2
!     /' Barrier S2,relative to SL '/ 
      DATA B_S2/0/
      REAL*4  B_S3
!     /' Barrier S3,relative to SL '/ 
      DATA B_S3/0/
      REAL*4  B_S4
      DATA B_S4/0/
      REAL*4  B_S11
!     /' Barrier S11,relative to SL '/ 
      DATA B_S11/0/
      REAL*4  B_S22
!     /' Barrier S22,relative to SL '/ 
      DATA B_S22/0/
      REAL*4  DES11ZPM
!     /' Mod. of eff. barrier due to ZPM in overlap '/ 
      DATA DES11ZPM/0/
      REAL*4  Delta_NZ_Pol
!     /' Polarization for 132Sn '/ 
      DATA Delta_NZ_Pol/0/
      REAL*4  Yield_Mode_0
!     /' Relative yield of SL '/ 
      DATA Yield_Mode_0/0/
      REAL*4  Yield_Mode_1
!     /' Relative yield of S1 '/ 
      DATA Yield_Mode_1/0/
      REAL*4  Yield_Mode_2
!     /' Relative yield of S2 '/ 
      DATA Yield_Mode_2/0/
      REAL*4  Yield_Mode_3
!     /' Relative yield of S3 '/ 
      DATA Yield_Mode_3/0/
      REAL*4  Yield_Mode_4
!     /' Relative yield of S4 '/ 
      DATA Yield_Mode_4/0/
      REAL*4  Yield_Mode_11
!     /' Relative yield of S11 '/ 
      DATA Yield_Mode_11/0/
      REAL*4  Yield_Mode_22
!     /' Relative yield of S22 '/ 
      DATA Yield_Mode_22/0/
      REAL*4  P_POL_CURV_S0
!     /' Stiffnes in N/Z '/ 
      DATA P_POL_CURV_S0/0/
      REAL*4  T_Coll_Mode_0
!     /' Effective collective temperature '/ 
      DATA T_Coll_Mode_0/0/
      REAL*4  E_Exc_S0
!     /' Energy over barrier of symmetric channel '/ 
      DATA E_Exc_S0/0/
      REAL*4  E_Exc_S1
!     /' Energy over barrier of S1 channel '/ 
      DATA E_Exc_S1/0/
      REAL*4  E_Exc_S2
!     /' Energy over barrier of S2 channel '/ 
      DATA E_Exc_S2/0/
      REAL*4  E_Exc_S3
!     /' Energy over barrier of S3 channel '/ 
      DATA E_Exc_S3/0/
      REAL*4  E_Exc_S4
!     /' Energy over barrier of S4 channel '/ 
      DATA E_Exc_S4/0/
      REAL*4  E_Exc_S11
!     /' Energy over barrier of S11 channel '/ 
      DATA E_Exc_S11/0/
      REAL*4  E_Exc_S22
!     /' Energy over barrier of S22 channel '/ 
      DATA E_Exc_S22/0/
      REAL*4  E_POT_SCISSION
!     /' Potential-energy gain saddle-scission '/ 
      DATA E_POT_SCISSION/0/
      REAL*4  EINTR_SCISSION
!     /' Intrinsic excitation energy at scission '/ 
      DATA EINTR_SCISSION/0/
      REAL*4  EeffS2
!     /' Governs S1 reduction by pairing '/ 
      DATA EeffS2/0/
      REAL*4  Sigpol_Mode_0
!     /' Width of isobaric Z distribution '/ 
      DATA Sigpol_Mode_0/0/
!     ' 
!     '  #Include Once "BEldmTF.bas" 
!     ' 
!     '  #Include Once "BEexp.bas" 
!     ' 
!     '  #Include Once "DEFO.bas" 
!     ' 
!     '  #Include Once "ShellMO.bas" 
!     ' 
!     ' 
!     ' 
!     ' 
!     ' 
      INTEGER*4  I_E_iso
!     ' numbered in sequence of increasing energy 
      REAL*4  Spin_CN
      REAL*4  Spin_pre_fission
      REAL*4  Spin_gs_light
      REAL*4  Spin_gs_heavy
!     ' 
!     /' Shell effects for the symmetric fission channel '/ 
      REAL*4  R_E_exc_used
      REAL*4  R_Z_mod
      REAL*4  T_Rusanov
      REAL*4  R_E_intr_S1,R_E_intr_S2,R_E_intr_S3
!     ' intrinsic exc. energies at barrier 
      REAL*4  R_E_intr_S4
      REAL*4 , DIMENSION(6) :: R_Att
!     ' attenuation of shell 
      REAL*4 , DIMENSION(6) :: R_Att_Sad
!     '  Dim As Single E_backshift 
!     '  E_backshift = -3 
!     ' 
!     ' 
      REAL*4  DZ_S1,DZ_S2,DZ_S3,DZ_S4
      REAL*4  EtotS2
      REAL*4  P_Z_Curv_S1_eff
      REAL*4  P_Z_Curv_S2_eff
      REAL*4  P_Z_Curv_S3_eff
      REAL*4  P_Z_Curv_S4_eff
      REAL*4  Etot,E1FG,E1ES
      REAL*4  Rincr1P,Rincr1N,Rincr2,Rincr2P,Rincr2N
      REAL*4  T1,T2,E1,E2
      REAL*4 , DIMENSION(0:6) :: E_coll_saddle
      REAL*4  Ediff
!     ' 
!     ' 
      REAL*4  DT
!     ' 
      REAL*4  AUCD
!     /' UCD fragment mass '/ 
      REAL*4  I_rigid_spher
!     /' I rigid for spherical shape '/ 
      REAL*4  I_rigid
!     /' I rigid for deformed scission shape '/ 
      REAL*4  I_eff
!     /' I with reduction due to pairing '/ 
      REAL*4  alph
!     /' deformation parameter '/ 
      REAL*4  E_exc
!     /' Excitation energy '/ 
      REAL*4  J_rms
!     /' rms angular momentum '/ 
      INTEGER*4  Ic,Jc
      REAL*4  R_Help,Zs,R_Sum
!     ' 
      REAL*4  R_Cut1,R_Cut2
      INTEGER*4  N_index,Z_index,A_index,M_index
      REAL*4  Ymin
!     ' Minimum yield to be stored 
      DATA Ymin/1.E-7/
      REAL*4  Eexc_mean,Eexc_sigma
      REAL*4  Eexc_intr,Eexc_coll
!     ' 
!      INCLUDE "RESULTS.FOR"      
!     ' 
!     ' 
      REAL*4  Rint
      REAL*4  RS
      REAL*4  RintE
      REAL*4 F1
      REAL*4 F2

end module GEFSUBdcl2


!*******************************************
!*   Declarataion of the RESULTS Module    *
!*******************************************
! Taken from RESULTS.FOR
module RESULTS

      INTEGER*4  N_cases
!     ' Number of cases in NZMkey,Etab,Jtab and Ytab 
!     ' (First dimension of NZMkey,Etab,Jtab and Ytab) 
      INTEGER*4 , DIMENSION(10000,3) :: NZMkey
!     ' Key (Mode,N,Z) for E*,spin and yield distr. of fragments 
      REAL*4 , DIMENSION(10000,1000) :: Etab
!     ' Excitation-energy distribution of fragments (0.1 MeV bins) 
      REAL*4 , DIMENSION(10000,100) :: Jtab
!     ' Spin distribution of fragments 
!     ' (0 to 100 hbar for even-A or 1/2 to 201/2 hbar for odd-A nuclei) 
      REAL*4 , DIMENSION(10000) :: Ytab

end module RESULTS


!*****************************************
!*       Init Function for Python        *
!*****************************************
!subroutine init_decay_module()

!    integer iii
!    iii = 1

!end


!*****************************************
!*              Initialize               *
!*****************************************
subroutine initialize(f_network, f_format, f_path)

    use iso
    use path_module
    implicit none

    character*256    f_network
    integer          f_format
    character*256    f_path
    character*10     re_import_keyphrase

    ! Make sure not to reimport the module if
    ! already imported and initialized.  This is to
    ! solve a problem when calling NuPyCEE multiple 
    ! times while it import the decay module each time.
    if (re_import_test .ne. 'keyphrase') then

        re_import_test = 'keyphrase'
        files_path = f_path
        input_file_network = trim(files_path) // 'decay_data/' // f_network
        abundance_file_format = f_format

        call init_variables()
        call read_elements()
        call init_network()
!        call special_fission() ! in special_fission.f95
    !    call init_n_fission() ! in n_fission.f95

        !print*, 'in init! ', files_path, f_path, re_import_test

    end if

end subroutine initialize


!*****************************************
!*              Run Decay                *
!*****************************************
subroutine run_decay(t_decay, t_steps, input_abundance)

    use iso
    use comm
    implicit none
    double precision t_decay
    integer          t_steps
    double precision input_abundance(max_number_isotopes)
    integer isot
 
    time_decay = t_decay
    time_steps = t_steps
    dt = time_decay / time_steps

    ! Re-initialize decay arrays
    do isot=0,max_entry
        i_entry(isot) = -1
    end do
    i_entry_max = 0
    do isot=1,number_isotopes
        abundance(isot) = 0.0
        initial_abundance(isot) = 0.0
    end do

    call init_abundances(input_abundance)
    call nucleosynthesis()

end


!*****************************************
!*         Get Non-Zero Indexes          *
!*****************************************
subroutine get_non_zero_indexes()

    use iso
    use comm

    i_entry_max = 0
    do isot=1,number_isotopes
      if (initial_abundance(isot) .gt. 0.0 .or. abundance(isot) .gt. 1.d-95) then
          i_entry(i_entry_max) = isot
          i_entry_max = i_entry_max + 1
      end if
    end do

end 


!*****************************************
!*            Get Iso Names              *
!*****************************************
subroutine get_iso_names(a_return)

    use iso
    implicit none
    character*6 a_return(max_number_isotopes)
    character*10 a_number
    integer temp
    integer isot
    integer ii

!f2py intent(out) a_return
    print*, 'Write function to calculate the len of non-zero isotopes'
!    do i=1,max_number_isotopes
    ii = 1
    do isot=1,number_isotopes
      if (initial_abundance(isot) .gt. 0.d0 .or. abundance(isot) .gt. 1.d-95) then
        print*, z(isot), z(isot)+n(isot), '  ',element_names(z(isot)), &
                   level(isot), abundance(isot), initial_abundance(isot)
        temp = z(isot)+n(isot)
        print*, z(isot)+n(isot)
        print*, temp
        write(a_number, '(i10)') temp
        print*, a_number
!        write(a_number, 10) temp
! 10     format (I4)
        a_return(ii) = trim(element_names(z(isot))) // '-' // trim(a_number)
        ii = ii + 1
      end if  
    end do

    do isot=1,ii
        print*, a_return(isot)
    end do

end


!*****************************************
!*            Init Variables             *
!*****************************************
subroutine init_variables()

    use iso
    implicit none
    integer i
    double precision en

    z = 0
    n = 0
!    level = 0.
    do i=1,max_number_isotopes
      level(i) = 0.0
    end do
    reactions = 0
    decay_constant = 0.
    decay = 0.
    production = 0.
    abundance = 0.
    initial_abundance = 0.
    s_fission_vector = 0.
    do i=1,1000
      en = 0.1d0 * (i - 0.5)
      mb_distribution(i) = dsqrt(en) * dexp(-en/kt)
    end do  
    mb_distribution = mb_distribution / sum(mb_distribution)

end


!*****************************************
!*            Read Elements              *
!*****************************************
subroutine read_elements()

    use iso
    use path_module
    implicit none
    integer iostats,i
    character*256 the_path

    the_path = trim(files_path) // trim('decay_data/element_symbols.txt')

    open(unit=13,file=the_path,status='unknown')
      iostats=0
      i=0
      element_names(0) = 'NN'
      do while (iostats.eq.0)
        i=i+1
        read(13,*,iostat=iostats) element_names(i)
      end do
    close(13)

end


!*****************************************
!*            Set Init Files             *
!*****************************************
subroutine set_init_files(f_network, f_format, t_decay, t_steps)

    ! Import list of parameters
    use iso

    ! Define the arguments
    implicit none
    character*256    f_network
    integer          f_format
    double precision t_decay
    integer          t_steps

    print*, 'in set_init_files'

    ! Set the value of the parameters
    input_file_network = f_network
    abundance_file_format = f_format
    time_decay = t_decay
    time_steps = t_steps

end


!*****************************************
!*            Init Abundances            *
!*****************************************
subroutine init_abundances(input_abundance)
    use iso
    implicit none
    double precision input_abundance(max_number_isotopes)
    integer iostats, ind

!    print*, 'in init_abundances'
!    input_file_abundance = 'data/iso_mass_U238.DAT'
!    input_file_abundance = 'data/iso_mass_Fe60.DAT'

!    open(unit=13,file=input_file_abundance,status='old')
!      select case (abundance_file_format)
!        case(1)  
!          call read_nugrid_abundances()
!        case default
!          print*,abundance_file_format, ' is no valid file format'  
!          stop
!      end select
!    close(13)  
!    abundance = initial_abundance

    abundance = input_abundance

end


!*****************************************
!*        Read NuGrid Abundances         *
!*****************************************
subroutine read_nugrid_abundances()

    use iso
    implicit none
    integer iostats, ind, al, zl, nl, ll, get_isotope
    double precision levell, abundancel
    character*42 c42 

    iostats=0
    do ind=1,7
      read(13,*)
    end do
    ind=0
    do while (iostats.eq.0) 
      ind = ind+1
      read(13,'(A42)',iostat=iostats) c42
      read(c42(7:11),*) zl
      read(c42(13:16),*) al
      nl = al-zl
      read(c42(19:21),*) ll
      levell = 0
      if (zl.eq.73 .and. al.eq.180 .and.  ll.eq.1) levell = 0.0771d0  ! 180Ta, isomer
      if (zl.eq.13 .and. al.eq.26  .and.  ll.eq.2) levell = 0.2283d0  ! 26Al, isomer
      if (zl.eq.36 .and. al.eq.85  .and.  ll.eq.2) levell = 0.305d0   ! 85Kr, isomer
      if (zl.eq.48 .and. al.eq.115 .and.  ll.eq.2) levell = 0.181d0    ! 115Cd, isomer
      if (zl.eq.71 .and. al.eq.176 .and.  ll.eq.2) levell = 0.1229d0   ! 176Lu, isomer
      read(c42(23:35),*) abundancel
      current_isotope = get_isotope(zl,nl,levell)
      if (current_isotope .eq. 0) then
        print*,'isotope not found: ',ind, zl, nl, levell, c42
        stop
      else
        initial_abundance(current_isotope) = abundancel
      end if  
    end do

end


!*****************************************
!*             Init Network              *
!*****************************************
subroutine init_network()

    use iso
    implicit none
    integer isomer, reac, get_isotope, zt, nt, fission_counter
    double precision norm

    !print*, 'in init_network'

    call read_network()
    norm = 0.
    fission_counter = 0
    index_p = get_isotope(1,0,norm)
    index_n = get_isotope(0,1,norm)
    index_a = get_isotope(2,2,norm)
    index_12c = get_isotope(6,6,norm)
    do isomer = 1, number_isotopes           ! loop over all species
      if (reactions(isomer,0) .gt. 0) then   ! unstable isotope
        norm = 1. / sum(decay_constant(isomer,1:reactions(isomer,0)))
        if (norm .le. 0.) then
          print *,norm
          print *,z(isomer),n(isomer)
          stop
        end if  
        decay_constant(isomer,1:reactions(isomer,0)) =  decay_constant(isomer,1:reactions(isomer,0))*norm  ! now all decay rates normalized to 1
        do reac = 1,reactions(isomer,0)      ! loop over all decay channels
          zt = z(isomer) + reaction_vector(1,reactions(isomer,reac)) ! Z of reaction product
          nt = n(isomer) + reaction_vector(2,reactions(isomer,reac)) ! N of reaction product
          product_isomer(isomer, reac) = get_isotope(zt,nt,level_product(isomer, reac))
          if (reactions(isomer,reac) .eq. 22) then
            fission_counter = fission_counter+1
            if (fission_counter .gt. max_number_fissile_isotopes) then
              print*,'number of fissile iostopes .gt. max_number_fissile_isotopes ', max_number_fissile_isotopes
              print*,'please change in the source code and recompile'
              stop
            end if  
            reactions(isomer,-1) = fission_counter                ! index for fission yields
            call init_fission(isomer)
          end if
        end do
      end if
    end do
!    print*, 'Number of fissile isotopes: ',fission_counter
!    dt = time_decay / time_steps

end


!*****************************************
!*             Read Network              *
!*****************************************
subroutine read_network()

    use iso
    implicit none
    integer iostats,io,ind, al,zl, nl, dm
    integer character_to_decay_mode, get_isotope
    double precision level_energy, hwzl, macs, target_level,dp
    double precision character_to_decay_probability,level_spin
    double precision get_spin
    character*4 decay_mode
    character*14 decay_prob
    character*159 c159

    open(unit=13,file=input_file_network,status='unknown')
      iostats=0
      ind=0
      read(13,*)
      do while (iostats.eq.0)
        ind=ind+1
        if (ind.gt.max_number_isotopes) then
          print*,'number of species greater than ', max_number_isotopes
          print*,'change in source code and recompile'
          stop
        end if  
        read(13,'(A159)',iostat=iostats) c159
        read(c159(8:11),*) zl
        read(c159(19:22),*) al
        nl = al-zl
        read(c159(37:54),*) level_energy
        level_spin = get_spin(c159(55:71))
        read(c159(72:85),*) hwzl
        if (max_hwz.lt.hwzl) hwzl = -1
        decay_mode = c159(109:112)
        decay_prob = c159(114:127)
     
        read(c159(145:159),*) target_level
        current_isotope = get_isotope(zl,nl,level_energy)
        if ((hwzl .gt. 0.) .and. (index(decay_mode,'?') .eq. 0.)) then
          ! unstable isotope, decay mode confirmed, bound nucleus
          dp         = character_to_decay_probability(decay_prob)
          dm         = character_to_decay_mode(decay_mode)
          if (current_isotope .eq. 0) then ! new unstable species
            call new_unstable(zl,nl,level_energy,level_spin,hwzl,dm,dp, target_level)
          else
            call add_unstable(dm,dp,target_level)
          end if  
        end if  
        if ((hwzl .eq. -1) ) then  ! stable isotope
          if (current_isotope .eq. 0) then ! new stable species
            call new_stable(zl,nl,level_energy)
          else 
            print*,'Warning: stable - second time??!! Entry:',ind, trim(element_names(zl)) , zl+nl
            print*,'will ignore entry'
!            stop
          end if  
        end if  
      end do 
    close(13)

end

!*****************************************
!*             Add Unstable              *
!*****************************************
subroutine add_unstable(dm,dp, target_level)

    use iso
    implicit none
    integer dm
    double precision target_level, dp

    reactions(current_isotope,0) = reactions(current_isotope,0) + 1
    reactions(current_isotope,reactions(current_isotope,0)) = dm
    decay_constant(current_isotope,reactions(current_isotope,0)) = dp  ! to be normalized later
    level_product(current_isotope,reactions(current_isotope,0)) = target_level

end


!*****************************************
!*             New Unstable              *
!*****************************************
subroutine new_unstable(zl,nl,level_energy,level_spin,hwzl,dm,dp, target_level)
    use iso
    implicit none
    integer zl,nl,dm
    double precision level_energy,hwzl,target_level, dp,level_spin

    number_isotopes                   = number_isotopes +1
    current_isotope                   = number_isotopes
    z(current_isotope)                = zl
    n(current_isotope)                = nl
    level(current_isotope)            = level_energy
    spin(current_isotope)             = level_spin
    reactions(current_isotope,0)      = 1
    reactions(current_isotope,1)      = dm
    decay_constant(current_isotope,0) = dlog(2.d0)/hwzl  ! lambda
    decay_constant(current_isotope,1) = dp  ! to be normalized later
    level_product(current_isotope,1)  = target_level

end


!*****************************************
!*        Character to Decay Mode        *
!*****************************************
integer function character_to_decay_mode(c4)

    use iso
    implicit none
    character*4 c4
    integer i

    character_to_decay_mode = 0
    do i=3,1,-1
      if (c4(i:i) .eq. ' ') then  ! get rid of leading or interspersed spaces
        c4(i:3) = c4(i+1:4)
        c4(4:4) = ' '
      end if  
    end do  
    do i=1, number_reaction_types
!      if (index(c4,trim(reaction_types(i))) .gt. 0) then
      if (trim(c4) .eq. trim(reaction_types(i)) ) then
        character_to_decay_mode = i
        exit
      end if   
    end do
    if (character_to_decay_mode .eq. 0) then
      print*,'Reaction type ', c4 ,' not found! '
      stop
    end if  

end


!*****************************************
!*    Character to Decay Probability     *
!*****************************************
double precision function character_to_decay_probability(c14)

    implicit none
    character*14 c14
    integer io

    if ((index(c14,'?') .gt. 0 ) .or. (trim(c14).eq.'') ) then
!      print*,c14
      character_to_decay_probability = 1.
    else
      c14(index(c14,'~'):index(c14,'~')) = ' ' 
      c14(index(c14,'='):index(c14,'=')) = ' ' 
      c14(index(c14,'<'):index(c14,'<')) = ' ' 
      c14(index(c14,'>'):index(c14,'>')) = ' ' 
      read(c14,*,iostat=io) character_to_decay_probability  
      if (io.ne.0) print*,'io',c14
    end if  
    if (character_to_decay_probability .eq. 0.) character_to_decay_probability = 1.d-10

end


!*****************************************
!*              Get Isotope              *
!*****************************************
integer function get_isotope(zl,nl,level_energy)

    use iso
    implicit none
    integer zl,nl,i
    double precision level_energy

    get_isotope = 0
    do i=1, number_isotopes
      if (zl.eq.z(i) .and. nl.eq.n(i) .and. level_energy.eq.level(i)) then
        get_isotope = i
        exit
      end if   
    end do

end


!*****************************************
!*               Get Spin                *
!*****************************************
double precision function get_spin(cin)

    implicit none
    integer i,j,number_symbols,char_length
    parameter(char_length=17)
    character*17 cin
    logical get_rid
    double precision n,z
    parameter(number_symbols=13)
    character*1 symbols(number_symbols)
    data symbols/ ' ', '(', ')', '[', ']', '+','-', 'G','E','L', '<','>','|'/ 

    get_spin = 0.
    do i=char_length,1,-1
      get_rid = .false.
      do j=1,number_symbols
        get_rid = get_rid .or. (cin(i:i) .eq. symbols(j))
      end do
      if (get_rid) then  ! get rid of leading or interspersed spaces and ()+-
        cin(i:char_length-1) = cin(i+1:char_length)
        cin(char_length:char_length) = ' '
      end if  
    end do  
    if (trim(cin) .ne. '') then
      i=index(cin,'/')
      if (i.eq.0) then
        read(cin,*) get_spin
      else
        read(cin(1:i-1),*) n 
        read(cin(i+1:i+1),*) z
        get_spin = n/z
      end if 
    end if

end


!*****************************************
!*             Init Fission              *
!*****************************************
subroutine init_fission(isomer)

    use iso
    use RESULTS
    use path_module
    implicit none
    character*256 fission_file
    character*3 chz, chn
    character*11 chl, chs
    integer isomer,M_index, N_index, Z_index, A_index,k,i,get_isotope,li
    integer fission_index, istatus
    double precision yield, dytab(10000)
    REAL*4    level_e, level_spin
    INTEGER*4 iso_n, iso_z, iso_a
    character*256 the_path

    the_path = trim(files_path) // trim('decay_data/fission/')

    iso_z = z(isomer)
    iso_n = n(isomer)
    level_e    = level(isomer) / 1000. ! keV -> MeV
    level_spin = spin(isomer)
    fission_index = reactions(isomer, -1)

!    iso_z = 92
!    iso_n = 143
!    iso_n = 146
!    level_e    = 0.
!    level_spin = 0.

    iso_a = iso_z+iso_n

    write(chz,'(I3.3)') iso_z
    write(chn,'(I3.3)') iso_n
    write(chl,'(ES11.5)') level_e
    write(chs,'(ES11.5)') level_spin
    fission_file = trim(the_path) // chz // '_' // chn // '_' // chl //  '_'  // chs // '.dat'
    open(unit=13,file=fission_file,status='old', iostat=istatus)
    if (istatus .ne. 0) then                      ! need to create the fission file
      print*,'Fission of: ',iso_a,element_names(iso_z), '  ', level_e,level_spin
      print*,'Creating: ',trim(fission_file)
      call GEFSUB(iso_z,iso_a,level_e,level_spin)
      dytab = Ytab
      dytab = 2./sum(dytab(1:N_cases)) * dytab   ! normalize to 2.
      call scission(fission_index,dytab)
      print*,'Neutron yield',s_fission_vector(fission_index,index_n)
      open(unit=13,file=fission_file,status='unknown')
        do li=1,number_isotopes
          if (s_fission_vector(fission_index,li) .gt. 0.0) then
            write(13,'(2I6,ES20.10)') z(li), n(li), s_fission_vector(fission_index,li)
          end if  
        end do
    else
      do 
        read(13,*,iostat=istatus) Z_index,N_index, yield
        if (istatus .eq. 0) then
          li = get_isotope(Z_index,N_index,0.0d0)
          if (li .eq. 0) then
            print*,Z_index+N_index,element_names(Z_index)
            print*,'Isotope with fission yield not in reaction network.'
            print*,'Consider deleting and recreating fission data with current network file.'
          else  
            s_fission_vector(fission_index,li) = yield
          end if  
        else
          exit
        end if    
      end do
    end if    
!    yield = sum(s_fission_vector(fission_index,1:index_n-1)) &
!          + sum(s_fission_vector(fission_index,index_n+1:number_isotopes) )
!    print*,  iso_z+iso_n,element_names(iso_z), yield
    close(13)

end


!*****************************************
!*              New Stable               *
!*****************************************
subroutine new_stable(zl,nl,level_energy)

    use iso
    implicit none
    integer zl,nl
    double precision level_energy

    number_isotopes              = number_isotopes +1
    current_isotope              = number_isotopes
    z(current_isotope)           = zl
    n(current_isotope)           = nl
    level(current_isotope)       = level_energy
    reactions(current_isotope,0) = 0

end


!*****************************************
!*               Scission                *
!*****************************************
subroutine scission(fission_index,dytab)

    use iso
    use RESULTS 
    implicit none
    integer isomer,M_index, N_index, Z_index, A_index,k,i,j,get_isotope,li,ch_sn, n_ch_n
    integer fission_index, istatus, ch_counter, li_ne, neutron_counter
    double precision neutron_yield, e_ave, e_sum, de_ave2,neutron_separation_energy
    double precision sn, e_excitation(2,1000), e_exc, en(1000),dytab(10000)
    REAL*4    level_e, level_spin
    INTEGER*4 iso_n, iso_z, iso_a

!    INCLUDE "RESULTS.FOR" 

    DO K = 1 , N_cases    
      e_excitation(1,:) = Etab(k,:)
      e_excitation(1,:) = e_excitation(1,:) / sum(e_excitation(1,:)) * dytab(K)
      e_excitation(2,:) = 0.
      M_index = NZMkey(K,1) 
      N_index = NZMkey(K,2) 
      Z_index = NZMkey(K,3) 
      A_index = N_index + Z_index 
      sn    = neutron_separation_energy(Z_index,N_index)
      li    = get_isotope(Z_index,N_index,0.0d0)
      li_ne = get_isotope(Z_index,N_index-1,0.0d0)
      if (li*li_ne.eq.0) then
        neutron_counter = 0
        do
          neutron_counter = neutron_counter + 1
          li    = get_isotope(Z_index,N_index-neutron_counter,0.0d0)
          li_ne = get_isotope(Z_index,N_index-neutron_counter-1,0.0d0)
          sn = sn + neutron_separation_energy(Z_index,N_index-neutron_counter)
          s_fission_vector(fission_index,index_n) = s_fission_vector(fission_index,index_n) + dytab(K)
          if (li*li_ne.ne.0) then
            exit
          end if
        end do   
      end if  
      ch_sn = int(sn*10. + 0.5)
      e_ave = 0.
      do i=1,1000
        if (e_excitation(1,i) .gt. 0.) then
          e_exc = 0.1d0 * (i-0.5)
          e_ave = e_ave + e_excitation(1,i) * e_exc
          if (e_exc.le.sn) then  !  below neutron separation threshold
            neutron_yield = 0.
          else  ! neutron emission
            neutron_yield  = sum(mb_distribution(ch_sn:i)) / sum(mb_distribution(1:i)) 
            n_ch_n         = i-ch_sn+1
            en(1:n_ch_n)   = mb_distribution(ch_sn:i)  / sum(mb_distribution(1:i))
            do j=1,n_ch_n
              e_excitation(2,i-j-ch_sn) = e_excitation(2,i-j-ch_sn) + e_excitation(1,i)*en(j)
            end do
          end if
          s_fission_vector(fission_index,li)      = s_fission_vector(fission_index,li)      + e_excitation(1,i)*(1-neutron_yield)
          s_fission_vector(fission_index,index_n) = s_fission_vector(fission_index,index_n) + e_excitation(1,i)*neutron_yield
          s_fission_vector(fission_index,li_ne)   = s_fission_vector(fission_index,li_ne)   + e_excitation(1,i)*neutron_yield
        end if  
      end do
   
!
!  repeat this loop until all energy is lt separation energy
!
      do 
        N_index = N_index - 1
        sn    = neutron_separation_energy(Z_index,N_index)
        ch_sn = int(sn*10. + 0.5)
        if (sum(e_excitation(2,ch_sn:1000)) .eq. 0.) exit
        e_excitation(1,:) = e_excitation(2,:) / sum(e_excitation(1,:))
        e_excitation(2,:) = 0.
        li = li_ne
        if (li.eq.0) then
          print*,'isotope after multi neutron emission not found!!!',Z_index,N_index-1
          stop
        end if
        do i=1,1000
          if (e_excitation(1,i) .gt. 0.) then
            e_exc = 0.1d0 * (i-0.5)
            if (e_exc.gt.sn) then  ! neutron emission
              neutron_yield = sum(mb_distribution(ch_sn:i)) / sum(mb_distribution(1:i))
              n_ch_n         = i-ch_sn+1
              en(1:n_ch_n)   = mb_distribution(ch_sn:i)  / sum(mb_distribution(1:i))
              do j=1,n_ch_n
                e_excitation(2,i-j-ch_sn) = e_excitation(2,i-j-ch_sn) + e_excitation(1,i)*en(j)
              end do
              s_fission_vector(fission_index,li)      = s_fission_vector(fission_index,li)      - e_excitation(1,i)*neutron_yield
              s_fission_vector(fission_index,index_n) = s_fission_vector(fission_index,index_n) + e_excitation(1,i)*neutron_yield
              s_fission_vector(fission_index,li_ne)   = s_fission_vector(fission_index,li_ne)   + e_excitation(1,i)*neutron_yield
            end if  
          end if  
        end do
      end do  
    END DO

end


!*****************************************
!*       Neutron Separation Energy       *
!*****************************************
double precision function neutron_separation_energy(z,n)
! Taken from particle_separation_energies.f95
!
!  adopted from K Vogt, T Hartmann, A Zilges, PLB 517 (2001) 255-260
!
!  return value: neutron separation energy in MeV

    implicit none
    integer z,n
    double precision a(3), a_pair, a_shell, d_pair, d_shell, nn, mass, n_over_z                         
    data a/ 6.29  , 3.43, 5.85 / ! parameters for equation as given in paper
    data a_pair/ 10.59 / ! parameters for equation as given in paper
    data a_shell/ 1.51 / ! parameters for equation as given in paper

    nn = 0.
    if (n .gt. 28)  nn = nn + 1
    if (n .gt. 50)  nn = nn + 1
    if (n .gt. 82)  nn = nn + 1
    if (n .gt. 126) nn = nn + 1
    d_shell = nn * a_shell

    mass = z+n
    if (modulo(n,2) .eq. 0 ) then    ! even N
      d_pair =  a_pair /dsqrt(mass)
    else  
      d_pair = -a_pair /dsqrt(mass)
    end if

    n_over_z = float(n)/float(z)
    neutron_separation_energy = a(1)
    neutron_separation_energy = neutron_separation_energy + a(2)*mass**(1./3.)
    neutron_separation_energy = neutron_separation_energy / n_over_z
    neutron_separation_energy = neutron_separation_energy - a(3)
    neutron_separation_energy = neutron_separation_energy + d_pair
    neutron_separation_energy = neutron_separation_energy - d_shell
    neutron_separation_energy = neutron_separation_energy 

end


!*****************************************
!*            Nucleosynthesis            *
!*****************************************
subroutine nucleosynthesis()

    use iso
    implicit none
    integer step, s_out
    double precision levell


    s_out=max(1,time_steps / 10)
    do step =1,time_steps
!      if ( modulo(step, s_out) .eq. 0) then
!        print*,step,' steps **********************************  '
!      end if  
      call make_production_rates()
      abundance = abundance + production - decay
    end do 
!    call equilibrium_test()

end 


!*****************************************
!*            Equilibrium Test           *
!*****************************************
subroutine equilibrium_test()

    use iso
    implicit none
    integer isomer, reac, get_isotope, zt, nt, fission_index
    double precision decayl

    do isomer = 1, number_isotopes  ! loop over all species
      if (abundance(isomer).gt.0.) then  ! isomer is abundant
        if (reactions(isomer,0).eq.0) then  ! isomer is stable
          print*, z(isomer)+n(isomer), element_names(z(isomer)), &
          decay_constant(isomer,0) , abundance(isomer), short_lived(isomer)
        else
          print*, z(isomer)+n(isomer), element_names(z(isomer)), &
          decay_constant(isomer,0) , abundance(isomer), short_lived(isomer)&
          , decay_constant(isomer,0) * abundance(isomer)
        end if  
      end if
    end do

end


!*****************************************
!*         Make Production Rates         *
!*****************************************
subroutine make_production_rates()

    use iso
    implicit none
    integer isomer, reac, get_isotope, zt, nt, fission_index
    double precision decayl, test_abundance
    integer short_lived_isotopes(max_number_isotopes), short1, short2, isomer_short1,isomer_short2, i
    logical hit

    decay      = 0.
    production = 0.   
    short_lived_counter = 0
    short_lived = .false.
    do isomer = 1, number_isotopes  ! loop over all species
      if (abundance(isomer).gt.0. .and. reactions(isomer,0).gt.0) then ! isomer is unstable and abundant
        if (decay_constant(isomer,0) * dt .gt. 1.d-3) then
!          decay(isomer) = (1.d0 - dexp(-decay_constant(isomer,0) * dt))*abundance(isomer)
          short_lived_counter = short_lived_counter + 1
          short_lived(isomer) = .true.
          short_lived_isotopes(short_lived_counter) = isomer
!          print*, element_names(z(isomer)), z(isomer)+n(isomer)
        else
          decay(isomer) = decay_constant(isomer,0) * dt * abundance(isomer)  ! linear approximation
          call make_production(isomer)
        end if  
      end if
    end do  
    short1 =  short_lived_counter
    do while (short_lived_counter .gt. 0)  ! loop over all short-lived species
      isomer_short1 = short_lived_isotopes(short1)
!      print*, z(isomer_short1)+n(isomer_short1),element_names(z(isomer_short1)), ' check .. ',short_lived_counter,isomer_short1
      hit = .false.
      do short2 = 1,short_lived_counter  ! loop over all short-lived species
        if (short2 .ne. short1) then
          isomer_short2 = short_lived_isotopes(short2)
          do reac = 1,reactions(isomer_short2,0)      ! loop over all decay channels
            hit = product_isomer(isomer_short2, reac) .eq. isomer_short1 
            if (hit) exit
          end do
        end if
        if (hit) exit  
      end do  
      if (.not. hit) then                                                           ! isotope is not produced by short-lived isotopes
!        print*, z(isomer_short1)+n(isomer_short1),element_names(z(isomer_short1)), ' found !! ',short_lived_counter
        decay(isomer_short1) = (1.d0 - dexp(-decay_constant(isomer_short1,0) * dt))* &
          (abundance(isomer_short1)-production(isomer_short1)/dt/&
           decay_constant(isomer_short1,0)) + production(isomer_short1)
        call make_production(isomer_short1)  
         do i=short1,short_lived_counter-1
          short_lived_isotopes(i) = short_lived_isotopes(i+1)
        end do                     
        short_lived_counter = short_lived_counter - 1
        short1              = short_lived_counter
!        stop
       else
        short1 = short1 - 1
        if (short1 .lt. 1) then
          print*,'no solution found', short1, short_lived_counter
          do i=1,short_lived_counter
            isomer_short2 = short_lived_isotopes(i)
            print*,z(isomer_short2)+n(isomer_short2),&
                   element_names(z(isomer_short2)),isomer_short2
          end do
          print*
          stop
        end if  
      end if
    end do

end


!*****************************************
!*           Make Production             *
!*****************************************
subroutine make_production(isomer)

     use iso
     implicit none
     integer isomer, reac, get_isotope, zt, nt, fission_index
     double precision decayl

     do reac = 1,reactions(isomer,0)      ! loop over all decay channels
       decayl = decay(isomer)*decay_constant(isomer, reac)
       if (reactions(isomer,reac) .ne. 22) then  ! not fission
         production(product_isomer(isomer, reac)) = &
             production(product_isomer(isomer, reac)) + decayl   
       end if
       select case (reactions(isomer,reac))
         case(3)  ! N
           production(index_n) = production(index_n) + decayl
         case(4)  ! P
           production(index_p) = production(index_p) + decayl
         case(5)  ! A
           production(index_a) = production(index_a) + decayl
         case(6)  ! BN
           production(index_n) = production(index_n) + decayl
         case(7)  ! EP
           production(index_p) = production(index_p) + decayl
         case(8)  ! BA
           production(index_a) = production(index_a) + decayl
         case(9)  ! EA
           production(index_a) = production(index_a) + decayl
         case(10)  ! 2N
           production(index_n) = production(index_n) + 2.*decayl
         case(11)  ! 2P
           production(index_p) = production(index_p) + 2.*decayl
         case(12)  ! 2A
           production(index_a) = production(index_a) + 2.*decayl
         case(13)  ! B2A
           production(index_a) = production(index_a) + 2.*decayl
         case(14)  ! B2N
           production(index_n) = production(index_n) + 2.*decayl
         case(15)  ! B3N
           production(index_n) = production(index_n) + 3.*decayl
         case(16)  ! B4N
           production(index_n) = production(index_n) + 4.*decayl
         case(17)  ! E2P
           production(index_p) = production(index_p) + 2.*decayl
         case(18)  ! BNA
           production(index_n) = production(index_n) + decayl
           production(index_a) = production(index_a) + decayl
         case(19)  ! EPA
           production(index_p) = production(index_p) + decayl
           production(index_a) = production(index_a) + decayl
         case(21)  ! 12C
           production(index_12c) = production(index_12c) + decayl
         case(22)  ! fission
           fission_index = reactions(isomer, -1)
           production = production + decayl * s_fission_vector(fission_index,:)
       end select
     end do

end


!*****************************************
!*                Output                 *
!*****************************************
subroutine output()

    use iso
    use path_module
    implicit none
    integer index1, index2, index3,get_isotope,isot
    double precision levell
    character*256 the_path

    levell = 0.
    index1 = get_isotope(36,51,levell) !87Kr
    index2 = get_isotope(37,50,levell) !87Rb 
    index3 = get_isotope(38,49,levell) !87Sr 
! 
!    print*,'Output - abu  :  ',abundance(index1),abundance(index2),abundance(index3)
    print*, 'Reaction file: ',trim(input_file_network)
    print*, 'Initial abundance file: ',trim(input_file_abundance)
    print*, 'Decay time: ',time_decay, &
         ' s           (',time_decay/365.25/24/3600,' y , ',&
         time_decay/365.25/24/3600/1.d9,' Gy)'
    print*, 'Simulation done in ', time_steps, ' steps'
    print*, '    Z     A Symbol   Level            Final-Abundance  Initial-Abundance'
    do isot=1,number_isotopes
       if (initial_abundance(isot) .gt. 0.d0 .or. abundance(isot) .gt. 1.d-95) then
         print*, z(isot), z(isot)+n(isot), '  ',&
               element_names(z(isot)), level(isot), abundance(isot), initial_abundance(isot)
       end if  
    end do  

end


!*****************************************
!*                 TEST                  *
!*****************************************
! Test function for F2Py
character*256 function get_float(init_fileee, fl_arr, NN)

    use iso

    character*256 init_fileee
    integer NN
    double precision fl_arr(NN)

!    print*, init_fileee
!    print*, init_file

    print*, fl_arr(1)

    get_float = init_fileee
    return

end
!end function get_float

!*****************************************
!*                GEFSUB                 *
!*****************************************
! Taken from GEFSUB.FOR file
subroutine GEFSUB(P_Z_CN,P_A_CN,P_E_EXC,P_J_CN)                

      use GEFSUB_FOR
      use GEFSUBdcl2
      use RESULTS
      implicit none

      INTEGER*4 P_Z_CN
      INTEGER*4 P_A_CN
      REAL*4 P_E_EXC
      REAL*4 P_J_CN

!     /' Input parameters: '/ 
!     /' Atomic number,mass number,excitation energy/MeV,spin/h_bar of CN '/ 
!     /' Results are stored in external arrays. '/ 
!     ' 
      !INCLUDE "GEFSUBdcl2.FOR" 

      ! Benoit, changed U_Delta_S0 by get_U_delta_S0
      xDelta_S0 = get_U_Delta_S0(P_Z_CN,P_A_CN) 
!     ' default values 
!     ' 
!     ' Use nominal parameter values: 
      P_DZ_Mean_S1 = xP_DZ_Mean_S1 
      P_DZ_Mean_S2 = xP_DZ_Mean_S2 
      P_DZ_Mean_S3 = xP_DZ_Mean_S3 
      P_DZ_Mean_S4 = xP_DZ_Mean_S4 
      P_Z_Curv_S1 = xP_Z_Curv_S1 
      P_Z_Curv_S2 = xP_Z_Curv_S2 
      P_A_Width_S2 = xP_A_Width_S2 
      P_Z_Curv_S3 = xP_Z_Curv_S3 
      P_Z_Curv_S4 = xP_Z_Curv_S4 
      Delta_S0 = xDelta_S0 
      P_Shell_S1 = xP_Shell_S1 
      P_Shell_S2 = xP_Shell_S2 
      P_Shell_S3 = xP_Shell_S3 
      P_Shell_S4 = xP_Shell_S4 
      T_low_S1 = xT_low_S1 
      T_low_S2 = xT_low_S2 
      T_low_S3 = xT_low_S3 
      T_low_S4 = xT_low_S4 
      T_low_SL = xT_low_SL 
      P_att_pol = xP_att_pol 
      HOMPOL = xHOMPOL 
      POLARadd = xPOLARadd 
!     ' 
      R_E_exc_used = P_E_exc 
      I_A_CN = P_A_CN 
      I_Z_CN = P_Z_CN 
!     ' 
!     /' Central Z values of fission modes '/ 
!     ' 
!     /' Fit to positions of fission channels (Boeckstiegel et al.,2008) '/ 
!     /' P_DZ_Mean_S1 and P_DZ_Mean_S2 allow for slight adjustments '/ 
!     '    Scope 
      R_Z_mod = I_Z_CN 
      ZC_Mode_0 = R_Z_mod * 0.5E0 
!     /' Central Z value of SL mode '/ 
      ZC_Mode_1 = (53.0E0 - 51.5E0) / (1.56E0 - 1.50E0) * &
          (R_Z_mod**1.3E0 / I_A_CN - 1.50E0) + 51.5E0 + P_DZ_Mean_S1 
      ZC_Mode_2 = (55.8E0 - 54.5E0) / (1.56E0 - 1.50E0) * &
          (R_Z_mod**1.3E0 / I_A_CN - 1.50E0) + 54.5E0 + P_DZ_Mean_S2 
      ZC_Mode_3 = ZC_Mode_2 + 4.5E0 + P_DZ_Mean_S3 
!     '  ZC_Mode_4 = 38.5 + P_DZ_Mean_S4  ' structure in nuclei with A around 190 for 201Tl 
!     '  ZC_Mode_4 = 35.5 + P_DZ_Mean_S4  ' for 180Hg  ( 36.2 for 208Po ) 
!     ' 
!     ' Do not delete these lines (,because this is a very good fit!): 
!     '    ZC_Mode_4 = 38.5 + (I_A_CN-I_Z_CN-110)*0.12 - (I_A_CN-I_Z_CN-110)^2 * 0.009   '                - (I_Z_CN-77)*0.34 + P_DZ_Mean_S4 
!     ' 
      ZC_Mode_4 = 38.5 + (I_A_CN-I_Z_CN-110)*0.12 - (I_A_CN-I_Z_CN-110)**2 * &
          0.009 - (I_Z_CN-77)*0.34 + P_DZ_Mean_S4 
!     ' assumption: mode position moves with Z and A (adjusted to exp. data 
!     ' of Itkis and Andreyev et al. 
!     ' 
      ZC_Mode_4L = 42.05 
!     ' enhances S1 ' 
!     '    End Scope 
!     ' 
!     ' 
!     ' 
      I_N_CN = I_A_CN - I_Z_CN 
!     /' Mean deformation at scission as a function of mass '/ 
!     ' 
!     /' Mode 0: liquid drop and mode 4: Z = 38 '/ 
      beta1_prev = 0.3 
      beta2_prev = 0.3 
      beta1_opt = beta1_prev 
      beta2_opt = beta2_prev 
      DO I = 10 , I_Z_CN - 10 
      IZ1 = I 
      Z1 = REAL(IZ1) 
      IZ2 = I_Z_CN - IZ1 
      Z2 = REAL(IZ2) 
      A1 = Z1 / REAL(I_Z_CN) * REAL(I_A_CN) 
      A2 = I_A_CN - A1 
!     ' 
      CALL Beta_Equi(A1,A2,Z1,Z2,dneck,beta1_prev,beta2_prev,beta1_opt,beta2_opt) 
!     ' 
!     'Print "Mode 0,Z1,Z2,beta1,beta2 ";Z1;" ";Z2;" ";beta1_opt,beta2_opt 
!     'Print Z1;" ";Z2;" ";beta1_opt,beta2_opt 
      Beta(0,1,IZ1) = beta1_opt 
!     /' "light" fragment '/ 
      Beta(4,1,IZ1) = beta1_opt 
      Beta(0,2,IZ2) = beta2_opt 
!     /' "heavy" fragment '/ 
      Beta(4,2,IZ2) = beta2_opt 
      beta1_prev = beta1_opt 
      beta2_prev = beta2_opt 
      E_defo = get_LyMass(Z1,A1,beta1_opt) - get_LyMass(Z1,A1,0.0) 
      Edefo(0,1,IZ1) = E_defo 
!     /' "light" fragment '/ 
      Edefo(4,1,IZ1) = E_defo 
      E_defo = get_LyMass(Z2,A2,beta2_opt) - get_LyMass(Z2,A2,0.0) 
      Edefo(0,2,IZ2) = E_defo 
!     /' "heavy" fragment '/ 
      Edefo(4,2,IZ2) = E_defo 
      END DO 
!     ' 
!     /' Mode 1: deformed shells (light) and spherical (heavy) '/ 
      DO I = 10 , I_Z_CN - 10 
      Z1 = I 
      Z2 = I_Z_CN - Z1 
      A1 = (Z1 - 0.5E0) / REAL(I_Z_CN) * REAL(I_A_CN) 
!     /' polarization roughly considered '/ 
      A2 = I_A_CN - A1 
      IF (  I_Z_CN * 0.5 .LT. ZC_Mode_1  ) THEN 
!     ' Beta_opt_light(A1,A2,Z1,Z2,dneck,0,rbeta_ld) 
!     /' nu_mean of Cf requires shells in the light fragment: '/ 
      rbeta = get_beta_light(I,betaL0,betaL1) - 0.1 
!     ' smaller than general deformation of light fragment 
!     '        (less neck influence due to spherical heavy fragment) 
      IF (  rbeta .LT. 0  )  rbeta = 0 
      Else 
      rbeta = get_beta_heavy(I,betaH0,betaH1) 
!     ' equal to S2 channel 
      IF (  rbeta .LT. 0  )  rbeta = 0 
      End If 
      Beta(1,1,I) = rbeta 
!     /' "light" fragment '/ 
      E_defo = get_LyMass(Z1,A1,rbeta) - get_LyMass(Z1,A1,0.0) 
      Edefo(1,1,I) = E_defo 
!     /' "light" fragment '/ 
      END DO 
!     ' 
      DO I = 10 , I_Z_CN - 10 
      rbeta = 0 
      Beta(1,2,I) = rbeta 
      Edefo(1,2,I) = 0 
!     /' "heavy" fragment (at S1 shell) '/ 
      END DO 
!     ' 
!     /' Mode 2: deformed shells (light and heavy) '/ 
      DO I = 10 , I_Z_CN - 10 
      Z1 = I 
      Z2 = I_Z_CN - Z1 
      A1 = (Z1 - 0.5E0) / REAL(I_Z_CN) * REAL(I_A_CN) 
!     /' polarization roughly considered '/ 
      A2 = I_A_CN - A1 
      IF (  I_Z_CN * 0.5 .LT. ZC_Mode_2  ) THEN 
!     ' Beta_opt_light(A1,A2,Z1,Z2,dneck,beta_heavy(Z2),rbeta_ld) 
      rbeta = get_beta_light(I,betaL0,betaL1) 
!     ' general deformation of light fragment 
      IF (  rbeta .LT. 0  )  rbeta = 0 
!     ' negative values replaced by 0 
      Else 
      rbeta = get_beta_heavy(I,betaH0,betaH1) 
!     ' equal to S2 channel 
      End If 
      Beta(2,1,I) = rbeta 
      E_defo = get_LyMass(Z1,A1,rbeta) - get_LyMass(Z1,A1,0.0) 
      Edefo(2,1,I) = E_defo 
      END DO 
      DO I = 10 , I_Z_CN - 10 
      rbeta = get_beta_heavy(I,betaH0,betaH1) 
!     /' "heavy" fragment (at S2 shell)'/ 
      IF (  rbeta .LT. 0  )  rbeta = 0 
!     ' negative values replaced by 0 
      Beta(2,2,I) = rbeta 
      Z1 = I 
      A1 = (Z1 + 0.5E0) / I_Z_CN * I_A_CN 
!     /' polarization roughly considered '/ 
      E_defo = get_LyMass(Z1,A1,rbeta) - get_LyMass(Z1,A1,0.0) 
      Edefo(2,2,I) = E_defo 
      END DO 
!     ' 
!     /' Mode 3 '/ 
      DO I = 10 , I_Z_CN - 10 
      Z1 = I 
      Z2 = I_Z_CN - Z1 
      A1 = (Z1 - 0.5E0) / REAL(I_Z_CN) * REAL(I_A_CN) 
!     /' polarization roughly considered '/ 
      A2 = I_A_CN - A1 
      rbeta = get_beta_light(I,betaL0,betaL1) 
      rbeta = Max(rbeta-0.10,0.0) 
!     /' for low nu-bar of lightest fragments '/ 
!     '  Beta_opt_light(A1,A2,Z1,Z2,dneck,beta_heavy(Z2,betaH0,betaH1),rbeta) 
      Beta(3,1,I) = rbeta 
      E_defo = get_LyMass(Z1,A1,rbeta) - get_LyMass(Z1,A1,0.0) 
      Edefo(3,1,I) = E_defo 
      END DO 
      DO I = 10 , I_Z_CN - 10 
      rbeta = get_beta_heavy(I,betaH0,betaH1) + 0.2 
!     /' for high nu-bar of heaviest fragments '/ 
      IF (  rbeta .LT. 0  )  rbeta = 0 
      Beta(3,2,I) = rbeta 
      Z1 = I 
      A1 = (Z1 + 0.5E0) / REAL(I_Z_CN) * REAL(I_A_CN) 
!     /' polarization roughly considered '/ 
      E_defo = get_LyMass(Z1,A1,rbeta) - get_LyMass(Z1,A1,0.0) 
      Edefo(3,2,I) = E_defo 
      END DO 
!     ' 
!     /' Mode 5: (Channel ST1 in both fragments) '/ 
      DO I = 10 , I_Z_CN - 10 
      Z1 = I 
      Z2 = I_Z_CN - Z1 
      rbeta = Beta(1,2,I) 
      IF (  rbeta .LT. 0  )  rbeta = 0 
      Beta(5,1,Int(Z1)) = rbeta 
      Beta(5,2,Int(Z1)) = rbeta 
      END DO 
!     ' 
!     /' Mode 6: (Channel ST2 in both fragments) '/ 
      DO I = 10 , I_Z_CN - 10 
      Z1 = I 
      Z2 = I_Z_CN - Z1 
      rbeta = Beta(2,2,I) 
      IF (  rbeta .LT. 0  )  rbeta = 0 
      Beta(6,1,Int(Z1)) = rbeta 
      Beta(6,2,Int(Z1)) = rbeta 
      END DO 
!     ' 
!     ' 
!     /' Mean Z as a function of mass '/ 
!     ' 
!     /' Mode 0 '/ 
      DO I = 10 , I_A_CN - 10 
      ZUCD = REAL(I) / REAL(I_A_CN) * REAL(I_Z_CN) 
      beta1 = Beta(0,1,Int(ZUCD + 0.5)) 
      beta2 = Beta(0,2,Int(I_Z_CN - ZUCD + 0.5)) 
      Z1 = get_Z_equi(I_Z_CN,I,I_A_CN - I,beta1,beta2,dneck,0,0.0,1.0) 
      Zmean(0,1,I) = Z1 
      Zshift(0,1,I) = Z1 - ZUCD 
      Zmean(0,2,I_A_CN - I) = I_Z_CN - Z1 
      Zshift(0,2,I_A_CN - I) = ZUCD - Z1 
      END DO 
!     ' 
!     /' Mode 1 '/ 
      DO I = 10 , I_A_CN - 10 
      ZUCD = REAL(I) / REAL(I_A_CN) * REAL(I_Z_CN) 
      Z = ZUCD + ZPOL1 
!     /' Charge polarisation is considered in a crude way '/ 
      beta1 = Beta(1,1,NINT(Z)) 
!     /' "light" fragment '/ 
      Z = ZUCD - ZPOL1 
      beta2 = Beta(1,2,NINT(I_Z_CN-Z)) 
!     /' "heavy" fragment  at S1 shell '/ 
      IF (  REAL(I_Z_CN) * 0.5 .LT. ZC_Mode_1  ) THEN 
      Z1 = get_Z_equi(I_Z_CN,I,I_A_CN - I,beta1,beta2,dneck,1,POLARadd,POLARfac) 
      Else 
      Z1 = get_Z_equi(I_Z_CN,I,I_A_CN - I,beta1,beta2,dneck,1,0.0,0.0) 
      End If 
      Z1 = Z1 + ZPOL1 
!     /' Charge polarization by shell '/ 
!     ' 
      IF (  I_Z_CN - Z1 .LT. 50 .AND. (I_Z_CN - Z1) .GT. Z1  ) THEN 
      Z1 = I_Z_CN - 50 
!     /' Z of mean heavy fragment not below 50 '/ 
      END IF 
!     ' 
      Zmean(1,1,I) = Z1 
      Zshift(1,1,I) = Z1 - ZUCD 
!     ' neutron-deficient 
      Zmean(1,2,I_A_CN - I) = I_Z_CN - Z1 
      Zshift(1,2,I_A_CN - I) = ZUCD - Z1 
!     ' neutron rich at shell 
      END DO 
!     ' 
!     /' Mode 2 '/ 
      DO I = 10 , I_A_CN - 10 
      ZUCD = REAL(I) / REAL(I_A_CN) * REAL(I_Z_CN) 
      Z = ZUCD 
!     /' Charge polarisation is here neglected '/ 
      beta1 = Beta(2,1,NINT(Z)) 
      beta2 = Beta(2,2,NINT(I_Z_CN-Z)) 
      IF (  REAL(I_Z_CN) * 0.5 .LT. ZC_Mode_2  ) THEN 
      Z1 = get_Z_equi(I_Z_CN,I,I_A_CN-I,beta1,beta2,dneck,2,POLARadd,POLARfac) 
      Else 
      Z1 = get_Z_equi(I_Z_CN,I,I_A_CN-I,beta1,beta2,dneck,2,0.0,0.0) 
      End If 
!     ' 
      Zmean(2,1,I) = Z1 
      Zshift(2,1,I) = Z1 - ZUCD 
!     ' neutron deficieint 
      Zmean(2,2,I_A_CN - I) = I_Z_CN - Z1 
      Zshift(2,2,I_A_CN - I) = ZUCD - Z1 
!     ' neutron rich at shell 
      END DO 
!     ' 
!     /' Mode 3 '/ 
      DO I = 10 , I_A_CN - 10 
      ZUCD = REAL(I) / REAL(I_A_CN) * REAL(I_Z_CN) 
      Z = ZUCD 
!     /' Charge polarisation is here neglected '/ 
      beta1 = Beta(3,1,NINT(Z)) 
      beta2 = Beta(3,2,NINT(I_Z_CN-Z)) 
      Z1 = get_Z_equi(I_Z_CN,I,I_A_CN - I,beta1,beta2,dneck,3,POLARadd,POLARfac) 
      Zmean(3,1,I) = Z1 
      Zshift(3,1,I) = Z1 - ZUCD 
      Zmean(3,2,I_A_CN - I) = I_Z_CN - Z1 
      Zshift(3,2,I_A_CN - I) = ZUCD - Z1 
      END DO 
!     ' 
!     /' Mode 4 (assumed to be equal to mode 0) '/ 
      DO I = 10 , I_A_CN - 10 
      Zmean(4,1,I) = Zmean(0,1,I) 
      Zshift(4,1,I) = Zshift(0,1,I) 
      Zmean(4,2,I_A_CN - I) = Zmean(0,2,I_A_CN - I) 
      Zshift(4,2,I_A_CN - I) = Zshift(0,2,I_A_CN - I) 
      END DO 
!     ' 
!     ' 
!     /' General relations between Z and A of fission channels '/ 
      RZpol = 0 
      DO I = 1 , 3 
      RA = (ZC_Mode_0 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
      RZpol = Zshift(0,2,NINT(RA)) 
      END DO 
      AC_Mode_0 = (ZC_Mode_0 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
!     /' mean position in mass '/ 
      NC_Mode_0 = AC_Mode_0 - ZC_Mode_0 
!     ' 
      RZpol = 0 
      DO I = 1 , 3 
      RA = (ZC_Mode_1 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
      RZpol = Zshift(1,2,NINT(RA)) 
      END DO 
      AC_Mode_1 = (ZC_Mode_1 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
      NC_Mode_1 = AC_Mode_1 - ZC_Mode_1 
!     ' 
      RZpol = 0 
      DO I = 1 , 3 
      RA = (ZC_Mode_2 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
      RZpol = Zshift(2,2,NINT(RA)) 
      END DO 
      AC_Mode_2 = (ZC_Mode_2 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
      NC_Mode_2 = AC_Mode_2 - ZC_Mode_2 
!     ' 
      RZpol = 0 
      DO I = 1 , 3 
      RA = (ZC_Mode_3 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
      RZpol = Zshift(3,2,NINT(RA)) 
      END DO 
      AC_Mode_3 = (ZC_Mode_3 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
      NC_Mode_3 = AC_Mode_3 - ZC_Mode_3 
!     ' 
      RZpol = 0 
      DO I = 1 , 3 
      RA = (ZC_Mode_4 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
      RZpol = Zshift(4,2,NINT(RA)) 
      END DO 
      AC_Mode_4 = (ZC_Mode_4 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
      NC_Mode_4 = AC_Mode_4 - ZC_Mode_4 
!     ' 
!     ' 
!     /' Potential curvatures of fission modes '/ 
!     ' 
!     ' For the width of the mass distribution (potential between saddle and scission): 
      R_Z_Curv_S0 = 8.E0 / REAL(I_Z_CN)**2 * get_Masscurv(REAL(I_Z_CN),REAL(I_A_CN),P_I_rms_CN,kappa) 
!     ' For the yields of the fission channels (potential near saddle): 
      R_Z_Curv1_S0 = 8.E0 / REAL(I_Z_CN)**2 * get_Masscurv1(REAL(I_Z_CN),REAL(I_A_CN),0.0,kappa) 
      R_A_Curv1_S0 = 8.E0 / REAL(I_A_CN)**2 * get_Masscurv1(REAL(I_Z_CN),REAL(I_A_CN),0.0,kappa) 
!     ' 
!     ' 
!     /' Energy transformation '/ 
!     ' 
      Select CASE( Emode) 
      CASE( 0) 
!     ' Energy above outer barrier given 
      R_E_exc_Eb = R_E_exc_used 
      R_E_exc_GS = R_E_exc_used + get_BFTFB(REAL(I_Z_CN),REAL(I_A_CN),1) 
      CASE( 1,3,-1) 
!     ' Energy above ground state given 
      R_E_exc_Eb = R_E_exc_used - get_BFTFB(REAL(I_Z_CN),REAL(I_A_CN),1) 
      R_E_exc_GS = R_E_exc_used 
      CASE( 2) 
!     ' kinetic energy of neutron given 
      SN = (get_U_MASS(REAL(I_Z_CN),REAL(I_A_CN-1)) + &
           get_LyPair(I_Z_CN,I_A_CN-1)) - &
           (get_U_MASS(REAL(I_Z_CN),REAL(I_A_CN)) + &
             get_LyPair(I_Z_CN,I_A_CN))
      R_E_exc_GS = R_E_exc_used + SN 
      R_E_exc_Eb = R_E_exc_GS - get_BFTFB(REAL(I_Z_CN),REAL(I_A_CN),1) 
      End Select 
!     ' 
!     ' 
!     /' Fission barriers -> global parameters '/ 
!     ' 
      B_F = get_BFTF(REAL(I_Z_CN),REAL(I_A_CN),1) 
      B_F_ld = get_BFTF(REAL(I_Z_CN),REAL(I_A_CN),0) 
      E_B = get_BFTFB(REAL(I_Z_CN),REAL(I_A_CN),1) 
      E_B_ld = get_BFTFB(REAL(I_Z_CN),REAL(I_A_CN),0) 
!     ' 
!     ' 
!     /' Barriers and excitation energies of the fission modes '/ 
!     ' 
      E_exc_S0_prov = R_E_exc_Eb 
!     ' 
!     ' 
!     /' Additional influence of N=82 assumed '/ 
      Delta_NZ_Pol = 82.E0/50.E0 - REAL(I_N_CN)/REAL(I_Z_CN) 
      R_Shell_S1_eff = P_Shell_S1 * (1.E0 - P_Att_Pol * Abs(Delta_NZ_Pol)) 
!     ' 
!     ' 
!     /' In Pu,the Z=50 shell meets Z=44 in the light fragment. '/ 
!     /' A deformed shell at Z=44 is assumed to explain the enhancement        of the S1 channel around Pu '/ 
!     /' This very same shell automatically produces the double-humped '/ 
!     /' mass distribution in 180Hg '/ 
      S1_enhance = P_Shell_SL4 + (REAL(I_Z_CN) - ZC_Mode_1 - ZC_Mode_4L)**2 * P_Z_Curv_SL4 
!     'Print "ZC_Mode_1,ZC_Mode_4",ZC_Mode_1,ZC_Mode_4 
!     'Print "Delta-Z S1-S4,S1_enhance",I_Z_CN-ZC_Mode_1 - ZC_Mode_4,S1_enhance 
      IF (  S1_enhance .GT. 0  )  S1_enhance = 0 
      R_Shell_S1_eff = R_Shell_S1_eff + S1_enhance 
!     ' 
!     /' The high TKE of S1 in 242Pu(sf) (and neighbours) is obtained by assuming '/ 
!     /' that the Z=44 shell reduces the deformation of the light fragment. '/ 
      DO I = 10 , I_Z_CN - 10 
      Z1 = I 
      A1 = (Z1 - 0.5E0) / REAL(I_Z_CN) * REAL(I_A_CN) 
!     /' polarization roughly considered '/ 
!     '      Beta(1,1,Z1) = Beta(1,1,Z1) + 0.15 * S1_enhance   /' "light" fragment '/ 
      Beta(1,1,I) = exp(S1_enhance) * Beta(1,1,I) + (1.E0-exp(S1_enhance)) * (Beta(1,1,I)-0.25) 
      Beta(1,1,I) = Max(Beta(1,1,I),0.0) 
      E_defo = get_LyMass(Z1,A1,Beta(1,1,I)) - get_LyMass(Z1,A1,0.0) 
      Edefo(1,1,I) = E_defo 
!     /' "light" fragment '/ 
      END DO 
!     ' 
!     ' Influence of S2 shell in complementary fragment 
!     ' May be called "S12 fission channel" 
      T_Asym_Mode_2 = 0.5 
      SigZ_Mode_2 = SQRT(0.5E0 * T_Asym_Mode_2/(P_Z_Curv_S2)) 
      SigA_Mode_2 = SigZ_Mode_2 * REAL(I_A_CN) / REAL(I_Z_CN) 
      S1_enhance = P_Shell_S2 * &
          get_U_Box(REAL(P_A_CN) - AC_Mode_2 - AC_Mode_1,SigA_Mode_2,P_A_Width_S2) *P_A_Width_S2 
      IF (  S1_enhance .LT. 0.01  ) THEN 
      R_Shell_S1_eff = R_Shell_S1_eff + S1_enhance 
      End If 
!     ' Modify deformation of complementary fragment in corresponding analyzer 
!     ' 
!     ' Overlap of S2 and shell in light fragment 
      R_Shell_S2_eff = P_Shell_S2 
!     '   S2_enhance = P_Shell_S4 +  '             (Csng(I_Z_CN) - ZC_Mode_2 - ZC_Mode_4)^2 * P_Z_Curv_S4 
!     '   If S2_enhance > 0 Then S2_enhance = 0 
!     '   R_Shell_S2_eff = R_Shell_S2_eff + S2_enhance 
!     ' 
!     ' Overlap of S3 and shell in light fragment 
      R_Shell_S3_eff = P_Shell_S3 * (1.E0 - PZ_S3_olap_curv * &
          (REAL(I_Z_CN) - ZC_Mode_3 - PZ_S3_olap_pos)**2) 
!     '        * (Csng(I_Z_CN) - 60.5E0 - PZ_S3_olap_pos)^2) 
      R_Shell_S3_eff = Min(R_Shell_S3_eff,0.0) 
!     ' 
!     '   R_Shell_S4_eff = 2.0 * (P_Shell_S4 + P_Z_Curv_S4*(ZC_Mode_4 - ZC_Mode_0)^2) 
      R_Shell_S4_eff = 2.0 * (P_Shell_S4 + P_Z_Curv_S4 * (ZC_Mode_4 - ZC_Mode_0)**2) 
!     ' overlap of S4 in both fragments 
      IF (  R_Shell_S4_eff .GT. P_Shell_S4  )  R_Shell_S4_eff = P_Shell_S4 
!     ' no overlap at large distance 
!     ' 
      E_ld_S1 = R_A_Curv1_S0 * (REAL(I_A_CN)/REAL(I_Z_CN)*(ZC_MODE_1 - ZC_MODE_0) )**2 
      B_S1 = E_ld_S1 + R_Shell_S1_eff 
      E_exc_S1_prov = E_Exc_S0_prov - B_S1 
!     ' 
      E_ld_S2 = R_A_Curv1_S0 * (REAL(I_A_CN)/REAL(I_Z_CN)*(ZC_MODE_2 - ZC_MODE_0) )**2 
      B_S2 = E_ld_S2 + R_Shell_S2_eff 
      E_exc_S2_prov = E_Exc_S0_prov - B_S2 
!     ' 
      E_ld_S3 = R_A_Curv1_S0 * (REAL(I_A_CN)/REAL(I_Z_CN)*(ZC_MODE_3 - ZC_MODE_0) )**2 
      B_S3 = E_ld_S3 + R_Shell_S3_eff 
      E_exc_S3_prov = E_Exc_S0_prov - B_S3 
!     ' 
      IF (  I_A_CN .LT. 220  ) THEN 
!     ' Only here S4 is close enough to symmetry to have a chance 
      E_ld_S4 = R_A_Curv1_S0 * (REAL(I_A_CN)/REAL(I_Z_CN)*(ZC_MODE_4 - ZC_MODE_0) )**2 
      B_S4 = E_ld_S4 + R_Shell_S4_eff 
      E_exc_S4_prov = E_Exc_S0_prov - B_S4 
      Else 
      B_S4 = 9999 
      E_exc_S4_prov = - 9999 
      End If 
!     ' 
!     /' Mode 11 (overlap of channel 1 in light and heavy fragment '/ 
!     /' Potential depth with respect to liquid-drop potential: B_S11 '/ 
      B_S11 = 2.E0 * (R_Shell_S1_eff + De_Defo_S1 + P_Z_Curv_S1 * &
          (ZC_Mode_1 - ZC_Mode_0)**2 ) - De_Defo_S1 
!     ' 
!     ' 
!     /' Lowering of effective barrier by lower ZPM due to larger width in 
!     partial overlap region (shells in light and heavy fragment) '/ 
      DES11ZPM = Level_S11 * Min(Abs(ZC_Mode_1 - ZC_Mode_0),4.E0*P_Z_Curv_S1) 
!     ' Print B_S11,DES11ZPM,ZC_Mode_1-ZC_Mode_0 
!     ' 
      B_S11 = B_S11 + DES11ZPM 
!     ' 
!     '  If B_S11 > R_Shell_S1_eff + 0.5E0 Then 
!     '   If B_S11 > R_Shell_S1_eff + Level_S11 Then 
!     '     B_S11 = 100   ' S1 and S11 are exclusive 
!     '   Else 
!     '     B_S11 = Min(B_S11,R_Shell_S1_eff) 
!     '   End If 
!     ' 
!     ' 
      E_exc_S11_prov = E_Exc_S0_prov - B_S11 
!     ' 
!     /' Mode 22 (overlap of channel 2 in light and heavy fragment '/ 
!     /' Potential depth with respect to liquid-drop potential: B_S22 '/ 
!     ' 
!     '   B_S22 = 2.E0 * (E_ld_S2 + P_Shell_S2)  '       + 2.E0 * P_Z_Curv_S2 * (ZC_Mode_2 - ZC_Mode_0)^2   /' Parabola '/ 
!     'Print E_ld_S2,P_Shell_S2,P_Z_Curv_S2,ZC_Mode_2,ZC_Mode_0 
      B_S22 = 2.E0 * R_Shell_S2_eff * &
          get_U_Box(REAL(P_A_CN)/2.0 - AC_Mode_2,SigA_Mode_2,P_A_Width_S2) * P_A_Width_S2 
!     ' The integral of U_Box is normalized,not the height! 
!     '    If Abs((P_A_CN/2.E0) - AC_Mode_2) > P_A_Width_S2 Then B_S22 = 9999 
      IF (  P_A_CN .LT. 226  )  B_S22 = 9999 
!     ' 
      E_exc_S22_prov = E_Exc_S0_prov - B_S22 
!     ' 
!     ' 
      E_Min_Barr = Min(0.0,B_S1) 
      E_Min_Barr = Min(E_Min_Barr,B_S2) 
      E_Min_Barr = Min(E_Min_Barr,B_S3) 
      E_Min_Barr = Min(E_Min_Barr,B_S4) 
      E_Min_Barr = Min(E_Min_Barr,B_S11) 
      E_Min_Barr = Min(E_Min_Barr,B_S22) 
!     ' 
!     /' Energy minus the height of the respective fission saddle '/ 
      E_exc_S0 = E_exc_S0_prov + E_Min_Barr - Delta_S0 
      E_exc_S1 = E_exc_S1_prov + E_Min_Barr 
      E_exc_S2 = E_exc_S2_prov + E_Min_Barr 
      E_exc_S3 = E_exc_S3_prov + E_Min_Barr 
      E_exc_S4 = E_exc_S4_prov + E_Min_Barr 
      E_exc_S11 = E_exc_S11_prov + E_Min_Barr 
      E_exc_S22 = E_exc_S22_prov + E_Min_Barr 
!     ' 
!     /' Energy above the lowest fission saddle '/ 
      E_exc_Barr = Max(E_Exc_S0,E_Exc_S1) 
      E_exc_Barr = Max(E_exc_Barr,E_Exc_S2) 
      E_exc_Barr = Max(E_exc_Barr,E_Exc_S3) 
      E_exc_Barr = Max(E_exc_Barr,E_Exc_S4) 
      E_exc_Barr = Max(E_exc_Barr,E_exc_S11) 
      E_exc_Barr = Max(E_exc_Barr,E_exc_S22) 
!     ' 
!     ' 
!     /' Collective temperature used for calculating the widths 
!     in mass asymmetry and charge polarization '/ 
!     ' 
      IF (  E_Exc_S0 .LT. 0  ) THEN 
      E_tunn = -E_Exc_S0
      Else
      E_tunn = 0
      END IF
      R_E_exc_eff = Max(0.1,E_Exc_S0) 
!     '  T_Coll_Mode_0 = TFCOLL * R_E_exc_eff + _  /' empirical,replaced by TRusanov '/ 
      T_Coll_Mode_0 = TCOLLFRAC * &
          (get_De_Saddle_Scission(REAL(I_Z_CN)**2 / REAL(I_A_CN)**0.33333E0,ESHIFTSASCI_coll) - &
           E_tunn) 
      T_Coll_Mode_0 = Max(T_Coll_Mode_0,0.0) 
!     ' 
!     ' 
!     /' Temperature description fitting to the empirical systematics of Rusanov et al. '/ 
!     /' Here from Ye. N. Gruzintsev et al.,Z. Phys. A 323 (1986) 307 '/ 
!     /' Empirical description of the nuclear temperature according to the '/ 
!     /' Fermi-gas description. Should be valid at higher excitation energies '/ 
      T_Rusanov = get_TRusanov(R_E_exc_eff,REAL(I_A_CN)) 
!     '  Print "Temperatures,(GEF,Total,Rusanov): ",T_Coll_Mode_0,TFCOLL * R_E_exc_eff,T_Rusanov 
      T_Coll_Mode_0 = Max(T_Coll_Mode_0,T_Rusanov) 
!     /' Transition vom const. temp. to Fermi gas occurs around 20 MeV by MAX function '/ 
!     '    T_Pol_Mode_0 = T_Pol_Red * T_Coll_Mode_0 
!     ' 
!     ' Application of the statistical model,intrinsic temperature at saddle 
      T_Pol_Mode_0 = get_U_Temp(0.5 * REAL(I_Z_CN),0.5 *REAL(I_A_CN),R_E_exc_eff,0,0,Tscale,Econd) 
      T_Asym_Mode_0 = SQRT(T_Coll_Mode_0**2 + (6E0*TCOLLMIN)**2) 
!     ' 
      E_pot_scission = (&
        get_De_Saddle_Scission(REAL(I_Z_CN)**2 / REAL(I_A_CN)**0.33333E0,ESHIFTSASCI_intr)-&
        E_tunn) 
!     ' 
!     ' 
      T_low_S1_used = T_low_S1 
!     ' 
      T_Coll_Mode_1 = TFCOLL * Max(E_exc_S1,0.E0) + &
          TCOLLFRAC * (get_De_Saddle_Scission(I_Z_CN**2 / &
          I_A_CN**0.33333E0,ESHIFTSASCI_coll) - E_tunn) 
      T_Coll_Mode_1 = Max(T_Coll_mode_1,0.0) 
!     '    T_Pol_Mode_1 = T_Pol_Red * T_Coll_Mode_1 
      T_Pol_Mode_1 = T_Pol_Mode_0 
      T_Asym_Mode_1 = SQRT(T_Coll_Mode_1**2 + (4.0*TCOLLMIN)**2) 
!     ' TCOLLMIN for ZPM 
!     ' 
      T_Coll_Mode_2 = TFCOLL * Max(E_exc_S2,0.E0) + &
          TCOLLFRAC * (get_De_Saddle_Scission(REAL(I_Z_CN)**2 / &
          REAL(I_A_CN)**0.33333E0,ESHIFTSASCI_coll) - E_tunn) 
      T_Coll_Mode_2 = Max(T_Coll_mode_2,0.0) 
!     '    T_Pol_Mode_2 = T_Pol_Red * T_Coll_Mode_2 
      T_Pol_Mode_2 = T_Pol_Mode_0 
      T_Asym_Mode_2 = SQRT(T_Coll_Mode_2**2 + TCOLLMIN**2) 
!     ' 
      T_Coll_Mode_3 = TFCOLL * Max(E_exc_S3,0.E0) + &
          TCOLLFRAC * (get_De_Saddle_Scission(REAL(I_Z_CN)**2 / &
          REAL(I_A_CN)**0.33333E0,ESHIFTSASCI_coll) - E_tunn) 
      T_Coll_Mode_3 = Max(T_Coll_mode_3,0.0) 
!     '    T_Pol_Mode_3 = T_Pol_Red * T_Coll_Mode_3 
      T_Pol_Mode_3 = T_Pol_Mode_0 
      T_Asym_Mode_3 = SQRT(T_Coll_Mode_3**2 + TCOLLMIN**2) 
!     ' 
      T_Coll_Mode_4 = TFCOLL * Max(E_exc_S4,0.E0) + &
          TCOLLFRAC * (get_De_Saddle_Scission(REAL(I_Z_CN)**2 / &
          REAL(I_A_CN)**0.33333E0,ESHIFTSASCI_coll) - E_tunn) 
      T_Coll_Mode_4 = Max(T_Coll_mode_4,0.0) 
!     '    T_Pol_Mode_4 = T_Pol_Red * T_Coll_Mode_4 
      T_Pol_Mode_4 = T_Pol_Mode_0 
      T_Asym_Mode_4 = SQRT(T_Coll_Mode_4**2 + 4.0*TCOLLMIN**2) 
!     ' ZPM like S1 
!     ' 
!     /' Stiffness in polarization '/ 
!     ' 
      RZ = REAL(I_Z_CN) * 0.5E0 
      RA = REAL(I_A_CN) * 0.5E0 
      beta1 = Beta(0,1,NINT(RZ)) 
      beta2 = Beta(0,2,NINT(RZ)) 
      R_Pol_Curv_S0 = ( get_LyMass( RZ - 1.E0,RA,beta1 ) + &
          get_LyMass( RZ + 1.0E0,RA,beta2 ) + get_LyMass( RZ + 1.0E0,RA,beta1 ) + &
          get_LyMass( RZ - 1.0E0,RA,beta2 ) + get_ECOUL( RZ - 1.0E0,RA,beta1,RZ + &
          1.0E0,RA,beta2,dneck) + &
          get_ECOUL( RZ + 1.0E0,RA,beta1,RZ - 1.0E0,RA,beta2,dneck) - &
          2.0E0*get_ECOUL( RZ,RA,beta1,RZ,RA,beta2,dneck) - &
          2.0E0*get_LyMass( RZ,RA,beta1 ) - &
          2.0E0*get_LyMass( RZ,RA,beta2)) * 0.5E0 
!     ' 
      P_Pol_Curv_S0 = R_Pol_Curv_S0 
!     ' 
      R_Pol_Curv_S1 = R_Pol_Curv_S0 
      R_Pol_Curv_S2 = R_Pol_Curv_S0 
      R_Pol_Curv_S3 = R_Pol_Curv_S0 
      R_Pol_Curv_S4 = R_Pol_Curv_S0 
!     ' 
!     ' 
!     ' 
!     /' Mean values and standard deviations of fission modes '/ 
!     ' 
      SIGZ_Mode_0 = SQRT(0.5E0 * T_Asym_Mode_0/R_Z_Curv_S0) 
      IF (  T_Pol_Mode_0 .GT. 1.E-2  ) THEN 
      SigPol_Mode_0 = SQRT(0.25E0 * HOMPOL / R_Pol_Curv_S0 / Tanh(HOMPOL/(2.E0 * T_Pol_Mode_0))) 
      Else 
      SigPol_Mode_0 = SQRT(0.25E0 * HOMPOL / R_Pol_Curv_S0) 
!     /' including influence of zero-point motion '/ 
      END IF 
!     ' 
!     ' 
      R_E_intr_S1 = Max(E_Exc_S1+get_LyPair(I_Z_CN,I_A_CN),0.0) 
      R_Att(1) = exp(-R_E_intr_S1/Shell_fading) 
      R_Att(5) = R_Att(1) 
      R_Att_Sad(1) = exp(-R_E_intr_S1/Shell_fading) 
      R_Att_Sad(5) = R_Att_Sad(1) 
      SIGZ_Mode_1 = SQRT(0.5E0 * T_Asym_Mode_1/(P_Z_Curv_S1*SQRT(R_Att(1)))) 
      IF (  T_Pol_Mode_1 .GT. 1.E-2  ) THEN 
      SigPol_Mode_1 = SQRT(0.25E0 * HOMPOL / R_Pol_Curv_S1 / Tanh(HOMPOL/(2.E0 * T_Pol_Mode_1))) 
      Else 
      SigPol_Mode_1 = SQRT(0.25E0 * HOMPOL / R_Pol_Curv_S1) 
      END IF 
!     ' 
      R_E_intr_S2 = Max(E_Exc_S2+get_LyPair(I_Z_CN,I_A_CN),0.0) 
      R_Att(2) = exp(-R_E_intr_S2/Shell_fading) 
      R_Att(6) = R_Att(2) 
      R_Att_Sad(2) = exp(-R_E_intr_S2/Shell_fading) 
      R_Att_Sad(6) = R_Att_Sad(2) 
      SIGZ_Mode_2 = SQRT(0.5E0 * T_Asym_Mode_2/(P_Z_Curv_S2*SQRT(R_Att(2)))) 
      IF (  T_Pol_Mode_2 .GT. 1.E-2  ) THEN 
      SigPol_Mode_2 = SQRT(0.25E0 * HOMPOL / R_Pol_Curv_S2 / Tanh(HOMPOL/(2.E0 * T_Pol_Mode_2))) 
      Else 
      SigPol_Mode_2 = SQRT(0.25E0 * HOMPOL / R_Pol_Curv_S2) 
      End If 
!     ' 
      R_E_intr_S3 = Max(E_exc_S3+get_LyPair(I_Z_CN,I_A_CN),0.0) 
      R_Att(3) = exp(-R_E_intr_S3/Shell_fading) 
      R_Att_Sad(3) = exp(-R_E_intr_S3/Shell_fading) 
      SIGZ_Mode_3 = SQRT(0.5E0 * T_Asym_Mode_3/(P_Z_Curv_S3*SQRT(R_Att(3)))) 
      IF (  T_Pol_Mode_3 .GT. 1.E-2  ) THEN 
      SigPol_Mode_3 = SQRT(0.25E0 * HOMPOL / R_Pol_Curv_S3 / Tanh(HOMPOL/(2.E0 * T_Pol_Mode_3))) 
      Else 
      SigPol_Mode_3 = SQRT(0.25E0 * HOMPOL / R_Pol_Curv_S3) 
      End if 
!     ' 
      R_E_intr_S4 = Max(E_exc_S4+get_LyPair(I_Z_CN,I_A_CN),0.0) 
      R_Att(4) = exp(-R_E_intr_S4/Shell_fading) 
      R_Att_Sad(4) = exp(-R_E_intr_S4/Shell_fading) 
      SIGZ_Mode_4 = SQRT(0.5E0 * T_Asym_Mode_4/(P_Z_Curv_S4*SQRT(R_Att(4)))) 
      IF (  T_Pol_Mode_4 .GT. 1.E-2  ) THEN 
      SigPol_Mode_4 = SQRT(0.25E0 * HOMPOL / R_Pol_Curv_S4 / Tanh(HOMPOL/(2.E0 * T_Pol_Mode_4))) 
      Else 
      SigPol_Mode_4 = SQRT(0.25E0 * HOMPOL / R_Pol_Curv_S4) 
      End if 
!     ' 
!     ' 
!     ' 
!     /' Energy-dependent shift of fission channels '/ 
!     '    Scope 
      P_Z_Curv_S1_eff = P_Z_Curv_S1 * P_Z_Curvmod_S1 
      P_Z_Curv_S2_eff = P_Z_Curv_S2 * P_Z_Curvmod_S2 
      P_Z_Curv_S3_eff = P_Z_Curv_S3 * P_Z_Curvmod_S3 
      P_Z_Curv_S4_eff = P_Z_Curv_S4 * P_Z_Curvmod_S4 
!     ' 
      DZ_S1 = ZC_Mode_1 * (P_Z_Curv_S1_eff*R_Att(1) / &
          (R_Z_Curv_S0 + P_Z_Curv_S1_eff*R_Att(1)) - &
          (P_Z_Curv_S1_eff / (R_Z_Curv_S0 + P_Z_Curv_S1_eff) ) ) 
      DZ_S2 = ZC_Mode_2 * (P_Z_Curv_S2_eff*R_Att(2) / &
          (R_Z_Curv_S0 + P_Z_Curv_S2_eff*R_Att(2)) - &
          (P_Z_Curv_S2_eff / (R_Z_Curv_S0 + P_Z_Curv_S2_eff) ) ) 
      DZ_S3 = ZC_Mode_3 * (P_Z_Curv_S3_eff*R_Att(3) / &
          (R_Z_Curv_S0 + P_Z_Curv_S3_eff*R_Att(3)) - &
          (P_Z_Curv_S3_eff / (R_Z_Curv_S0 + P_Z_Curv_S3_eff) ) ) 
      DZ_S4 = SIGN(1.0,ZC_Mode_4 - ZC_Mode_0) * ZC_Mode_4 * &
          (P_Z_Curv_S4_eff*R_Att(4) / (R_Z_Curv_S0 + &
           P_Z_Curv_S4_eff*R_Att(4)) - (P_Z_Curv_S4_eff / &
          (R_Z_Curv_S0 + P_Z_Curv_S4_eff) ) ) 
!     ' 
!     ' Empirical shift of S2 channel at low excitation energy at scission 
!     ' for better reproduction of 238U(s,f) and some data for Th isotopes. 
!     ' Does not solve the problem of 229Th(nth,f). 
      EtotS2 = Max(E_Exc_S2 + EDISSFRAC * E_pot_scission,0.0) 
      IF (  EtotS2 .LT. 5.E0  ) THEN 
      DZ_S2 = DZ_S2 + (5.E0 - EtotS2) * 0.1 
      End If 
!     ' 
!     '   DZ_S1 = 0 
!     '   DZ_S2 = 0 
!     '   DZ_S3 = 0 
!     '   DZ_S4 = 0 
!     ' 
!     ' 
      P_Z_Mean_S0 = ZC_Mode_0 
      ZC_Mode_1 = ZC_Mode_1 + DZ_S1 
      P_Z_Mean_S1 = ZC_Mode_1 
!     /' Copy to global parameter '/ 
      ZC_Mode_2 = ZC_Mode_2 + DZ_S2 
      P_Z_Mean_S2 = ZC_Mode_2 
!     /'             "            '/ 
      ZC_Mode_3 = ZC_Mode_3 + DZ_S3 
      P_Z_Mean_S3 = ZC_Mode_3 
!     '   ZC_Mode_4 = ZC_Mode_4 + DZ_S4 
!     ' shift is very small,because S4 exists only close to symmetry 
      P_Z_Mean_S4 = ZC_Mode_4 
!     '    End Scope 
!     ' 
!     /' Energy dependence of charge polarization '/ 
!     /' Due to washing out of shells '/ 
!     ' 
      DO I = 10 , I_A_CN - 10 
!     ' mass number 
      DO J = 1 , 4 
!     ' fission channel 
      DO K = 1 , 2 
!     ' light - heavy group 
      Zshift(J,K,I) = Zshift(0,K,I) + (Zshift(J,K,I) - Zshift(0,K,I))*R_Att(J) 
      END DO 
      END DO 
      END DO 
!     ' 
!     ' 
!     /' Energy dependence of shell-induced deformation '/ 
!     /' Due to washing out of shells '/ 
!     /' (Under development) '/ 
!     /'For I = 10 To I_Z_CN - 10  ' mass number 
!     For J = 1 To 4           ' fission channel 
!     For K = 1 To 2         ' light - heavy group 
!     beta(J,K,I) = beta(0,K,I) + (beta(J,K,I) - beta(0,K,I))*R_Att_Sad(J) 
!     if beta(J,K,I) < 0 Then 
!     beta(J,K,I) = 0 
!     End If 
!     Z1 = I 
!     Z2 = I_Z_CN - Z1 
!     A1 = Z1 / Csng(I_Z_CN) * Csng(I_A_CN) 
!     A2 = I_A_CN - A1 
!     E_defo = Lymass(Z1,A1,beta(J,K,I)) - Lymass(Z1,A1,0.0) 
!     Edefo(J,K,I) = E_defo 
!     Next 
!     Next 
!     Next  '/ 
!     ' 
!     ' 
!     ' 
!     ' 
!     /' General relations between Z and A of fission channels '/ 
!     /' 2nd iteration '/ 
!     ' 
      RZpol = 0 
      DO I = 1 , 3 
      RA = (ZC_Mode_0 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
      RZpol = Zshift(0,2,NINT(RA)) 
      END DO 
      AC_Mode_0 = (ZC_Mode_0 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
!     /' mean position in mass '/ 
      NC_Mode_0 = AC_Mode_0 - ZC_Mode_0 
!     ' 
      RZpol = 0 
      DO I = 1 , 3 
      RA = (ZC_Mode_1 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
      RZpol = Zshift(1,2,NINT(RA)) 
      END DO 
      AC_Mode_1 = (ZC_Mode_1 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
      NC_Mode_1 = AC_Mode_1 - ZC_Mode_1 
!     ' 
      RZpol = 0 
      DO I = 1 , 3 
      RA = (ZC_Mode_2 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
      RZpol = Zshift(2,2,NINT(RA)) 
      END DO 
      AC_Mode_2 = (ZC_Mode_2 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
      NC_Mode_2 = AC_Mode_2 - ZC_Mode_2 
!     ' 
      RZpol = 0 
      DO I = 1 , 3 
      RA = (ZC_Mode_3 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
      RZpol = Zshift(3,2,NINT(RA)) 
      END DO 
      AC_Mode_3 = (ZC_Mode_3 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
      NC_Mode_3 = AC_Mode_3 - ZC_Mode_3 
!     ' 
      RZpol = 0 
      DO I = 1 , 3 
      RA = (ZC_Mode_4 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
      RZpol = Zshift(4,2,NINT(RA)) 
      END DO 
      AC_Mode_4 = (ZC_Mode_4 - RZPol) * REAL(I_A_CN) / REAL(I_Z_CN) 
      NC_Mode_4 = AC_Mode_4 - ZC_Mode_4 
!     ' 
!     ' 
!     ' 
!     /' Yields of the fission modes '/ 
!     ' 
      Yield_Mode_0 = get_Getyield(E_exc_S0,E_exc_S0,T_low_SL,get_TEgidy(REAL(I_A_CN),0.E0,Tscale)) 
!     ' 
      Yield_Mode_1 = get_Getyield(E_exc_S1,E_exc_S0,T_low_S1_used,get_TEgidy(REAL(I_A_CN),R_Shell_S1_eff + dE_Defo_S1,Tscale)) 
!     ' 
      Yield_Mode_2 = get_Getyield(E_exc_S2,E_exc_S0,T_low_S2,get_TEgidy(REAL(I_A_CN),R_Shell_S2_eff + dE_Defo_S2,Tscale)) 
!     ' 
      Yield_Mode_3 = get_Getyield(E_exc_S3,E_exc_S0,T_low_S3,get_TEgidy(REAL(I_A_CN),R_Shell_S3_eff + dE_Defo_S3,Tscale)) 
!     ' 
      Yield_Mode_4 = get_Getyield(E_exc_S4,E_exc_S0,T_low_S4,get_TEgidy(REAL(I_A_CN),R_Shell_S4_eff + dE_Defo_S4,Tscale)) 
!     ' 
!     ' 
      IF (  B_S11 .GT. B_S1  ) THEN 
      Yield_Mode_11 = 0.0 
      Else 
      Yield_Mode_11 = get_Getyield(E_exc_S11,E_exc_S0,T_low_S11,get_TEgidy(REAL(I_A_CN),R_Shell_S1_eff + 2.E0 * dE_Defo_S1,Tscale)) 
      End If 
!     ' 
      IF (  B_S22 .GT. B_S2  ) THEN 
      Yield_Mode_22 = 0.0 
      Else 
      Yield_Mode_22 = get_Getyield(E_exc_S22,E_exc_S0,T_low_S2,get_TEgidy(REAL(I_A_CN),R_Shell_S2_eff,Tscale)) 
      End If 
!     ' 
!     ' 
      Yield_Norm = Yield_Mode_0 + Yield_Mode_1 + Yield_Mode_2 + Yield_Mode_3 + Yield_Mode_4 + Yield_Mode_11 + Yield_Mode_22 
      Yield_Mode_0 = Yield_Mode_0 / Yield_Norm 
      Yield_Mode_1 = Yield_Mode_1 / Yield_Norm 
      Yield_Mode_2 = Yield_Mode_2 / Yield_Norm 
      Yield_Mode_3 = Yield_Mode_3 / Yield_Norm 
      Yield_Mode_4 = Yield_Mode_4 / Yield_Norm 
      Yield_Mode_11 = Yield_Mode_11 / Yield_Norm 
      Yield_Mode_22 = Yield_Mode_22 / Yield_Norm 
!     ' 
!     ' 
!     /' Mass widhts of the fission channels '/ 
!     ' 
      SigA_Mode_0 = SigZ_Mode_0 * REAL(I_A_CN) / REAL(I_Z_CN) 
!     /' width in mass '/ 
      SigA_Mode_1 = SigZ_Mode_1 * REAL(I_A_CN) / REAL(I_Z_CN) 
      SigA_Mode_1 = Min(SigA_Mode_1,SigA_Mode_0) 
!     ' not broader than liquid-drop 
      SigA_Mode_2 = SigZ_Mode_2 * REAL(I_A_CN) / REAL(I_Z_CN) 
      SigA_Mode_2 = Min(SigA_Mode_2,SigA_Mode_0) 
!     ' not broader than liquid-drop 
      SigA_Mode_3 = SigZ_Mode_3 * REAL(I_A_CN) / REAL(I_Z_CN) 
      SigA_Mode_3 = Min(SigA_Mode_3,SigA_Mode_0) 
      SigA_Mode_4 = SigZ_mode_4 * REAL(I_A_CN) / REAL(I_Z_CN) 
      SigA_Mode_4 = Min(SigA_Mode_4,SigA_Mode_0) 
      SigA_Mode_11 = SigZ_Mode_1 * SQRT(2.E0) * REAL(I_A_CN) / REAL(I_Z_CN) 
      SigA_Mode_11 = Min(SigA_Mode_11,SigA_Mode_0) 
      SigA_Mode_22 = SigZ_Mode_2 * SQRT(2.E0) * REAL(I_A_CN) / REAL(I_Z_CN) 
      SigA_Mode_22 = Min(SigA_Mode_22,SigA_Mode_0) 
!     ' 
!     ' 
!     ' 
!     /' Shell effects of different fission channels '/ 
!     /' This is the "real" microscopic shell effect,not the effective shell-correction energy '/ 
!     /' EShell acts on the level density and determines the T parameter '/ 
!     ' 
      DO I = 1 , I_A_CN - 1 
      DO J = 0 , 4 
      EShell(J,1,I) = 0 
!     /' Shells in "light" fragment assumed to be zero '/ 
      END DO 
      DU0 = 0 
      EShell(0,2,I) = 0 
!     /' Shell = 0 in symmetric mode '/ 
      DU1 = R_Shell_S1_eff + dE_Defo_S1 
!     /' + R_A_Curv1_S1 * (AC_Mode_1 - Float(I,6))**2; '/ 
      DU1 = MIN(DU1,0.E0) 
!     /' Technical limit '/ 
      EShell(1,2,I) = DU1 
!     ' 
      DU2 = R_Shell_S2_eff + dE_Defo_S2 
!     /' + R_A_Curv1_S2 * (AC_Mode_2 - Float(I,6))**2; '/ 
      DU2 = Min(DU2,0.E0) 
!     /' Technical limit '/ 
      EShell(2,2,I) = DU2 
!     ' 
      DU3 = R_Shell_S3_eff + dE_Defo_S3 
!     /' + R_A_Curv1_S3 * (AC_Mode_3 - Float(I,6))**2; '/ 
      DU3 = Min(DU3,0.E0) 
!     /' Technical limit '/ 
      EShell(3,2,I) = DU3 
!     ' 
      DU4 = R_Shell_S4_eff + dE_Defo_S4 
!     /' + R_A_Curv1_S4 * (AC_Mode_4 - Float(I,6))**2; '/ 
      DU4 = Min(DU4,0.E0) 
!     /' Technical limit '/ 
      EShell(4,2,I) = DU4 
!     ' 
      END DO 
!     ' 
!     ' 
!     /' Intrinsic temperatures of fragments at scission '/ 
!     ' 
!     /' Mean values '/ 
      T_intr_Mode_0 = get_TEgidy(AC_Mode_0,0.0,0.8) 
      T_intr_Mode_1_heavy = get_TEgidy(AC_Mode_1,R_Shell_S1_eff + dE_Defo_S1,Tscale) 
      T_intr_Mode_1_light = get_TEgidy(REAL(I_A_CN) - AC_Mode_1,0.0,Tscale) 
      T_intr_Mode_2_heavy = get_TEgidy(AC_Mode_2,R_Shell_S2_eff + dE_Defo_S2,Tscale) 
      T_intr_Mode_2_light = get_TEgidy(REAL(I_A_CN) - AC_Mode_2,0.0,Tscale) 
      T_intr_Mode_3_heavy = get_TEgidy(AC_Mode_3,R_Shell_S3_eff + dE_Defo_S3,Tscale) 
      T_intr_Mode_3_light = get_TEgidy(REAL(I_A_CN) - AC_Mode_3,0.0,Tscale) 
      T_intr_Mode_4_heavy = get_TEgidy(AC_Mode_4,R_Shell_S4_eff + dE_Defo_S4,Tscale) 
      T_intr_Mode_4_light = get_TEgidy(REAL(I_A_CN) - AC_Mode_4,0.0,Tscale) 
!     ' 
!     ' 
!     /' Mass-dependent values of individual fragments '/ 
!     /' Mode 0 '/ 
      DO I = 1 , I_A_CN - 1 
      T = get_TEgidy(REAL(I),EShell(0,1,I),Tscale) 
      Temp(0,1,I) = T 
!     /' "light" fragment at freeze-out (somewhere before scission) '/ 
      T = get_TEgidy(REAL(I),EShell(0,2,I),Tscale) 
      Temp(0,2,I) = T 
!     /' "heavy" fragment at freeze-out (somewhere before scission) '/ 
!     ' 
      T = get_TEgidy(REAL(I),0.0,1.0) 
      TempFF(0,1,I) = T 
!     ' FF in their ground state 
      TempFF(0,2,I) = T 
!     ' FF in their ground state 
      END DO 
!     ' 
!     /' Mode 1 '/ 
      DO I = 1 , I_A_CN - 1 
      T = get_TEgidy(REAL(I),EShell(1,1,I),Tscale) 
      Temp(1,1,I) = T 
!     /' "light" fragment '/ 
      T = get_TEgidy(REAL(I),EShell(1,2,I),Tscale) 
      Temp(1,2,I) = T 
!     /' "heavy" fragment '/ 
!     ' 
      T = get_TEgidy(REAL(I),0.0,1.0) 
      TempFF(1,1,I) = T 
!     ' FF in their ground state 
      TempFF(1,2,I) = T 
!     ' FF in their ground state 
      END DO 
!     ' 
!     /' Mode 2 '/ 
      DO I = 1 , I_A_CN - 1 
      T = get_TEgidy(REAL(I),EShell(2,1,I),Tscale) 
      Temp(2,1,I) = T 
!     /' "light" fragment '/ 
      T = get_TEgidy(REAL(I),EShell(2,2,I),Tscale) 
      Temp(2,2,I) = T 
!     /' "heavy" fragment '/ 
!     ' 
!     /' The next section is introduced,because energy sorting is not strong enough, 
!     when shells are only introduced in the heavy fragment. 
!     Ad hoc assumption: For Mode 2 there are shells in both fragments of about 
!     equal size. Technically,we neglect the shells in both fragments. 
!     This has about the same effect for the energy sorting. '/ 
      T = get_TEgidy(REAL(I),0.0,Tscale) 
!     ' FF at scssion 
      Temp(2,1,I) = T 
!     /' "light" fragment '/ 
      T = get_TEgidy(REAL(I),0.0,Tscale) 
!     ' FF at scission 
      Temp(2,2,I) = T 
!     /' "heavy" fragment '/ 
!     ' 
      T = get_TEgidy(REAL(I),0.0,1.0) 
!     ' shell effect neglected 
      TempFF(2,1,I) = T 
!     ' FFs in their ground state 
      TempFF(2,2,I) = T 
!     ' FFs in their ground state 
      END DO 
!     ' 
!     /' Mode 3 '/ 
      DO I = 1 , I_A_CN -1 
      T = get_TEgidy(REAL(I),0.0,Tscale) 
      Temp(3,1,I) = T 
      T = get_TEgidy(REAL(I),0.0,Tscale) 
      Temp(3,2,I) = T 
!     ' 
      T = get_TEgidy(REAL(I),0.0,1.0) 
      TempFF(3,1,I) = T 
!     ' FF in their ground state 
      TempFF(3,2,I) = T 
!     ' FF in their ground state 
      END DO 
!     ' 
!     /' Mode 4 '/ 
      DO I = 1 , I_A_CN -1 
      T = get_TEgidy(REAL(I),0.0,Tscale) 
      Temp(4,1,I) = T 
      T = get_TEgidy(REAL(I),0.0,Tscale) 
      Temp(4,2,I) = T 
!     ' 
      T = get_TEgidy(REAL(I),0.0,1.0) 
      TempFF(4,1,I) = T 
!     ' FF in their ground state 
      TempFF(4,2,I) = T 
!     ' FF in their ground state 
      END DO 
!     ' 
!     ' 
!     /'** Intrinsic excitation energy at saddle and at scission as well as   **'/ 
!     /'** Even-odd effect in proton and neutron number for each fission mode **'/ 
      DO I_Mode = 0 , 6 
      E_coll_saddle(I_Mode) = 0 
      IF (  I_Mode .EQ. 0  )  Etot = E_exc_S0 
      IF (  I_Mode .EQ. 1  )  Etot = E_exc_S1 
      IF (  I_Mode .EQ. 2  )  Etot = E_exc_S2 
      IF (  I_Mode .EQ. 3  )  Etot = E_exc_S3 
      IF (  I_Mode .EQ. 4  )  Etot = E_exc_S4 
      IF (  I_Mode .EQ. 5  )  Etot = E_exc_S11 
      IF (  I_Mode .EQ. 6  )  Etot = E_exc_S22 
!     ' 
      IF (   MOD(I_Z_CN,2)  +  MOD(I_N_CN,2)  .EQ. 0  ) THEN 
!     /' Even-even CN '/ 
      IF (  Etot .GT. 0 .AND. Etot .LT. 2.E0 * 14.E0/SQRT(REAL(I_A_CN))  ) THEN 
      E_coll_saddle(I_Mode) = Etot 
      Etot = 0 
!     /' Excitation below the pairing gap in even-even CN goes into collective excitations '/ 
      End If 
      End If 
!     ' 
!     '    If I_Z_CN Mod 2 + I_N_CN Mod 2 = 0 Then    ' even-even 
!     '      Ediff = Min(Etot,14.0/sqr(Csng(I_A_CN))) 
!     '    End If 
!     '    If I_Z_CN Mod 2 + I_N_CN Mod 2 = 1 Then    ' even-odd or odd-even 
!     '       Ediff = Min(Etot,2.0 * 14.0/sqr(Csng(I_A_CN))) 
!     '    End If 
!     '    Ediff = Max(Ediff,0.0) 
!     '    Etot = Etot - Ediff 
!     ' 
!     ' 
      IF (  Etot .LT. 0  ) THEN 
      E_tunn = -Etot
      Else
      E_tunn = 0
      END IF
      Etot = Max(Etot,0.0) 
!     ' 
      E_pot_scission = (get_De_Saddle_Scission(REAL(I_Z_CN)**2 / REAL(I_A_CN)**0.33333E0,ESHIFTSASCI_intr) ) 
      Etot = Etot + EDISSFRAC * (E_pot_scission - E_tunn) 
!     /' All excitation energy at saddle and part of the potential-energy gain to scission 
!     go into intrinsic excitation energy at scission '/ 
!     ' 
!     ' 
!     ' 
      IF (  I_Mode .EQ. 2  ) THEN 
      EINTR_SCISSION = Etot 
!     /' (For Mode 2) Global parameter '/ 
      End If 
!     ' 
      DO IA1 = 40 , I_A_CN - 40 
!     ' 
      IA2 = I_A_CN - IA1 
      IF (  I_Mode .LE. 4  ) THEN 
      T1 = Temp(I_Mode,1,IA1) 
      T2 = Temp(I_Mode,2,IA2) 
      End If 
      IF (  I_Mode .EQ. 5  ) THEN 
      T1 = Temp(1,2,IA1) 
      T2 = Temp(1,2,IA2) 
      End If 
      IF (  I_Mode .EQ. 6  ) THEN 
      T1 = Temp(2,2,IA1) 
      T2 = Temp(2,2,IA2) 
      End If 
      DT = ABS(T2 - T1) 
!     ' 
!     /' Even-odd effect '/ 
      IF (   MOD(I_Z_CN,2)  .EQ. 0  ) THEN 
      Rincr1P = Exp(-Etot/PZ_EO_symm) 
      Else 
      Rincr1P = 0 
      End If 
      IF (   MOD(I_N_CN,2)  .EQ. 0  ) THEN 
      Rincr1N = Exp(-Etot/PN_EO_symm) 
      Else 
      Rincr1N = 0 
      End If 
      PEOZ(I_Mode,1,IA1) = Rincr1P 
      PEOZ(I_Mode,2,IA2) = Rincr1P 
      PEON(I_Mode,1,IA1) = Rincr1N 
      PEON(I_Mode,2,IA2) = Rincr1N 
!     ' 
      Rincr2 = get_Gaussintegral(DT/Etot-R_EO_Thresh,R_EO_Sigma*(DT+0.0001)) 
!     /' even-odd effect due to asymmetry '/ 
      Rincr2P = (R_EO_MAX - Rincr1P) * Rincr2 
      Rincr2N = (R_EO_MAX - Rincr1N) * Rincr2 
!     ' 
      IF (  IA1 .LT. IA2  ) THEN 
!     ' A1 is lighter 
      PEOZ(I_Mode,1,IA1) = PEOZ(I_Mode,1,IA1) + Rincr2P 
      IF (   MOD(I_Z_CN,2)  .EQ. 0  ) THEN 
      PEOZ(I_Mode,2,IA2) = PEOZ(I_Mode,2,IA2) + Rincr2P 
      Else 
      PEOZ(I_Mode,2,IA2) = PEOZ(I_Mode,2,IA2) - Rincr2P 
      End if 
      PEON(I_Mode,1,IA1) = PEON(I_Mode,1,IA1) + Rincr2N 
      IF (   MOD(I_N_CN,2)  .EQ. 0  ) THEN 
      PEON(I_Mode,2,IA2) = PEON(I_Mode,2,IA2) + Rincr2N 
      Else 
      PEON(I_Mode,2,IA2) = PEON(I_Mode,2,IA2) - Rincr2N 
      End if 
      Else 
      PEOZ(I_Mode,1,IA1) = PEOZ(I_Mode,2,IA1) 
      PEON(I_Mode,1,IA1) = PEON(I_Mode,2,IA1) 
      PEOZ(I_Mode,2,IA2) = PEOZ(I_Mode,1,IA2) 
      PEON(I_Mode,2,IA2) = PEON(I_Mode,1,IA2) 
      End If 
!     ' 
!     ' 
!     /'  Else 
!     PEOZ(I_Mode,2,IA2) =                PEOZ(I_Mode,1,IA2) + Rincr2P 
!     IF I_Z_CN Mod 2 = 0 Then 
!     PEOZ(I_Mode,1,IA1) =                 PEOZ(I_Mode,1,IA1) + Rincr2P 
!     Else 
!     PEOZ(I_Mode,1,IA1) =                 PEOZ(I_Mode,1,IA1) - Rincr2P 
!     End if 
!     PEON(I_Mode,2,IA2) =              PEON(I_Mode,2,IA2) + Rincr2N 
!     IF I_N_CN Mod 1 = 0 Then 
!     PEON(I_Mode,1,IA1) =                 PEON(I_Mode,1,IA1) + Rincr2N 
!     Else 
!     PEON(I_Mode,1,IA1) =                 PEON(I_Mode,1,IA1) - Rincr2N 
!     End if 
!     End If  '/ 
!     ' 
      PEOZ(I_Mode,1,IA1) = PEOZ(I_Mode,1,IA1) * EOscale 
      PEOZ(I_Mode,2,IA2) = PEOZ(I_Mode,2,IA2) * EOscale 
      PEON(I_Mode,1,IA1) = PEON(I_Mode,1,IA1) * EOscale 
      PEON(I_Mode,2,IA2) = PEON(I_Mode,2,IA2) * EOscale 
!     ' 
!     /' Energy sorting '/ 
!     /' E1 = Etot * get_Gaussintegral(T2-T1,0.03); '/ 
      IF (  Abs(T1-T2) .LT. 1.E-6  ) THEN 
      E1 = 0.5E0 * Etot 
      Else 
      E1ES = Csort * T1 * T2 / ( Abs(T1 - T2) ) 
      E1ES = Min(E1ES,0.5E0*Etot) 
!     /' Asymptotic value after "complete" energy sorting '/ 
      E1FG = Etot * IA1 / I_A_CN 
!     /' in Fermi-gas regime '/ 
      IF (  Etot .LT. 13  )  E1 = E1ES 
!     ' complete energy sorting 
      IF (  Etot .GE. 13 .AND. Etot .LE. 20  ) THEN 
!     ' transition region 
      E1 = E1ES + (Etot-13)/7*(E1FG-E1ES) 
      End If 
      IF (  Etot .GT. 20  )  E1 = E1FG 
!     ' Fermi-gas regime 
      End If 
      E2 = Etot - E1 
      EPART(I_Mode,1,IA1) = Max(E1,0.0) 
!     /' Mean E* in light fragment '/ 
      EPART(I_Mode,2,IA2) = Max(E2,0.0) 
!     /' Mean E* in heavy fragment '/ 
      END DO 
      END DO 
!     ' 
!     ' 
!     /'** RMS angular momentum of fission fragments **'/ 
!     /' Following Naik et al.,EPJ A 31 (2007) 195 and  '/ 
!     /' S. G. Kadmensky,Phys. At. Nucl. 71 (2008) 1193 '/ 
!     ' 
!     '   Scope 
      Spin_CN = P_J_CN 
      P_I_rms_CN = P_J_CN 
      Spin_pre_fission = SPIN_CN 
!     ' CN ground-state spin 
!     ' 
      DO IZ1 = 10 , I_Z_CN - 10 
      AUCD = Int(REAL(IZ1) * REAL(I_A_CN) / REAL(I_Z_CN)) 
      DO IA1 = Int(AUCD - 15) , Int(AUCD + 15) 
      IN1 = IA1 - IZ1 
      IF (  IA1 - IZ1 .GE. 10  ) THEN 
!     /' Rigid momentum of inertia for spherical nucleus '/ 
      I_rigid_spher = 1.16E0**2 * REAL(IA1)**1.6667E0 / 103.8415E0 
!     /' unit: hbar^2/MeV '/ 
      DO I_Mode = 0 , 6 
!     ' 
!     /' First (normally light) fission fragment: '/ 
!     ' 
      beta1 = Beta(I_Mode,1,IZ1) 
      alph = beta1 / SQRT(4.E0 * pi / 5.E0) 
      I_rigid = I_rigid_spher * (1.E0 + 0.5E0*alph + 9.E0/7.E0*alph**2) 
!     /' From Hasse & Myers,Geometrical Relationships ... '/ 
      E_exc = EPART(I_Mode,1,IA1) 
      IF (  E_exc .LT. 0  )  E_exc = 0 
      T = get_U_Temp(REAL(IZ1),REAL(IA1),E_exc,1,1,Tscale,Econd) 
!     '   T = sqr(T^2 + 0.8^2)       ' For ZPM 
!     '   T = T_orbital 
!     '   T =  sqr(T^2 + T_orbital^2) 
      IF (  T_orbital .GT. 0.1  ) THEN 
      T = T_orbital / tanh(T_orbital/T) 
!     ' T_orbital represents the ZPM 
      End If 
      I_eff = I_rigid * (1.E0 - 0.8E0 * exp(-0.693E0 * E_exc / 5.E0)) 
      J_rms = SQRT(2.E0 * I_eff * T) 
!     ' 
      J_rms = J_rms * Jscaling 
!     ' 
      IF (   MOD(IZ1,2)  .EQ. 1 .OR.  MOD(IN1,2)  .EQ. 1  )  J_rms = J_rms + Spin_odd * (REAL(IA1)/140.0)**0.66667 
!     '                * Max(0,1 - (E_exc-1)/9) /' empirical '/ 
!     /' Additional angular momentum of unpaired proton. '/ 
!     /' See also Tomar et al.,Pramana 68 (2007) 111 '/ 
!     ' 
!     ' Print Z1,I_Mode,beta1,T,E_exc,Spin_CN 
!     ' Print " ",I_rigid_spher,I_rigid,I_eff,J_rms 
!     ' 
      J_rms = SQRT(J_rms**2 + (IA1/I_A_CN * Spin_pre_fission)**2) 
!     ' 
      SpinRMSNZ(I_Mode,1,IA1-IZ1,IZ1) = J_rms 
!     ' 
!     '     Print 
!     '     Print IA1,T,E_exc,I_rigid_spher,I_rigid,I_eff,J_rms 
!     ' 
!     /' Second (normally heavy) fission fragment: '/ 
!     ' 
      beta2 = Beta(I_Mode,2,IZ1) 
      alph = beta2 / SQRT(4.E0 * pi / 5.E0) 
      I_rigid = I_rigid_spher * (1.E0 + 0.5E0*alph + 9.E0/7.E0*alph**2) 
!     /' From Hasse & Myers,Geometrical Relationships ... '/ 
      E_exc = EPART(I_Mode,2,IA1) 
      IF (  E_exc .LT. 0  )  E_exc = 0 
      T = get_U_Temp(REAL(IZ1),REAL(IA1),E_exc,1,1,Tscale,Econd) 
!     '    T = sqr(T^2 + 0.8^2)       ' For ZPM 
!     '    T = T_orbital 
!     '    T =  sqr(T^2 + T_orbital^2) 
      IF (  T_orbital .GT. 0.1  ) THEN 
      T = T_orbital / tanh(T_orbital/T) 
!     ' T_orbital represents the ZPM 
      End If 
      I_eff = I_rigid * (1.E0 - 0.8E0 * exp(-0.693E0 * E_exc / 5.E0)) 
      J_rms = SQRT(2.E0 * I_eff * T) 
!     ' 
      J_rms = J_rms * Jscaling 
!     ' 
      IF (   MOD(IZ1,2)  .EQ. 1 .OR.  MOD(IN1,2)  .EQ. 1  )  J_rms = J_rms + Spin_odd * (REAL(IA1)/140.0)**0.66667 
!     '                 * Max(0,1 - (E_exc-1)/9) /' empirical '/ 
!     /' Additional angular momentum of unpaired proton. '/ 
!     /' See also Tomar et al.,Pramana 68 (2007) 111 '/ 
!     ' 
      J_rms = SQRT(J_rms**2 + (IA1/I_A_CN * Spin_pre_fission)**2) 
!     ' 
      SpinRMSNZ(I_Mode,2,IA1-IZ1,IZ1) = J_rms 
!     ' 
!     '      Print IA1,T,E_exc,I_rigid_spher,I_rigid,I_eff,J_rms 
!     ' 
      END DO 
      ENd If 
      END DO 
      END DO 
!     '   End Scope 
!     ' 
!     ' **************************************************************** 
!     ' *** Filling arrays with results in the folding mode (GEFSUB) *** 
!     ' **************************************************************** 
!     ' 
      DO I = 10 , I_A_CN - P_Z_CN - 10 
      DO J = 10 , P_Z_CN - 10 
      DO K = 0 , 6 
      NZMPRE(K,I,J) = 0.0 
      END DO 
      END DO 
      END DO 
!     ' 
!     ' Mode 0 
      DO I = 20 , I_A_CN - 20 
      Ic = I_A_CN - I 
      R_Help = Yield_Mode_0 * (get_U_Gauss_mod(AC_Mode_0 - REAL(I),SigA_Mode_0) + &
          get_U_Gauss_mod(AC_Mode_0 - REAL(Ic),SigA_Mode_0)) 
!     ' Mass yield 
      IF (  I .LT. Ic  ) THEN 
      Zs = ZShift(0,1,I) 
      Else 
      Zs = -ZShift(0,1,Ic) 
      End If 
      DO J = 10 , P_Z_CN - 10 
      Jc = P_Z_CN - J 
      IF (  I-J .GE. 0 .AND. Ic-Jc .GE. 0 .AND. I-J .LE. 200 .AND. Ic-Jc .LE. 200  ) THEN 
      NZMPRE(0,I-J,J) = R_Help * &
          get_U_Gauss_mod(REAL(P_Z_CN)/REAL(I_A_CN)*REAL(I) + Zs - REAL(J),SigPol_Mode_0) * &
          get_U_Even_Odd(J,PEOZ(0,1,I)) * get_U_Even_Odd(I-J,PEON(0,1,I)) 
      End If 
      END DO 
      END DO 
!     ' 
!     ' Mode 1 
      DO I = 20 , I_A_CN - 20 
      Ic = I_A_CN - I 
      R_Help = Yield_Mode_1 * (get_U_Gauss_mod(AC_Mode_1 - REAL(I),SigA_Mode_1) + &
          get_U_Gauss_mod(AC_Mode_1 - REAL(Ic),SigA_Mode_1)) 
!     ' Mass yield 
      IF (  I .LT. Ic  ) THEN 
      Zs = ZShift(1,1,I) 
      Else 
      Zs = -ZShift(1,1,Ic) 
      End If 
      DO J = 10 , P_Z_CN - 10 
      Jc = P_Z_CN - J 
      IF (  I-J .GE. 0 .AND. Ic-Jc .GE. 0 .AND. I-J .LE. 200 .AND. Ic-Jc .LE. 200  ) THEN 
      NZMPRE(1,I-J,J) = R_Help * get_U_Gauss_mod(REAL(P_Z_CN)/REAL(I_A_CN)*REAL(I) + &
          Zs - REAL(J),SigPol_Mode_1)* get_U_Even_Odd(J,PEOZ(1,1,I)) * &
          get_U_Even_Odd(I-J,PEON(1,1,I)) 
      End If 
      END DO 
      END DO 
!     ' 
!     ' Mode 2 
      DO I = 20 , I_A_CN - 20 
      Ic = I_A_CN - I 
      R_Help = Yield_Mode_2 * &
          (get_U_Box2(AC_Mode_2-REAL(I),SQRT(2.0)*S2leftmod*SigA_Mode_2,SQRT(2.0)*SigA_Mode_2,P_A_Width_S2) + &
          get_U_Box2(AC_Mode_2 - REAL(Ic),SQRT(2.0)* &
          S2leftmod*SigA_Mode_2,SQRT(2.0)*SigA_Mode_2,P_A_Width_S2)) 
      IF (  I .LT. Ic  ) THEN 
      Zs = ZShift(2,1,I) 
      Else 
      Zs = -ZShift(2,1,Ic) 
      End If 
      DO J = 10 , P_Z_CN - 10 
      Jc = P_Z_CN - J 
      IF (  I-J .GE. 0 .AND. Ic-Jc .GE. 0 .AND. I-J .LE. 200 .AND. Ic-Jc .LE. 200  ) THEN 
      R_Cut1 = R_Help 
      R_Cut2 = R_Help 
      IF (  J .GT. Jc  ) THEN 
      R_Cut1 = R_Help * get_Gaussintegral(REAL(J)-ZTRUNC50,FTRUNC50*SigZ_Mode_2) 
      Else 
      R_Cut2 = R_Help * get_Gaussintegral(REAL(J)-ZTRUNC50,FTRUNC50*SigZ_Mode_2) 
      End If 
      NZMPRE(2,I-J,J) = R_Help * &
          get_U_Gauss_mod(REAL(P_Z_CN)/REAL(I_A_CN)*REAL(I)+Zs-REAL(J),SigPol_Mode_2) * &
          get_U_Even_Odd(J,PEOZ(2,1,I)) * get_U_Even_Odd(I-J,PEON(2,1,I)) 
      End If 
      END DO 
      END DO 
!     ' 
!     ' Mode 3 
      DO I = 20 , I_A_CN - 20 
      Ic = I_A_CN - I 
      R_Help = Yield_Mode_3 * (get_U_Gauss_mod(AC_Mode_3 - REAL(I),SigA_Mode_3) + &
          get_U_Gauss_mod(AC_Mode_3 - REAL(Ic),SigA_Mode_3)) 
!     ' Mass yield 
      IF (  I .LT. Ic  ) THEN 
      Zs = ZShift(3,1,I) 
      Else 
      Zs = -ZShift(3,1,Ic) 
      End If 
      DO J = 10 , P_Z_CN - 10 
      Jc = P_Z_CN - J 
      IF (  I-J .GE. 0 .AND. Ic-Jc .GE. 0 .AND. I-J .LE. 200 .AND. Ic-Jc .LE. 200  ) THEN 
      NZMPRE(3,I-J,J) = R_Help * &
          get_U_Gauss_mod(REAL(P_Z_CN)/REAL(I_A_CN)*REAL(I)+Zs-REAL(J),SigPol_Mode_3) * &
          get_U_Even_Odd(J,PEOZ(3,1,I)) * get_U_Even_Odd(I-J,PEON(3,1,I)) 
      End If 
      END DO 
      END DO 
!     ' 
!     ' Mode 4 
      DO I = 20 , I_A_CN - 20 
      Ic = I_A_CN - I 
      R_Help = Yield_Mode_4 * (get_U_Gauss_mod(AC_Mode_4 - REAL(I),SigA_Mode_4) + &
          get_U_Gauss_mod(AC_Mode_4 - REAL(Ic),SigA_Mode_4)) 
!     ' Mass yield 
      IF (  I .LT. Ic  ) THEN 
      Zs = ZShift(3,1,I) 
      Else 
      Zs = -ZShift(3,1,Ic) 
      End If 
      DO J = 10 , P_Z_CN - 10 
      Jc = P_Z_CN - J 
      IF (  I-J .GE. 0 .AND. Ic-Jc .GE. 0 .AND. I-J .LE. 200 .AND. Ic-Jc .LE. 200  ) THEN 
      NZMPRE(4,I-J,J) = R_Help * &
          get_U_Gauss_mod(REAL(P_Z_CN)/REAL(I_A_CN)*REAL(I)+Zs-REAL(J),SigPol_Mode_4) * &
          get_U_Even_Odd(J,PEOZ(4,1,I)) * get_U_Even_Odd(I-J,PEON(4,1,I)) 
      End If 
      END DO 
      END DO 
!     ' 
!     ' Mode 11 
      DO I = 20 , I_A_CN - 20 
      Ic = I_A_CN - I 
      R_Help = Yield_Mode_11 * (get_U_Gauss_mod(AC_Mode_0 - REAL(I),SigA_Mode_11) + &
          get_U_Gauss_mod(AC_Mode_0 - REAL(Ic),SigA_Mode_11)) 
!     ' Mass yield 
      DO J = 10 , P_Z_CN - 10 
      Jc = P_Z_CN - J 
      IF (  I-J .GE. 0 .AND. Ic-Jc .GE. 0 .AND. I-J .LE. 200 .AND. Ic-Jc .LE. 200  ) THEN 
      NZMPRE(5,I-J,J) = R_Help * &
          get_U_Gauss_mod(REAL(P_Z_CN)/REAL(I_A_CN)*REAL(I)-REAL(J),SigPol_Mode_0) * &
          get_U_Even_Odd(J,PEOZ(5,1,I)) * get_U_Even_Odd(I-J,PEON(5,1,I)) 
      End If 
      END DO 
      END DO 
!     ' 
!     ' Mode 22 
      DO I = 20 , I_A_CN - 20 
      Ic = I_A_CN - I 
      R_Help = Yield_Mode_22 * (get_U_Gauss_mod(AC_Mode_0 - REAL(I),SigA_Mode_22) + &
          get_U_Gauss_mod(AC_Mode_0 - REAL(Ic),SigA_Mode_22)) 
!     ' Mass yield 
      DO J = 10 , P_Z_CN - 10 
      Jc = P_Z_CN - J 
      IF (  I-J .GE. 0 .AND. Ic-Jc .GE. 0 .AND. I-J .LE. 200 .AND. Ic-Jc .LE. 200  ) THEN 
      NZMPRE(6,I-J,J) = R_Help * &
          get_U_Gauss_mod(REAL(P_Z_CN)/REAL(I_A_CN)*REAL(I)-REAL(J),SigPol_Mode_0) * &
          get_U_Even_Odd(J,PEOZ(6,1,I)) * get_U_Even_Odd(I-J,PEON(6,1,I)) 
      End If 
      END DO 
      END DO 
!     ' 
!     ' 
!     ' Normalization 
      R_Sum = 0 
      DO I = 10 , (I_A_CN - P_Z_CN) - 10 
      DO J = 10 , P_Z_CN - 10 
      NZPRE(I,J) = 0 
      DO K = 0 , 6 
      IF (  NZMPRE(K,I,J) .GT. 0  ) THEN 
      R_Sum = R_Sum + NZMPRE(K,I,J) 
      NZPRE(I,J) = NZPRE(I,J) + NZMPRE(K,I,J) 
!     ' sum of all modes 
      End If 
      END DO 
      END DO 
      END DO 
!     ' Print R_Sum 
      DO I = 10 , (I_A_CN - P_Z_CN) - 10 
      DO J = 10 , P_Z_CN - 10 
      NZPRE(I,J) = NZPRE(I,J) / R_Sum 
      DO K = 0 , 6 
      NZMPRE(K,I,J) = NZMPRE(K,I,J) / R_Sum 
      END DO 
      END DO 
      END DO 
!     ' 
!     ' Calculate and store distributions of fragment excitation energy and spin 
!     ' 
      N_cases = 0 
      DO N_index = 10 , (I_A_CN - P_Z_CN) - 10 
!     ' Neutron number 
      DO Z_index = 10 , P_Z_CN - 10 
!     ' Atomic number 
      DO M_index = 0 , 6 
      IF (  NZMPRE(M_index,N_index,Z_index) .GT. Ymin  ) THEN 
      N_cases = N_cases + 1 
      IF (  N_cases .EQ. Ubound(NZMkey,1)  ) THEN 
!     '           Print "Upper bound of NZkey reached" 
!     '           Print "Result will be incomplete" 
      End If 
      NZMkey(N_cases,1) = M_index 
!     ' Fission mode 
      NZMkey(N_cases,2) = N_index 
!     ' Neutron number of fragment 
      NZMkey(N_cases,3) = Z_index 
!     ' Atomic number of fragment 
      End If 
      END DO 
      END DO 
      END DO 
!     ' Print "N_cases  ",N_cases 
!      WRITE (*,*) "N_cases ",N_cases 
!     ' 
      DO K = 1 , N_cases 
      M_index = NZMkey(K,1) 
!     ' fission mode 
      N_index = NZMkey(K,2) 
!     ' neutron number 
      Z_index = NZMkey(K,3) 
!     ' atomic number 
      A_index = N_index + Z_index 
!     ' 
!     ' Yield 
      Ytab(K) = NZMpre(M_index,N_index,Z_index) 
!     ' 
!     ' Angular momentum: 
      DO I = 1 , 100 
      IF (  M_index .LE. 4  ) THEN 
      IF (  Z_index .LT. 0.5 * P_Z_CN  ) THEN 
      Jtab(K,I) = get_U_LinGauss(REAL(I),SpinRMSNZ(M_index,1,N_index,Z_index)/SQRT(2.0))
      Else 
      Jtab(K,I) = get_U_LinGauss(REAL(I),SpinRMSNZ(M_index,2,N_index,Z_index)/SQRT(2.0))
      End If 
      End If 
      IF (  M_index .EQ. 5  ) THEN 
      Jtab(K,I) = get_U_LinGauss(REAL(I),SpinRMSNZ(1,2,N_index,Z_index)/SQRT(2.0)) 
      End If 
      IF (  M_index .EQ. 6  ) THEN 
      Jtab(K,I) = get_U_LinGauss(REAL(I),SpinRMSNZ(2,2,N_index,Z_index)/SQRT(2.0)) 
      End If 
      END DO 
!     ' 
!     ' Normalize numerically (due to non-continuous values) 
!     '   Scope 
      Rint = 0 
      DO I = 1 , 100 
      Rint = Rint + Jtab(K,I) 
      END DO 
      IF (  Rint .GT. 0  ) THEN 
      DO I = 1 , 100 
      Jtab(K,I) = Jtab(K,I) / Rint 
      END DO 
      End If 
!     '   End Scope 
!     ' 
!     ' 
!     ' Excitation energy: 
!     ' 1. Deformation energy at scission 
      IF (  M_index .EQ. 0  ) THEN 
      IF (  Z_index .LT. 0.5 * P_Z_CN  ) THEN 
      Eexc_mean = Edefo(M_index,1,Z_index) 
      Eexc_sigma = ( &
          get_LyMass(REAL(Z_index),REAL(A_index),beta(M_index,1,Z_index) + SIGDEFO_0) - &
          get_LyMass(REAL(Z_index),REAL(A_index),beta(M_index,1,Z_index) )) 
      Else 
      Eexc_mean = Edefo(M_index,2,Z_index) 
      Eexc_sigma = ( &
          get_LyMass(REAL(Z_index),REAL(A_index),beta(M_index,2,Z_index) + SIGDEFO_0) - &
          get_LyMass(REAL(Z_index),REAL(A_index),beta(M_index,2,Z_index) )) 
      End If 
      End If 
      IF (  M_index .GT. 0 .AND. M_index .LE. 4  ) THEN 
      IF (  Z_index .LT. 0.5 * P_Z_CN  ) THEN 
      Eexc_mean = Edefo(M_index,1,Z_index) 
      RS = SIGDEFO/SQRT(R_Att_Sad(M_index)) 
      Eexc_sigma = ( &
          get_LyMass(REAL(Z_index),REAL(A_index),beta(M_index,1,Z_index) + RS) - &
          get_LyMass(REAL(Z_index),REAL(A_index),beta(M_index,1,Z_index) )) 
      Else 
      Eexc_mean = Edefo(M_index,2,Z_index) 
      RS = SIGDEFO/SQRT(R_Att_Sad(M_index)) 
      Eexc_sigma = ( &
          get_LyMass(REAL(Z_index),REAL(A_index),beta(M_index,2,Z_index) + RS) - &
          get_LyMass(REAL(Z_index),REAL(A_index),beta(M_index,2,Z_index) )) 
      End If 
      End If 
      IF (  M_index .EQ. 5  ) THEN 
      Eexc_mean = Edefo(1,2,Z_index) 
      RS = SIGDEFO/SQRT(R_Att_Sad(M_index)) 
      Eexc_sigma = ( &
          get_LyMass(REAL(Z_index),REAL(A_index),beta(1,2,Z_index) + RS) - &
          get_LyMass(REAL(Z_index),REAL(A_index),beta(1,2,Z_index) )) 
      End If 
      IF (  M_index .EQ. 6  ) THEN 
      Eexc_mean = Edefo(2,2,Z_index) 
      RS = SIGDEFO/SQRT(R_Att_Sad(M_index)) 
      Eexc_sigma = ( &
          get_LyMass(REAL(Z_index),REAL(A_index),beta(2,2,Z_index) + RS) - &
          get_LyMass(REAL(Z_index),REAL(A_index),beta(2,2,Z_index) )) 
      End If 
      Eexc_mean = Max(Eexc_mean,0.0) 
!     ' 
!     ' 2. Intrinsic excitation energy at scission 
      IF (  Z_index .LT. 0.5 * REAL(P_Z_CN)  ) THEN 
      Eexc_intr = EPART(M_index,1,A_index) 
      Else 
      Eexc_intr = EPART(M_index,2,A_index) 
      End If 
      IF (  M_index .EQ. 0  ) THEN 
!     ' add shell and pairing of final fragment 
      Eexc_intr = Eexc_intr - get_AME2012(Z_index,A_index) + &
          get_LDMass(REAL(Z_index),REAL(A_index),0.) - 2.0 * 12.0 / SQRT(REAL(A_index)) 
!     ' general shift 
      End If 
      Eexc_intr = Max(Eexc_intr,0.0) 
      Eexc_mean = Eexc_mean + Eexc_intr 
      Eexc_sigma = SQRT(Eexc_sigma**2 + (EexcSIGrel * Eexc_intr)**2) 
!     ' 
!     ' 3. Pairing staggering 
      Eexc_mean = Eexc_mean - get_LyPair(Z_index,A_index) 
!     ' 
!     ' 4. Collective energy 
      Eexc_coll = 0.5 * ECOLLFRAC * &
       (get_De_Saddle_Scission(REAL(P_Z_CN)**2/REAL(I_A_CN)**0.33333E0,ESHIFTSASCI_coll)-&
        E_tunn) 
      Eexc_coll = Max(Eexc_coll,0.0) 
      Eexc_sigma = SQRT(Eexc_sigma**2 + 0.5*(EexcSIGrel*Eexc_coll)**2) 
      Eexc_mean = Eexc_mean + Eexc_coll + 0.5 * E_coll_saddle(M_index) 
!     ' 
!     ' 5. Total excitation energy distribution of fragments (all contributions summed up) 
      DO I = 0 , 1000 
!     ' 100 keV bins up to 100 MeV 
      Etab(K,I) = exp(-(0.1*REAL(I)-Eexc_mean)**2/(2.0 * Eexc_sigma)) 
      END DO 
!     ' 
!     ' Normalize excitation-energy distribution 
!     '   Scope 
      RintE = 0 
      DO I = 0 , 1000 
      RintE = RintE + Etab(K,I) 
      END DO 
      IF (  RintE .GT. 0  ) THEN 
      DO I = 0 , 1000 
      Etab(K,I) = Etab(K,I) / RintE 
      END DO 
      End If 
!     '   End Scope 
!     ' 
      END DO 

end


!*****************************************
!*            special_fission            *
!*****************************************
! Taken from special_fission.f95
subroutine special_fission()

    use iso
    use RESULTS
    use path_module
    implicit none
    integer isomer, get_isotope, n_fission_counter,fission_index, li
    integer istatus
    double precision neutron_separation_energy
    character*3 chz, chn 
    character*256 fission_file
    double precision yield, dytab(10000)
    INTEGER*4 z_t, n_t, a_t
    REAL*4    level_e, level_spin 
    character*256 the_path

    the_path = trim(files_path) // trim('decay_data/')

    n_fission_counter = 0
    fission_index     = 0
    do isomer = 1, number_isotopes           ! loop over all species
      if (z(isomer) .eq. 92 .and. n(isomer).eq.146) then            ! only 238U for now
        z_t        = z(isomer)  ! compound nucleus
        n_t        = n(isomer)  ! compound nucleus
        a_t        = n_t + z_t  ! compound nucleus
        level_e    = 0.0              ! GS
        level_spin = spin(isomer)           ! 
        n_fission_counter = n_fission_counter + 1
        write(chz,'(I3.3)') z(isomer)
        write(chn,'(I3.3)') n(isomer)
        fission_file = the_path // chz // '_' // chn // '_spontaneous_no_scission.dat'
!
!        open(unit=13,file=fission_file,status='old', iostat=istatus)
!        if (istatus .ne. 0) then                      ! need to create the fission file
          print*,'s-Fission of: ',a_t,element_names(z(isomer)), '  ', level_e,level_spin
          print*,'Creating: ',trim(fission_file)
          call GEFSUB(z_t,a_t,level_e,level_spin)
          dytab = Ytab
          dytab = 2./sum(dytab(1:N_cases)) * dytab   ! normalize to 2.
          s_fission_vector(fission_index,:) = 0.
          call special_scission(fission_index,N_cases,NZMkey,dytab,Etab)
          print*,'Neutron yield',s_fission_vector(fission_index,index_n) &
                , sum(s_fission_vector(fission_index,:))
          open(unit=13,file=fission_file,status='unknown')
          do li=1,number_isotopes
            if (s_fission_vector(fission_index,li) .gt. 0.0) then
              write(13,'(2I6,ES20.10)') z(li), n(li), s_fission_vector(fission_index,li)
            end if  
          end do
!        end if
        close(13)
      end if
    end do
    print*,'Number of isotopes with neutron-induced fission: ', n_fission_counter

end


!*****************************************
!*            Special Scission           *
!*****************************************
! Taken from special_fission.f95
subroutine special_scission(fission_index,dytab)

    use iso
    use RESULTS
    use path_module
    implicit none
    integer isomer,M_index, N_index, Z_index, A_index,k,i,j,get_isotope,li,ch_sn, n_ch_n
    integer fission_index, istatus, ch_counter, li_ne, neutron_counter
    double precision neutron_yield, e_ave, e_sum, de_ave2,neutron_separation_energy
    double precision sn, e_excitation(2,1000), e_exc, en(1000),dytab(10000)
    double precision yield_local(300)
    REAL*4    level_e, level_spin
    INTEGER*4 iso_n, iso_z, iso_a
    character*256 the_path

    the_path = trim(files_path) // trim('decay_data/debug/u238sf_A_yield_GLE_no-scission.csv')

    yield_local = 0.
    DO K = 1 , N_cases    
      e_excitation(1,:) = Etab(k,:)
      e_excitation(1,:) = e_excitation(1,:) / sum(e_excitation(1,:)) * dytab(K)
      e_excitation(2,:) = 0.
      M_index = NZMkey(K,1) 
      N_index = NZMkey(K,2) 
      Z_index = NZMkey(K,3) 
      A_index = N_index + Z_index 
      yield_local(A_index) = yield_local(A_index) + dytab(K)
      sn    = neutron_separation_energy(Z_index,N_index)
      li    = get_isotope(Z_index,N_index,0.0d0)
      li_ne = get_isotope(Z_index,N_index-1,0.0d0)
      if (li*li_ne.eq.0) then
        neutron_counter = 0
        do
          neutron_counter = neutron_counter + 1
          li    = get_isotope(Z_index,N_index-neutron_counter,0.0d0)
          li_ne = get_isotope(Z_index,N_index-neutron_counter-1,0.0d0)
          sn = sn + neutron_separation_energy(Z_index,N_index-neutron_counter)
          s_fission_vector(fission_index,index_n) = &
              s_fission_vector(fission_index,index_n) + dytab(K)
          if (li*li_ne.ne.0) then
            exit
          end if
        end do   
      end if  
      sn = 1.d12
      ch_sn = int(sn*10. + 0.5)
      e_ave = 0.
      do i=1,1000
        if (e_excitation(1,i) .gt. 0.) then
          e_exc = 0.1d0 * (i-0.5)
          e_ave = e_ave + e_excitation(1,i) * e_exc
          if (e_exc.le.sn) then  !  below neutron separation threshold
            neutron_yield = 0.
          else  ! neutron emission
            neutron_yield  = sum(mb_distribution(ch_sn:i)) / sum(mb_distribution(1:i)) 
            n_ch_n         = i-ch_sn+1
            en(1:n_ch_n)   = mb_distribution(ch_sn:i)  / sum(mb_distribution(1:i))
            do j=1,n_ch_n
              e_excitation(2,i-j-ch_sn) = e_excitation(2,i-j-ch_sn) + e_excitation(1,i)*en(j)
            end do
          end if
          s_fission_vector(fission_index,li) = &
              s_fission_vector(fission_index,li) + e_excitation(1,i)*(1-neutron_yield)
          s_fission_vector(fission_index,index_n) = &
              s_fission_vector(fission_index,index_n) + e_excitation(1,i)*neutron_yield
          s_fission_vector(fission_index,li_ne) = &
              s_fission_vector(fission_index,li_ne) + e_excitation(1,i)*neutron_yield
        end if  
      end do
   
!
!  repeat this loop until all energy is lt separation energy
!
      do 
        N_index = N_index - 1
        sn    = neutron_separation_energy(Z_index,N_index)
        ch_sn = int(sn*10. + 0.5)
        if (sum(e_excitation(2,ch_sn:1000)) .eq. 0.) exit
        e_excitation(1,:) = e_excitation(2,:) / sum(e_excitation(1,:))
        e_excitation(2,:) = 0.
        li = li_ne
        if (li.eq.0) then
          print*,'isotope after multi neutron emission not found!!!',Z_index,N_index-1
          stop
        end if
        do i=1,1000
          if (e_excitation(1,i) .gt. 0.) then
            e_exc = 0.1d0 * (i-0.5)
            if (e_exc.gt.sn) then ! neutron emission
              neutron_yield = sum(mb_distribution(ch_sn:i)) / sum(mb_distribution(1:i))
              n_ch_n         = i-ch_sn+1
              en(1:n_ch_n)   = mb_distribution(ch_sn:i)  / sum(mb_distribution(1:i))
              do j=1,n_ch_n
                e_excitation(2,i-j-ch_sn) = e_excitation(2,i-j-ch_sn) + e_excitation(1,i)*en(j)
              end do
              s_fission_vector(fission_index,li) = &
                  s_fission_vector(fission_index,li) - e_excitation(1,i)*neutron_yield
              s_fission_vector(fission_index,index_n) = &
                  s_fission_vector(fission_index,index_n) + e_excitation(1,i)*neutron_yield
              s_fission_vector(fission_index,li_ne) = &
                  s_fission_vector(fission_index,li_ne) + e_excitation(1,i)*neutron_yield
            end if  
          end if  
        end do
      end do  
    END DO
    open(unit=23,file=the_path,status='unknown') 
      do A_index=1,300
        if (yield_local(A_index).gt.0.) write(23,*) A_index,yield_local(A_index)
      end do
    close(23)

end


!*****************************************
!*             Init N Fission            *
!*****************************************
! Taken from n_fission.f95
subroutine init_n_fission()

    use iso
    use RESULTS
    use path_module
    implicit none
    integer isomer, get_isotope, n_fission_counter,fission_index, li
    integer istatus
    double precision neutron_separation_energy
    character*3 chz, chn 
    character*256 fission_file
    double precision yield, dytab(10000)
    INTEGER*4 z_t, n_t, a_t
    REAL*4    level_e, level_spin 
    character*256 the_path

    the_path = trim(files_path) // trim('decay_data/n_fission_Sn_plus_500keV/')

    n_fission_counter = 0
    fission_index     = 0
    do isomer = 1, number_isotopes ! loop over all species
      if (z(isomer) .ge. 84) then ! Z=84 is Po. Below no fission is considered to be of importance
        z_t        = z(isomer)  ! compound nucleus
        n_t        = n(isomer) + 1  ! compound nucleus
        a_t        = n_t + z_t  ! compound nucleus
        level_e    = neutron_separation_energy(z(isomer),n(isomer) + 1) + 0.5 ! neutron separation energy of target comp-nucleus + average neutron energy
        if ( level_e .lt. 0.) then
          print*,'Sn+0.1 MeV .lt. 0.)'
          level_e = 0.
        end if  
        level_spin = spin(isomer) ! in principle, it should be spin(isomer) +- spin_of_neutron (0.5, 1.5, ...)
        n_fission_counter = n_fission_counter + 1
        write(chz,'(I3.3)') z(isomer)
        write(chn,'(I3.3)') n(isomer)
        fission_file = the_path // chz // '_' // chn // '_n_induced.dat'
!
        open(unit=13,file=fission_file,status='old', iostat=istatus)
        if (istatus .ne. 0) then                      ! need to create the fission file
          print*,'n-Fission of: ',a_t,element_names(z(isomer)), '  ', level_e,level_spin
          print*,'Creating: ',trim(fission_file)
          call GEFSUB(z_t,a_t,level_e,level_spin)
          dytab = Ytab
          dytab = 2./sum(dytab(1:N_cases)) * dytab   ! normalize to 2.
          s_fission_vector(fission_index,:) = 0.
          call scission(fission_index,N_cases,NZMkey,dytab,Etab)
          print*,'Neutron yield',s_fission_vector(fission_index,index_n) &
                , sum(s_fission_vector(fission_index,:))
          open(unit=13,file=fission_file,status='unknown')
          do li=1,number_isotopes
            if (s_fission_vector(fission_index,li) .gt. 0.0) then
              write(13,'(2I6,ES20.10)') z(li), n(li), s_fission_vector(fission_index,li)
            end if  
          end do
        end if
        close(13)
      end if
    end do
    print*,'Number of isotopes with neutron-induced fission: ', n_fission_counter

end


