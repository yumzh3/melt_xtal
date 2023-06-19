# melting + crystallization portal
# Jan 18, 2023
# written by Mingzhen Yu
# last modified:
     
import numpy as np
import pandas as pd
import math
import sympy  
import copy
from scipy.optimize import fsolve  
import matplotlib.pyplot as plt
from melting_function2023 import *
from olonly_function2023 import *
from wl1989stoich_2023 import *
from wl1989kdcalc_2023 import *
from wl1989models_2023 import *
from wlState_2023 import *


# default parameters with default values
S_mantle = 200  # all values for sulfur related default parameters are from Zhao etal 2022 and references therein
S_mantlesulfide = 369000
SCSS_QFM_MORB = 1200
f_sulfideliquid = 4/5
f_mss = 1-f_sulfideliquid
deltaQFM = 0
Fe2Fet_Haw = 0.85
Fe2Fet_MORB = 0.9
Po_high =45
Po_low = 20
F_target_Haw = 0.06
F_target_MORB = 0.10
xtalization_model = 'fractional'
melting_model_Haw = 'polybaric'
melting_model_MORB = 'polybaric'
S_mode_Haw = 'N'
S_mode_MORB = 'N'
#source_wt_single_pe = {'SiO2':45.4, 'TiO2':0.17, 'Al2O3':4.22, 'FeO':7.9,'CaO':3.53,'MgO':38,'MnO':0.136,'K2O':0.013,'Na2O':0.32,'P2O5':0.013,'Cr2O3':0.38,'NiO':0.248}  # by looking at Ionov pe data with MgO = 38
#source_wt_Haw_pe = {'SiO2':45.8, 'TiO2':0.17, 'Al2O3':4.30, 'FeO':8,'CaO':3.65,'MgO':37.3,'MnO':0.130,'K2O':0.013,'Na2O':0.35,'P2O5':0.013,'Cr2O3':0.37,'NiO':0.250}  # modifed after source_wt_single_pe, by looking at Ionov pe data with MgO = 37.3, set NiO and MnO for Haw
#source_wt_MORB_pe = {'SiO2':45.1, 'TiO2':0.15, 'Al2O3':3.7, 'FeO':8,'CaO':3.15,'MgO':39,'MnO':0.135,'K2O':0.013,'Na2O':0.28, 'P2O5':0.013,'Cr2O3':0.38,'NiO':0.250}  # modifed after source_wt_single_pe, by looking at Ionov pe data with MgO = 39, set NiO and MnO for MORB
#source_wt_Haw_eclope = {'SiO2':46.1, 'TiO2':0.33, 'Al2O3':4.6, 'FeO':8,'CaO':3.7,'MgO':36.2,'MnO':0.129,'K2O':0.02,'Na2O':0.458, 'P2O5':0.027,'Cr2O3':0.36,'NiO':0.243}  # add 3% eclogite melt (A177-82 with melting degree of 7.4% from Pertermann and Hirschmann 2003) to source_wt_Haw_pe, calculate by mixng mass balance
source_wt_Haw_MORB_pe = {'SiO2':45.1, 'TiO2':0.14, 'Al2O3':3.7, 'FeO':8.1,'CaO':3,'MgO':39.5,'MnO':0.140,'K2O':0.013,'Na2O':0.27, 'P2O5':0.014,'Cr2O3':0.38,'NiO':0.255} 
source_wt_Haw_MORB_eclope = {'SiO2':45.5, 'TiO2':0.27, 'Al2O3':4, 'FeO':8.1,'CaO':3.1,'MgO':38.4,'MnO':0.138,'K2O':0.021,'Na2O':0.40, 'P2O5':0.029,'Cr2O3':0.37,'NiO':0.248} 
source_wt_Haw = source_wt_Haw_MORB_eclope
source_wt_MORB = source_wt_Haw_MORB_pe
source_phase_highP_pe_KLB = {'ol':60,'opx':8,'cpx':23,'gt':9,'sp':0}  # Davis etal 2011 EPSL Table1 subsolidus mode of KLB-1 at 3 GPa (60ol,8opx,23cpx,9gt)
source_phase_lowP_pe_KLB = {'ol':60,'opx':24,'cpx':14,'gt':0,'sp':2}  # subsolidus mode of KLB-1 at 1.5 GPa (Hirose and Kushiro 1993, Herzberg etal 1990 JGR solid earth, Davis etal 2009 Table4 American Mineralogist)
source_phase_highP_pe = {'ol':55.3,'opx':8,'cpx':28.4,'gt':8.3,'sp':0}  # modifed after Davis etal 2011 EPSL 3 GPa subsolidus mode of KLB-1
source_phase_highP_eclope = {'ol':53.5,'opx':11.0,'cpx':27.5,'gt':8,'sp':0}  # add 3% eclogite melt to source_phase_highP_pe (adding solidus eclogite melt will first increase opx in peridotite, Yaxley and Green 1998)
source_phase_Haw = source_phase_highP_eclope
source_phase_MORB = source_phase_lowP_pe_KLB
# source_wt_Haw_Putirka2011 = {'SiO2':45.7, 'TiO2':0.29, 'Al2O3':4.9, 'FeO':8.24,'CaO':3.9,'MgO':36.07,'MnO':0.143,'K2O':0.027,'Na2O':0.55, 'P2O5':0.030,'Cr2O3':0.34,'NiO':0.229}  # 10.8% N-MORB + 89.2% peridotite for MORB and Haw
# source_phase_Haw_Putirka2011 = {'ol':49.5,'opx':7.1,'cpx':34.1,'gt':9.3,'sp':0}  # 10.8% N-MORB (80%cpx-20%grt) + 89.2% peridotite for MORB and Haw (source_phase_highP_pe)
# source_wt_Haw_Putirka2011 = {'SiO2':45.5, 'TiO2':0.29, 'Al2O3':5.43, 'FeO':8,'CaO':4.36,'MgO':34.9,'MnO':0.14,'K2O':0.03,'Na2O':0.55, 'P2O5':0.030,'Cr2O3':0.33,'NiO':0.225}  # 10.8% MORB + 89.2% S&S(2004) DM from Putirka et al. 2011 Table A5
# source_phase_Haw_Putirka2011 = {'ol':60,'opx':13.9,'cpx':15.5,'gt':10.6,'sp':0}  # 10.8% MORB + 89.2% DM from Putirka et al. 2011 Table A5
# source_phase_Haw_Putirka2011 = {'ol':44.2,'opx':4.4,'cpx':34.4,'gt':17,'sp':0}  # calculated modes for Putirka et al. 2011 Table A5 compositions by using mineral compositions from run 40.02 of Walter 1998
# source_phase_Haw_Putirka2011 = {'ol':52.4,'opx':7.2,'cpx':25,'gt':15.4,'sp':0}  # calculated modes for Putirka et al. 2011 Table A5 compositions by using mineral compositions from Davis et al. 2011 3GPa KLB-1 subsolidus
# source_phase_Haw_Putirka2011 = {'ol':50.4,'opx':15.2,'cpx':18.6,'gt':15.8,'sp':0}  # calculated modes for Putirka et al. 2011 Table A5 compositions by using average mineral compositions from Ionov et al. 1993 Table 5
# source_wt_MORB_Putirka2011 = {'SiO2':44.9, 'TiO2':0.13, 'Al2O3':4.28, 'FeO':7.75,'CaO':3.5,'MgO':38.22,'MnO':0.135,'K2O':0.007,'Na2O':0.29, 'P2O5':0.009,'Cr2O3':0.365,'NiO':0.249}  # Salter and Stracke (2004) DM used by Putirka et al. 2011 in their MORB-DM mixing model
# source_phase_MORB_Putirka2011 = {'ol':54.9,'opx':24.5,'cpx':15.8,'gt':0,'sp':4.8}  # calculated modes for Salter&Stracke (2004) DM for MORB by using mineral compositions from run 20 of Baker and Stolper (1994)
# source_phase_MORB_Putirka2011 = {'ol':56.1,'opx':28.1,'cpx':13.3,'gt':0,'sp':2.5}  # calculated modes for Salter&Stracke (2004) DM for MORB by using mineral compositions from Workman and Hart (2005) Table 3
# source_wt_3eclmelt97DMmix = {'SiO2':45.27, 'TiO2':0.299, 'Al2O3':4.6, 'FeO':7.75,'CaO':3.6,'MgO':37.14,'MnO':0.133,'K2O':0.015,'Na2O':0.399, 'P2O5':0.024,'Cr2O3':0.356,'NiO':0.242}  # 3% eclogite melt (run A177-82, Pertermann and Hirschmann 2003) + 97% DM (Salter and Stracke 2004)
# source_phase_3eclmelt97DMmix = {'ol':50.5,'opx':7.7,'cpx':28,'gt':13.8,'sp':0} # modes calculted using minerals from run 40.02 of Walter 1998
# source_wt_5eclmelt95DMmix = {'SiO2':45.52, 'TiO2':0.409, 'Al2O3':4.8, 'FeO':7.74,'CaO':3.7,'MgO':36.42,'MnO':0.132,'K2O':0.021,'Na2O':0.472, 'P2O5':0.034,'Cr2O3':0.35,'NiO':0.238}  # 5% eclogite melt (run A177-82, Pertermann and Hirschmann 2003) + 95% DM (Salter and Stracke 2004)
# source_phase_5eclmelt95DMmix = {'ol':50.1,'opx':6.3,'cpx':28.6,'gt':15,'sp':0} # modes calculted using minerals from run 40.02 of Walter 1998
# source_wt_Walter1998 = {'SiO2':45.86, 'TiO2':0.11, 'Al2O3':4.36, 'FeO':7.69,'CaO':3.55,'MgO':38.12,'MnO':0.137,'K2O':0.03,'Na2O':0.23, 'P2O5':0.030,'Cr2O3':0.41,'NiO':0.249}  # run 30.05
# source_phase_Walter1998 = {'ol':53.1,'opx':17.7,'cpx':27.3,'gt':1.9,'sp':0}   # run 30.05
# source_phase_Walter1998_lowP = {'ol':52.3,'opx':30.8,'cpx':15,'gt':0,'sp':2.0}   # run 30.05
# source_wt_Walter1998 = {'SiO2':45.44, 'TiO2':0.04, 'Al2O3':3.11, 'FeO':7.41,'CaO':2.39,'MgO':41.34,'MnO':0.129,'K2O':0.03,'Na2O':0.10, 'P2O5':0.030,'Cr2O3':0.42,'NiO':0.275}  # run 30.12
# source_phase_Walter1998 = {'ol':60.8,'opx':22.9,'cpx':16.2,'gt':0,'sp':0}   # run 30.12
# source_phase_Walter1998_lowP = {'ol':62.4,'opx':28.4,'cpx':8.5,'gt':0,'sp':0.6}   # run 30.12
# source_wt_Walter1998 = {'SiO2':44.85, 'TiO2':0.03, 'Al2O3':2.79, 'FeO':7.39,'CaO':1.90,'MgO':42.35,'MnO':0.129,'K2O':0.03,'Na2O':0.08, 'P2O5':0.030,'Cr2O3':0.46,'NiO':0.283}  # run 30.07
# source_phase_Walter1998 = {'ol':63.2,'opx':25.3,'cpx':11.5,'gt':0,'sp':0}   # run 30.07
# source_phase_Walter1998_lowP = {'ol':66.4,'opx':26.4,'cpx':6.5,'gt':0,'sp':0.6}   # run 30.07
# source_wt_Walter1998 = {'SiO2':44.41, 'TiO2':0.02, 'Al2O3':1.90, 'FeO':7.16,'CaO':1.08,'MgO':43.82,'MnO':0.130,'K2O':0.03,'Na2O':0.04, 'P2O5':0.030,'Cr2O3':0.45,'NiO':0.294}  # run 30.14
# source_phase_Walter1998 = {'ol':66.8,'opx':33.2,'cpx':0,'gt':0,'sp':0}   # run 30.14
# source_phase_Walter1998_lowP = {'ol':72,'opx':24.9,'cpx':3.1,'gt':0,'sp':0}   # run 30.14
# source_wt_Walter1998 = {'SiO2':43.41, 'TiO2':0.01, 'Al2O3':0.66, 'FeO':7.02,'CaO':0.38,'MgO':48.53,'MnO':0.108,'K2O':0.03,'Na2O':0.01, 'P2O5':0.030,'Cr2O3':0.32,'NiO':0.332}  # run 30.1
# source_phase_Walter1998 = {'ol':84.2,'opx':15.8,'cpx':0,'gt':0,'sp':0}   # run 30.1
# source_phase_Walter1998_lowP = {'ol':89.2,'opx':10.8,'cpx':0,'gt':0,'sp':0}   # run 30.1
# source_wt_Walter1998 = {'SiO2':40.96, 'TiO2':0, 'Al2O3':0.19, 'FeO':6.74,'CaO':0.20,'MgO':51.99,'MnO':0.090,'K2O':0.03,'Na2O':0, 'P2O5':0.030,'Cr2O3':0.27,'NiO':0.359}  # run 30.11
# source_phase_Walter1998 = {'ol':100,'opx':0,'cpx':0,'gt':0,'sp':0}   # run 30.11
# source_phase_Walter1998_lowP = {'ol':100,'opx':0,'cpx':0,'gt':0,'sp':0}   # run 30.11
# source_wt_Walter1998 = {'SiO2':45.65, 'TiO2':0.13, 'Al2O3':4.31, 'FeO':7.87,'CaO':3.50,'MgO':37.88,'MnO':0.141,'K2O':0.03,'Na2O':0.25, 'P2O5':0.030,'Cr2O3':0.34,'NiO':0.237}  # run 40.02
# source_phase_Walter1998 = {'ol':53.6,'opx':5.6,'cpx':27.9,'gt':12.9,'sp':0}   # run 40.02
# source_phase_Walter1998_lowP = {'ol':52.4,'opx':30.1,'cpx':15.4,'gt':0,'sp':2.1}   # run 40.02
# source_wt_Walter1998 = {'SiO2':45.37, 'TiO2':0.11, 'Al2O3':4.32, 'FeO':7.71,'CaO':3.56,'MgO':37.73,'MnO':0.137,'K2O':0.03,'Na2O':0.23, 'P2O5':0.030,'Cr2O3':0.33,'NiO':0.246}  # run 40.08
# source_phase_Walter1998 = {'ol':53.3,'opx':0,'cpx':35.7,'gt':11.0,'sp':0}   # run 40.08
# source_phase_Walter1998_lowP = {'ol':52.9,'opx':28.6,'cpx':16.2,'gt':11.0,'sp':2.3}   # run 40.08
# source_wt_Walter1998 = {'SiO2':45.27, 'TiO2':0.06, 'Al2O3':3.76, 'FeO':7.42,'CaO':3.10,'MgO':39.68,'MnO':0.137,'K2O':0.03,'Na2O':0.18, 'P2O5':0.030,'Cr2O3':0.36,'NiO':0.262}  # run 40.06
# source_phase_Walter1998 = {'ol':57.6,'opx':0,'cpx':33.6,'gt':8.8,'sp':0}   # run 40.06
# source_phase_Walter1998_lowP = {'ol':58.6,'opx':27,'cpx':12.9,'gt':0,'sp':1.5}   # run 40.06
# source_wt_Walter1998 = {'SiO2':45.34, 'TiO2':0.04, 'Al2O3':3.39, 'FeO':7.40,'CaO':2.64,'MgO':40.34,'MnO':0.125,'K2O':0.03,'Na2O':0.18, 'P2O5':0.030,'Cr2O3':0.40,'NiO':0.267}  # run 40.07
# source_phase_Walter1998 = {'ol':59,'opx':8.2,'cpx':26.8,'gt':6.0,'sp':0}   # run 40.07
# source_phase_Walter1998_lowP = {'ol':60,'opx':28.4,'cpx':10.5,'gt':0,'sp':1.1}   # run 40.07
# source_wt_Walter1998 = {'SiO2':43.88, 'TiO2':0.01, 'Al2O3':1.17, 'FeO':6.64,'CaO':0.55,'MgO':46.67,'MnO':0.110,'K2O':0.03,'Na2O':0.02, 'P2O5':0.030,'Cr2O3':0.31,'NiO':0.317}  # run 40.05
# source_phase_Walter1998 = {'ol':76.8,'opx':23.2,'cpx':0,'gt':0,'sp':0}   # run 40.05
# source_phase_Walter1998_lowP = {'ol':81.3,'opx':18.7,'cpx':0,'gt':0,'sp':0}   # run 40.05
# source_wt_Walter1998 = {'SiO2':44.81, 'TiO2':0.11, 'Al2O3':4.25, 'FeO':8.04,'CaO':3.77,'MgO':37.26,'MnO':0.140,'K2O':0.03,'Na2O':0.25, 'P2O5':0.030,'Cr2O3':0.34,'NiO':0.243}  # run 45.07
# source_phase_Walter1998 = {'ol':53,'opx':0,'cpx':34,'gt':13,'sp':0}   # run 45.07
# source_phase_Walter1998_lowP = {'ol':54.2,'opx':24.7,'cpx':18.4,'gt':0,'sp':2.7}   # run 45.07
# source_wt_Walter1998 = {'SiO2':44.81, 'TiO2':0.07, 'Al2O3':3.76, 'FeO':7.14,'CaO':2.89,'MgO':39.92,'MnO':0.137,'K2O':0.03,'Na2O':0.20, 'P2O5':0.030,'Cr2O3':0.33,'NiO':0.264}  # run 45.03
# source_phase_Walter1998 = {'ol':59,'opx':0,'cpx':29.4,'gt':11.6,'sp':0}   # run 45.03
# source_phase_Walter1998_lowP = {'ol':60,'opx':25.9,'cpx':12.3,'gt':0,'sp':1.8}   # run 45.03
# source_wt_Walter1998 = {'SiO2':44.88, 'TiO2':0.01, 'Al2O3':1.53, 'FeO':6.44,'CaO':0.86,'MgO':45.43,'MnO':0.111,'K2O':0.03,'Na2O':0.05, 'P2O5':0.030,'Cr2O3':0.31,'NiO':0.307}  # run 45.02
# source_phase_Walter1998 = {'ol':70.1,'opx':28.3,'cpx':0,'gt':1.6,'sp':0}   # run 45.02
# source_phase_Walter1998_lowP = {'ol':74.1,'opx':25.6,'cpx':0.3,'gt':0,'sp':0}   # run 45.02
# source_wt_Walter1998 = {'SiO2':45.29, 'TiO2':0.06, 'Al2O3':3.98, 'FeO':7.28,'CaO':2.92,'MgO':39.35,'MnO':0.137,'K2O':0.03,'Na2O':0.18, 'P2O5':0.030,'Cr2O3':0.37,'NiO':0.259}  # run 50.01
# source_phase_Walter1998 = {'ol':56.2,'opx':0,'cpx':31.4,'gt':12.3,'sp':0}   # run 50.01
# source_phase_Walter1998_lowP = {'ol':56.6,'opx':29.3,'cpx':12.2,'gt':0,'sp':1.9}   # run 50.01
# source_wt_IonovMgO36 = {'SiO2':45.92, 'TiO2':0.2, 'Al2O3':4.98, 'FeO':7.824,'CaO':4.19,'MgO':36,'MnO':0.1375,'K2O':0.013,'Na2O':0.386, 'P2O5':0.013,'Cr2O3':0.38,'NiO':0.233}  # Ionov peridotite when MgO=36
# source_wt_IonovMgO36 = {'SiO2':45.92, 'TiO2':0.2, 'Al2O3':4.98, 'FeO':7.89,'CaO':4.19,'MgO':36,'MnO':0.1365,'K2O':0.013,'Na2O':0.386, 'P2O5':0.013,'Cr2O3':0.38,'NiO':0.233}  # Ionov peridotite when MgO=36
# source_phase_IonovMgO36 = {'ol':49.5,'opx':0,'cpx':34.9,'gt':15.6,'sp':0}   # phase mode for Ionov pe MgO36 calculated using minerals from Walter 1998 run 40.2
# source_phase_IonovMgO36 = {'ol':54.9,'opx':7.1,'cpx':24.4,'gt':13.6,'sp':0}   # phase mode for Ionov pe MgO36 calculated using minerals from Davis et al 2011 3 GPa KLB-1 subsolidus
# source_phase_IonovMgO36 = {'ol':49.8,'opx':18.7,'cpx':18.2,'gt':13.3,'sp':0}   # phase mode for Ionov pe MgO36 calculated using minerals from Ionov et al. 1993 average minerals
# source_phase_IonovMgO36_lowP = {'ol':37,'opx':39,'cpx':19.8,'gt':0,'sp':4.2}   # phase mode for Ionov pe MgO36 calculated using minerals from Baker and Stolper 1994 run 20
# source_phase_IonovMgO36_lowP = {'ol':48,'opx':33,'cpx':17,'gt':0,'sp':2}   # phase mode for Ionov pe MgO36 calculated using minerals from Workman and Hart 2005 mantle minerals
# source_wt_IonovMgO385 = {'SiO2':45.27, 'TiO2':0.158, 'Al2O3':4.03, 'FeO':7.872,'CaO':3.36,'MgO':38.5,'MnO':0.136,'K2O':0.013,'Na2O':0.306, 'P2O5':0.013,'Cr2O3':0.38,'NiO':0.252}  # Ionov peridotite when MgO=38.5
source_wt_IonovMgO385 = {'SiO2':45.27, 'TiO2':0.158, 'Al2O3':4.03, 'FeO':7.872,'CaO':3.36,'MgO':38.5,'MnO':0.1362,'K2O':0.013,'Na2O':0.306, 'P2O5':0.013,'Cr2O3':0.38,'NiO':0.252}  # Ionov peridotite when MgO=38.5
# source_phase_IonovMgO385 = {'ol':54.9,'opx':6.7,'cpx':26.8,'gt':11.6,'sp':0}   # phase mode for Ionov pe MgO38.5 calculated using minerals from Walter 1998 run 40.2
# source_phase_IonovMgO385 = {'ol':55,'opx':7.5,'cpx':28,'gt':9.5,'sp':0}   # modifed after phase mode calculated by Walter 1998 run 40.2 minerals
# source_phase_IonovMgO385 = {'ol':59.3,'opx':12.4,'cpx':18.7,'gt':9.6,'sp':0}   # phase mode for Ionov pe MgO38.5 calculated using minerals from Davis et al 2011 3 GPa KLB-1 subsolidus
# source_phase_IonovMgO385 = {'ol':58.4,'opx':16.4,'cpx':14.5,'gt':10.7,'sp':0}   # phase mode for Ionov pe MgO38.5 calculated using minerals from Ionov et al. 1993 average minerals
# source_phase_IonovMgO385_lowP = {'ol':46.9,'opx':34.6,'cpx':15.4,'gt':0,'sp':3.1}   # phase mode for Ionov pe MgO38.5 calculated using minerals from Baker and Stolper 1994 run 20
source_phase_IonovMgO385_lowP = {'ol':56.5,'opx':27.5,'cpx':14,'gt':0,'sp':2}   # phase mode for Ionov pe MgO38.5 calculated using minerals from Workman and Hart 2005 mantle minerals
# source_wt_IonovMgO41 = {'SiO2':44.62, 'TiO2':0.107, 'Al2O3':3.08, 'FeO':7.728,'CaO':2.54,'MgO':41,'MnO':0.1322,'K2O':0.013,'Na2O':0.226, 'P2O5':0.013,'Cr2O3':0.38,'NiO':0.272}  # Ionov peridotite when MgO=41
# source_wt_IonovMgO41 = {'SiO2':44.62, 'TiO2':0.107, 'Al2O3':3.08, 'FeO':7.776,'CaO':2.54,'MgO':41,'MnO':0.1345,'K2O':0.013,'Na2O':0.226, 'P2O5':0.013,'Cr2O3':0.38,'NiO':0.272}  # Ionov peridotite when MgO=41
# source_phase_IonovMgO41 = {'ol':64.3,'opx':8,'cpx':19.3,'gt':8.4,'sp':0}   # phase mode for Ionov pe MgO41 calculated using minerals from Walter 1998 run 40.2
# source_phase_IonovMgO41 = {'ol':64.4,'opx':16.8,'cpx':12.9,'gt':5.9,'sp':0}   # phase mode for Ionov pe MgO41 calculated using minerals from Davis et al 2011 3 GPa KLB-1 subsolidus
# source_phase_IonovMgO41 = {'ol':68.1,'opx':13.0,'cpx':10.8,'gt':8.1,'sp':0}   # phase mode for Ionov pe MgO41 calculated using minerals from Ionov et al. 1993 average minerals
# source_phase_IonovMgO41_lowP = {'ol':56.8,'opx':30.1,'cpx':11,'gt':0,'sp':2.1}   # phase mode for Ionov pe MgO41 calculated using minerals from Baker and Stolper 1994 run 20
# source_phase_IonovMgO41_lowP = {'ol':62.6,'opx':26.5,'cpx':9.9,'gt':0,'sp':1}   # phase mode for Ionov pe MgO41 calculated using minerals from Workman and Hart 2005 mantle minerals
source_wt_IonovMgO385_eclope = {'SiO2':45.6, 'TiO2':0.3, 'Al2O3':4.37, 'FeO':7.87,'CaO':3.5,'MgO':37.4,'MnO':0.135,'K2O':0.021,'Na2O':0.415, 'P2O5':0.028,'Cr2O3':0.37,'NiO':0.245} 
source_phase_IonovMgO385_eclope = {'ol':53.2,'opx':10.5,'cpx':27.1,'gt':9.2,'sp':0}  # add 3% eclogite melt to a modifed mineral mode for MgO 38.5 peridotite which is 55ol+7.5opx+28cpx+9.5grt (adding solidus eclogite melt will first increase opx in peridotite, Yaxley and Green 1998) 
source_wt_Haw = source_wt_IonovMgO385_eclope
source_phase_Haw = source_phase_IonovMgO385_eclope
source_wt_MORB = source_wt_IonovMgO385
source_phase_MORB = source_phase_IonovMgO385_lowP

## high-P melting
# input parameters:
# source compositions in wt.%, initial mineral phases in percent, initial pressure Po in kbar, melting model (polybaric or isobaric), consider Sulfur or not
#source_wt = {'SiO2':45.2, 'TiO2':0.16, 'Al2O3':4, 'FeO':8.1,'CaO':3.3,'MgO':38.8,'MnO':0.137,'K2O':0.013,'Na2O':0.3, 'P2O5':0.014,'Cr2O3':0.38,'NiO':0.244}  ## peridotite for both MORB and Haw for old KdMg_oll
#source_wt = {'SiO2':45.6, 'TiO2':0.39, 'Al2O3':4.4, 'FeO':8.1,'CaO':3.4,'MgO':37.5,'MnO':0.135,'K2O':0.032,'Na2O':0.43, 'P2O5':0.031,'Cr2O3':0.37,'NiO':0.236}  ## add 3.5% eclogite with peridotite for MORB for old KdMg_oll
#source_wt = {'SiO2':46.1, 'TiO2':0.33, 'Al2O3':4.6, 'FeO':8,'CaO':3.7,'MgO':36.2,'MnO':0.129,'K2O':0.02,'Na2O':0.458, 'P2O5':0.027,'Cr2O3':0.36,'NiO':0.243} 
#source_wt = {'SiO2':45.5, 'TiO2':0.37, 'Al2O3':4.3, 'FeO':8.1,'CaO':3.4,'MgO':37.7,'MnO':0.135,'K2O':0.03,'Na2O':0.41, 'P2O5':0.029,'Cr2O3':0.37,'NiO':0.237}  ## add 3% eclogite with peridotite for MORB for old KdMg_oll
#source_wt = {'SiO2':45.8, 'TiO2':0.39, 'Al2O3':4.4, 'FeO':7.62,'CaO':3.6,'MgO':36.7,'MnO':0.135,'K2O':0.032,'Na2O':0.45, 'P2O5':0.031,'Cr2O3':0.37,'NiO':0.241} ## add 3.5% eclogite with peridotite for MORB for new KdMg_oll
#source_wt = {'SiO2':44.8, 'TiO2':0.35, 'Al2O3':3.9, 'FeO':7.74,'CaO':3.6,'MgO':38.1,'MnO':0.124,'K2O':0.037,'Na2O':0.41, 'P2O5':0.044,'Cr2O3':0.30,'NiO':0.244}  ## add 3% eclogite with KLB-1
#source_wt = {'SiO2':45.8, 'TiO2':0.18, 'Al2O3':4.4, 'FeO':8,'CaO':3.6,'MgO':37,'MnO':0.132,'K2O':0.013,'Na2O':0.32, 'P2O5':0.014,'Cr2O3':0.38,'NiO':0.255}  ## ideal peridotite for Haw
#source_phase = {'ol':53.5,'opx':11.0,'cpx':27.5,'gt':8,'sp':0}  ## add 3% eclogite melt adjust the mineral phase (56ol,7.5opx,28cpx,8.5gt, adding solidus eclogite melt will first increase opx in peridotite, Yaxley and Green 1998)
#source_phase = {'ol':58,'opx':11,'cpx':22.3,'gt':8.7,'sp':0}  ## add 3% eclogite melt adjust the mineral phase of Davis etal 2011 EPSL Table1 subsolidus mode of KLB-1 at 3 GPa (60ol,8opx,23cpx,9gt)
#source_phase = {'ol':51.6,'opx':9.1,'cpx':26.9,'gt':12.4,'sp':0}  ## add 3.5% eclogite melt adjust the mineral phase of Walter 1998 4 GPa (53.6ol,5.6opx,27.9cpx,12.9gt)
#source_phase = {'ol':51.9,'opx':8.6,'cpx':27,'gt':12.5,'sp':0}  ## add 3% eclogite melt adjust the mineral phase of Walter 1998 4 GPa (53.6ol,5.6opx,27.9cpx,12.9gt)
#source_phase = {'ol':53.6,'opx':5.6,'cpx':27.9,'gt':12.9,'sp':0}  ## the mineral phase of Walter 1998 4 GPa (53.6ol,5.6opx,27.9cpx,12.9gt)
source_wt = source_wt_Haw
source_phase = source_phase_Haw
Po = Po_high  # >=30 is high-pressure, <30 is low-pressure
melting_model = melting_model_Haw
S_mode = S_mode_Haw 

# parameters used in calculating mineral phases during isobaric equilibrium melting
source_phase2 = source_phase 
source_phase3 = source_phase
source_phase4 = source_phase
f_gt0 = 0.0000001
f_cpx0 = 0.0000001
f_sp0 = 0.0000001

# calculate melting
source_cm, mgnumber_source = wttocm(source_wt)
if melting_model == 'polybaric':  # polybaric fractional melting
    ## calculate all parameters near solidus assuming the the extent of melting is 0.0000001
    f = 0.0000001  
    f_step = 0.0000001
    T = 13*Po+1140+600*(1-Po/88)*f+20*(mgnumber_source-89)
    crust_thickness = 0
    p_remain = -Po
    P = Po
    if S_mode == 'Y':
        S_res = S_mantle 
        f_sulfide,S_res = S_polyfrac(S_mantlesulfide,S_res,f_step,deltaQFM,SCSS_QFM_MORB)
        f_mineral, phase_tot = mineral_phase_polyfrac(Po,P,f_step,f_mineral=source_phase)
        f_mineral, f_sulfide = mineral_phase_S(f_sulfide,f_mineral)
        keys = ['MgO','FeO','TiO2', 'Na2O', 'K2O','NiO','MnO']
        res = {key:source_wt[key] for key in keys}
        res['MgO'] = source_cm['MgO']*100
        res['FeO'] = source_cm['FeO']*100
        res['mgnumber'] = 0.0
        cl_wt = {key:0.0 for key in keys}
        cl_wt['SiO2'] = 0.0
        cl_cm = {'MgO':0.0,'FeO':0.0,'Na2O':0.0,'K2O':0.0}
        bulkD = {'K2O':0.005,'Na2O':0.0,'TiO2':0.0,'Ni':0.0,'Mn':0.0}
        cl_wt,bulkD,cl_cm,res = liquid_wt_polyfrac(res,f_step,f_mineral,P,T,cl_wt,cl_cm,bulkD,Po)
        cl_molar = {'SiO2':0.0,'Na2O':0.0,'K2O':0.0}
        ol = {'MgOcm':0.0,'FeOcm':0.0,'Fo':0.0,'NiOwt':0.0,'MnOwt':0.0}
        cl_molar,clSiO2_adjust,kdMgO_oll_cm,ol,kdFe2Mg_oll = KDFeMg(T,P,f_step,cl_wt,cl_cm,res,cl_molar,ol)
        ol,cl_cm,res,kdFeO_oll_cm,cl_wt = MgOFeO_polyfrac(ol,kdMgO_oll_cm,f_step,kdFe2Mg_oll,res,cl_cm,cl_wt)
        kdNi_wt, bulkD, cl_wt, ol, res = Ni_polyfrac_S(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol,f_sulfideliquid,f_sulfide)
        kdMn_wt, bulkD, cl_wt, ol, res = Mn_polyfrac(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol,Po,kdFeO_oll_cm)
    else:
        f_mineral, phase_tot = mineral_phase_polyfrac(Po,P,f_step,f_mineral=source_phase)
        keys = ['MgO','FeO','TiO2', 'Na2O', 'K2O','NiO','MnO']
        res = {key:source_wt[key] for key in keys}
        res['MgO'] = source_cm['MgO']*100
        res['FeO'] = source_cm['FeO']*100
        res['mgnumber'] = 0.0
        cl_wt = {key:0.0 for key in keys}
        cl_wt['SiO2'] = 0.0
        cl_cm = {'MgO':0.0,'FeO':0.0,'Na2O':0.0,'K2O':0.0}
        bulkD = {'K2O':0.005,'Na2O':0.0,'TiO2':0.0,'Ni':0.0,'Mn':0.0}
        cl_wt,bulkD,cl_cm,res = liquid_wt_polyfrac(res,f_step,f_mineral,P,T,cl_wt,cl_cm,bulkD,Po)
        cl_molar = {'SiO2':0.0,'Na2O':0.0,'K2O':0.0}
        ol = {'MgOcm':0.0,'FeOcm':0.0,'Fo':0.0,'NiOwt':0.0,'MnOwt':0.0}
        cl_molar,clSiO2_adjust,kdMgO_oll_cm,ol,kdFe2Mg_oll = KDFeMg(T,P,f_step,cl_wt,cl_cm,res,cl_molar,ol)
        ol,cl_cm,res,kdFeO_oll_cm,cl_wt = MgOFeO_polyfrac(ol,kdMgO_oll_cm,f_step,kdFe2Mg_oll,res,cl_cm,cl_wt)
        kdNi_wt, bulkD, cl_wt, ol, res = Ni_polyfrac(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol)
        kdMn_wt, bulkD, cl_wt, ol, res = Mn_polyfrac(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol,Po,kdFeO_oll_cm)
    ## format data
    F_mineral = {element: [] for element in f_mineral}
    Res = {element: [] for element in res}  
    Cl_wt = {element: [] for element in cl_wt}
    Cl_cm = {element: [] for element in cl_cm}
    KdNi_wt = {element: [] for element in kdNi_wt}
    KdMn_wt = {element: [] for element in kdMn_wt}
    BulkD = {element: [] for element in bulkD}
    Cl_molar = {element: [] for element in cl_molar}
    ClSiO2_adjust = []
    Ol = {element: [] for element in ol}
    Phase_tot = []
    T_melting = []
    P_melting = []
    F_melting = []
    F_step = []
    Crust_thickness = []
    P_remain = []
    KdMgO_oll_cm = []
    KdFeO_oll_cm = []
    KdFe2Mg_oll = []
    S_Res = []
    F_sulfide = []
    for element in F_mineral:
        F_mineral[element].append(f_mineral[element])
    for element in Res:
        Res[element].append(res[element])
    for element in Cl_wt:
        Cl_wt[element].append(cl_wt[element])
    for element in Cl_cm:
        Cl_cm[element].append(cl_cm[element])
    for element in KdNi_wt:
        KdNi_wt[element].append(kdNi_wt[element])
    for element in KdMn_wt:
        KdMn_wt[element].append(kdMn_wt[element])
    for element in BulkD:
        BulkD[element].append(bulkD[element])
    for element in Cl_molar:
        Cl_molar[element].append(cl_molar[element])
    for element in Ol:
        Ol[element].append(ol[element])
    ClSiO2_adjust.append(clSiO2_adjust)
    Phase_tot.append(phase_tot)
    T_melting.append(T)
    P_melting.append(P)
    F_melting.append(f)
    F_step.append(f_step)
    Crust_thickness.append(crust_thickness)
    P_remain.append(p_remain)
    KdMgO_oll_cm.append(kdMgO_oll_cm)
    KdFeO_oll_cm.append(kdFeO_oll_cm)
    KdFe2Mg_oll.append(kdFe2Mg_oll)
    if S_mode == 'Y':
        S_Res.append(S_res)
        F_sulfide.append(f_sulfide)
    ## melting stops when the top of melting column reaches to the bottom of the crust
    while p_remain <=0:
        T, P, f, f_step, crust_thickness, p_remain = TPF_polyfrac(P,f,mgnumber_source,Po)
        if S_mode == 'Y':
            f_sulfide,S_res = S_polyfrac(S_mantlesulfide,S_res,f_step,deltaQFM,SCSS_QFM_MORB)
            f_mineral, phase_tot = mineral_phase_polyfrac(Po,P,f_step,f_mineral)
            f_mineral, f_sulfide = mineral_phase_S(f_sulfide,f_mineral)
            cl_wt,bulkD,cl_cm,res = liquid_wt_polyfrac(res,f_step,f_mineral,P,T,cl_wt,cl_cm,bulkD,Po)
            cl_molar,clSiO2_adjust,kdMgO_oll_cm,ol,kdFe2Mg_oll = KDFeMg(T,P,f_step,cl_wt,cl_cm,res,cl_molar,ol)
            ol,cl_cm,res,kdFeO_oll_cm,cl_wt = MgOFeO_polyfrac(ol,kdMgO_oll_cm,f_step,kdFe2Mg_oll,res,cl_cm,cl_wt)
            kdNi_wt, bulkD, cl_wt, ol, res = Ni_polyfrac_S(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol,f_sulfideliquid,f_sulfide)
            kdMn_wt, bulkD, cl_wt, ol, res = Mn_polyfrac(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol,Po,kdFeO_oll_cm)
        else:
            f_mineral, phase_tot = mineral_phase_polyfrac(Po,P,f_step,f_mineral)
            cl_wt,bulkD,cl_cm,res = liquid_wt_polyfrac(res,f_step,f_mineral,P,T,cl_wt,cl_cm,bulkD,Po)
            cl_molar,clSiO2_adjust,kdMgO_oll_cm,ol,kdFe2Mg_oll = KDFeMg(T,P,f_step,cl_wt,cl_cm,res,cl_molar,ol)
            ol,cl_cm,res,kdFeO_oll_cm,cl_wt = MgOFeO_polyfrac(ol,kdMgO_oll_cm,f_step,kdFe2Mg_oll,res,cl_cm,cl_wt)
            kdNi_wt, bulkD, cl_wt, ol, res = Ni_polyfrac(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol)
            kdMn_wt, bulkD, cl_wt, ol, res = Mn_polyfrac(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol,Po,kdFeO_oll_cm)
        for element in F_mineral:
            F_mineral[element].append(f_mineral[element])
        for element in Res:
            Res[element].append(res[element])
        for element in Cl_wt:
            Cl_wt[element].append(cl_wt[element])
        for element in Cl_cm:
            Cl_cm[element].append(cl_cm[element])
        for element in KdNi_wt:
            KdNi_wt[element].append(kdNi_wt[element])
        for element in KdMn_wt:
            KdMn_wt[element].append(kdMn_wt[element])
        for element in BulkD:
            BulkD[element].append(bulkD[element])
        for element in Cl_molar:
            Cl_molar[element].append(cl_molar[element])
        for element in Ol:
            Ol[element].append(ol[element])
        ClSiO2_adjust.append(clSiO2_adjust)
        Phase_tot.append(phase_tot)
        T_melting.append(T)
        P_melting.append(P)
        F_melting.append(f)
        F_step.append(f_step)
        Crust_thickness.append(crust_thickness)
        P_remain.append(p_remain)
        KdMgO_oll_cm.append(kdMgO_oll_cm)
        KdFeO_oll_cm.append(kdFeO_oll_cm)
        KdFe2Mg_oll.append(kdFe2Mg_oll)
        if S_mode == 'Y':
            S_Res.append(S_res)
            F_sulfide.append(f_sulfide)         
    ## calcluate the accumulated melt compositions for polybaric fractional melting    
    Cl_wt_itg1,Cl_wt_itg2,F_melting_itg1,F_melting_itg2 = itg(Cl_wt,F_step,F_melting,Po) 

elif melting_model == 'isobaric':  # isobaric equilibrium melting 
    P = Po  # pressure will be constant during the melting
    f = 0.0000001  # calculate all parameters near solidus assuming the the extent of melting is 0.0000001 
    f_step = 0.0000001
    T = 13*Po+1140+600*(1-Po/88)*f+20*(mgnumber_source-89)
    if S_mode == 'Y':
        S_res = S_mantle 
        f_sulfide,S_res = S_isoequ(S_mantle,S_mantlesulfide,S_res,f,deltaQFM,SCSS_QFM_MORB)
        f_mineral = source_phase.copy()
        f_mineral, phase_tot,source_phase2,source_phase3,source_phase4,f_gt0,f_cpx0,f_sp0 = mineral_phase_isoequ(Po,P,f,source_phase,source_phase2,f_gt0,source_phase3,source_phase4,f_cpx0,f_sp0,f_mineral)
        f_mineral, f_sulfide = mineral_phase_S(f_sulfide,f_mineral)
        keys = ['MgO','FeO','TiO2','Na2O', 'K2O','NiO','MnO']
        res = {key:source_wt[key] for key in keys}
        res['MgO'] = source_cm['MgO']*100
        res['FeO'] = source_cm['FeO']*100
        res['mgnumber'] = 0.0
        cl_wt = {key:0.0 for key in keys}
        cl_wt['SiO2'] = 0.0
        cl_cm = {'MgO':0.0,'FeO':0.0,'Na2O':0.0,'K2O':0.0}
        bulkD = {'K2O':0.005,'Na2O':0.0,'TiO2':0.0,'Ni':0.0,'Mn':0.0}
        cl_wt,bulkD,cl_cm,res = liquid_wt_isoequ(source_wt,f,f_mineral,P,T,cl_wt,cl_cm,bulkD,res,Po)
        cl_molar = {'SiO2':0.0,'Na2O':0.0,'K2O':0.0}
        ol = {'MgOcm':0.0,'FeOcm':0.0,'Fo':0.0,'NiOwt':0.0,'MnOwt':0.0}
        cl_molar,clSiO2_adjust,kdMgO_oll_cm,ol,kdFe2Mg_oll = KDFeMg(T,P,f_step,cl_wt,cl_cm,res,cl_molar,ol)
        ol,cl_cm,res,kdFeO_oll_cm,cl_wt = MgOFeO_isoequ(ol,kdMgO_oll_cm,f,kdFe2Mg_oll,res,cl_cm,cl_wt,source_cm)
        kdNi_wt, bulkD, cl_wt, ol, res = Ni_isoequ_S(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt,f_sulfideliquid,f_sulfide)
        kdMn_wt, bulkD, cl_wt, ol, res = Mn_isoequ(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt,Po,kdFeO_oll_cm)        
    else:
        f_mineral = source_phase.copy()
        f_mineral, phase_tot,source_phase2,source_phase3,source_phase4,f_gt0,f_cpx0,f_sp0 = mineral_phase_isoequ(Po,P,f,source_phase,source_phase2,f_gt0,source_phase3,source_phase4,f_cpx0,f_sp0,f_mineral)
        keys = ['MgO','FeO','TiO2','Na2O', 'K2O','NiO','MnO']
        res = {key:source_wt[key] for key in keys}
        res['MgO'] = source_cm['MgO']*100
        res['FeO'] = source_cm['FeO']*100
        res['mgnumber'] = 0.0
        cl_wt = {key:0.0 for key in keys}
        cl_wt['SiO2'] = 0.0
        cl_cm = {'MgO':0.0,'FeO':0.0,'Na2O':0.0,'K2O':0.0}
        bulkD = {'K2O':0.005,'Na2O':0.0,'TiO2':0.0,'Ni':0.0,'Mn':0.0}
        cl_wt,bulkD,cl_cm,res = liquid_wt_isoequ(source_wt,f,f_mineral,P,T,cl_wt,cl_cm,bulkD,res,Po)
        cl_molar = {'SiO2':0.0,'Na2O':0.0,'K2O':0.0}
        ol = {'MgOcm':0.0,'FeOcm':0.0,'Fo':0.0,'NiOwt':0.0,'MnOwt':0.0}
        cl_molar,clSiO2_adjust,kdMgO_oll_cm,ol,kdFe2Mg_oll = KDFeMg(T,P,f_step,cl_wt,cl_cm,res,cl_molar,ol)
        ol,cl_cm,res,kdFeO_oll_cm,cl_wt = MgOFeO_isoequ(ol,kdMgO_oll_cm,f,kdFe2Mg_oll,res,cl_cm,cl_wt,source_cm)
        kdNi_wt, bulkD, cl_wt, ol, res = Ni_isoequ(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt)
        kdMn_wt, bulkD, cl_wt, ol, res = Mn_isoequ(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt,Po,kdFeO_oll_cm)
    ## format data
    F_mineral = {element: [] for element in f_mineral}
    Res = {element: [] for element in res}  
    Cl_wt = {element: [] for element in cl_wt}
    Cl_cm = {element: [] for element in cl_cm}
    KdNi_wt = {element: [] for element in kdNi_wt}
    KdMn_wt = {element: [] for element in kdMn_wt}
    BulkD = {element: [] for element in bulkD}
    Cl_molar = {element: [] for element in cl_molar}
    ClSiO2_adjust = []
    Ol = {element: [] for element in ol}
    Phase_tot = []
    T_melting = []
    P_melting = []
    F_melting = []
    F_step = []
    KdMgO_oll_cm = []
    KdFeO_oll_cm = []
    KdFe2Mg_oll = []
    S_Res = []
    F_sulfide = []
    for element in F_mineral:
        F_mineral[element].append(f_mineral[element])
    for element in Res:
        Res[element].append(res[element])
    for element in Cl_wt:
        Cl_wt[element].append(cl_wt[element])
    for element in Cl_cm:
        Cl_cm[element].append(cl_cm[element])
    for element in KdNi_wt:
        KdNi_wt[element].append(kdNi_wt[element])
        KdMn_wt[element].append(kdMn_wt[element])
    for element in BulkD:
        BulkD[element].append(bulkD[element])
    for element in Cl_molar:
        Cl_molar[element].append(cl_molar[element])
    for element in Ol:
        Ol[element].append(ol[element])
    ClSiO2_adjust.append(clSiO2_adjust)
    Phase_tot.append(phase_tot)
    T_melting.append(T)
    P_melting.append(Po)
    F_melting.append(f)
    F_step.append(f_step)
    KdMgO_oll_cm.append(kdMgO_oll_cm)
    KdFeO_oll_cm.append(kdFeO_oll_cm)
    KdFe2Mg_oll.append(kdFe2Mg_oll)
    if S_mode == 'Y':
        S_Res.append(S_res)
        F_sulfide.append(f_sulfide)
    ## melting stops when the extent of melting reaches to about 50%
    while f <=0.5:
        T,f,f_step = TPF_isoequ(P,f,mgnumber_source)
        if S_mode == 'Y':
            f_sulfide,S_res = S_isoequ(S_mantle,S_mantlesulfide,S_res,f,deltaQFM,SCSS_QFM_MORB)
            f_mineral, phase_tot,source_phase2,source_phase3,source_phase4,f_gt0,f_cpx0,f_sp0 = mineral_phase_isoequ(Po,P,f,source_phase,source_phase2,f_gt0,source_phase3,source_phase4,f_cpx0,f_sp0,f_mineral)
            f_mineral, f_sulfide = mineral_phase_S(f_sulfide,f_mineral)
            cl_wt,bulkD,cl_cm,res = liquid_wt_isoequ(source_wt,f,f_mineral,P,T,cl_wt,cl_cm,bulkD,res,Po)
            cl_molar,clSiO2_adjust,kdMgO_oll_cm,ol,kdFe2Mg_oll = KDFeMg(T,P,f_step,cl_wt,cl_cm,res,cl_molar,ol)
            ol,cl_cm,res,kdFeO_oll_cm,cl_wt = MgOFeO_isoequ(ol,kdMgO_oll_cm,f,kdFe2Mg_oll,res,cl_cm,cl_wt,source_cm)
            kdNi_wt, bulkD, cl_wt, ol, res = Ni_isoequ_S(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt,f_sulfideliquid,f_sulfide)
            kdMn_wt, bulkD, cl_wt, ol, res = Mn_isoequ(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt,Po,kdFeO_oll_cm)
        else:
            f_mineral, phase_tot,source_phase2,source_phase3,source_phase4,f_gt0,f_cpx0,f_sp0 = mineral_phase_isoequ(Po,P,f,source_phase,source_phase2,f_gt0,source_phase3,source_phase4,f_cpx0,f_sp0,f_mineral)
            cl_wt,bulkD,cl_cm,res = liquid_wt_isoequ(source_wt,f,f_mineral,P,T,cl_wt,cl_cm,bulkD,res,Po)
            cl_molar,clSiO2_adjust,kdMgO_oll_cm,ol,kdFe2Mg_oll = KDFeMg(T,P,f_step,cl_wt,cl_cm,res,cl_molar,ol)
            ol,cl_cm,res,kdFeO_oll_cm,cl_wt = MgOFeO_isoequ(ol,kdMgO_oll_cm,f,kdFe2Mg_oll,res,cl_cm,cl_wt,source_cm)
            kdNi_wt, bulkD, cl_wt, ol, res = Ni_isoequ(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt)
            kdMn_wt, bulkD, cl_wt, ol, res = Mn_isoequ(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt,Po,kdFeO_oll_cm)
        for element in F_mineral:
            F_mineral[element].append(f_mineral[element])
        for element in Res:
            Res[element].append(res[element])
        for element in Cl_wt:
            Cl_wt[element].append(cl_wt[element])
        for element in Cl_cm:
            Cl_cm[element].append(cl_cm[element])
        for element in KdNi_wt:
            KdNi_wt[element].append(kdNi_wt[element])
            KdMn_wt[element].append(kdMn_wt[element])
        for element in BulkD:
            BulkD[element].append(bulkD[element])
        for element in Cl_molar:
            Cl_molar[element].append(cl_molar[element])
        for element in Ol:
            Ol[element].append(ol[element])
        ClSiO2_adjust.append(clSiO2_adjust)
        Phase_tot.append(phase_tot)
        T_melting.append(T)
        P_melting.append(Po)
        F_melting.append(f)
        F_step.append(f_step)
        KdMgO_oll_cm.append(kdMgO_oll_cm)
        KdFeO_oll_cm.append(kdFeO_oll_cm)
        KdFe2Mg_oll.append(kdFe2Mg_oll)
        if S_mode == 'Y':
            S_Res.append(S_res)
            F_sulfide.append(f_sulfide)    
            
# output:
dfname = str(Po)+melting_model  # the name of the output file
BulkD_df = pd.DataFrame(BulkD)
BulkD_df.columns = ['DK2O','DNa2O','DTiO2','DNiO','DMnO']
Cl_cm_df = pd.DataFrame(Cl_cm)
Cl_cm_df.columns = ['clMgO_cm','clFeO_cm','clNa2O_cm','clK2O_cm']
Cl_molar_df = pd.DataFrame(Cl_molar)
Cl_molar_df.columns = ['clSiO2_molar','clNa2O_molar','clK2O_molar']
ClSiO2_adjust = {'clSiO2_adjust':ClSiO2_adjust}
ClSiO2_adjust = pd.DataFrame(ClSiO2_adjust)
F_mineral = pd.DataFrame(F_mineral)
F_step = {'f_step':F_step}
F_step = pd.DataFrame(F_step)
KDFeMg_oll = {'kdMgO_oll_cm':KdMgO_oll_cm,'kdFeO_oll_cm':KdFeO_oll_cm,'KDFe2Mg_oll':KdFe2Mg_oll}
KDFeMg_oll = pd.DataFrame(KDFeMg_oll)
KdMn_wt_df = pd.DataFrame(KdMn_wt)
KdMn_wt_df.columns = ['KdMn_oll_wt','KdMn_opxl_wt','KdMn_cpxl_wt','KdMn_gtl_wt','KdMn_spl_wt','KdMn_opxol_wt','KdMn_cpxol_wt','KdMn_gtol_wt','KdMn_spol_wt']
Ol_df = pd.DataFrame(Ol)
Ol_df.columns = ['olMgO_cm','olFeO_cm','Fo','olNiO_wt','olMnO_wt']
P_melting = {'P kbar':P_melting}
P_melting = pd.DataFrame(P_melting)
Phase_tot = {'mineral_phase_tot':Phase_tot}
Phase_tot = pd.DataFrame(Phase_tot)
Res = pd.DataFrame(Res)
Res.columns = ['resMgO_cm','resFeO_cm','resTiO2_wt','resNa2O_wt','resK2O_wt','resNiO_wt','resMnO_wt','resMgnumber']
T_melting = {'T Celsius':T_melting}
T_melting = pd.DataFrame(T_melting)
if melting_model == 'polybaric':
    Cl_wt_itg1_df = pd.DataFrame(Cl_wt_itg1)
    Cl_wt_itg1_df.columns = ['clMgO_wt_itg1','clFeO_wt_itg1','clTiO2_wt_itg1','clNa2O_wt_itg1','clK2O_wt_itg1','clNiO_wt_itg1','clMnO_wt_itg1','clSiO2_wt_itg1']
    Cl_wt_itg2_df = pd.DataFrame(Cl_wt_itg2)
    Cl_wt_itg2_df.columns = ['clMgO_wt_itg2','clFeO_wt_itg2','clTiO2_wt_itg2','clNa2O_wt_itg2','clK2O_wt_itg2','clNiO_wt_itg2','clMnO_wt_itg2','clSiO2_wt_itg2']
    F_melting_itg1 = {'F_liq_itg1':F_melting_itg1}
    F_melting_itg1 = pd.DataFrame(F_melting_itg1)
    F_melting_itg2 = {'F_liq_itg2':F_melting_itg2}
    F_melting_itg2 = pd.DataFrame(F_melting_itg2)
    if S_mode == 'Y':
        KdNi_wt_df = pd.DataFrame(KdNi_wt)
        KdNi_wt_df.columns = ['KdNi_oll_wt','KdNi_opxl_wt','KdNi_cpxl_wt','KdNi_gtl_wt','KdNi_spl_wt','KdNi_surfl_wt','KdNi_opxol_wt','KdNi_cpxol_wt','KdNi_gtol_wt','KdNi_spol_wt']
        F_sulfide = {'F_sulfide':F_sulfide}
        F_sulfide = pd.DataFrame(F_sulfide)
        S_Res = {'S_Res':S_Res}
        S_Res = pd.DataFrame(S_Res)
        melting_df_highP = pd.concat([T_melting,P_melting,F_step,F_mineral,F_sulfide,Phase_tot,F_melting_itg2,Cl_wt_itg2_df,F_melting_itg1,Cl_wt_itg1_df,Ol_df,Cl_cm_df,Cl_molar_df,ClSiO2_adjust,Res,S_Res,KDFeMg_oll,BulkD_df,KdNi_wt_df,KdMn_wt_df],axis=1)
    else:
        KdNi_wt_df = pd.DataFrame(KdNi_wt)
        KdNi_wt_df.columns = ['KdNi_oll_wt','KdNi_opxl_wt','KdNi_cpxl_wt','KdNi_gtl_wt','KdNi_spl_wt','KdNi_opxol_wt','KdNi_cpxol_wt','KdNi_gtol_wt','KdNi_spol_wt']
        melting_df_highP = pd.concat([T_melting,P_melting,F_step,F_mineral,Phase_tot,F_melting_itg2,Cl_wt_itg2_df,F_melting_itg1,Cl_wt_itg1_df,Ol_df,Cl_cm_df,Cl_molar_df,ClSiO2_adjust,Res,KDFeMg_oll,BulkD_df,KdNi_wt_df,KdMn_wt_df],axis=1)        
else:
    if S_mode == 'Y':
        KdNi_wt_df = pd.DataFrame(KdNi_wt)
        KdNi_wt_df.columns = ['KdNi_oll_wt','KdNi_opxl_wt','KdNi_cpxl_wt','KdNi_gtl_wt','KdNi_spl_wt','KdNi_surfl_wt','KdNi_opxol_wt','KdNi_cpxol_wt','KdNi_gtol_wt','KdNi_spol_wt']
        F_sulfide = {'F_sulfide':F_sulfide}
        F_sulfide = pd.DataFrame(F_sulfide)
        S_Res = {'S_Res':S_Res}
        S_Res = pd.DataFrame(S_Res)
    else:
        KdNi_wt_df = pd.DataFrame(KdNi_wt)
        KdNi_wt_df.columns = ['KdNi_oll_wt','KdNi_opxl_wt','KdNi_cpxl_wt','KdNi_gtl_wt','KdNi_spl_wt','KdNi_opxol_wt','KdNi_cpxol_wt','KdNi_gtol_wt','KdNi_spol_wt']
    Cl_wt_df = pd.DataFrame(Cl_wt)
    Cl_wt_df.columns = ['clMgO_wt','clFeO_wt','clTiO2_wt','clNa2O_wt','clK2O_wt','clNiO_wt','clMnO_wt','clSiO2_wt']
    F_melting = {'F_liq':F_melting}
    F_melting = pd.DataFrame(F_melting)
    melting_df_highP = pd.concat([T_melting,P_melting,F_melting,F_step,F_mineral,Phase_tot,Cl_wt_df,Ol_df,Cl_cm_df,Cl_molar_df,ClSiO2_adjust,Res,KDFeMg_oll,BulkD_df,KdNi_wt_df,KdMn_wt_df],axis=1)

# crystallization
F_target = F_target_Haw
if melting_model == 'polybaric':
    ip_magma = abs(melting_df_highP['F_liq_itg1']-F_target).idxmin()
    magma = {'MgO':float(melting_df_highP.loc[ip_magma,'clMgO_wt_itg1']),'FeO':float(melting_df_highP.loc[ip_magma,'clFeO_wt_itg1']),\
             'SiO2':float(melting_df_highP.loc[ip_magma,'clSiO2_wt_itg1']),'Na2O':float(melting_df_highP.loc[ip_magma,'clNa2O_wt_itg1']),\
                 'K2O':float(melting_df_highP.loc[ip_magma,'clK2O_wt_itg1']),'NiO':float(melting_df_highP.loc[ip_magma,'clNiO_wt_itg1']),\
                     'MnO':float(melting_df_highP.loc[ip_magma,'clMnO_wt_itg1'])}
else:
    ip_magma = abs(melting_df_highP['F_liq']-F_target).idxmin()
    magma = {'MgO':float(melting_df_highP.loc[ip_magma,'clMgO_wt']),'FeO':float(melting_df_highP.loc[ip_magma,'clFeO_wt']),\
             'SiO2':float(melting_df_highP.loc[ip_magma,'clSiO2_wt']),'Na2O':float(melting_df_highP.loc[ip_magma,'clNa2O_wt']),\
                 'K2O':float(melting_df_highP.loc[ip_magma,'clK2O_wt']),'NiO':float(melting_df_highP.loc[ip_magma,'clNiO_wt']),\
                     'MnO':float(melting_df_highP.loc[ip_magma,'clMnO_wt'])}
    
# default values
cm_mass = {'MgO':40.304,'FeO':71.844,'SiO2':60.083,'Na2O':30.99,'K2O':47.098} # relative molecular mass, e.g., SiO2, MgO, NaO1.5  
cm_tot = 1.833  # sum of relative cation mole mass, e.g., NaO0.5, SiO2, MgO, to converse between cation mole and wt%, estimated from melt compositions of Walter 1998 and Baker and Stolper 1994
molar_tot = 1.65  # sum of relative molecular mass, e.g., Na2O, SiO2, MgO, to calculate molar mass of SiO2, K2O and Na2O, estimated from melt compositions of Walter 1998 and Baker and Stolper 1994

# input:
# magma compositions in wt%: MgO,FeO,SiO2,Na2O,K2O,NiO,MnO
# magma used in Matzen et al. (2017b)
#magma = {'MgO':16.9,'FeO':9.55*0.85,'SiO2':46.17,'Na2O':0.96,'K2O':0.56,'NiO':0.082,'MnO':0.169}
#magma = {'MgO':17.58,'FeO':8.75*0.85,'SiO2':46.66,'Na2O':0.93,'K2O':0.41,'NiO':0.087,'MnO':0.171}
#magma = {'MgO':18.22,'FeO':8.86*0.85,'SiO2':46.91,'Na2O':0.82,'K2O':0.34,'NiO':0.092,'MnO':0.173}
#magma = {'MgO':19.71,'FeO':9.45*0.85,'SiO2':48.98,'Na2O':0.77,'K2O':0.23,'NiO':0.101,'MnO':0.165}
#magma = {'MgO':23.89,'FeO':9.19*0.85,'SiO2':47.96,'Na2O':0.52,'K2O':0.22,'NiO':0.131,'MnO':0.162}
#magma = {'MgO':18.58,'FeO':10.65*0.85,'SiO2':46.38,'Na2O':0.93,'K2O':0.83,'NiO':0.100,'MnO':0.170}
#magma = {'MgO':19.89,'FeO':10.65*0.85,'SiO2':45.52,'Na2O':1.08,'K2O':0.7,'NiO':0.110,'MnO':0.181}
#magma = {'MgO':22.31,'FeO':9.67*0.85,'SiO2':46.17,'Na2O':0.4,'K2O':0.22,'NiO':0.129,'MnO':0.174}
#magma = {'MgO':20.02,'FeO':11.72*0.85,'SiO2':45.97,'Na2O':1.11,'K2O':0.99,'NiO':0.112,'MnO':0.177}
#magma = {'MgO':24.37,'FeO':10.12*0.85,'SiO2':46.01,'Na2O':0.58,'K2O':0.29,'NiO':0.141,'MnO':0.182}
# magma from mixing between envolved magma at Fo80 and primitive magma for Haw
# magma = {'MgO':7.33,'FeO':9.65,'SiO2':49.77,'Na2O':4.1,'K2O':0.45,'NiO':0.0185,'MnO':0.1721}  # proportion of primitive magma is 10%
# magma = {'MgO':8.51,'FeO':9.69,'SiO2':49.34,'Na2O':3.96,'K2O':0.44,'NiO':0.0300,'MnO':0.1718}  # proportion of primitive magma is 20%
# magma = {'MgO':9.68,'FeO':9.74,'SiO2':48.91,'Na2O':3.83,'K2O':0.42,'NiO':0.0416,'MnO':0.1716}  # proportion of primitive magma is 30%
# magma = {'MgO':10.85,'FeO':9.78,'SiO2':48.48,'Na2O':3.69,'K2O':0.41,'NiO':0.0531,'MnO':0.1714}  # proportion of primitive magma is 40%
# magma = {'MgO':12.03,'FeO':9.83,'SiO2':48.05,'Na2O':3.56,'K2O':0.39,'NiO':0.0647,'MnO':0.1711}  # proportion of primitive magma is 50%
# magma = {'MgO':13.2,'FeO':9.87,'SiO2':47.62,'Na2O':3.43,'K2O':0.38,'NiO':0.0762,'MnO':0.1709}  # proportion of primitive magma is 60%
# magma = {'MgO':14.37,'FeO':9.92,'SiO2':47.2,'Na2O':3.29,'K2O':0.36,'NiO':0.0878,'MnO':0.1706}  # proportion of primitive magma is 70%
# magma = {'MgO':15.54,'FeO':9.96,'SiO2':46.77,'Na2O':3.16,'K2O':0.35,'NiO':0.0993,'MnO':0.1704}  # proportion of primitive magma is 80%
# magma = {'MgO':16.72,'FeO':10.01,'SiO2':46.34,'Na2O':3.02,'K2O':0.33,'NiO':0.1109,'MnO':0.1702}  # proportion of primitive magma is 90%
# magma = {'MgO':17.89,'FeO':10.05,'SiO2':45.91,'Na2O':2.89,'K2O':0.32,'NiO':0.1224,'MnO':0.1699}
# crystallization pressure in kbar
P = 20
cm_magma = cationmole_magma(magma)
if xtalization_model == 'fractional':
    clcm_olonly = cm_magma
    clppm_olonly = {'Ni':magma['NiO']*58.6934/74.69*10**4,'Mn':magma['MnO']*54.938/70.94*10**4}
    clmolar_olonly = {'SiO2':0,'Na2O':0,'K2O':0}
    clmolar_olonly['SiO2'] = 0.01*clcm_olonly['SiO2']*cm_tot/molar_tot
    clmolar_olonly['Na2O'] = 0.01*clcm_olonly['Na2O']*cm_tot*cm_mass['Na2O']/(cm_mass['Na2O']*2)/molar_tot
    clmolar_olonly['K2O'] = 0.01*clcm_olonly['K2O']*cm_tot*cm_mass['K2O']/(cm_mass['K2O']*2)/molar_tot
    if clmolar_olonly['SiO2'] <= 0.6:
        molarSiO2_adjust = 100*clmolar_olonly['SiO2']+100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O'])*((0.46*100/(100-100*clmolar_olonly['SiO2'])-0.93)*100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O'])-5.33*100/(100-100*clmolar_olonly['SiO2'])+9.69)
    else:
        molarSiO2_adjust = 100*clmolar_olonly['SiO2']+100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O'])*(11-5.5*100/(100-100*clmolar_olonly['SiO2']))*math.exp(-0.13*100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O']))
    T = get_firstT_olonly(clcm_olonly,P,molarSiO2_adjust)
    liquidusT_olonly = T
    cm_kdMg_oll_olonly = math.exp(6921/(T+273.15)+0.034*clcm_olonly['Na2O']+0.063*clcm_olonly['K2O']+0.01154*P-3.27)
    #cm_kdMg_oll_olonly = math.exp(6604/(T+273.15)+0.03014*clcm_olonly['Na2O']+0.1454*clcm_olonly['K2O']+0.010076*P-3.1174)
    kdFe2Mg_oll_olonly = math.exp(-6766/(8.3144*(T+273.15))-7.34/8.3144+math.log(0.036*molarSiO2_adjust-0.22)+3000*(1-2*clcm_olonly['MgO']*cm_kdMg_oll_olonly/66.67)/(8.3144*(T+237.15))+0.035*(P*10**3-1)/(8.3144*(T+273.15)))
    cm_kdFe2_oll_olonly = kdFe2Mg_oll_olonly*cm_kdMg_oll_olonly
    olcm_olonly = {'MgO':0,'FeO':0}
    olcm_olonly['MgO'] = clcm_olonly['MgO']*cm_kdMg_oll_olonly
    olcm_olonly['FeO'] = clcm_olonly['FeO']*cm_kdFe2_oll_olonly
    ol_stoich_olonly = olcm_olonly['MgO']+olcm_olonly['FeO']
    fo_olonly = 100*olcm_olonly['MgO']/66.67
    #wt_kdNi_oll_olonly = (3.346*cm_kdMg_oll_olonly-3.665)*1.09
    #wt_kdNi_oll_olonly = math.exp(4505/(T+273.15)-2.075)*(cm_kdMg_oll_olonly*1.09)
    #wt_kdNi_oll_olonly = math.exp(4288/(T+273.15)+0.01804*(clcm_olonly['SiO2']*cm_tot*cm_mass['SiO2']/100)-2.8799)*(cm_kdMg_oll_olonly*1.09)
    #wt_kdNi_oll_olonly = math.exp(4449/(T+273.15)+0.01137*(clcm_olonly['SiO2']*cm_tot*cm_mass['SiO2']/100)-2.6345)*(cm_kdMg_oll_olonly*1.09)
    wt_kdNi_oll_olonly = math.exp(4272/(T+273.15)+0.01582*(clcm_olonly['SiO2']*cm_tot*cm_mass['SiO2']/100)-2.7622)*(cm_kdMg_oll_olonly*1.09)
    #wt_kdNi_oll_olonly = math.exp(4146/(T+273.15)+0.01559*(clcm_olonly['SiO2']*cm_tot*cm_mass['SiO2']/100)-2.6742)*(cm_kdMg_oll_olonly*1.09)
    olppm_olonly = {'Ni':0,'Mn':0}
    olppm_olonly['Ni'] = clppm_olonly['Ni']*wt_kdNi_oll_olonly
    #wt_kdMn_oll_olonly = 0.78
    #wt_kdMn_oll_olonly = (0.259*cm_kdMg_oll_olonly-0.049)*1.09
    #wt_kdMn_oll_olonly = math.exp(-2.76+3583/T)
    #wt_kdMn_oll_olonly = math.exp(0.00877960466828*(cm_tot*40.32*clcm_olonly['MgO']/100)-1.50316580917181)*(cm_kdMg_oll_olonly*1.09)
    wt_kdMn_oll_olonly = 0.79*cm_kdFe2_oll_olonly*1.09
    olppm_olonly['Mn'] = clppm_olonly['Mn']*wt_kdMn_oll_olonly
    f_step_olonly = 1
    f_olonly = 1
    
    ## format data
    T_olonly = []
    F_olonly = []
    F_step_olonly = []
    cmKdMgoll_olonly = []
    cmKdFe2oll_olonly = []
    KdFe2Mgoll_olonly = []
    Clcm_olonly = {element: [] for element in clcm_olonly}
    Olcm_olonly = {element: [] for element in olcm_olonly}
    Olstoich_olonly = []
    Fo_olonly = []
    Clmolar_olonly = {element: [] for element in clmolar_olonly}
    MolarSiO2_adjust = []
    wtKdNioll_olonly = []
    wtKdMnoll_olonly = []
    Clppm_olonly = {element: [] for element in clppm_olonly}
    Olppm_olonly = {element: [] for element in olppm_olonly}

    T_olonly.append(T)
    F_olonly.append(f_olonly)
    F_step_olonly.append(f_step_olonly)
    cmKdMgoll_olonly.append(cm_kdMg_oll_olonly)
    cmKdFe2oll_olonly.append(cm_kdFe2_oll_olonly)
    KdFe2Mgoll_olonly.append(kdFe2Mg_oll_olonly)
    for element in Clcm_olonly:
        Clcm_olonly[element].append(clcm_olonly[element])
    for element in Olcm_olonly:
        Olcm_olonly[element].append(olcm_olonly[element])
    Olstoich_olonly.append(ol_stoich_olonly)
    Fo_olonly.append(fo_olonly)
    for element in Clmolar_olonly:
        Clmolar_olonly[element].append(clmolar_olonly[element])
    MolarSiO2_adjust.append(molarSiO2_adjust)
    wtKdNioll_olonly.append(wt_kdNi_oll_olonly)
    wtKdMnoll_olonly.append(wt_kdMn_oll_olonly)
    for element in Clppm_olonly:
        Clppm_olonly[element].append(clppm_olonly[element])
    for element in Olppm_olonly:
        Olppm_olonly[element].append(olppm_olonly[element])

    while T>liquidusT_olonly-350:
        T,f_step_olonly,f_olonly,cm_kdMg_oll_olonly,kdFe2Mg_oll_olonly,cm_kdFe2_oll_olonly,clmolar_olonly,molarSiO2_adjust = TF_olonly(T,clmolar_olonly,clcm_olonly,P,f_olonly)
        clcm_olonly,olcm_olonly,ol_stoich_olonly,fo_olonly = concentration_olonly(clcm_olonly,cm_kdMg_oll_olonly,f_step_olonly,cm_kdFe2_oll_olonly,olcm_olonly)
        wt_kdNi_oll_olonly,clppm_olonly,olppm_olonly,wt_kdMn_oll_olonly = NiMn_olonly(T,cm_kdMg_oll_olonly,clppm_olonly,f_step_olonly,olppm_olonly,clcm_olonly,cm_kdFe2_oll_olonly,Po)
        T_olonly.append(T)
        F_olonly.append(f_olonly)
        F_step_olonly.append(f_step_olonly)
        cmKdMgoll_olonly.append(cm_kdMg_oll_olonly)
        cmKdFe2oll_olonly.append(cm_kdFe2_oll_olonly)
        KdFe2Mgoll_olonly.append(kdFe2Mg_oll_olonly)
        for element in Clcm_olonly:
            Clcm_olonly[element].append(clcm_olonly[element])
        for element in Olcm_olonly:
            Olcm_olonly[element].append(olcm_olonly[element])
        Olstoich_olonly.append(ol_stoich_olonly)
        Fo_olonly.append(fo_olonly)
        for element in Clmolar_olonly:
            Clmolar_olonly[element].append(clmolar_olonly[element])
        MolarSiO2_adjust.append(molarSiO2_adjust)
        wtKdNioll_olonly.append(wt_kdNi_oll_olonly)
        wtKdMnoll_olonly.append(wt_kdMn_oll_olonly)
        for element in Clppm_olonly:
            Clppm_olonly[element].append(clppm_olonly[element])
        for element in Olppm_olonly:
            Olppm_olonly[element].append(olppm_olonly[element])
elif xtalization_model == 'equilibrium':
    clcm_olonly = cm_magma
    clppm_olonly = {'Ni':magma['NiO']*58.6934/74.69*10**4,'Mn':magma['MnO']*54.938/70.94*10**4}
    clmolar_olonly = {'SiO2':0,'Na2O':0,'K2O':0}
    clmolar_olonly['SiO2'] = 0.01*clcm_olonly['SiO2']*cm_tot/molar_tot
    clmolar_olonly['Na2O'] = 0.01*clcm_olonly['Na2O']*cm_tot*cm_mass['Na2O']/(cm_mass['Na2O']*2)/molar_tot
    clmolar_olonly['K2O'] = 0.01*clcm_olonly['K2O']*cm_tot*cm_mass['K2O']/(cm_mass['K2O']*2)/molar_tot
    if clmolar_olonly['SiO2'] <= 0.6:
        molarSiO2_adjust = 100*clmolar_olonly['SiO2']+100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O'])*((0.46*100/(100-100*clmolar_olonly['SiO2'])-0.93)*100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O'])-5.33*100/(100-100*clmolar_olonly['SiO2'])+9.69)
    else:
        molarSiO2_adjust = 100*clmolar_olonly['SiO2']+100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O'])*(11-5.5*100/(100-100*clmolar_olonly['SiO2']))*math.exp(-0.13*100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O']))
    T = get_firstT_olonly(clcm_olonly,P,molarSiO2_adjust)
    liquidusT_olonly = T
    cm_kdMg_oll_olonly = math.exp(6921/(T+273.15)+0.034*clcm_olonly['Na2O']+0.063*clcm_olonly['K2O']+0.01154*P-3.27)
    #cm_kdMg_oll_olonly = math.exp(6604/(T+273.15)+0.03014*clcm_olonly['Na2O']+0.1454*clcm_olonly['K2O']+0.010076*P-3.1174)
    kdFe2Mg_oll_olonly = math.exp(-6766/(8.3144*(T+273.15))-7.34/8.3144+math.log(0.036*molarSiO2_adjust-0.22)+3000*(1-2*clcm_olonly['MgO']*cm_kdMg_oll_olonly/66.67)/(8.3144*(T+237.15))+0.035*(P*10**3-1)/(8.3144*(T+273.15)))
    cm_kdFe2_oll_olonly = kdFe2Mg_oll_olonly*cm_kdMg_oll_olonly
    olcm_olonly = {'MgO':0,'FeO':0}
    olcm_olonly['MgO'] = clcm_olonly['MgO']*cm_kdMg_oll_olonly
    olcm_olonly['FeO'] = clcm_olonly['FeO']*cm_kdFe2_oll_olonly
    ol_stoich_olonly = olcm_olonly['MgO']+olcm_olonly['FeO']
    fo_olonly = 100*olcm_olonly['MgO']/66.67
    #wt_kdNi_oll_olonly = math.exp(4505/(T+273.15)-2.075)*(cm_kdMg_oll_olonly*1.09)
    #wt_kdNi_oll_olonly = math.exp(4288/(T+273.15)+0.01804*(clcm_olonly['SiO2']*cm_tot*cm_mass['SiO2']/100)-2.8799)*(cm_kdMg_oll_olonly*1.09)
    #wt_kdNi_oll_olonly = math.exp(4449/(T+273.15)+0.01137*(clcm_olonly['SiO2']*cm_tot*cm_mass['SiO2']/100)-2.6345)*(cm_kdMg_oll_olonly*1.09)
    wt_kdNi_oll_olonly = math.exp(4272/(T+273.15)+0.01582*(clcm_olonly['SiO2']*cm_tot*cm_mass['SiO2']/100)-2.7622)*(cm_kdMg_oll_olonly*1.09)
    #wt_kdNi_oll_olonly = math.exp(4146/(T+273.15)+0.01559*(clcm_olonly['SiO2']*cm_tot*cm_mass['SiO2']/100)-2.6742)*(cm_kdMg_oll_olonly*1.09)
    olppm_olonly = {'Ni':0,'Mn':0}
    olppm_olonly['Ni'] = clppm_olonly['Ni']*wt_kdNi_oll_olonly
    #wt_kdMn_oll_olonly = 0.78
    #wt_kdMn_oll_olonly = (0.259*cm_kdMg_oll_olonly-0.049)*1.09
    #wt_kdMn_oll_olonly = math.exp(-2.76+3583/T)
    #wt_kdMn_oll_olonly = math.exp(0.00877960466828*(cm_tot*40.32*clcm_olonly['MgO']/100)-1.50316580917181)*(cm_kdMg_oll_olonly*1.09)
    wt_kdMn_oll_olonly = 0.79*cm_kdFe2_oll_olonly*1.09
    olppm_olonly['Mn'] = clppm_olonly['Mn']*wt_kdMn_oll_olonly
    f_step_olonly = 0
    f_olonly = 1
    
    ## format data
    T_olonly = []
    F_olonly = []
    F_step_olonly = []
    cmKdMgoll_olonly = []
    cmKdFe2oll_olonly = []
    KdFe2Mgoll_olonly = []
    Clcm_olonly = {element: [] for element in clcm_olonly}
    Olcm_olonly = {element: [] for element in olcm_olonly}
    Olstoich_olonly = []
    Fo_olonly = []
    Clmolar_olonly = {element: [] for element in clmolar_olonly}
    MolarSiO2_adjust = []
    wtKdNioll_olonly = []
    wtKdMnoll_olonly = []
    Clppm_olonly = {element: [] for element in clppm_olonly}
    Olppm_olonly = {element: [] for element in olppm_olonly}

    T_olonly.append(T)
    F_olonly.append(f_olonly)
    F_step_olonly.append(f_step_olonly)
    cmKdMgoll_olonly.append(cm_kdMg_oll_olonly)
    cmKdFe2oll_olonly.append(cm_kdFe2_oll_olonly)
    KdFe2Mgoll_olonly.append(kdFe2Mg_oll_olonly)
    for element in Clcm_olonly:
        Clcm_olonly[element].append(clcm_olonly[element])
    for element in Olcm_olonly:
        Olcm_olonly[element].append(olcm_olonly[element])
    Olstoich_olonly.append(ol_stoich_olonly)
    Fo_olonly.append(fo_olonly)
    for element in Clmolar_olonly:
        Clmolar_olonly[element].append(clmolar_olonly[element])
    MolarSiO2_adjust.append(molarSiO2_adjust)
    wtKdNioll_olonly.append(wt_kdNi_oll_olonly)
    wtKdMnoll_olonly.append(wt_kdMn_oll_olonly)
    for element in Clppm_olonly:
        Clppm_olonly[element].append(clppm_olonly[element])
    for element in Olppm_olonly:
        Olppm_olonly[element].append(olppm_olonly[element])
    
    while T>liquidusT_olonly-350:
        cm_magma = cationmole_magma(magma)
        clppm_magma = {'Ni':magma['NiO']*58.6934/74.69*10**4,'Mn':magma['MnO']*54.938/70.94*10**4}
        T,f_step_olonly,f_olonly,cm_kdMg_oll_olonly,kdFe2Mg_oll_olonly,cm_kdFe2_oll_olonly,clmolar_olonly,molarSiO2_adjust = TF_olonly_equ(T,clmolar_olonly,clcm_olonly,P,f_olonly,cm_magma)
        clcm_olonly,olcm_olonly,ol_stoich_olonly,fo_olonly = concentration_olonly_equ(clcm_olonly,cm_kdMg_oll_olonly,f_step_olonly,cm_kdFe2_oll_olonly,olcm_olonly,cm_magma,f_olonly)
        wt_kdNi_oll_olonly,clppm_olonly,olppm_olonly,wt_kdMn_oll_olonly = NiMn_olonly_equ(T,cm_kdMg_oll_olonly,clppm_olonly,f_step_olonly,olppm_olonly,clcm_olonly,clppm_magma,f_olonly,cm_kdFe2_oll_olonly)
        T_olonly.append(T)
        F_olonly.append(f_olonly)
        F_step_olonly.append(f_step_olonly)
        cmKdMgoll_olonly.append(cm_kdMg_oll_olonly)
        cmKdFe2oll_olonly.append(cm_kdFe2_oll_olonly)
        KdFe2Mgoll_olonly.append(kdFe2Mg_oll_olonly)
        for element in Clcm_olonly:
            Clcm_olonly[element].append(clcm_olonly[element])
        for element in Olcm_olonly:
            Olcm_olonly[element].append(olcm_olonly[element])
        Olstoich_olonly.append(ol_stoich_olonly)
        Fo_olonly.append(fo_olonly)
        for element in Clmolar_olonly:
            Clmolar_olonly[element].append(clmolar_olonly[element])
        MolarSiO2_adjust.append(molarSiO2_adjust)
        wtKdNioll_olonly.append(wt_kdNi_oll_olonly)
        wtKdMnoll_olonly.append(wt_kdMn_oll_olonly)
        for element in Clppm_olonly:
            Clppm_olonly[element].append(clppm_olonly[element])
        for element in Olppm_olonly:
            Olppm_olonly[element].append(olppm_olonly[element])
        
# output
clwtMgOarray_olonly = np.asarray(Clcm_olonly['MgO'])*cm_tot*cm_mass['MgO']/100
clwtFeOarray_olonly = np.asarray(Clcm_olonly['FeO'])*cm_tot*cm_mass['FeO']/100
clwtFeOtarray_olonly = clwtFeOarray_olonly/Fe2Fet_Haw
clwtMnOarray_olonly = np.asarray(Clppm_olonly['Mn'])/(10**4)*70.94/54.938
clwtFeOtMnOarray_olonly = clwtFeOarray_olonly/Fe2Fet_Haw/clwtMnOarray_olonly
clwtSiO2array_olonly = np.asarray(Clcm_olonly['SiO2'])*cm_tot*cm_mass['SiO2']/100
Clwt_olonly = {'MgO':clwtMgOarray_olonly.tolist(),'FeO':clwtFeOarray_olonly.tolist(),\
               'FeOt':clwtFeOtarray_olonly.tolist(),'MnO':clwtMnOarray_olonly.tolist(),\
                   'FeOt/MnO':clwtFeOtMnOarray_olonly.tolist(),\
                       'SiO2':clwtSiO2array_olonly.tolist()}
T_olonly = {'T Celsius':T_olonly}
T_olonly = pd.DataFrame(T_olonly)
F_olonly = {'melt fraction':F_olonly}
F_olonly = pd.DataFrame(F_olonly)
F_step_olonly = {'F_step':F_step_olonly}
F_step_olonly = pd.DataFrame(F_step_olonly)
cmKdMgoll_olonly = {'cmkdMgoll':cmKdMgoll_olonly}
cmKdMgoll_olonly = pd.DataFrame(cmKdMgoll_olonly)
cmKdFe2oll_olonly = {'cmkdFe2oll':cmKdFe2oll_olonly}
cmKdFe2oll_olonly = pd.DataFrame(cmKdFe2oll_olonly)
KdFe2Mgoll_olonly = {'KDFe2Mgoll':KdFe2Mgoll_olonly}
KdFe2Mgoll_olonly = pd.DataFrame(KdFe2Mgoll_olonly)
Clcm_olonly = pd.DataFrame(Clcm_olonly)
Clcm_olonly.columns = ['clcm_MgO','clcm_FeO','clcm_SiO2','clcm_Na2O','clcm_K2O']
Olcm_olonly = pd.DataFrame(Olcm_olonly)
Olcm_olonly.columns = ['olcm_MgO','olcm_FeO']
Olstoich_olonly = {'(MgO+FeO)ol':Olstoich_olonly}
Olstoich_olonly = pd.DataFrame(Olstoich_olonly)
Fo_olonly = {'Fo':Fo_olonly}
Fo_olonly = pd.DataFrame(Fo_olonly)
Clmolar_olonly = pd.DataFrame(Clmolar_olonly)
Clmolar_olonly.columns = ['clmolar_SiO2','clmolar_Na2O','clmolar_K2O']
MolarSiO2_adjust = {'molarSiO2_adjust':MolarSiO2_adjust}
MolarSiO2_adjust = pd.DataFrame(MolarSiO2_adjust)
wtKdNioll_olonly = {'wtkdNioll':wtKdNioll_olonly}
wtKdNioll_olonly = pd.DataFrame(wtKdNioll_olonly)
wtKdMnoll_olonly = {'wtkdMnoll':wtKdMnoll_olonly}
wtKdMnoll_olonly = pd.DataFrame(wtKdMnoll_olonly)
Clppm_olonly = pd.DataFrame(Clppm_olonly)
Clppm_olonly.columns = ['clppm_Ni','clppm_Mn']
Olppm_olonly = pd.DataFrame(Olppm_olonly)
Olppm_olonly.columns = ['olppm_Ni','olppm_Mn']
Clwt_olonly = pd.DataFrame(Clwt_olonly)
Clwt_olonly.columns = ['clwt_MgO','clwt_FeO','clwt_FeOt','clwt_MnO','clwt_FeOt/MnO','clwt_SiO2'] 
olonly_xtalization = pd.concat([T_olonly,F_olonly,F_step_olonly,Clwt_olonly,Clppm_olonly,Fo_olonly,Olppm_olonly,\
                                Olcm_olonly,Olstoich_olonly,cmKdMgoll_olonly,cmKdFe2oll_olonly,\
                                KdFe2Mgoll_olonly,wtKdNioll_olonly,wtKdMnoll_olonly,Clcm_olonly,\
                                    Clmolar_olonly,MolarSiO2_adjust],axis=1)

    
## low-P melting
# input parameters:
# source compositions in wt.%, initial mineral phases in percent, initial pressure Po in kbar, melting model (polybaric or isobaric), consider Sulfur or not
#source_wt = {'SiO2':45.2, 'TiO2':0.16, 'Al2O3':4, 'FeO':8.1,'CaO':3.3,'MgO':38.8,'MnO':0.137,'K2O':0.013,'Na2O':0.3, 'P2O5':0.014,'Cr2O3':0.38,'NiO':0.244}  ## peridotite for both MORB and Haw for old KdMg_oll
#source_wt = {'SiO2':45.1, 'TiO2':0.15, 'Al2O3':3.7, 'FeO':8,'CaO':3.15,'MgO':39,'MnO':0.135,'K2O':0.013,'Na2O':0.28, 'P2O5':0.013,'Cr2O3':0.38,'NiO':0.25} 
#source_wt = {'SiO2':45.4, 'TiO2':0.16, 'Al2O3':4, 'FeO':7.60,'CaO':3.5,'MgO':38,'MnO':0.137,'K2O':0.013,'Na2O':0.32, 'P2O5':0.014,'Cr2O3':0.38,'NiO':0.250}  ## peridotite for both MORB and Haw for new KdMg_oll
#source_wt = {'SiO2':44.48, 'TiO2':0.16, 'Al2O3':3.59, 'FeO':7.78,'CaO':3.44,'MgO':39.22,'MnO':0.125,'K2O':0.02,'Na2O':0.3, 'P2O5':0.03,'Cr2O3':0.31,'NiO':0.25} ## KLB-1
#source_wt = {'SiO2':45.1, 'TiO2':0.15, 'Al2O3':3.7, 'FeO':8.1,'CaO':3.15,'MgO':39,'MnO':0.139,'K2O':0.013,'Na2O':0.28, 'P2O5':0.014,'Cr2O3':0.38,'NiO':0.24}  ## ideal peridotite for MORB 
#source_phase = {'ol':60,'opx':24,'cpx':14,'gt':0,'sp':2}  ## mineral phases of KLB-1 (Hirose and Kushiro 1993, Herzberg etal 1990 JGR solid earth, Davis etal 2009 Table4 American Mineralogist)
source_wt = source_wt_MORB
source_phase = source_phase_MORB
Po = Po_low  # >=30 is high-pressure, <30 is low-pressure
melting_model = melting_model_MORB
S_mode = S_mode_MORB

# parameters used in calculating mineral phases during isobaric equilibrium melting
source_phase2 = source_phase 
source_phase3 = source_phase
source_phase4 = source_phase
f_gt0 = 0.0000001
f_cpx0 = 0.0000001
f_sp0 = 0.0000001

# calculate melting
source_cm, mgnumber_source = wttocm(source_wt)
if melting_model == 'polybaric':  # polybaric fractional melting
    ## calculate all parameters near solidus assuming the the extent of melting is 0.0000001
    f = 0.0000001  
    f_step = 0.0000001
    T = 13*Po+1140+600*(1-Po/88)*f+20*(mgnumber_source-89)
    crust_thickness = 0
    p_remain = -Po
    P = Po
    if S_mode == 'Y':
        S_res = S_mantle 
        f_sulfide,S_res = S_polyfrac(S_mantlesulfide,S_res,f_step,deltaQFM,SCSS_QFM_MORB)
        f_mineral, phase_tot = mineral_phase_polyfrac(Po,P,f_step,f_mineral=source_phase)
        f_mineral, f_sulfide = mineral_phase_S(f_sulfide,f_mineral)
        keys = ['MgO','FeO','TiO2', 'Na2O', 'K2O','NiO','MnO']
        res = {key:source_wt[key] for key in keys}
        res['MgO'] = source_cm['MgO']*100
        res['FeO'] = source_cm['FeO']*100
        res['mgnumber'] = 0.0
        cl_wt = {key:0.0 for key in keys}
        cl_wt['SiO2'] = 0.0
        cl_cm = {'MgO':0.0,'FeO':0.0,'Na2O':0.0,'K2O':0.0}
        bulkD = {'K2O':0.005,'Na2O':0.0,'TiO2':0.0,'Ni':0.0,'Mn':0.0}
        cl_wt,bulkD,cl_cm,res = liquid_wt_polyfrac(res,f_step,f_mineral,P,T,cl_wt,cl_cm,bulkD,Po)
        cl_molar = {'SiO2':0.0,'Na2O':0.0,'K2O':0.0}
        ol = {'MgOcm':0.0,'FeOcm':0.0,'Fo':0.0,'NiOwt':0.0,'MnOwt':0.0}
        cl_molar,clSiO2_adjust,kdMgO_oll_cm,ol,kdFe2Mg_oll = KDFeMg(T,P,f_step,cl_wt,cl_cm,res,cl_molar,ol)
        ol,cl_cm,res,kdFeO_oll_cm,cl_wt = MgOFeO_polyfrac(ol,kdMgO_oll_cm,f_step,kdFe2Mg_oll,res,cl_cm,cl_wt)
        kdNi_wt, bulkD, cl_wt, ol, res = Ni_polyfrac_S(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol,f_sulfideliquid,f_sulfide)
        kdMn_wt, bulkD, cl_wt, ol, res = Mn_polyfrac(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol,Po,kdFeO_oll_cm)
    else:
        f_mineral, phase_tot = mineral_phase_polyfrac(Po,P,f_step,f_mineral=source_phase)
        keys = ['MgO','FeO','TiO2', 'Na2O', 'K2O','NiO','MnO']
        res = {key:source_wt[key] for key in keys}
        res['MgO'] = source_cm['MgO']*100
        res['FeO'] = source_cm['FeO']*100
        res['mgnumber'] = 0.0
        cl_wt = {key:0.0 for key in keys}
        cl_wt['SiO2'] = 0.0
        cl_cm = {'MgO':0.0,'FeO':0.0,'Na2O':0.0,'K2O':0.0}
        bulkD = {'K2O':0.005,'Na2O':0.0,'TiO2':0.0,'Ni':0.0,'Mn':0.0}
        cl_wt,bulkD,cl_cm,res = liquid_wt_polyfrac(res,f_step,f_mineral,P,T,cl_wt,cl_cm,bulkD,Po)
        cl_molar = {'SiO2':0.0,'Na2O':0.0,'K2O':0.0}
        ol = {'MgOcm':0.0,'FeOcm':0.0,'Fo':0.0,'NiOwt':0.0,'MnOwt':0.0}
        cl_molar,clSiO2_adjust,kdMgO_oll_cm,ol,kdFe2Mg_oll = KDFeMg(T,P,f_step,cl_wt,cl_cm,res,cl_molar,ol)
        ol,cl_cm,res,kdFeO_oll_cm,cl_wt = MgOFeO_polyfrac(ol,kdMgO_oll_cm,f_step,kdFe2Mg_oll,res,cl_cm,cl_wt)
        kdNi_wt, bulkD, cl_wt, ol, res = Ni_polyfrac(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol)
        kdMn_wt, bulkD, cl_wt, ol, res = Mn_polyfrac(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol,Po,kdFeO_oll_cm)
    ## format data
    F_mineral = {element: [] for element in f_mineral}
    Res = {element: [] for element in res}  
    Cl_wt = {element: [] for element in cl_wt}
    Cl_cm = {element: [] for element in cl_cm}
    KdNi_wt = {element: [] for element in kdNi_wt}
    KdMn_wt = {element: [] for element in kdMn_wt}
    BulkD = {element: [] for element in bulkD}
    Cl_molar = {element: [] for element in cl_molar}
    ClSiO2_adjust = []
    Ol = {element: [] for element in ol}
    Phase_tot = []
    T_melting = []
    P_melting = []
    F_melting = []
    F_step = []
    Crust_thickness = []
    P_remain = []
    KdMgO_oll_cm = []
    KdFeO_oll_cm = []
    KdFe2Mg_oll = []
    S_Res = []
    F_sulfide = []
    for element in F_mineral:
        F_mineral[element].append(f_mineral[element])
    for element in Res:
        Res[element].append(res[element])
    for element in Cl_wt:
        Cl_wt[element].append(cl_wt[element])
    for element in Cl_cm:
        Cl_cm[element].append(cl_cm[element])
    for element in KdNi_wt:
        KdNi_wt[element].append(kdNi_wt[element])
    for element in KdMn_wt:
        KdMn_wt[element].append(kdMn_wt[element])
    for element in BulkD:
        BulkD[element].append(bulkD[element])
    for element in Cl_molar:
        Cl_molar[element].append(cl_molar[element])
    for element in Ol:
        Ol[element].append(ol[element])
    ClSiO2_adjust.append(clSiO2_adjust)
    Phase_tot.append(phase_tot)
    T_melting.append(T)
    P_melting.append(P)
    F_melting.append(f)
    F_step.append(f_step)
    Crust_thickness.append(crust_thickness)
    P_remain.append(p_remain)
    KdMgO_oll_cm.append(kdMgO_oll_cm)
    KdFeO_oll_cm.append(kdFeO_oll_cm)
    KdFe2Mg_oll.append(kdFe2Mg_oll)
    if S_mode == 'Y':
        S_Res.append(S_res)
        F_sulfide.append(f_sulfide)
    ## melting stops when the top of melting column reaches to the bottom of the crust
    while p_remain <=0:
        T, P, f, f_step, crust_thickness, p_remain = TPF_polyfrac(P,f,mgnumber_source,Po)
        if S_mode == 'Y':
            f_sulfide,S_res = S_polyfrac(S_mantlesulfide,S_res,f_step,deltaQFM,SCSS_QFM_MORB)
            f_mineral, phase_tot = mineral_phase_polyfrac(Po,P,f_step,f_mineral)
            f_mineral, f_sulfide = mineral_phase_S(f_sulfide,f_mineral)
            cl_wt,bulkD,cl_cm,res = liquid_wt_polyfrac(res,f_step,f_mineral,P,T,cl_wt,cl_cm,bulkD,Po)
            cl_molar,clSiO2_adjust,kdMgO_oll_cm,ol,kdFe2Mg_oll = KDFeMg(T,P,f_step,cl_wt,cl_cm,res,cl_molar,ol)
            ol,cl_cm,res,kdFeO_oll_cm,cl_wt = MgOFeO_polyfrac(ol,kdMgO_oll_cm,f_step,kdFe2Mg_oll,res,cl_cm,cl_wt)
            kdNi_wt, bulkD, cl_wt, ol, res = Ni_polyfrac_S(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol,f_sulfideliquid,f_sulfide)
            kdMn_wt, bulkD, cl_wt, ol, res = Mn_polyfrac(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol,Po,kdFeO_oll_cm)
        else:
            f_mineral, phase_tot = mineral_phase_polyfrac(Po,P,f_step,f_mineral)
            cl_wt,bulkD,cl_cm,res = liquid_wt_polyfrac(res,f_step,f_mineral,P,T,cl_wt,cl_cm,bulkD,Po)
            cl_molar,clSiO2_adjust,kdMgO_oll_cm,ol,kdFe2Mg_oll = KDFeMg(T,P,f_step,cl_wt,cl_cm,res,cl_molar,ol)
            ol,cl_cm,res,kdFeO_oll_cm,cl_wt = MgOFeO_polyfrac(ol,kdMgO_oll_cm,f_step,kdFe2Mg_oll,res,cl_cm,cl_wt)
            kdNi_wt, bulkD, cl_wt, ol, res = Ni_polyfrac(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol)
            kdMn_wt, bulkD, cl_wt, ol, res = Mn_polyfrac(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol,Po,kdFeO_oll_cm)
        for element in F_mineral:
            F_mineral[element].append(f_mineral[element])
        for element in Res:
            Res[element].append(res[element])
        for element in Cl_wt:
            Cl_wt[element].append(cl_wt[element])
        for element in Cl_cm:
            Cl_cm[element].append(cl_cm[element])
        for element in KdNi_wt:
            KdNi_wt[element].append(kdNi_wt[element])
        for element in KdMn_wt:
            KdMn_wt[element].append(kdMn_wt[element])
        for element in BulkD:
            BulkD[element].append(bulkD[element])
        for element in Cl_molar:
            Cl_molar[element].append(cl_molar[element])
        for element in Ol:
            Ol[element].append(ol[element])
        ClSiO2_adjust.append(clSiO2_adjust)
        Phase_tot.append(phase_tot)
        T_melting.append(T)
        P_melting.append(P)
        F_melting.append(f)
        F_step.append(f_step)
        Crust_thickness.append(crust_thickness)
        P_remain.append(p_remain)
        KdMgO_oll_cm.append(kdMgO_oll_cm)
        KdFeO_oll_cm.append(kdFeO_oll_cm)
        KdFe2Mg_oll.append(kdFe2Mg_oll)
        if S_mode == 'Y':
            S_Res.append(S_res)
            F_sulfide.append(f_sulfide)         
    ## calcluate the accumulated melt compositions for polybaric fractional melting    
    Cl_wt_itg1,Cl_wt_itg2,F_melting_itg1,F_melting_itg2 = itg(Cl_wt,F_step,F_melting,Po) 

elif melting_model == 'isobaric':  # isobaric equilibrium melting 
    P = Po  # pressure will be constant during the melting
    f = 0.0000001  # calculate all parameters near solidus assuming the the extent of melting is 0.0000001 
    f_step = 0.0000001
    T = 13*Po+1140+600*(1-Po/88)*f+20*(mgnumber_source-89)
    if S_mode == 'Y':
        S_res = S_mantle 
        f_sulfide,S_res = S_isoequ(S_mantle,S_mantlesulfide,S_res,f,deltaQFM,SCSS_QFM_MORB)
        f_mineral = source_phase.copy()
        f_mineral, phase_tot,source_phase2,source_phase3,source_phase4,f_gt0,f_cpx0,f_sp0 = mineral_phase_isoequ(Po,P,f,source_phase,source_phase2,f_gt0,source_phase3,source_phase4,f_cpx0,f_sp0,f_mineral)
        f_mineral, f_sulfide = mineral_phase_S(f_sulfide,f_mineral)
        keys = ['MgO','FeO','TiO2','Na2O', 'K2O','NiO','MnO']
        res = {key:source_wt[key] for key in keys}
        res['MgO'] = source_cm['MgO']*100
        res['FeO'] = source_cm['FeO']*100
        res['mgnumber'] = 0.0
        cl_wt = {key:0.0 for key in keys}
        cl_wt['SiO2'] = 0.0
        cl_cm = {'MgO':0.0,'FeO':0.0,'Na2O':0.0,'K2O':0.0}
        bulkD = {'K2O':0.005,'Na2O':0.0,'TiO2':0.0,'Ni':0.0,'Mn':0.0}
        cl_wt,bulkD,cl_cm,res = liquid_wt_isoequ(source_wt,f,f_mineral,P,T,cl_wt,cl_cm,bulkD,res,Po)
        cl_molar = {'SiO2':0.0,'Na2O':0.0,'K2O':0.0}
        ol = {'MgOcm':0.0,'FeOcm':0.0,'Fo':0.0,'NiOwt':0.0,'MnOwt':0.0}
        cl_molar,clSiO2_adjust,kdMgO_oll_cm,ol,kdFe2Mg_oll = KDFeMg(T,P,f_step,cl_wt,cl_cm,res,cl_molar,ol)
        ol,cl_cm,res,kdFeO_oll_cm,cl_wt = MgOFeO_isoequ(ol,kdMgO_oll_cm,f,kdFe2Mg_oll,res,cl_cm,cl_wt,source_cm)
        kdNi_wt, bulkD, cl_wt, ol, res = Ni_isoequ_S(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt,f_sulfideliquid,f_sulfide)
        kdMn_wt, bulkD, cl_wt, ol, res = Mn_isoequ(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt,Po,kdFeO_oll_cm)        
    else:
        f_mineral = source_phase.copy()
        f_mineral, phase_tot,source_phase2,source_phase3,source_phase4,f_gt0,f_cpx0,f_sp0 = mineral_phase_isoequ(Po,P,f,source_phase,source_phase2,f_gt0,source_phase3,source_phase4,f_cpx0,f_sp0,f_mineral)
        keys = ['MgO','FeO','TiO2','Na2O', 'K2O','NiO','MnO']
        res = {key:source_wt[key] for key in keys}
        res['MgO'] = source_cm['MgO']*100
        res['FeO'] = source_cm['FeO']*100
        res['mgnumber'] = 0.0
        cl_wt = {key:0.0 for key in keys}
        cl_wt['SiO2'] = 0.0
        cl_cm = {'MgO':0.0,'FeO':0.0,'Na2O':0.0,'K2O':0.0}
        bulkD = {'K2O':0.005,'Na2O':0.0,'TiO2':0.0,'Ni':0.0,'Mn':0.0}
        cl_wt,bulkD,cl_cm,res = liquid_wt_isoequ(source_wt,f,f_mineral,P,T,cl_wt,cl_cm,bulkD,res,Po)
        cl_molar = {'SiO2':0.0,'Na2O':0.0,'K2O':0.0}
        ol = {'MgOcm':0.0,'FeOcm':0.0,'Fo':0.0,'NiOwt':0.0,'MnOwt':0.0}
        cl_molar,clSiO2_adjust,kdMgO_oll_cm,ol,kdFe2Mg_oll = KDFeMg(T,P,f_step,cl_wt,cl_cm,res,cl_molar,ol)
        ol,cl_cm,res,kdFeO_oll_cm,cl_wt = MgOFeO_isoequ(ol,kdMgO_oll_cm,f,kdFe2Mg_oll,res,cl_cm,cl_wt,source_cm)
        kdNi_wt, bulkD, cl_wt, ol, res = Ni_isoequ(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt)
        kdMn_wt, bulkD, cl_wt, ol, res = Mn_isoequ(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt,Po,kdFeO_oll_cm)
    ## format data
    F_mineral = {element: [] for element in f_mineral}
    Res = {element: [] for element in res}  
    Cl_wt = {element: [] for element in cl_wt}
    Cl_cm = {element: [] for element in cl_cm}
    KdNi_wt = {element: [] for element in kdNi_wt}
    KdMn_wt = {element: [] for element in kdMn_wt}
    BulkD = {element: [] for element in bulkD}
    Cl_molar = {element: [] for element in cl_molar}
    ClSiO2_adjust = []
    Ol = {element: [] for element in ol}
    Phase_tot = []
    T_melting = []
    P_melting = []
    F_melting = []
    F_step = []
    KdMgO_oll_cm = []
    KdFeO_oll_cm = []
    KdFe2Mg_oll = []
    S_Res = []
    F_sulfide = []
    for element in F_mineral:
        F_mineral[element].append(f_mineral[element])
    for element in Res:
        Res[element].append(res[element])
    for element in Cl_wt:
        Cl_wt[element].append(cl_wt[element])
    for element in Cl_cm:
        Cl_cm[element].append(cl_cm[element])
    for element in KdNi_wt:
        KdNi_wt[element].append(kdNi_wt[element])
        KdMn_wt[element].append(kdMn_wt[element])
    for element in BulkD:
        BulkD[element].append(bulkD[element])
    for element in Cl_molar:
        Cl_molar[element].append(cl_molar[element])
    for element in Ol:
        Ol[element].append(ol[element])
    ClSiO2_adjust.append(clSiO2_adjust)
    Phase_tot.append(phase_tot)
    T_melting.append(T)
    P_melting.append(Po)
    F_melting.append(f)
    F_step.append(f_step)
    KdMgO_oll_cm.append(kdMgO_oll_cm)
    KdFeO_oll_cm.append(kdFeO_oll_cm)
    KdFe2Mg_oll.append(kdFe2Mg_oll)
    if S_mode == 'Y':
        S_Res.append(S_res)
        F_sulfide.append(f_sulfide)
    ## melting stops when the extent of melting reaches to about 50%
    while f <=0.5:
        T,f,f_step = TPF_isoequ(P,f,mgnumber_source)
        if S_mode == 'Y':
            f_sulfide,S_res = S_isoequ(S_mantle,S_mantlesulfide,S_res,f,deltaQFM,SCSS_QFM_MORB)
            f_mineral, phase_tot,source_phase2,source_phase3,source_phase4,f_gt0,f_cpx0,f_sp0 = mineral_phase_isoequ(Po,P,f,source_phase,source_phase2,f_gt0,source_phase3,source_phase4,f_cpx0,f_sp0,f_mineral)
            f_mineral, f_sulfide = mineral_phase_S(f_sulfide,f_mineral)
            cl_wt,bulkD,cl_cm,res = liquid_wt_isoequ(source_wt,f,f_mineral,P,T,cl_wt,cl_cm,bulkD,res,Po)
            cl_molar,clSiO2_adjust,kdMgO_oll_cm,ol,kdFe2Mg_oll = KDFeMg(T,P,f_step,cl_wt,cl_cm,res,cl_molar,ol)
            ol,cl_cm,res,kdFeO_oll_cm,cl_wt = MgOFeO_isoequ(ol,kdMgO_oll_cm,f,kdFe2Mg_oll,res,cl_cm,cl_wt,source_cm)
            kdNi_wt, bulkD, cl_wt, ol, res = Ni_isoequ_S(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt,f_sulfideliquid,f_sulfide)
            kdMn_wt, bulkD, cl_wt, ol, res = Mn_isoequ(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt,Po,kdFeO_oll_cm)
        else:
            f_mineral, phase_tot,source_phase2,source_phase3,source_phase4,f_gt0,f_cpx0,f_sp0 = mineral_phase_isoequ(Po,P,f,source_phase,source_phase2,f_gt0,source_phase3,source_phase4,f_cpx0,f_sp0,f_mineral)
            cl_wt,bulkD,cl_cm,res = liquid_wt_isoequ(source_wt,f,f_mineral,P,T,cl_wt,cl_cm,bulkD,res,Po)
            cl_molar,clSiO2_adjust,kdMgO_oll_cm,ol,kdFe2Mg_oll = KDFeMg(T,P,f_step,cl_wt,cl_cm,res,cl_molar,ol)
            ol,cl_cm,res,kdFeO_oll_cm,cl_wt = MgOFeO_isoequ(ol,kdMgO_oll_cm,f,kdFe2Mg_oll,res,cl_cm,cl_wt,source_cm)
            kdNi_wt, bulkD, cl_wt, ol, res = Ni_isoequ(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt)
            kdMn_wt, bulkD, cl_wt, ol, res = Mn_isoequ(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt,Po,kdFeO_oll_cm)
        for element in F_mineral:
            F_mineral[element].append(f_mineral[element])
        for element in Res:
            Res[element].append(res[element])
        for element in Cl_wt:
            Cl_wt[element].append(cl_wt[element])
        for element in Cl_cm:
            Cl_cm[element].append(cl_cm[element])
        for element in KdNi_wt:
            KdNi_wt[element].append(kdNi_wt[element])
            KdMn_wt[element].append(kdMn_wt[element])
        for element in BulkD:
            BulkD[element].append(bulkD[element])
        for element in Cl_molar:
            Cl_molar[element].append(cl_molar[element])
        for element in Ol:
            Ol[element].append(ol[element])
        ClSiO2_adjust.append(clSiO2_adjust)
        Phase_tot.append(phase_tot)
        T_melting.append(T)
        P_melting.append(Po)
        F_melting.append(f)
        F_step.append(f_step)
        KdMgO_oll_cm.append(kdMgO_oll_cm)
        KdFeO_oll_cm.append(kdFeO_oll_cm)
        KdFe2Mg_oll.append(kdFe2Mg_oll)
        if S_mode == 'Y':
            S_Res.append(S_res)
            F_sulfide.append(f_sulfide)    
            
# output:
dfname = str(Po)+melting_model  # the name of the output file
BulkD_df = pd.DataFrame(BulkD)
BulkD_df.columns = ['DK2O','DNa2O','DTiO2','DNiO','DMnO']
Cl_cm_df = pd.DataFrame(Cl_cm)
Cl_cm_df.columns = ['clMgO_cm','clFeO_cm','clNa2O_cm','clK2O_cm']
Cl_molar_df = pd.DataFrame(Cl_molar)
Cl_molar_df.columns = ['clSiO2_molar','clNa2O_molar','clK2O_molar']
ClSiO2_adjust = {'clSiO2_adjust':ClSiO2_adjust}
ClSiO2_adjust = pd.DataFrame(ClSiO2_adjust)
F_mineral = pd.DataFrame(F_mineral)
F_step = {'f_step':F_step}
F_step = pd.DataFrame(F_step)
KDFeMg = {'kdMgO_oll_cm':KdMgO_oll_cm,'kdFeO_oll_cm':KdFeO_oll_cm,'KDFe2Mg_oll':KdFe2Mg_oll}
KDFeMg = pd.DataFrame(KDFeMg)
KdMn_wt_df = pd.DataFrame(KdMn_wt)
KdMn_wt_df.columns = ['KdMn_oll_wt','KdMn_opxl_wt','KdMn_cpxl_wt','KdMn_gtl_wt','KdMn_spl_wt','KdMn_opxol_wt','KdMn_cpxol_wt','KdMn_gtol_wt','KdMn_spol_wt']
Ol_df = pd.DataFrame(Ol)
Ol_df.columns = ['olMgO_cm','olFeO_cm','Fo','olNiO_wt','olMnO_wt']
P_melting = {'P kbar':P_melting}
P_melting = pd.DataFrame(P_melting)
Phase_tot = {'mineral_phase_tot':Phase_tot}
Phase_tot = pd.DataFrame(Phase_tot)
Res = pd.DataFrame(Res)
Res.columns = ['resMgO_cm','resFeO_cm','resTiO2_wt','resNa2O_wt','resK2O_wt','resNiO_wt','resMnO_wt','resMgnumber']
T_melting = {'T Celsius':T_melting}
T_melting = pd.DataFrame(T_melting)
if melting_model == 'polybaric':
    Cl_wt_itg1_df = pd.DataFrame(Cl_wt_itg1)
    Cl_wt_itg1_df.columns = ['clMgO_wt_itg1','clFeO_wt_itg1','clTiO2_wt_itg1','clNa2O_wt_itg1','clK2O_wt_itg1','clNiO_wt_itg1','clMnO_wt_itg1','clSiO2_wt_itg1']
    Cl_wt_itg2_df = pd.DataFrame(Cl_wt_itg2)
    Cl_wt_itg2_df.columns = ['clMgO_wt_itg2','clFeO_wt_itg2','clTiO2_wt_itg2','clNa2O_wt_itg2','clK2O_wt_itg2','clNiO_wt_itg2','clMnO_wt_itg2','clSiO2_wt_itg2']
    F_melting_itg1 = {'F_liq_itg1':F_melting_itg1}
    F_melting_itg1 = pd.DataFrame(F_melting_itg1)
    F_melting_itg2 = {'F_liq_itg2':F_melting_itg2}
    F_melting_itg2 = pd.DataFrame(F_melting_itg2)
    if S_mode == 'Y':
        KdNi_wt_df = pd.DataFrame(KdNi_wt)
        KdNi_wt_df.columns = ['KdNi_oll_wt','KdNi_opxl_wt','KdNi_cpxl_wt','KdNi_gtl_wt','KdNi_spl_wt','KdNi_surfl_wt','KdNi_opxol_wt','KdNi_cpxol_wt','KdNi_gtol_wt','KdNi_spol_wt']
        F_sulfide = {'F_sulfide':F_sulfide}
        F_sulfide = pd.DataFrame(F_sulfide)
        S_Res = {'S_Res':S_Res}
        S_Res = pd.DataFrame(S_Res)
        melting_df_lowP = pd.concat([T_melting,P_melting,F_step,F_mineral,F_sulfide,Phase_tot,F_melting_itg2,Cl_wt_itg2_df,F_melting_itg1,Cl_wt_itg1_df,Ol_df,Cl_cm_df,Cl_molar_df,ClSiO2_adjust,Res,S_Res,KDFeMg,BulkD_df,KdNi_wt_df,KdMn_wt_df],axis=1)
    else:
        KdNi_wt_df = pd.DataFrame(KdNi_wt)
        KdNi_wt_df.columns = ['KdNi_oll_wt','KdNi_opxl_wt','KdNi_cpxl_wt','KdNi_gtl_wt','KdNi_spl_wt','KdNi_opxol_wt','KdNi_cpxol_wt','KdNi_gtol_wt','KdNi_spol_wt']
        melting_df_lowP = pd.concat([T_melting,P_melting,F_step,F_mineral,Phase_tot,F_melting_itg2,Cl_wt_itg2_df,F_melting_itg1,Cl_wt_itg1_df,Ol_df,Cl_cm_df,Cl_molar_df,ClSiO2_adjust,Res,KDFeMg,BulkD_df,KdNi_wt_df,KdMn_wt_df],axis=1)        
else:
    if S_mode == 'Y':
        KdNi_wt_df = pd.DataFrame(KdNi_wt)
        KdNi_wt_df.columns = ['KdNi_oll_wt','KdNi_opxl_wt','KdNi_cpxl_wt','KdNi_gtl_wt','KdNi_spl_wt','KdNi_surfl_wt','KdNi_opxol_wt','KdNi_cpxol_wt','KdNi_gtol_wt','KdNi_spol_wt']
        F_sulfide = {'F_sulfide':F_sulfide}
        F_sulfide = pd.DataFrame(F_sulfide)
        S_Res = {'S_Res':S_Res}
        S_Res = pd.DataFrame(S_Res)
    else:
        KdNi_wt_df = pd.DataFrame(KdNi_wt)
        KdNi_wt_df.columns = ['KdNi_oll_wt','KdNi_opxl_wt','KdNi_cpxl_wt','KdNi_gtl_wt','KdNi_spl_wt','KdNi_opxol_wt','KdNi_cpxol_wt','KdNi_gtol_wt','KdNi_spol_wt']
    Cl_wt_df = pd.DataFrame(Cl_wt)
    Cl_wt_df.columns = ['clMgO_wt','clFeO_wt','clTiO2_wt','clNa2O_wt','clK2O_wt','clNiO_wt','clMnO_wt','clSiO2_wt']
    F_melting = {'F_liq':F_melting}
    F_melting = pd.DataFrame(F_melting)
    melting_df_lowP = pd.concat([T_melting,P_melting,F_melting,F_step,F_mineral,Phase_tot,Cl_wt_df,Ol_df,Cl_cm_df,Cl_molar_df,ClSiO2_adjust,Res,KDFeMg,BulkD_df,KdNi_wt_df,KdMn_wt_df],axis=1)    

# crystallization
F_target = F_target_MORB
if melting_model == 'polybaric':
    ip_magma = abs(melting_df_lowP['F_liq_itg2']-F_target).idxmin()
    magma = {'SiO2':float(melting_df_lowP.loc[ip_magma,'clSiO2_wt_itg2']),'TiO2':float(melting_df_lowP.loc[ip_magma,'clTiO2_wt_itg2']),\
             'Al2O3':14.8,'FeO':float(melting_df_lowP.loc[ip_magma,'clFeO_wt_itg2']),'MgO':float(melting_df_lowP.loc[ip_magma,'clMgO_wt_itg2']),\
                 'K2O':float(melting_df_lowP.loc[ip_magma,'clK2O_wt_itg2']),'MnO':float(melting_df_lowP.loc[ip_magma,'clMnO_wt_itg2']),\
                     'Na2O':float(melting_df_lowP.loc[ip_magma,'clNa2O_wt_itg2']),'P2O5':0.06,'CaO':11.5,\
                         'NiO':float(melting_df_lowP.loc[ip_magma,'clNiO_wt_itg2'])}
else:
    ip_magma = abs(melting_df_lowP['F_liq']-F_target).idxmin()
    magma = {'SiO2':float(melting_df_lowP.loc[ip_magma,'clSiO2_wt']),'TiO2':float(melting_df_lowP.loc[ip_magma,'clTiO2_wt']),\
             'Al2O3':14.8,'FeO':float(melting_df_lowP.loc[ip_magma,'clFeO_wt']),'MgO':float(melting_df_lowP.loc[ip_magma,'clMgO_wt']),\
                 'K2O':float(melting_df_lowP.loc[ip_magma,'clK2O_wt']),'MnO':float(melting_df_lowP.loc[ip_magma,'clMnO_wt']),\
                     'Na2O':float(melting_df_lowP.loc[ip_magma,'clNa2O_wt']),'P2O5':0.06,'CaO':11.5,\
                         'NiO':float(melting_df_lowP.loc[ip_magma,'clNiO_wt'])}
system_components = magma
T_system_components = oxideToComponent(system_components)
t_start = get_first_T(T_system_components, P = 1., kdCalc = kdCalc_langmuir1992)
t_stop = t_start -250
fl,fa_dict,major_oxide_dict,major_phase_oxide_dict = frac_model_trange(t_start, t_stop,system_components,P=1.,kdCalc = kdCalc_langmuir1992) 

# output
T_df = pd.DataFrame(np.arange(t_start,t_stop,-1))
T_df = T_df-273.15
T_df.columns = ['T_C']
fl_dict = {'fl':fl}
fl_df = pd.DataFrame(fl_dict)
fa_df = pd.DataFrame(fa_dict)
major_oxide_df = pd.DataFrame(major_oxide_dict)
major_ol_oxide_df = pd.DataFrame(major_phase_oxide_dict['ol'])
major_cpx_oxide_df = pd.DataFrame(major_phase_oxide_dict['cpx'])
major_plg_oxide_df = pd.DataFrame(major_phase_oxide_dict['plg'])

LLD_df = pd.concat([T_df,fl_df,fa_df,major_oxide_df,major_ol_oxide_df,major_cpx_oxide_df,major_plg_oxide_df],axis=1)
LLD_df.columns = ['T_C','f_liq','f_plg','f_cpx','f_ol','liq_SiO2','liq_TiO2','liq_Al2O3','liq_FeO',\
               'liq_MgO','liq_K2O','liq_MnO','liq_Na2O','liq_P2O5','liq_CaO','liq_NiO','olSiO2',\
                   'olTiO2','olAl2O3','olFeO','olMgO','olK2O','olMnO','olNa2O','olP2O5','olCaO','olNiO',\
                       'cpxSiO2','cpxTiO2','cpxAl2O3','cpxFeO','cpxMgO','cpxK2O','cpxMnO','cpxNa2O',\
                           'cpxP2O5','cpxCaO','cpxNiO','plgSiO2','plgTiO2','plgAl2O3','plgFeO','plgMgO',\
                               'plgK2O','plgMnO','plgNa2O','plgP2O5','plgCaO','plgNiO']
LLD_df['liq_FeOt'] = LLD_df['liq_FeO']/Fe2Fet_MORB
LLD_df['Fo'] = 100/(1+LLD_df['olFeO']/LLD_df['olMgO']*40.3/71.84)
LLD_df['olNippm'] = LLD_df['olNiO']*58.6934/74.69*10**4
LLD_df['olMnppm'] = LLD_df['olMnO']*54.938/70.94*10**4
LLD_df['liq_Nippm'] = LLD_df['liq_NiO']*58.6934/74.69*10**4
LLD_df['liq_FeOtMnO'] = LLD_df['liq_FeOt']/LLD_df['liq_MnO']
#LLD_df.to_csv('/Users/apple/Desktop/NiMnpaper/model parameters and results/model_variation_LLDoutput0517/{}.csv'.format(LLDinput.loc[k,['PFK_condition']][0]),sep=',',index=True,header=True)

F_target = F_target_MORB
if melting_model == 'polybaric':
    ip_magma = abs(melting_df_lowP['F_liq_itg2']-F_target).idxmin()
    magma = {'MgO':float(melting_df_lowP.loc[ip_magma,'clMgO_wt_itg2']),'FeO':float(melting_df_lowP.loc[ip_magma,'clFeO_wt_itg2']),\
             'SiO2':float(melting_df_lowP.loc[ip_magma,'clSiO2_wt_itg2']),'Na2O':float(melting_df_lowP.loc[ip_magma,'clNa2O_wt_itg2']),\
                 'K2O':float(melting_df_lowP.loc[ip_magma,'clK2O_wt_itg2']),'NiO':float(melting_df_lowP.loc[ip_magma,'clNiO_wt_itg2']),\
                     'MnO':float(melting_df_lowP.loc[ip_magma,'clMnO_wt_itg2'])}
else:
    ip_magma = abs(melting_df_lowP['F_liq']-F_target).idxmin()
    magma = {'MgO':float(melting_df_lowP.loc[ip_magma,'clMgO_wt']),'FeO':float(melting_df_lowP.loc[ip_magma,'clFeO_wt']),\
             'SiO2':float(melting_df_lowP.loc[ip_magma,'clSiO2_wt']),'Na2O':float(melting_df_lowP.loc[ip_magma,'clNa2O_wt']),\
                 'K2O':float(melting_df_lowP.loc[ip_magma,'clK2O_wt']),'NiO':float(melting_df_lowP.loc[ip_magma,'clNiO_wt']),\
                     'MnO':float(melting_df_lowP.loc[ip_magma,'clMnO_wt'])}  
# default values
cm_mass = {'MgO':40.304,'FeO':71.844,'SiO2':60.083,'Na2O':30.99,'K2O':47.098} # relative molecular mass, e.g., SiO2, MgO, NaO1.5  
cm_tot = 1.833  # sum of relative cation mole mass, e.g., NaO0.5, SiO2, MgO, to converse between cation mole and wt%, estimated from melt compositions of Walter 1998 and Baker and Stolper 1994
molar_tot = 1.65  # sum of relative molecular mass, e.g., Na2O, SiO2, MgO, to calculate molar mass of SiO2, K2O and Na2O, estimated from melt compositions of Walter 1998 and Baker and Stolper 1994

# input:
# magma compositions in wt%: MgO,FeO,SiO2,Na2O,K2O,NiO,MnO
#magma = {'MgO':5.7,'FeO':4.1*0.9,'SiO2':55.5,'Na2O':7.4,'K2O':0.43,'NiO':0.018,'MnO':0.065}
#magma = {'MgO':10.7,'FeO':6*0.9,'SiO2':49.7,'Na2O':2.8,'K2O':0.06,'NiO':0.036,'MnO':0.114}
#magma = {'MgO':10.7,'FeO':6*0.9,'SiO2':50.3,'Na2O':3.2,'K2O':0.08,'NiO':0.036,'MnO':0.114}
#magma = {'MgO':11.4,'FeO':6.1*0.9,'SiO2':50.3,'Na2O':2.6,'K2O':0.07,'NiO':0.038,'MnO':0.119}
#magma = {'MgO':12.3,'FeO':6.5*0.9,'SiO2':49.6,'Na2O':2,'K2O':0.05,'NiO':0.043,'MnO':0.125}
#magma = {'MgO':13,'FeO':6.7*0.9,'SiO2':49.7,'Na2O':1.6,'K2O':0.07,'NiO':0.048,'MnO':0.128}
#magma = {'MgO':13.3,'FeO':6.8*0.9,'SiO2':49.8,'Na2O':1.58,'K2O':0.02,'NiO':0.049,'MnO':0.129}
#magma = {'MgO':13.9,'FeO':7.1*0.9,'SiO2':50.4,'Na2O':1.28,'K2O':0,'NiO':0.053,'MnO':0.131}
#magma = {'MgO':14.5,'FeO':7.3*0.9,'SiO2':50.5,'Na2O':1.12,'K2O':0,'NiO':0.056,'MnO':0.134}
#magma = {'MgO':15.7,'FeO':7.4*0.9,'SiO2':51,'Na2O':1,'K2O':0,'NiO':0.063,'MnO':0.138}
# crystallization pressure in kbar
P = 0.001
cm_magma = cationmole_magma(magma)
clcm_olonly = cm_magma
clppm_olonly = {'Ni':magma['NiO']*58.6934/74.69*10**4,'Mn':magma['MnO']*54.938/70.94*10**4}
clmolar_olonly = {'SiO2':0,'Na2O':0,'K2O':0}
clmolar_olonly['SiO2'] = 0.01*clcm_olonly['SiO2']*cm_tot/molar_tot
clmolar_olonly['Na2O'] = 0.01*clcm_olonly['Na2O']*cm_tot*cm_mass['Na2O']/(cm_mass['Na2O']*2)/molar_tot
clmolar_olonly['K2O'] = 0.01*clcm_olonly['K2O']*cm_tot*cm_mass['K2O']/(cm_mass['K2O']*2)/molar_tot
if clmolar_olonly['SiO2'] <= 0.6:
    molarSiO2_adjust = 100*clmolar_olonly['SiO2']+100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O'])*((0.46*100/(100-100*clmolar_olonly['SiO2'])-0.93)*100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O'])-5.33*100/(100-100*clmolar_olonly['SiO2'])+9.69)
else:
    molarSiO2_adjust = 100*clmolar_olonly['SiO2']+100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O'])*(11-5.5*100/(100-100*clmolar_olonly['SiO2']))*math.exp(-0.13*100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O']))
T = get_firstT_olonly(clcm_olonly,P,molarSiO2_adjust)
liquidusT_olonly = T
cm_kdMg_oll_olonly = math.exp(6921/(T+273.15)+0.034*clcm_olonly['Na2O']+0.063*clcm_olonly['K2O']+0.01154*P-3.27)
#cm_kdMg_oll_olonly = math.exp(6604/(T+273.15)+0.03014*clcm_olonly['Na2O']+0.1454*clcm_olonly['K2O']+0.010076*P-3.1174)
kdFe2Mg_oll_olonly = math.exp(-6766/(8.3144*(T+273.15))-7.34/8.3144+math.log(0.036*molarSiO2_adjust-0.22)+3000*(1-2*clcm_olonly['MgO']*cm_kdMg_oll_olonly/66.67)/(8.3144*(T+237.15))+0.035*(P*10**3-1)/(8.3144*(T+273.15)))
cm_kdFe2_oll_olonly = kdFe2Mg_oll_olonly*cm_kdMg_oll_olonly
olcm_olonly = {'MgO':0,'FeO':0}
olcm_olonly['MgO'] = clcm_olonly['MgO']*cm_kdMg_oll_olonly
olcm_olonly['FeO'] = clcm_olonly['FeO']*cm_kdFe2_oll_olonly
ol_stoich_olonly = olcm_olonly['MgO']+olcm_olonly['FeO']
fo_olonly = 100*olcm_olonly['MgO']/66.67
#wt_kdNi_oll_olonly = (3.346*cm_kdMg_oll_olonly-3.665)*1.09
#wt_kdNi_oll_olonly = math.exp(4505/(T+273.15)-2.075)*(cm_kdMg_oll_olonly*1.09)
#wt_kdNi_oll_olonly = math.exp(4288/(T+273.15)+0.01804*(clcm_olonly['SiO2']*cm_tot*cm_mass['SiO2']/100)-2.8799)*(cm_kdMg_oll_olonly*1.09)
#wt_kdNi_oll_olonly = math.exp(4449/(T+273.15)+0.01137*(clcm_olonly['SiO2']*cm_tot*cm_mass['SiO2']/100)-2.6345)*(cm_kdMg_oll_olonly*1.09)
wt_kdNi_oll_olonly = math.exp(4272/(T+273.15)+0.01582*(clcm_olonly['SiO2']*cm_tot*cm_mass['SiO2']/100)-2.7622)*(cm_kdMg_oll_olonly*1.09)
#wt_kdNi_oll_olonly = math.exp(4146/(T+273.15)+0.01559*(clcm_olonly['SiO2']*cm_tot*cm_mass['SiO2']/100)-2.6742)*(cm_kdMg_oll_olonly*1.09)
olppm_olonly = {'Ni':0,'Mn':0}
olppm_olonly['Ni'] = clppm_olonly['Ni']*wt_kdNi_oll_olonly
#wt_kdMn_oll_olonly = 0.78
#wt_kdMn_oll_olonly = (0.259*cm_kdMg_oll_olonly-0.049)*1.09
#wt_kdMn_oll_olonly = math.exp(-2.76+3583/T)
#wt_kdMn_oll_olonly = math.exp(0.00877960466828*(cm_tot*40.32*clcm_olonly['MgO']/100)-1.50316580917181)*(cm_kdMg_oll_olonly*1.09)
wt_kdMn_oll_olonly = 0.79*cm_kdFe2_oll_olonly*1.09

olppm_olonly['Mn'] = clppm_olonly['Mn']*wt_kdMn_oll_olonly
f_step_olonly = 1
f_olonly = 1

## format data
T_olonly = []
F_olonly = []
F_step_olonly = []
cmKdMgoll_olonly = []
cmKdFe2oll_olonly = []
KdFe2Mgoll_olonly = []
Clcm_olonly = {element: [] for element in clcm_olonly}
Olcm_olonly = {element: [] for element in olcm_olonly}
Olstoich_olonly = []
Fo_olonly = []
Clmolar_olonly = {element: [] for element in clmolar_olonly}
MolarSiO2_adjust = []
wtKdNioll_olonly = []
wtKdMnoll_olonly = []
Clppm_olonly = {element: [] for element in clppm_olonly}
Olppm_olonly = {element: [] for element in olppm_olonly}

T_olonly.append(T)
F_olonly.append(f_olonly)
F_step_olonly.append(f_step_olonly)
cmKdMgoll_olonly.append(cm_kdMg_oll_olonly)
cmKdFe2oll_olonly.append(cm_kdFe2_oll_olonly)
KdFe2Mgoll_olonly.append(kdFe2Mg_oll_olonly)
for element in Clcm_olonly:
    Clcm_olonly[element].append(clcm_olonly[element])
for element in Olcm_olonly:
    Olcm_olonly[element].append(olcm_olonly[element])
Olstoich_olonly.append(ol_stoich_olonly)
Fo_olonly.append(fo_olonly)
for element in Clmolar_olonly:
    Clmolar_olonly[element].append(clmolar_olonly[element])
MolarSiO2_adjust.append(molarSiO2_adjust)
wtKdNioll_olonly.append(wt_kdNi_oll_olonly)
wtKdMnoll_olonly.append(wt_kdMn_oll_olonly)
for element in Clppm_olonly:
    Clppm_olonly[element].append(clppm_olonly[element])
for element in Olppm_olonly:
    Olppm_olonly[element].append(olppm_olonly[element])

while T>liquidusT_olonly-250:
    T,f_step_olonly,f_olonly,cm_kdMg_oll_olonly,kdFe2Mg_oll_olonly,cm_kdFe2_oll_olonly,clmolar_olonly,molarSiO2_adjust = TF_olonly(T,clmolar_olonly,clcm_olonly,P,f_olonly)
    clcm_olonly,olcm_olonly,ol_stoich_olonly,fo_olonly = concentration_olonly(clcm_olonly,cm_kdMg_oll_olonly,f_step_olonly,cm_kdFe2_oll_olonly,olcm_olonly)
    wt_kdNi_oll_olonly,clppm_olonly,olppm_olonly,wt_kdMn_oll_olonly = NiMn_olonly(T,cm_kdMg_oll_olonly,clppm_olonly,f_step_olonly,olppm_olonly,clcm_olonly,cm_kdFe2_oll_olonly,Po)
    T_olonly.append(T)
    F_olonly.append(f_olonly)
    F_step_olonly.append(f_step_olonly)
    cmKdMgoll_olonly.append(cm_kdMg_oll_olonly)
    cmKdFe2oll_olonly.append(cm_kdFe2_oll_olonly)
    KdFe2Mgoll_olonly.append(kdFe2Mg_oll_olonly)
    for element in Clcm_olonly:
        Clcm_olonly[element].append(clcm_olonly[element])
    for element in Olcm_olonly:
        Olcm_olonly[element].append(olcm_olonly[element])
    Olstoich_olonly.append(ol_stoich_olonly)
    Fo_olonly.append(fo_olonly)
    for element in Clmolar_olonly:
        Clmolar_olonly[element].append(clmolar_olonly[element])
    MolarSiO2_adjust.append(molarSiO2_adjust)
    wtKdNioll_olonly.append(wt_kdNi_oll_olonly)
    wtKdMnoll_olonly.append(wt_kdMn_oll_olonly)
    for element in Clppm_olonly:
        Clppm_olonly[element].append(clppm_olonly[element])
    for element in Olppm_olonly:
        Olppm_olonly[element].append(olppm_olonly[element])
        
# output
clwtMgOarray_olonly = np.asarray(Clcm_olonly['MgO'])*cm_tot*cm_mass['MgO']/100
clwtFeOarray_olonly = np.asarray(Clcm_olonly['FeO'])*cm_tot*cm_mass['FeO']/100
clwtFeOtarray_olonly = clwtFeOarray_olonly/Fe2Fet_MORB
clwtMnOarray_olonly = np.asarray(Clppm_olonly['Mn'])/(10**4)*70.94/54.938
clwtFeOtMnOarray_olonly = clwtFeOarray_olonly/Fe2Fet_MORB/clwtMnOarray_olonly
clwtSiO2array_olonly = np.asarray(Clcm_olonly['SiO2'])*cm_tot*cm_mass['SiO2']/100
Clwt_olonly = {'MgO':clwtMgOarray_olonly.tolist(),'FeO':clwtFeOarray_olonly.tolist(),\
               'FeOt':clwtFeOtarray_olonly.tolist(),'MnO':clwtMnOarray_olonly.tolist(),\
                   'FeOt/MnO':clwtFeOtMnOarray_olonly.tolist(),\
                       'SiO2':clwtSiO2array_olonly.tolist()}
T_olonly = {'T Celsius':T_olonly}
T_olonly = pd.DataFrame(T_olonly)
F_olonly = {'melt fraction':F_olonly}
F_olonly = pd.DataFrame(F_olonly)
F_step_olonly = {'F_step':F_step_olonly}
F_step_olonly = pd.DataFrame(F_step_olonly)
cmKdMgoll_olonly = {'cmkdMgoll':cmKdMgoll_olonly}
cmKdMgoll_olonly = pd.DataFrame(cmKdMgoll_olonly)
cmKdFe2oll_olonly = {'cmkdFe2oll':cmKdFe2oll_olonly}
cmKdFe2oll_olonly = pd.DataFrame(cmKdFe2oll_olonly)
KdFe2Mgoll_olonly = {'KDFe2Mgoll':KdFe2Mgoll_olonly}
KdFe2Mgoll_olonly = pd.DataFrame(KdFe2Mgoll_olonly)
Clcm_olonly = pd.DataFrame(Clcm_olonly)
Clcm_olonly.columns = ['clcm_MgO','clcm_FeO','clcm_SiO2','clcm_Na2O','clcm_K2O']
Olcm_olonly = pd.DataFrame(Olcm_olonly)
Olcm_olonly.columns = ['olcm_MgO','olcm_FeO']
Olstoich_olonly = {'(MgO+FeO)ol':Olstoich_olonly}
Olstoich_olonly = pd.DataFrame(Olstoich_olonly)
Fo_olonly = {'Fo':Fo_olonly}
Fo_olonly = pd.DataFrame(Fo_olonly)
Clmolar_olonly = pd.DataFrame(Clmolar_olonly)
Clmolar_olonly.columns = ['clmolar_SiO2','clmolar_Na2O','clmolar_K2O']
MolarSiO2_adjust = {'molarSiO2_adjust':MolarSiO2_adjust}
MolarSiO2_adjust = pd.DataFrame(MolarSiO2_adjust)
wtKdNioll_olonly = {'wtkdNioll':wtKdNioll_olonly}
wtKdNioll_olonly = pd.DataFrame(wtKdNioll_olonly)
wtKdMnoll_olonly = {'wtkdMnoll':wtKdMnoll_olonly}
wtKdMnoll_olonly = pd.DataFrame(wtKdMnoll_olonly)
Clppm_olonly = pd.DataFrame(Clppm_olonly)
Clppm_olonly.columns = ['clppm_Ni','clppm_Mn']
Olppm_olonly = pd.DataFrame(Olppm_olonly)
Olppm_olonly.columns = ['olppm_Ni','olppm_Mn']
Clwt_olonly = pd.DataFrame(Clwt_olonly)
Clwt_olonly.columns = ['clwt_MgO','clwt_FeO','clwt_FeOt','clwt_MnO','clwt_FeOt/MnO','clwt_SiO2'] 
olonly_xtalization_lowP = pd.concat([T_olonly,F_olonly,F_step_olonly,Clwt_olonly,Clppm_olonly,Fo_olonly,Olppm_olonly,\
                                Olcm_olonly,Olstoich_olonly,cmKdMgoll_olonly,cmKdFe2oll_olonly,\
                                KdFe2Mgoll_olonly,wtKdNioll_olonly,wtKdMnoll_olonly,Clcm_olonly,\
                                    Clmolar_olonly,MolarSiO2_adjust],axis=1)
    

## plot results    
fig_data = pd.read_csv('/Users/apple/Desktop/mdlresults_figdata20230425.csv')

color_Haw = 'navajowhite'
color_MORB = 'skyblue'
color_segmean = 'deepskyblue'
color_DNEMORB = 'purple'
color_abpe = 'slategray'
area = np.pi*2**2
color_mdlHaw = 'blue'
color_mdlMORB = 'red'
color_hotspotMORB = 'dimgray'
color_mdlhotspotMORB = 'orange'

# figure 4a Ni-Fo
plt.figure(figsize=(6.5,5))
plt.scatter(fig_data['Fo_HawOL'],fig_data['Nippm_HawOL'],s=area*0.8,c=color_Haw,edgecolor='black',linewidths=0.1,label='Hawaiian olivine')
plt.scatter(fig_data['Fo_MORBOL'][821:1684],fig_data['Nippm_MORBOL'][821:1684],s=area*0.8,marker='^',c=color_hotspotMORB,edgecolor='white',linewidths=0.1,label='MAR 43° N and FAMOUS olivine')
plt.scatter(fig_data['Fo_MORBOL'][0:821],fig_data['Nippm_MORBOL'][0:821],s=area*0.8,c=color_MORB,edgecolor='black',linewidths=0.1,label='MORB olivine')
plt.plot(olonly_xtalization['Fo'],olonly_xtalization['olppm_Ni'],c=color_mdlHaw,linestyle='-.')
plt.plot(olonly_xtalization_lowP['Fo'],olonly_xtalization_lowP['olppm_Ni'],c=color_mdlMORB,linestyle='-.')
plt.xlabel('Fo mol%',fontsize=12)
plt.ylabel('Ni ppm',fontsize=12)
plt.tick_params(labelsize=11)
plt.xlim(xmax=92,xmin=81)
plt.ylim(ymax=5000,ymin=1000)
plt.legend(loc='upper left',edgecolor='none',fontsize=8,labelspacing=0.5,handlelength=0.6,handletextpad=0.4,borderaxespad=0.15)
#plt.savefig('/Users/apple/Desktop/Figure_1_1.png',dpi=300)
plt.show()

# figure 4c Mn-Fo
plt.figure(figsize=(6.5,5))
plt.scatter(fig_data['Fo_HawOL'],fig_data['Mnppm_HawOL'],s=area*0.8,c=color_Haw,edgecolor='black',linewidths=0.1)
plt.scatter(fig_data['Fo_MORBOL'][821:1684],fig_data['Mnppm_MORBOL'][821:1684],s=area*0.8,marker='^',c=color_hotspotMORB,edgecolor='white',linewidths=0.1)
plt.scatter(fig_data['Fo_MORBOL'][0:821],fig_data['Mnppm_MORBOL'][0:821],s=area*0.8,c=color_MORB,edgecolor='black',linewidths=0.1)
plt.plot(olonly_xtalization['Fo'],olonly_xtalization['olppm_Mn'],c=color_mdlHaw,linestyle='-.')
plt.plot(olonly_xtalization_lowP['Fo'],olonly_xtalization_lowP['olppm_Mn'],c=color_mdlMORB,linestyle='-.')
plt.xlabel('Fo mol%',fontsize=14)
plt.ylabel('Mn ppm',fontsize=14)
plt.tick_params(labelsize=12)
plt.xlim(xmax=92,xmin=81)
plt.ylim(ymax=2400,ymin=800)
#plt.legend(loc='lower right',edgecolor='none',fontsize=12,labelspacing=0.5,handlelength=0.6,handletextpad=0.4,borderaxespad=0.15)
#plt.savefig('/Users/apple/Desktop/Figure_1_1.png',dpi=300)
plt.show()

# figure 4b Ni-MgO
plt.figure(figsize=(6.5,5))
plt.scatter(fig_data['MgO_100_Haw'],fig_data['Ni_Haw'],s=area,c=color_Haw,edgecolor='black',linewidths=0.1,label='Hawaiian lava')
plt.scatter(fig_data['MgO_MORB'],fig_data['Ni_MORB'],s=area,c=color_MORB,edgecolor='black',linewidths=0.1,label='MORB')
#plt.scatter(fig_data['MgO_segmean'],fig_data['Ni_segmean'],s=area,c='none',edgecolor=color_segmean,linewidths=0.5,label='MORB segment mean')
plt.errorbar(fig_data['MgO_DNEMORB'],fig_data['Ni_DNEMORB'],xerr=fig_data['std_MgO_DNEMORB'],yerr=fig_data['std_Ni_DNEMORB'],fmt='o',mfc=color_DNEMORB,mec='none',ms=area*0.3,capsize=1,elinewidth=0.6,label='average MORB')
# plt.plot(model_result['3eclomelt6Ffrac_MgO_L'],model_result['3eclomelt6Ffrac_Ni_L'],c=color_mdlHaw,label='modeled Hawaiian LLD')
# plt.plot(model_result['3eclomelt6Fequ_MgO_L'],model_result['3eclomelt6Fequ_Ni_L'],c=color_mdlHaw,linestyle='--')
plt.plot(olonly_xtalization['clwt_MgO'],olonly_xtalization['clppm_Ni'],c=color_mdlHaw,linestyle='-.')
# plt.plot(model_result['mdlMORB_MgO_L'],model_result['mdlMORB_Ni_L'],c=color_mdlMORB,label='modeled MORB LLD')
#plt.plot(LLD_df['liq_MgO'],LLD_df['liq_Nippm'],c=color_mdlMORB,linestyle='-.')
plt.plot(olonly_xtalization_lowP['clwt_MgO'],olonly_xtalization_lowP['clppm_Ni'],c=color_mdlMORB,linestyle='-.')
plt.xlabel('MgO wt%',fontsize=12)
plt.ylabel('Ni ppm',fontsize=12)
plt.tick_params(labelsize=11)
plt.xlim(xmax=20,xmin=4)
plt.ylim(ymax=1000,ymin=0)
# plt.xlim(xmax=10,xmin=6)
# plt.ylim(ymax=250,ymin=0)
plt.legend(loc='upper left',edgecolor='none',fontsize=8,labelspacing=0.5,handlelength=0.6,handletextpad=0.4,borderaxespad=0.15)
#plt.savefig('/Users/apple/Desktop/Figure_1_1.png',dpi=300)
plt.show()

# figure 4d MnO-MgO
plt.figure(figsize=(6.5,5))
plt.scatter(fig_data['MgO_100_Haw'],fig_data['MnO_100_Haw'],s=area,c=color_Haw,label='Hawaiian lava',edgecolor='black',linewidths=0.1)
plt.scatter(fig_data['MgO_MORB'],fig_data['MnO_MORB'],s=area,c=color_MORB,label='MORB',edgecolor='black',linewidths=0.1)
plt.errorbar(fig_data['MgO_DNEMORB'],fig_data['MnO_DNEMORB'],xerr=fig_data['std_MgO_DNEMORB'],yerr=fig_data['std_MnO_DNEMORB'],fmt='o',mfc=color_DNEMORB,mec='none',ms=area*0.3,capsize=1,elinewidth=0.6)
# plt.plot(model_result['3eclomelt6Ffrac_MgO_L'],model_result['3eclomelt6Ffrac_MnO_L'],c=color_mdlHaw,label='modeled Hawaiian LLD')
# plt.plot(model_result['3eclomelt6Fequ_MgO_L'],model_result['3eclomelt6Fequ_MnO_L'],c=color_mdlHaw,linestyle='--')
plt.plot(olonly_xtalization['clwt_MgO'],olonly_xtalization['clwt_MnO'],c=color_mdlHaw,linestyle='-.')
# plt.plot(model_result['mdlMORB_MgO_L'],model_result['mdlMORB_MnO_L'],c=color_mdlMORB,label='modeled MORB LLD')
plt.plot(LLD_df['liq_MgO'],LLD_df['liq_MnO'],c=color_mdlMORB,linestyle='-.')
plt.xlabel('MgO wt%',fontsize=14)
plt.ylabel('MnO wt%',fontsize=14)
plt.tick_params(labelsize=12)
plt.xlim(xmax=20,xmin=4)
plt.ylim(ymax=0.28,ymin=0.1)
#plt.legend(loc='lower right',edgecolor='none',fontsize=12,labelspacing=0.5,handlelength=0.6,handletextpad=0.4,borderaxespad=0.15)
#plt.savefig('/Users/apple/Desktop/Figure_1_1.png',dpi=300)
plt.show()

# figure 4e FeOt-MgO
plt.figure(figsize=(6.5,5))
plt.scatter(fig_data['MgO_100_Haw'],fig_data['FeOt_100_Haw'],s=area,c=color_Haw,label='Hawaiian lava',edgecolor='black',linewidths=0.1)
plt.scatter(fig_data['MgO_MORB'],fig_data['FeOt_MORB'],s=area,c=color_MORB,label='MORB',edgecolor='black',linewidths=0.1)
plt.errorbar(fig_data['MgO_DNEMORB'],fig_data['FeOt_DNEMORB'],xerr=fig_data['std_MgO_DNEMORB'],yerr=fig_data['std_FeOt_DNEMORB'],fmt='o',mfc=color_DNEMORB,mec='none',ms=area*0.3,capsize=1,elinewidth=0.6)
# plt.plot(model_result['3eclomelt6Ffrac_MgO_L'],model_result['3eclomelt6Ffrac_FeOt_L'],c=color_mdlHaw,label='modeled Hawaiian LLD')
# plt.plot(model_result['3eclomelt6Fequ_MgO_L'],model_result['3eclomelt6Fequ_FeOt_L'],c=color_mdlHaw,linestyle='--')
plt.plot(olonly_xtalization['clwt_MgO'],olonly_xtalization['clwt_FeOt'],c=color_mdlHaw,linestyle='-.')
# plt.plot(model_result['mdlMORB_MgO_L'],model_result['mdlMORB_FeOt_L'],c=color_mdlMORB,label='modeled MORB LLD')
plt.plot(LLD_df['liq_MgO'],LLD_df['liq_FeOt'],c=color_mdlMORB,linestyle='-.')
plt.xlabel('MgO wt%',fontsize=14)
plt.ylabel('FeO wt%',fontsize=14)
plt.tick_params(labelsize=12)
plt.xlim(xmax=20,xmin=4)
plt.ylim(ymax=16,ymin=6)
#plt.legend(loc='lower right',edgecolor='none',fontsize=12,labelspacing=0.5,handlelength=0.6,handletextpad=0.4,borderaxespad=0.15)
#plt.savefig('/Users/apple/Desktop/Figure_1_1.png',dpi=300)
plt.show()

# figure 4f FeOtMnO-MgO
plt.figure(figsize=(6.5,5))
plt.scatter(fig_data['MgO_100_Haw'],fig_data['FeOMnO_100_Haw'],s=area,c=color_Haw,label='Hawaiian lava',edgecolor='black',linewidths=0.1)
plt.scatter(fig_data['MgO_MORB'],fig_data['FeOMnO_MORB'],s=area,c=color_MORB,label='MORB',edgecolor='black',linewidths=0.1)
plt.scatter(fig_data['MgO_DNEMORB'],fig_data['FeOtMnO_DNEMORB'],s=area,c=color_DNEMORB,edgecolor='black',linewidths=0.1)
# plt.plot(model_result['3eclomelt6Ffrac_MgO_L'],model_result['3eclomelt6Ffrac_FeOtMnO_L'],c=color_mdlHaw,label='modeled Hawaiian LLD')
# plt.plot(model_result['3eclomelt6Fequ_MgO_L'],model_result['3eclomelt6Fequ_FeOtMnO_L'],c=color_mdlHaw,linestyle='--')
plt.plot(olonly_xtalization['clwt_MgO'],olonly_xtalization['clwt_FeOt/MnO'],c=color_mdlHaw,linestyle='-.')
# plt.plot(model_result['mdlMORB_MgO_L'],model_result['mdlMORB_FeOtMnO_L'],c=color_mdlMORB,label='modeled MORB LLD')
plt.plot(LLD_df['liq_MgO'],LLD_df['liq_FeOtMnO'],c=color_mdlMORB,linestyle='-.')
plt.xlabel('MgO wt%',fontsize=14)
plt.ylabel('FeO/MnO',fontsize=14)
plt.tick_params(labelsize=12)
plt.xlim(xmax=20,xmin=4)
plt.ylim(ymax=90,ymin=40)
#plt.legend(loc='lower right',edgecolor='none',fontsize=12,labelspacing=0.5,handlelength=0.6,handletextpad=0.4,borderaxespad=0.15)
#plt.savefig('/Users/apple/Desktop/Figure_1_1.png',dpi=300)
plt.show()

















