# melting + crystallization portal, see readme file for an introduction of inputs and outputs.
# Mantle source compositions and modes, melting and crystallization starting pressures, extent of melting and crystallization,
# melting and crystallization types can be changed for both Hawaii and MORB.
# Melting results for Hawaii are in dataframe 'melting_df_highP'. Melting results for MORB are in dataframe 'melting_df_lowP'.
# Olivine-only crystallization results for Hawaii are in dataframe 'olonly_xtalization'. Olivine-only crystallization results for MORB are in dataframe 'olonly_xtalization_lowP'. Ol-Pl-Cpx crystallization results for MORB are in dataframe 'LLD_df'.
# Figures shown Hawaii and MORB olivine data with modeld CLDs and Hawaii basalts and MORB glass data with modeled LLDs will be plotted at the end.
# Jan 18, 2023
# written by Mingzhen Yu
# last modified: Jun 20, 2023
     
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


## default parameters with default values
Fe2Fet_Haw = 0.85  # assume ferrous/total Fe = 0.85 for Hawaiian basalts (Rhodes and Vollinger 2005, Brounce et al. 2017, Berry et al. 2018, Zhang et al. 2018, Brounce et al. 2022)
Fe2Fet_MORB = 0.9  # assume ferrous/total Fe = 0.9 for Hawaiian basalts (Rhodes and Vollinger 2005, Brounce et al. 2017, Berry et al. 2018, Zhang et al. 2018, Brounce et al. 2022)
Po_high =45  # starting pressure of polybaric melting modeled for Hawaii
Po_low = 20  # starting pressure of polybaric melting modeled for MORB
F_target_Haw = 0.06  # extent of melting modeled for Hawaii
F_target_MORB = 0.10  # extent of melting modeled for MORB
melting_model_Haw = 'polybaric'   # melting type for Hawaii, 'polybaric' represents polybaric fractionaly melting,can be changed to 'isobaric', meaning isobaric equilibrium melting
melting_model_MORB = 'polybaric'   # melting type for MORB, 'polybaric' represents polybaric fractionaly melting,can be changed to 'isobaric', meaning isobaric equilibrium melting
xtalization_model = 'fractional'  # crystallization type, can be changed to 'equilibrium'

## mantle source compositions for Hawaii and MORB and their corresponding mineral modes
'''
mantle source compositions with modes listed here including: 10.8% MORB + 89.2% DM (S&S(2004))  from Putirka et al. 2011 Table A5,
depleted mantle from Salter and Stracke (2004),
depleted, normal, and fertile peridotite with MgO 36, 38.5 and 41 wt% observed from peridotite data of Carlson and Ionov (2019),
eclogite melt fertilized peridotite formed by mixing 3% eclogite melt (run A177-82 from Pertermann and Hirschmann 2003) with 97% normal peridotite with 38.5 wt% MgO observed by peridotite data of Carlson and Ionov (2019)
'lowP' represents 'low pressure' which is pressure < 30 kbar, here we use 'lowP' to refer to MORB conditions
'''
# source_wt_Haw_Putirka2011 = {'SiO2':45.5, 'TiO2':0.29, 'Al2O3':5.43, 'FeO':8,'CaO':4.36,'MgO':34.9,'MnO':0.14,'K2O':0.03,'Na2O':0.55, 'P2O5':0.030,'Cr2O3':0.33,'NiO':0.225}  # 10.8% MORB + 89.2% S&S(2004) DM from Putirka et al. 2011 Table A5
# source_phase_Haw_Putirka2011 = {'ol':60,'opx':13.9,'cpx':15.5,'gt':10.6,'sp':0}  # 10.8% MORB + 89.2% DM from Putirka et al. 2011 Table A5
# source_phase_Haw_Putirka2011 = {'ol':52.4,'opx':7.2,'cpx':25,'gt':15.4,'sp':0}  # calculated modes for Putirka et al. 2011 Table A5 compositions by using mineral compositions from Davis et al. 2011 3GPa KLB-1 subsolidus
# source_wt_MORB_Putirka2011 = {'SiO2':44.9, 'TiO2':0.13, 'Al2O3':4.28, 'FeO':7.75,'CaO':3.5,'MgO':38.22,'MnO':0.135,'K2O':0.007,'Na2O':0.29, 'P2O5':0.009,'Cr2O3':0.365,'NiO':0.249}  # Salter and Stracke (2004) DM used by Putirka et al. 2011 in their MORB-DM mixing model
# source_phase_MORB_Putirka2011 = {'ol':56.1,'opx':28.1,'cpx':13.3,'gt':0,'sp':2.5}  # calculated modes for Salter&Stracke (2004) DM for MORB by using mineral compositions from Workman and Hart (2005) Table 3
# source_wt_IonovMgO36 = {'SiO2':45.92, 'TiO2':0.2, 'Al2O3':4.98, 'FeO':7.89,'CaO':4.19,'MgO':36,'MnO':0.1365,'K2O':0.013,'Na2O':0.386, 'P2O5':0.013,'Cr2O3':0.38,'NiO':0.233}  # Ionov peridotite when MgO=36
# source_phase_IonovMgO36 = {'ol':49.5,'opx':0,'cpx':34.9,'gt':15.6,'sp':0}   # phase mode for Ionov pe MgO36 calculated using minerals from Walter 1998 run 40.2
# source_phase_IonovMgO36_lowP = {'ol':48,'opx':33,'cpx':17,'gt':0,'sp':2}   # phase mode for Ionov pe MgO36 calculated using minerals from Workman and Hart 2005 mantle minerals
source_wt_IonovMgO385 = {'SiO2':45.27, 'TiO2':0.158, 'Al2O3':4.03, 'FeO':7.872,'CaO':3.36,'MgO':38.5,'MnO':0.1362,'K2O':0.013,'Na2O':0.306, 'P2O5':0.013,'Cr2O3':0.38,'NiO':0.252}  # Ionov peridotite when MgO=38.5
# source_phase_IonovMgO385 = {'ol':55,'opx':7.5,'cpx':28,'gt':9.5,'sp':0}   # modifed after phase mode calculated by Walter 1998 run 40.2 minerals
source_phase_IonovMgO385_lowP = {'ol':56.5,'opx':27.5,'cpx':14,'gt':0,'sp':2}   # phase mode for Ionov pe MgO38.5 calculated using minerals from Workman and Hart 2005 mantle minerals
# source_wt_IonovMgO41 = {'SiO2':44.62, 'TiO2':0.107, 'Al2O3':3.08, 'FeO':7.776,'CaO':2.54,'MgO':41,'MnO':0.1345,'K2O':0.013,'Na2O':0.226, 'P2O5':0.013,'Cr2O3':0.38,'NiO':0.272}  # Ionov peridotite when MgO=41
# source_phase_IonovMgO41 = {'ol':64.3,'opx':8,'cpx':19.3,'gt':8.4,'sp':0}   # phase mode for Ionov pe MgO41 calculated using minerals from Walter 1998 run 40.2
# source_phase_IonovMgO41_lowP = {'ol':62.6,'opx':26.5,'cpx':9.9,'gt':0,'sp':1}   # phase mode for Ionov pe MgO41 calculated using minerals from Workman and Hart 2005 mantle minerals
source_wt_IonovMgO385_eclope = {'SiO2':45.6, 'TiO2':0.3, 'Al2O3':4.37, 'FeO':7.87,'CaO':3.5,'MgO':37.4,'MnO':0.135,'K2O':0.021,'Na2O':0.415, 'P2O5':0.028,'Cr2O3':0.37,'NiO':0.245}  # mix 3% eclogite melt (run A177-82 from Pertermann and Hirschmann 2003) with 97% peridotite (MgO 38.5 wt%)
source_phase_IonovMgO385_eclope = {'ol':53.2,'opx':10.5,'cpx':27.1,'gt':9.2,'sp':0}  # add 3% eclogite melt to a modifed mineral mode for MgO 38.5 peridotite which is 55ol+7.5opx+28cpx+9.5grt (adding solidus eclogite melt will first increase opx in peridotite, Yaxley and Green 1998) 
source_wt_Haw = source_wt_IonovMgO385_eclope   # mantle source for Hawaii used in the melting modeling
source_phase_Haw = source_phase_IonovMgO385_eclope   # mantle modes for Hawaii used in the melting modeling
source_wt_MORB = source_wt_IonovMgO385   # mantle source for MORB used in the melting modeling
source_phase_MORB = source_phase_IonovMgO385_lowP   # mantle modes for Hawaii used in the melting modeling

## high-pressure melting, melting modeling for Hawaii
# input parameters: source compositions in wt.%, initial mineral phases in percent, initial pressure Po in kbar, melting model (polybaric or isobaric)
source_wt = source_wt_Haw
source_phase = source_phase_Haw
Po = Po_high  # >=30 is high-pressure, <30 is low-pressure
melting_model = melting_model_Haw

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
    ## melting stops when the top of melting column reaches to the bottom of the crust
    while p_remain <=0:
        T, P, f, f_step, crust_thickness, p_remain = TPF_polyfrac(P,f,mgnumber_source,Po)
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
    ## calcluate the accumulated melt compositions for polybaric fractional melting    
    Cl_wt_itg1,Cl_wt_itg2,F_melting_itg1,F_melting_itg2 = itg(Cl_wt,F_step,F_melting,Po) 

elif melting_model == 'isobaric':  # isobaric equilibrium melting 
    P = Po  # pressure will be constant during the melting
    f = 0.0000001  # calculate all parameters near solidus assuming the the extent of melting is 0.0000001 
    f_step = 0.0000001
    T = 13*Po+1140+600*(1-Po/88)*f+20*(mgnumber_source-89)
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
    ## melting stops when the extent of melting reaches to about 50%
    while f <=0.5:
        T,f,f_step = TPF_isoequ(P,f,mgnumber_source)
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
            
# melting results output: 
'''
The melting results for Hawaii are saved in a dataframe named 'melting_df_highP', see readme file for an inroduction of each column.
'''
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
    KdNi_wt_df = pd.DataFrame(KdNi_wt)
    KdNi_wt_df.columns = ['KdNi_oll_wt','KdNi_opxl_wt','KdNi_cpxl_wt','KdNi_gtl_wt','KdNi_spl_wt','KdNi_opxol_wt','KdNi_cpxol_wt','KdNi_gtol_wt','KdNi_spol_wt']
    melting_df_highP = pd.concat([T_melting,P_melting,F_step,F_mineral,Phase_tot,F_melting_itg2,Cl_wt_itg2_df,F_melting_itg1,Cl_wt_itg1_df,Ol_df,Cl_cm_df,Cl_molar_df,ClSiO2_adjust,Res,KDFeMg_oll,BulkD_df,KdNi_wt_df,KdMn_wt_df],axis=1)        
else:
    KdNi_wt_df = pd.DataFrame(KdNi_wt)
    KdNi_wt_df.columns = ['KdNi_oll_wt','KdNi_opxl_wt','KdNi_cpxl_wt','KdNi_gtl_wt','KdNi_spl_wt','KdNi_opxol_wt','KdNi_cpxol_wt','KdNi_gtol_wt','KdNi_spol_wt']
    Cl_wt_df = pd.DataFrame(Cl_wt)
    Cl_wt_df.columns = ['clMgO_wt','clFeO_wt','clTiO2_wt','clNa2O_wt','clK2O_wt','clNiO_wt','clMnO_wt','clSiO2_wt']
    F_melting = {'F_liq':F_melting}
    F_melting = pd.DataFrame(F_melting)
    melting_df_highP = pd.concat([T_melting,P_melting,F_melting,F_step,F_mineral,Phase_tot,Cl_wt_df,Ol_df,Cl_cm_df,Cl_molar_df,ClSiO2_adjust,Res,KDFeMg_oll,BulkD_df,KdNi_wt_df,KdMn_wt_df],axis=1)

# olivine-only crystallization
F_target = F_target_Haw  # the extent of melting, determining the magma compositions for crystallization
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

# input: magma compositions in wt%: MgO,FeO,SiO2,Na2O,K2O,NiO,MnO
P = 0.001  # crystallization pressure in kbar
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
    kdFe2Mg_oll_olonly = math.exp(-6766/(8.3144*(T+273.15))-7.34/8.3144+math.log(0.036*molarSiO2_adjust-0.22)+3000*(1-2*clcm_olonly['MgO']*cm_kdMg_oll_olonly/66.67)/(8.3144*(T+237.15))+0.035*(P*10**3-1)/(8.3144*(T+273.15)))
    cm_kdFe2_oll_olonly = kdFe2Mg_oll_olonly*cm_kdMg_oll_olonly
    olcm_olonly = {'MgO':0,'FeO':0}
    olcm_olonly['MgO'] = clcm_olonly['MgO']*cm_kdMg_oll_olonly
    olcm_olonly['FeO'] = clcm_olonly['FeO']*cm_kdFe2_oll_olonly
    ol_stoich_olonly = olcm_olonly['MgO']+olcm_olonly['FeO']
    fo_olonly = 100*olcm_olonly['MgO']/66.67
    wt_kdNi_oll_olonly = math.exp(4272/(T+273.15)+0.01582*(clcm_olonly['SiO2']*cm_tot*cm_mass['SiO2']/100)-2.7622)*(cm_kdMg_oll_olonly*1.09)
    olppm_olonly = {'Ni':0,'Mn':0}
    olppm_olonly['Ni'] = clppm_olonly['Ni']*wt_kdNi_oll_olonly
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

    while T>liquidusT_olonly-350:  # 350 means temperature decreases by 350 Celsius, determining when will the calculation stop
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
    kdFe2Mg_oll_olonly = math.exp(-6766/(8.3144*(T+273.15))-7.34/8.3144+math.log(0.036*molarSiO2_adjust-0.22)+3000*(1-2*clcm_olonly['MgO']*cm_kdMg_oll_olonly/66.67)/(8.3144*(T+237.15))+0.035*(P*10**3-1)/(8.3144*(T+273.15)))
    cm_kdFe2_oll_olonly = kdFe2Mg_oll_olonly*cm_kdMg_oll_olonly
    olcm_olonly = {'MgO':0,'FeO':0}
    olcm_olonly['MgO'] = clcm_olonly['MgO']*cm_kdMg_oll_olonly
    olcm_olonly['FeO'] = clcm_olonly['FeO']*cm_kdFe2_oll_olonly
    ol_stoich_olonly = olcm_olonly['MgO']+olcm_olonly['FeO']
    fo_olonly = 100*olcm_olonly['MgO']/66.67
    wt_kdNi_oll_olonly = math.exp(4272/(T+273.15)+0.01582*(clcm_olonly['SiO2']*cm_tot*cm_mass['SiO2']/100)-2.7622)*(cm_kdMg_oll_olonly*1.09)
    olppm_olonly = {'Ni':0,'Mn':0}
    olppm_olonly['Ni'] = clppm_olonly['Ni']*wt_kdNi_oll_olonly
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
    
    while T>liquidusT_olonly-350:  # 350 means temperature decreases by 350 Celsius, determining when will the calculation stop
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
        
# olivine-only crystallization results output:
'''
Olivine-only crystallization results for Hawaii are saved in dataframe named 'olonly_xtalization', see readme file for an introduction of each column.
'''
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

    
## low-pressure melting, melting modeling for MORB
# input parameters: source compositions in wt.%, initial mineral phases in percent, initial pressure Po in kbar, melting model (polybaric or isobaric)
source_wt = source_wt_MORB
source_phase = source_phase_MORB
Po = Po_low  # >=30 is high-pressure, <30 is low-pressure
melting_model = melting_model_MORB

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
    ## melting stops when the top of melting column reaches to the bottom of the crust
    while p_remain <=0:
        T, P, f, f_step, crust_thickness, p_remain = TPF_polyfrac(P,f,mgnumber_source,Po)
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
    ## calcluate the accumulated melt compositions for polybaric fractional melting    
    Cl_wt_itg1,Cl_wt_itg2,F_melting_itg1,F_melting_itg2 = itg(Cl_wt,F_step,F_melting,Po) 

elif melting_model == 'isobaric':  # isobaric equilibrium melting 
    P = Po  # pressure will be constant during the melting
    f = 0.0000001  # calculate all parameters near solidus assuming the the extent of melting is 0.0000001 
    f_step = 0.0000001
    T = 13*Po+1140+600*(1-Po/88)*f+20*(mgnumber_source-89)
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
    ## melting stops when the extent of melting reaches to about 50%
    while f <=0.5:
        T,f,f_step = TPF_isoequ(P,f,mgnumber_source)
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
            
# melting results output:
'''
Melting results for MORB are saved in dataframe named 'melting_df_lowP', see readme file for an introduction of each column.
'''
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
    KdNi_wt_df = pd.DataFrame(KdNi_wt)
    KdNi_wt_df.columns = ['KdNi_oll_wt','KdNi_opxl_wt','KdNi_cpxl_wt','KdNi_gtl_wt','KdNi_spl_wt','KdNi_opxol_wt','KdNi_cpxol_wt','KdNi_gtol_wt','KdNi_spol_wt']
    melting_df_lowP = pd.concat([T_melting,P_melting,F_step,F_mineral,Phase_tot,F_melting_itg2,Cl_wt_itg2_df,F_melting_itg1,Cl_wt_itg1_df,Ol_df,Cl_cm_df,Cl_molar_df,ClSiO2_adjust,Res,KDFeMg,BulkD_df,KdNi_wt_df,KdMn_wt_df],axis=1)        
else:
    KdNi_wt_df = pd.DataFrame(KdNi_wt)
    KdNi_wt_df.columns = ['KdNi_oll_wt','KdNi_opxl_wt','KdNi_cpxl_wt','KdNi_gtl_wt','KdNi_spl_wt','KdNi_opxol_wt','KdNi_cpxol_wt','KdNi_gtol_wt','KdNi_spol_wt']
    Cl_wt_df = pd.DataFrame(Cl_wt)
    Cl_wt_df.columns = ['clMgO_wt','clFeO_wt','clTiO2_wt','clNa2O_wt','clK2O_wt','clNiO_wt','clMnO_wt','clSiO2_wt']
    F_melting = {'F_liq':F_melting}
    F_melting = pd.DataFrame(F_melting)
    melting_df_lowP = pd.concat([T_melting,P_melting,F_melting,F_step,F_mineral,Phase_tot,Cl_wt_df,Ol_df,Cl_cm_df,Cl_molar_df,ClSiO2_adjust,Res,KDFeMg,BulkD_df,KdNi_wt_df,KdMn_wt_df],axis=1)    

# ol-pl-cpx crystallization
F_target = F_target_MORB  # extent of melting, determining the magma compositions for crystallization
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
t_stop = t_start -250  # 250 means temperature decreases by 250 Celsius, determining when will the crystallization stop
fl,fa_dict,major_oxide_dict,major_phase_oxide_dict = frac_model_trange(t_start, t_stop,system_components,P=1.,kdCalc = kdCalc_langmuir1992) 

# ol-pl-cpx crystallization output:
'''
Results of crystallization involved olivine, plagioclase and clinopyroxene for MORB are saved in dataframe 'LLD_df'. See readme file for an introduction of each column. 
'''
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

# olivine-only crystallization
F_target = F_target_MORB  # extent of melting, determining the magma compositions for crystallization
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

# input: magma compositions in wt%: MgO,FeO,SiO2,Na2O,K2O,NiO,MnO
P = 0.001  # crystallization pressure in kbar
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
kdFe2Mg_oll_olonly = math.exp(-6766/(8.3144*(T+273.15))-7.34/8.3144+math.log(0.036*molarSiO2_adjust-0.22)+3000*(1-2*clcm_olonly['MgO']*cm_kdMg_oll_olonly/66.67)/(8.3144*(T+237.15))+0.035*(P*10**3-1)/(8.3144*(T+273.15)))
cm_kdFe2_oll_olonly = kdFe2Mg_oll_olonly*cm_kdMg_oll_olonly
olcm_olonly = {'MgO':0,'FeO':0}
olcm_olonly['MgO'] = clcm_olonly['MgO']*cm_kdMg_oll_olonly
olcm_olonly['FeO'] = clcm_olonly['FeO']*cm_kdFe2_oll_olonly
ol_stoich_olonly = olcm_olonly['MgO']+olcm_olonly['FeO']
fo_olonly = 100*olcm_olonly['MgO']/66.67
wt_kdNi_oll_olonly = math.exp(4272/(T+273.15)+0.01582*(clcm_olonly['SiO2']*cm_tot*cm_mass['SiO2']/100)-2.7622)*(cm_kdMg_oll_olonly*1.09)
olppm_olonly = {'Ni':0,'Mn':0}
olppm_olonly['Ni'] = clppm_olonly['Ni']*wt_kdNi_oll_olonly
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

while T>liquidusT_olonly-250:  # 250 means temperature decreases by 250 Celsius, determining when will the crystallization stop
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
        
# olivine-only crystallization output:
'''
Olivine-only crystallization results for MORB are saved in dataframe 'olonly_xtalization_lowP'. See readme file for an introduction of each column.  
'''
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
    

## plot results, compare natural data with CLDs and LLDs    
fig_data = pd.read_csv('.../olivine_glass_data.csv')  # must modify the data local address here

# color parameters
color_Haw = 'navajowhite'
color_MORB = 'skyblue'
area = np.pi*2**2
color_mdlHaw = 'blue'
color_mdlMORB = 'red'

# figure CLD Ni-Fo
plt.figure(figsize=(6.5,5))
plt.scatter(fig_data['Fo_HawOL'],fig_data['Nippm_HawOL'],s=area*0.8,c=color_Haw,edgecolor='black',linewidths=0.1,label='Hawaiian olivine')  # Hawaiian olivine data
plt.scatter(fig_data['Fo_MORBOL'],fig_data['Nippm_MORBOL'],s=area*0.8,c=color_MORB,edgecolor='black',linewidths=0.1,label='MORB olivine')  # MORB olivine data
plt.plot(olonly_xtalization['Fo'],olonly_xtalization['olppm_Ni'],c=color_mdlHaw,linestyle='-.',label='fractional crystallization CLD for Hawaiian olivine')  # plot olivine-only 1 atm fractional crystallization results
plt.plot(olonly_xtalization_lowP['Fo'],olonly_xtalization_lowP['olppm_Ni'],c=color_mdlMORB,linestyle='-.',label='fractional crystallization CLD for MORB olivine')  # plot olivine-only 1 atm fractional crystallization results
plt.xlabel('Fo mol%',fontsize=12)
plt.ylabel('Ni ppm',fontsize=12)
plt.tick_params(labelsize=11)
plt.xlim(xmax=92,xmin=81)
plt.ylim(ymax=5000,ymin=1000)
plt.legend(loc='upper left',edgecolor='none',fontsize=10,labelspacing=0.5,handlelength=0.6,handletextpad=0.4,borderaxespad=0.15)
plt.show()

# figure CLD Mn-Fo
plt.figure(figsize=(6.5,5))
plt.scatter(fig_data['Fo_HawOL'],fig_data['Mnppm_HawOL'],s=area*0.8,c=color_Haw,edgecolor='black',linewidths=0.1,label='Hawaiian olivine')  # Hawaiian olivine data
plt.scatter(fig_data['Fo_MORBOL'],fig_data['Mnppm_MORBOL'],s=area*0.8,c=color_MORB,edgecolor='black',linewidths=0.1,label='MORB olivine')  # MORB olivine data
plt.plot(olonly_xtalization['Fo'],olonly_xtalization['olppm_Mn'],c=color_mdlHaw,linestyle='-.',label='fractional crystallization CLD for Hawaiian olivine')  # plot olivine-only 1 atm fractional crystallization results
plt.plot(olonly_xtalization_lowP['Fo'],olonly_xtalization_lowP['olppm_Mn'],c=color_mdlMORB,linestyle='-.',label='fractional crystallization CLD for MORB olivine')  # plot olivine-only 1 atm fractional crystallization results
plt.xlabel('Fo mol%',fontsize=14)
plt.ylabel('Mn ppm',fontsize=14)
plt.tick_params(labelsize=12)
plt.xlim(xmax=92,xmin=81)
plt.ylim(ymax=2400,ymin=800)
plt.legend(loc='lower left',edgecolor='none',fontsize=10,labelspacing=0.5,handlelength=0.6,handletextpad=0.4,borderaxespad=0.15,facecolor='none')
plt.show()

# figure LLD Ni-MgO
plt.figure(figsize=(6.5,5))
plt.scatter(fig_data['MgO_100_Haw'],fig_data['Ni_Haw'],s=area,c=color_Haw,edgecolor='black',linewidths=0.1,label='Hawaiian lava')  # Hawaiian basalts data
plt.scatter(fig_data['MgO_MORB'],fig_data['Ni_MORB'],s=area,c=color_MORB,edgecolor='black',linewidths=0.1,label='MORB')  # MORB glasses data
plt.plot(olonly_xtalization['clwt_MgO'],olonly_xtalization['clppm_Ni'],c=color_mdlHaw,linestyle='-.',label='fractional crystallization LLD for Hawaiian basalts')  # plot olivine-only 1 atm fractional crystallization results
plt.plot(LLD_df['liq_MgO'],LLD_df['liq_Nippm'],c=color_mdlMORB,linestyle='-.',label='fractional crystallization LLD for MORB')  # plot ol-pl-cpx 1 atm fractional crystallization results
# plt.plot(olonly_xtalization_lowP['clwt_MgO'],olonly_xtalization_lowP['clppm_Ni'],c=color_mdlMORB,linestyle='-.')
plt.xlabel('MgO wt%',fontsize=12)
plt.ylabel('Ni ppm',fontsize=12)
plt.tick_params(labelsize=11)
plt.xlim(xmax=14,xmin=4)
plt.ylim(ymax=600,ymin=0)
plt.legend(loc='upper left',edgecolor='none',fontsize=10,labelspacing=0.5,handlelength=0.6,handletextpad=0.4,borderaxespad=0.15)
plt.show()

# figure LLD MnO-MgO
plt.figure(figsize=(6.5,5))
plt.scatter(fig_data['MgO_100_Haw'],fig_data['MnO_100_Haw'],s=area,c=color_Haw,label='Hawaiian lava',edgecolor='black',linewidths=0.1)  # Hawaiian basalts data
plt.scatter(fig_data['MgO_MORB'],fig_data['MnO_MORB'],s=area,c=color_MORB,label='MORB',edgecolor='black',linewidths=0.1)  # MORB glasses data
plt.plot(olonly_xtalization['clwt_MgO'],olonly_xtalization['clwt_MnO'],c=color_mdlHaw,linestyle='-.',label='fractional crystallization LLD for Hawaiian basalts')  # plot olivine-only 1 atm fractional crystallization results
plt.plot(LLD_df['liq_MgO'],LLD_df['liq_MnO'],c=color_mdlMORB,linestyle='-.',label='fractional crystallization LLD for MORB')  # plot ol-pl-cpx 1 atm fractional crystallization results
plt.xlabel('MgO wt%',fontsize=14)
plt.ylabel('MnO wt%',fontsize=14)
plt.tick_params(labelsize=12)
plt.xlim(xmax=14,xmin=4)
plt.ylim(ymax=0.28,ymin=0.1)
plt.legend(loc='lower left',edgecolor='none',fontsize=10,labelspacing=0.5,handlelength=0.6,handletextpad=0.4,borderaxespad=0.15,facecolor='none')
plt.show()

# figure LLD FeOt-MgO
plt.figure(figsize=(6.5,5))
plt.scatter(fig_data['MgO_100_Haw'],fig_data['FeOt_100_Haw'],s=area,c=color_Haw,label='Hawaiian lava',edgecolor='black',linewidths=0.1)  # Hawaiian basalts data
plt.scatter(fig_data['MgO_MORB'],fig_data['FeOt_MORB'],s=area,c=color_MORB,label='MORB',edgecolor='black',linewidths=0.1)  # MORB glasses data
plt.plot(olonly_xtalization['clwt_MgO'],olonly_xtalization['clwt_FeOt'],c=color_mdlHaw,linestyle='-.',label='fractional crystallization LLD for Hawaiian basalts')  # plot olivine-only 1 atm fractional crystallization results
plt.plot(LLD_df['liq_MgO'],LLD_df['liq_FeOt'],c=color_mdlMORB,linestyle='-.',label='fractional crystallization LLD for MORB')  # plot ol-pl-cpx 1 atm fractional crystallization results
plt.xlabel('MgO wt%',fontsize=14)
plt.ylabel('FeOt wt%',fontsize=14)
plt.tick_params(labelsize=12)
plt.xlim(xmax=14,xmin=4)
plt.ylim(ymax=16,ymin=6)
plt.legend(loc='lower left',edgecolor='none',fontsize=10,labelspacing=0.5,handlelength=0.6,handletextpad=0.4,borderaxespad=0.15,facecolor='none')
plt.show()

# figure LLD FeOt/MnO-MgO
plt.figure(figsize=(6.5,5))
plt.scatter(fig_data['MgO_100_Haw'],fig_data['FeOMnO_100_Haw'],s=area,c=color_Haw,label='Hawaiian lava',edgecolor='black',linewidths=0.1)  # Hawaiian basalts data
plt.scatter(fig_data['MgO_MORB'],fig_data['FeOMnO_MORB'],s=area,c=color_MORB,label='MORB',edgecolor='black',linewidths=0.1)  # MORB glasses data
plt.plot(olonly_xtalization['clwt_MgO'],olonly_xtalization['clwt_FeOt/MnO'],c=color_mdlHaw,linestyle='-.',label='fractional crystallization LLD for Hawaiian basalts')  # plot olivine-only 1 atm fractional crystallization results
plt.plot(LLD_df['liq_MgO'],LLD_df['liq_FeOtMnO'],c=color_mdlMORB,linestyle='-.',label='fractional crystallization LLD for MORB')  # plot ol-pl-cpx 1 atm fractional crystallization results
plt.xlabel('MgO wt%',fontsize=14)
plt.ylabel('FeOt/MnO',fontsize=14)
plt.tick_params(labelsize=12)
plt.xlim(xmax=14,xmin=4)
plt.ylim(ymax=90,ymin=40)
plt.legend(loc='upper left',edgecolor='none',fontsize=10,labelspacing=0.5,handlelength=0.6,handletextpad=0.4,borderaxespad=0.15,facecolor='none')
plt.show()

















