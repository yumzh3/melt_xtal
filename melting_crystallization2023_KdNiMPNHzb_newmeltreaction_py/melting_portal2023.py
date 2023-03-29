# melting portal
# Jan 16, 2023
# written by Mingzhen Yu
# last modified:
     
import numpy as np
import pandas as pd
import math
import sympy  
import copy
from melting_function2023 import *


# default parameters with default values
S_mantle = 200  # all values for sulfur related default parameters are from Zhao etal 2022 and references therein
S_mantlesulfide = 369000
SCSS_QFM_MORB = 1200
f_sulfideliquid = 4/5
f_mss = 1-f_sulfideliquid
deltaQFM = 0

# input parameters:
# source compositions in wt.%, initial mineral phases in percent, initial pressure Po in kbar, melting model (polybaric or isobaric), consider Sulfur or not
source_wt = {'SiO2':45.6, 'TiO2':0.35, 'Al2O3':4.3, 'FeO':8.1,'CaO':3.5,'MgO':37.5,'MnO':0.137,'K2O':0.028,'Na2O':0.41, 'P2O5':0.029,'Cr2O3':0.37,'NiO':0.233} ## add 3% eclogite with peridotite above
melting_model = 'polybaric'
Po = 40  # >=30 is high-pressure, <30 is low-pressure
source_phase = {'ol':53,'opx':7.5,'cpx':31,'gt':8.5,'sp':0}  ## add 3% or 3.5% eclogite melt adjust the mineral phase
S_mode = 'N'  ## question for Charlie: If consider sulfur, does the starting Ni contents need to be increased?

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
        kdMn_wt, bulkD, cl_wt, ol, res = Mn_polyfrac(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol)
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
        kdMn_wt, bulkD, cl_wt, ol, res = Mn_polyfrac(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol)
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
            kdMn_wt, bulkD, cl_wt, ol, res = Mn_polyfrac(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol)
        else:
            f_mineral, phase_tot = mineral_phase_polyfrac(Po,P,f_step,f_mineral)
            cl_wt,bulkD,cl_cm,res = liquid_wt_polyfrac(res,f_step,f_mineral,P,T,cl_wt,cl_cm,bulkD,Po)
            cl_molar,clSiO2_adjust,kdMgO_oll_cm,ol,kdFe2Mg_oll = KDFeMg(T,P,f_step,cl_wt,cl_cm,res,cl_molar,ol)
            ol,cl_cm,res,kdFeO_oll_cm,cl_wt = MgOFeO_polyfrac(ol,kdMgO_oll_cm,f_step,kdFe2Mg_oll,res,cl_cm,cl_wt)
            kdNi_wt, bulkD, cl_wt, ol, res = Ni_polyfrac(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol)
            kdMn_wt, bulkD, cl_wt, ol, res = Mn_polyfrac(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol)
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
        kdMn_wt, bulkD, cl_wt, ol, res = Mn_isoequ(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt)        
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
        kdMn_wt, bulkD, cl_wt, ol, res = Mn_isoequ(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt)
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
            kdMn_wt, bulkD, cl_wt, ol, res = Mn_isoequ(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt)
        else:
            f_mineral, phase_tot,source_phase2,source_phase3,source_phase4,f_gt0,f_cpx0,f_sp0 = mineral_phase_isoequ(Po,P,f,source_phase,source_phase2,f_gt0,source_phase3,source_phase4,f_cpx0,f_sp0,f_mineral)
            cl_wt,bulkD,cl_cm,res = liquid_wt_isoequ(source_wt,f,f_mineral,P,T,cl_wt,cl_cm,bulkD,res,Po)
            cl_molar,clSiO2_adjust,kdMgO_oll_cm,ol,kdFe2Mg_oll = KDFeMg(T,P,f_step,cl_wt,cl_cm,res,cl_molar,ol)
            ol,cl_cm,res,kdFeO_oll_cm,cl_wt = MgOFeO_isoequ(ol,kdMgO_oll_cm,f,kdFe2Mg_oll,res,cl_cm,cl_wt,source_cm)
            kdNi_wt, bulkD, cl_wt, ol, res = Ni_isoequ(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt)
            kdMn_wt, bulkD, cl_wt, ol, res = Mn_isoequ(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt)
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
        melting_df = pd.concat([T_melting,P_melting,F_step,F_mineral,F_sulfide,Phase_tot,F_melting_itg2,Cl_wt_itg2_df,F_melting_itg1,Cl_wt_itg1_df,Ol_df,Cl_cm_df,Cl_molar_df,ClSiO2_adjust,Res,S_Res,KDFeMg,BulkD_df,KdNi_wt_df,KdMn_wt_df],axis=1)
    else:
        KdNi_wt_df = pd.DataFrame(KdNi_wt)
        KdNi_wt_df.columns = ['KdNi_oll_wt','KdNi_opxl_wt','KdNi_cpxl_wt','KdNi_gtl_wt','KdNi_spl_wt','KdNi_opxol_wt','KdNi_cpxol_wt','KdNi_gtol_wt','KdNi_spol_wt']
        melting_df = pd.concat([T_melting,P_melting,F_step,F_mineral,Phase_tot,F_melting_itg2,Cl_wt_itg2_df,F_melting_itg1,Cl_wt_itg1_df,Ol_df,Cl_cm_df,Cl_molar_df,ClSiO2_adjust,Res,KDFeMg,BulkD_df,KdNi_wt_df,KdMn_wt_df],axis=1)        
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
    melting_df = pd.concat([T_melting,P_melting,F_melting,F_step,F_mineral,Phase_tot,Cl_wt_df,Ol_df,Cl_cm_df,Cl_molar_df,ClSiO2_adjust,Res,KDFeMg,BulkD_df,KdNi_wt_df,KdMn_wt_df],axis=1)
























