# crystalization functions for olivine-only crystallization
# only target to calculate MgO, FeO (and FeOt), MnO and NiO in both olivines and melts during fractional or equilibrium crystallization
# 'equ' refers to 'equilibrium crystalliztaion'. Otherwise, fractional crystalliztaion is calculated
# Jan 17, 2023
# written by: Mingzhen Yu
# last modified: Jun 20, 2023

import numpy as np
import pandas as pd
import math
import sympy  
import copy  
from scipy.optimize import fsolve  

cm_mass = {'MgO':40.304,'FeO':71.844,'SiO2':60.083,'Na2O':30.99,'K2O':47.098}  # relative molecular mass, e.g., SiO2, MgO, NaO1.5  
cm_tot = 1.833  # sum of relative cation mole mass, e.g., NaO0.5, SiO2, MgO, to converse between cation mole and wt%, estimated from melt compositions of Walter 1998 and Baker and Stolper 1994
molar_tot = 1.65  # sum of relative molecular mass, e.g., Na2O, SiO2, MgO, to calculate molar mass of SiO2, K2O and Na2O, estimated from melt compositions of Walter 1998 and Baker and Stolper 1994

# convert unit of magma concentrations from wt% to cation mole percent
def cationmole_magma(magma):  
    cm_magma = {element:100*magma[element]/cm_mass[element]/cm_tot for element in cm_mass}
    return cm_magma

# calculate the liquidus temperature based on olivine MgO+olivine FeO=66.67, T in Celsius    
def get_firstT_olonly(clcm_olonly,P,molarSiO2_adjust):
    constantA = clcm_olonly['MgO']
    constantB = 0.034*clcm_olonly['Na2O']+0.063*clcm_olonly['K2O']+0.01154*P-3.27
    constantC = clcm_olonly['FeO']
    constantD = -7.34/8.3144+math.log(0.036*molarSiO2_adjust-0.22)
    TK = fsolve(lambda tk: constantA*math.exp(6921/tk+constantB)+constantC*math.exp(6921/tk+constantB)*math.exp((-3766-6000*constantA/66.67*math.exp(6921/tk+constantB))/(8.3144*tk)+constantD)-66.67, 1600)
    T = TK[0]-273.15
    return T

# calculate the extent of fractional crysatallization per each step, The decrease of temperature by 1 Celsius is set as one step.
def TF_olonly(T,clmolar_olonly,clcm_olonly,P,f_olonly):
    T = T-1  # 1 Celsius per step
    clmolar_olonly['SiO2'] = 0.01*clcm_olonly['SiO2']*cm_tot/molar_tot  
    clmolar_olonly['Na2O'] = 0.01*clcm_olonly['Na2O']*cm_tot*cm_mass['Na2O']/(cm_mass['Na2O']*2)/molar_tot
    clmolar_olonly['K2O'] = 0.01*clcm_olonly['K2O']*cm_tot*cm_mass['K2O']/(cm_mass['K2O']*2)/molar_tot
    if clmolar_olonly['SiO2'] <= 0.6:
        molarSiO2_adjust = 100*clmolar_olonly['SiO2']+100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O'])*((0.46*100/(100-100*clmolar_olonly['SiO2'])-0.93)*100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O'])-5.33*100/(100-100*clmolar_olonly['SiO2'])+9.69)
    else:
        molarSiO2_adjust = 100*clmolar_olonly['SiO2']+100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O'])*(11-5.5*100/(100-100*clmolar_olonly['SiO2']))*math.exp(-0.13*100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O']))
    cm_kdMg_oll_olonly = math.exp(6921/(T+273.15)+0.034*clcm_olonly['Na2O']+0.063*clcm_olonly['K2O']+0.01154*P-3.27)  # KdMg(ol/l) refers to Langmuir et al. 1992
    kdFe2Mg_oll_olonly = math.exp(-6766/(8.3144*(T+273.15))-7.34/8.3144+math.log(0.036*molarSiO2_adjust-0.22)+3000*(1-2*clcm_olonly['MgO']*cm_kdMg_oll_olonly/66.67)/(8.3144*(T+237.15))+0.035*(P*10**3-1)/(8.3144*(T+273.15)))  # KDFeMg(ol/l) refers to Toplis 2005
    cm_kdFe2_oll_olonly = kdFe2Mg_oll_olonly*cm_kdMg_oll_olonly
    a_olonly = 66.67*(1-cm_kdMg_oll_olonly)*(1-cm_kdFe2_oll_olonly)
    b_olonly = (66.67-clcm_olonly['FeO'])*cm_kdFe2_oll_olonly*(1-cm_kdMg_oll_olonly)+(66.67-clcm_olonly['MgO'])*cm_kdMg_oll_olonly*(1-cm_kdFe2_oll_olonly)
    c_olonly = (66.67-clcm_olonly['MgO']-clcm_olonly['FeO'])*cm_kdMg_oll_olonly*cm_kdFe2_oll_olonly
    d_olonly = (b_olonly**2-4*a_olonly*c_olonly)**(0.5)
    f_step_olonly = (-b_olonly-d_olonly)/(2*a_olonly)
    f_olonly = f_olonly*f_step_olonly
    return T,f_step_olonly,f_olonly,cm_kdMg_oll_olonly,kdFe2Mg_oll_olonly,cm_kdFe2_oll_olonly,clmolar_olonly,molarSiO2_adjust

# calculate MgO, FeO, SiO2 in the melts and olivines and Na2O and K2O in the melts during fractional crystallization    
def concentration_olonly(clcm_olonly,cm_kdMg_oll_olonly,f_step_olonly,cm_kdFe2_oll_olonly,olcm_olonly):
    clcm_olonly['MgO'] = clcm_olonly['MgO']/(cm_kdMg_oll_olonly*(1-f_step_olonly)+f_step_olonly)
    clcm_olonly['FeO'] = clcm_olonly['FeO']/(cm_kdFe2_oll_olonly*(1-f_step_olonly)+f_step_olonly)
    olcm_olonly['MgO'] = clcm_olonly['MgO']*cm_kdMg_oll_olonly
    olcm_olonly['FeO'] = clcm_olonly['FeO']*cm_kdFe2_oll_olonly
    ol_stoich_olonly = olcm_olonly['MgO']+olcm_olonly['FeO']
    fo_olonly = 100*olcm_olonly['MgO']/66.67
    clcm_olonly['Na2O'] = clcm_olonly['Na2O']/f_step_olonly
    clcm_olonly['K2O'] = clcm_olonly['K2O']/f_step_olonly
    clcm_olonly['SiO2'] = (clcm_olonly['SiO2']-(1-f_step_olonly)*(100-66.67))/f_step_olonly
    return clcm_olonly,olcm_olonly,ol_stoich_olonly,fo_olonly

# calculate Ni and Mn in the melts and olivines during fractional crystallization 
def NiMn_olonly(T,cm_kdMg_oll_olonly,clppm_olonly,f_step_olonly,olppm_olonly,clcm_olonly,cm_kdFe2_oll_olonly,Po):
    wt_kdNi_oll_olonly = math.exp(4272/(T+273.15)+0.01582*(clcm_olonly['SiO2']*cm_tot*cm_mass['SiO2']/100)-2.7622)*(cm_kdMg_oll_olonly*1.09) ## fitted by MPN+Hzb dataset (Eqn. 1 in the paper), *1.09 to convert from cmf to wt%, observed from Walter 1998 
    clppm_olonly['Ni'] = clppm_olonly['Ni']/(wt_kdNi_oll_olonly*(1-f_step_olonly)+f_step_olonly)
    olppm_olonly['Ni'] = clppm_olonly['Ni']*wt_kdNi_oll_olonly
    wt_kdMn_oll_olonly = 0.79*cm_kdFe2_oll_olonly*1.09  # KDMnFe(ol/l) from Davis et al. (2013), *1.09 to convert from cmf to wt%, observed from Walter 1998
    clppm_olonly['Mn'] = clppm_olonly['Mn']/(wt_kdMn_oll_olonly*(1-f_step_olonly)+f_step_olonly)
    olppm_olonly['Mn'] = clppm_olonly['Mn']*wt_kdMn_oll_olonly
    return wt_kdNi_oll_olonly,clppm_olonly,olppm_olonly,wt_kdMn_oll_olonly

# calculate the extent of equilibrium crysatallization per each step, The decrease of temperature by 1 Celsius is set as one step.
def TF_olonly_equ(T,clmolar_olonly,clcm_olonly,P,f_olonly,cm_magma):
    T = T-1  # 1 Celsius per step
    clmolar_olonly['SiO2'] = 0.01*clcm_olonly['SiO2']*cm_tot/molar_tot
    clmolar_olonly['Na2O'] = 0.01*clcm_olonly['Na2O']*cm_tot*cm_mass['Na2O']/(cm_mass['Na2O']*2)/molar_tot
    clmolar_olonly['K2O'] = 0.01*clcm_olonly['K2O']*cm_tot*cm_mass['K2O']/(cm_mass['K2O']*2)/molar_tot
    if clmolar_olonly['SiO2'] <= 0.6:
        molarSiO2_adjust = 100*clmolar_olonly['SiO2']+100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O'])*((0.46*100/(100-100*clmolar_olonly['SiO2'])-0.93)*100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O'])-5.33*100/(100-100*clmolar_olonly['SiO2'])+9.69)
    else:
        molarSiO2_adjust = 100*clmolar_olonly['SiO2']+100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O'])*(11-5.5*100/(100-100*clmolar_olonly['SiO2']))*math.exp(-0.13*100*(clmolar_olonly['Na2O']+clmolar_olonly['K2O']))
    cm_kdMg_oll_olonly = math.exp(6921/(T+273.15)+0.034*clcm_olonly['Na2O']+0.063*clcm_olonly['K2O']+0.01154*P-3.27)  # KdMg(ol/l) refers to Langmuir et al. 1992
    kdFe2Mg_oll_olonly = math.exp(-6766/(8.3144*(T+273.15))-7.34/8.3144+math.log(0.036*molarSiO2_adjust-0.22)+3000*(1-2*clcm_olonly['MgO']*cm_kdMg_oll_olonly/66.67)/(8.3144*(T+237.15))+0.035*(P*10**3-1)/(8.3144*(T+273.15)))  # KDFeMg(ol/l) refers to Toplis 2005
    cm_kdFe2_oll_olonly = kdFe2Mg_oll_olonly*cm_kdMg_oll_olonly
    a_olonly = 66.67*(1-cm_kdMg_oll_olonly)*(1-cm_kdFe2_oll_olonly)
    b_olonly = (66.67-cm_magma['FeO'])*cm_kdFe2_oll_olonly*(1-cm_kdMg_oll_olonly)+(66.67-cm_magma['MgO'])*cm_kdMg_oll_olonly*(1-cm_kdFe2_oll_olonly)
    c_olonly = (66.67-cm_magma['MgO']-cm_magma['FeO'])*cm_kdMg_oll_olonly*cm_kdFe2_oll_olonly
    d_olonly = (b_olonly**2-4*a_olonly*c_olonly)**(0.5)
    f_last = f_olonly
    f_olonly = (-b_olonly-d_olonly)/(2*a_olonly)
    f_step_olonly = f_last-f_olonly
    return T,f_step_olonly,f_olonly,cm_kdMg_oll_olonly,kdFe2Mg_oll_olonly,cm_kdFe2_oll_olonly,clmolar_olonly,molarSiO2_adjust

# calculate MgO, FeO, SiO2 in the melts and olivines and Na2O and K2O in the melts during equilibrium crystallization   
def concentration_olonly_equ(clcm_olonly,cm_kdMg_oll_olonly,f_step_olonly,cm_kdFe2_oll_olonly,olcm_olonly,cm_magma,f_olonly):
    clcm_olonly['MgO'] = cm_magma['MgO']/(cm_kdMg_oll_olonly*(1-f_olonly)+f_olonly)
    clcm_olonly['FeO'] = cm_magma['FeO']/(cm_kdFe2_oll_olonly*(1-f_olonly)+f_olonly)
    olcm_olonly['MgO'] = clcm_olonly['MgO']*cm_kdMg_oll_olonly
    olcm_olonly['FeO'] = clcm_olonly['FeO']*cm_kdFe2_oll_olonly
    ol_stoich_olonly = olcm_olonly['MgO']+olcm_olonly['FeO']
    fo_olonly = 100*olcm_olonly['MgO']/66.67
    clcm_olonly['Na2O'] = cm_magma['Na2O']/f_olonly
    clcm_olonly['K2O'] = cm_magma['K2O']/f_olonly
    clcm_olonly['SiO2'] = (cm_magma['SiO2']-(1-f_olonly)*(100-66.67))/f_olonly
    return clcm_olonly,olcm_olonly,ol_stoich_olonly,fo_olonly 

# calculate Ni and Mn in the melts and olivines during equilibrium crystallization 
def NiMn_olonly_equ(T,cm_kdMg_oll_olonly,clppm_olonly,f_step_olonly,olppm_olonly,clcm_olonly,clppm_magma,f_olonly,cm_kdFe2_oll_olonly):
    wt_kdNi_oll_olonly = math.exp(4272/(T+273.15)+0.01582*(clcm_olonly['SiO2']*cm_tot*cm_mass['SiO2']/100)-2.7622)*(cm_kdMg_oll_olonly*1.09) ## fitted by MPN+Hzb dataset (Eqn.1 in the paper), *1.09 to convert from cmf to wt%, observed from Walter 1998 
    clppm_olonly['Ni'] = clppm_magma['Ni']/(wt_kdNi_oll_olonly*(1-f_olonly)+f_olonly)
    olppm_olonly['Ni'] = clppm_olonly['Ni']*wt_kdNi_oll_olonly
    wt_kdMn_oll_olonly = 0.79*cm_kdFe2_oll_olonly*1.09  # KDMnFe(ol/l) from Davis et al. (2013), *1.09 to convert from cmf to wt%, observed from Walter 1998
    clppm_olonly['Mn'] = clppm_magma['Mn']/(wt_kdMn_oll_olonly*(1-f_olonly)+f_olonly)
    olppm_olonly['Mn'] = clppm_olonly['Mn']*wt_kdMn_oll_olonly
    return wt_kdNi_oll_olonly,clppm_olonly,olppm_olonly,wt_kdMn_oll_olonly
       

