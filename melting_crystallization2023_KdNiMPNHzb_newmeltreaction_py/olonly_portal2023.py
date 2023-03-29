# run olivine-only crystallization 
# Jan 17, 2023
# written by: Mingzhen Yu
# last modified:
    
import numpy as np
import pandas as pd
import math
import sympy  
import copy    
from olonly_function2023 import *

# default values
cm_mass = {'MgO':40.304,'FeO':71.844,'SiO2':60.083,'Na2O':30.99,'K2O':47.098} # relative molecular mass, e.g., SiO2, MgO, NaO1.5  
cm_tot = 1.833  # sum of relative cation mole mass, e.g., NaO0.5, SiO2, MgO, to converse between cation mole and wt%, estimated from melt compositions of Walter 1998 and Baker and Stolper 1994
molar_tot = 1.65  # sum of relative molecular mass, e.g., Na2O, SiO2, MgO, to calculate molar mass of SiO2, K2O and Na2O, estimated from melt compositions of Walter 1998 and Baker and Stolper 1994

# input:
# magma compositions in wt%: MgO,FeO,SiO2,Na2O,K2O,NiO,MnO
# crystallization pressure in bar
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
kdFe2Mg_oll_olonly = math.exp(-6766/(8.3144*(T+273.15))-7.34/8.3144+math.log(0.036*molarSiO2_adjust-0.22)+3000*(1-2*clcm_olonly['MgO']*cm_kdMg_oll_olonly/66.67)/(8.3144*(T+237.15))+0.035*(P*10**3-1)/(8.3144*(T+273.15)))
cm_kdFe2_oll_olonly = kdFe2Mg_oll_olonly*cm_kdMg_oll_olonly
olcm_olonly = {'MgO':0,'FeO':0}
olcm_olonly['MgO'] = clcm_olonly['MgO']*cm_kdMg_oll_olonly
olcm_olonly['FeO'] = clcm_olonly['FeO']*cm_kdFe2_oll_olonly
ol_stoich_olonly = olcm_olonly['MgO']+olcm_olonly['FeO']
fo_olonly = 100*olcm_olonly['MgO']/66.67
#wt_kdNi_oll_olonly = math.exp(4505/(T+273.15)-2.075)*(cm_kdMg_oll_olonly*1.09)
wt_kdNi_oll_olonly = math.exp(4288/(T+273.15)+0.01804*(clcm_olonly['SiO2']*cm_tot*cm_mass['SiO2']/100)-2.8799)*(cm_kdMg_oll_olonly*1.09)
olppm_olonly = {'Ni':0,'Mn':0}
olppm_olonly['Ni'] = clppm_olonly['Ni']*wt_kdNi_oll_olonly
wt_kdMn_oll_olonly = math.exp(0.00877960466828*(cm_tot*40.32*clcm_olonly['MgO']/100)-1.50316580917181)*(cm_kdMg_oll_olonly*1.09)
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

while T>liquidusT_olonly-300:
    T,f_step_olonly,f_olonly,cm_kdMg_oll_olonly,kdFe2Mg_oll_olonly,cm_kdFe2_oll_olonly,clmolar_olonly,molarSiO2_adjust = TF_olonly(T,clmolar_olonly,clcm_olonly,P,f_olonly)
    clcm_olonly,olcm_olonly,ol_stoich_olonly,fo_olonly = concentration_olonly(clcm_olonly,cm_kdMg_oll_olonly,f_step_olonly,cm_kdFe2_oll_olonly,olcm_olonly)
    wt_kdNi_oll_olonly,clppm_olonly,olppm_olonly,wt_kdMn_oll_olonly = NiMn_olonly(T,cm_kdMg_oll_olonly,clppm_olonly,f_step_olonly,olppm_olonly,clcm_olonly)
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
clwtFeOtarray_olonly = clwtFeOarray_olonly/0.85
clwtMnOarray_olonly = np.asarray(Clppm_olonly['Mn'])/(10**4)*70.94/54.938
clwtFeOtMnOarray_olonly = clwtFeOarray_olonly/0.85/clwtMnOarray_olonly
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


    




















