# define all functions needed to calculate the mantle melting
# 'polyfrac' means the function is for polybaric fractional melting, 'isoequ' means the function is for isobaric equilibrium melting
# 'cmf' means calculate based on the unit as cation mole fraction, 'wt' means calculate based on the unit as wt%
# Jan 16, 2023
# written by: Mingzhen Yu
# last modified: Jun 20, 2023
    
import numpy as np
import pandas as pd
import math
import sympy  
import copy


# default parameters with default values
cm_mass = {'SiO2':60.083, 'TiO2':79.865, 'Al2O3':50.98, 'FeO':71.844,'CaO':56.077,\
           'MgO':40.304,'MnO':70.937,'K2O':47.098, 
         'Na2O':30.99, 'P2O5':70.972,'Cr2O3':75.99,'NiO':74.69}  # relative molecular mass, e.g., SiO2, MgO, NaO1.5  
molar_tot = 1.65  # sum of relative molecular mass, e.g., Na2O, SiO2, MgO, to calculate molar mass of SiO2, K2O and Na2O, estimated from melt compositions of Walter 1998 and Baker and Stolper 1994
cm_tot = 1.833  # sum of relative cation mole mass, e.g., NaO0.5, SiO2, MgO, to converse between cation mole and wt%, estimated from melt compositions of Walter 1998 and Baker and Stolper 1994

# change the unit of source compositions from wt% to cation mole fraction, and calculate the Mg number of the source
def wttocm(source_wt):  
    source_cm1 = {element:(source_wt[element])/cm_mass[element] for element in source_wt}
    cmtot = sum(source_cm1.values())
    source_cm = {element: source_cm1[element]/cmtot for element in source_cm1}
    mgnumber_source = 100*source_cm['MgO']/(source_cm['MgO']+source_cm['FeO'])
    return source_cm, mgnumber_source

# calcluate the liquid compositions of K2O, Na2O, TiO2 and SiO2 during the fractional melting 
# using equation liquid = bulk/(f+(1-f)*D), while 'liquid' is the liquid composition, 
# 'bulk' is the residual composition from the last melting cell, 
# 'f' is the melting fraction during the current melting cell, 
# 'D' is the bulk distribution coefficients of the composition
def liquid_wt_polyfrac(res,f_step,f_mineral,P,T,cl_wt,cl_cm,bulkD,Po): 
    cl_wt['K2O'] = res['K2O']/(bulkD['K2O']*(1-f_step)+f_step)  # bulk D of K2O is assumed to be 0.005
    if Po >= 30:
        bulkD['Na2O'] = 0.015+0.6*f_mineral['cpx']*0.01  # Na partitioning are from looking experimental data (e.g., Walter 1998, Davis et al. 2013, Salters and Longhi 1999),here the contribution from opx, grt and ol to the bulk D of Na2O is simplified to be 0.015, KdNa2O(cpx/l)=0.6
    else:
        bulkD['Na2O'] = 0.015+0.4*f_mineral['cpx']*0.01  # Na partitioning are from looking experimental data (e.g., Walter 1998, Davis et al. 2013, Salters and Longhi 1999),here the contribution from opx, grt and ol to the bulk D of Na2O is simplified to be 0.015, KdNa2O(cpx/l)=0.4    
    cl_wt['Na2O'] = res['Na2O']/(bulkD['Na2O']*(1-f_step)+f_step)
    bulkD['TiO2'] = 0.015*f_mineral['ol']*0.01+0.086*f_mineral['opx']*0.01+0.35*f_mineral['gt']*0.01+f_mineral['cpx']*0.2*0.01  # Ti partitioning for ol and opx are from Salters & Stracke (2004), for grt (0.35) and cpx (0.2) are from high P experimental data (Takahashi 1986, Herzberg & Zhang 1996, Walter 1998)
    cl_wt['TiO2'] = res['TiO2']/(bulkD['TiO2']*(1-f_step)+f_step)
    if Po >= 30: # the relationship between the SiO2 in liquids and pressure is observed from looking at the experimental data, e.g., Hirose and Kushiro 1993, Kushiro 2001
        cl_wt['SiO2'] = 55.7-0.233*P  
    else:
        cl_wt['SiO2'] = 53.17-0.233*P  
    cl_cm['Na2O'] = cl_wt['Na2O']/(cm_mass['Na2O']*cm_tot)*100
    cl_cm['K2O'] = cl_wt['K2O']/(cm_mass['K2O']*cm_tot)*100
    res['K2O'] = (res['K2O']-f_step*cl_wt['K2O'])/(1-f_step)
    res['Na2O'] = (res['Na2O']-f_step*cl_wt['Na2O'])/(1-f_step)
    res['TiO2'] = (res['TiO2']-f_step*cl_wt['TiO2'])/(1-f_step)
    return cl_wt,bulkD,cl_cm,res

# calcluate the liquid compositions of K2O, Na2O, TiO2 and SiO2 during the equilibrium melting 
# using equation liquid = bulk/(f+(1-f)*D), while 'liquid' is the liquid composition, 
# 'bulk' is the source composition, 
# 'f' is the melting fraction of the system, 
# 'D' is the bulk distribution coefficients of the composition
def liquid_wt_isoequ(source_wt,f,f_mineral,P,T,cl_wt,cl_cm,bulkD,res,Po):  
    cl_wt['K2O'] = source_wt['K2O']/(bulkD['K2O']*(1-f)+f)  # bulk D of K2O is assumed to be 0.005
    if Po >= 30:
        bulkD['Na2O'] = 0.015+0.6*f_mineral['cpx']*0.01  # Na partitioning are from looking experimental data (e.g., Walter 1998, Davis et al. 2013, Salters and Longhi 1999),here the contribution from opx, grt and ol to the bulk D of Na2O is simplified to be 0.015, KdNa2O(cpx/l)=0.6
    else:
        bulkD['Na2O'] = 0.015+0.4*f_mineral['cpx']*0.01  # Na partitioning are from looking experimental data (e.g., Walter 1998, Davis et al. 2013, Salters and Longhi 1999),here the contribution from opx, grt and ol to the bulk D of Na2O is simplified to be 0.015, KdNa2O(cpx/l)=0.4 
    cl_wt['Na2O'] = source_wt['Na2O']/(bulkD['Na2O']*(1-f)+f)
    bulkD['TiO2'] = 0.015*f_mineral['ol']*0.01+0.086*f_mineral['opx']*0.01+0.35*f_mineral['gt']*0.01+f_mineral['cpx']*0.2*0.01  # Ti partitioning for ol and opx are from Salters & Stracke (2004), for grt (0.35) and cpx (0.2) are from high P experimental data (Takahashi 1986, Herzberg & Zhang 1996, Walter 1998)
    cl_wt['TiO2'] = source_wt['TiO2']/(bulkD['TiO2']*(1-f)+f)
    if Po >= 30:  # the relationship between the SiO2 in liquids and pressure and melting extent is observed  from looking at the experimental data, e.g., Hirose and Kushiro 1993 Fig 7, Kushiro 2001
        if f <= 0.37:
            cl_wt['SiO2'] = (55.7-0.233*P)+12*(f-0)
        else:
            cl_wt['SiO2'] = (55.7-0.233*P)+12*(0.37-0)-6.4*(f-0.37) 
    else:
        if f <= 0.135:
            cl_wt['SiO2'] = (53.17-0.233*P)-12*(f-0)
        else:
            cl_wt['SiO2'] = (53.17-0.233*P)-12*(0.135-0)+6.2*(f-0.135)
    cl_cm['Na2O'] = cl_wt['Na2O']/(cm_mass['Na2O']*cm_tot)*100
    cl_cm['K2O'] = cl_wt['K2O']/(cm_mass['K2O']*cm_tot)*100
    res['K2O'] = (source_wt['K2O']-f*cl_wt['K2O'])/(1-f)
    res['Na2O'] = (source_wt['Na2O']-f*cl_wt['Na2O'])/(1-f)
    res['TiO2'] = (source_wt['TiO2']-f*cl_wt['TiO2'])/(1-f)
    return cl_wt,bulkD,cl_cm,res

# calculate Fe2-Mg exchange coefficient between olivine and liquid 
# using equations developed by Toplis 2005, 
# then use olivine stoichiometry (MgO+FeO=66.67) to calculate the FeO concentration in olivine
def KDFeMg(T,P,f_step,cl_wt,cl_cm,res,cl_molar,ol):  
    cl_molar['SiO2'] = cl_wt['SiO2']/cm_mass['SiO2']/molar_tot
    cl_molar['Na2O'] = cl_wt['Na2O']/(cm_mass['Na2O']*2)/molar_tot
    cl_molar['K2O'] = cl_wt['K2O']/(cm_mass['K2O']*2)/molar_tot
    if cl_molar['SiO2'] <= 0.6:
        clSiO2_adjust = 100*cl_molar['SiO2']+100*(cl_molar['Na2O']+cl_molar['K2O'])*((0.46*100/(100-100*cl_molar['SiO2'])-0.93)*100*(cl_molar['Na2O']+cl_molar['K2O'])-5.33*100/(100-100*cl_molar['SiO2'])+9.69)
    else:
        clSiO2_adjust = 100*cl_molar['SiO2']+100*(cl_molar['Na2O']+cl_molar['K2O'])*(11-5.5*100/(100-100*cl_molar['SiO2']))*math.exp(-0.13*100*(cl_molar['Na2O']+cl_molar['K2O']))
    kdMgO_oll_cm = math.exp(6921/(T+273.15)+0.034*cl_cm['Na2O']+0.063*cl_cm['K2O']+0.01154*P-3.27)  # Mg partition coefficient between olivine and liquid refers to Langmuir et al. 1992
    a = -6766/(8.3144*(T+273.15))-7.34/8.3144+math.log(0.036*clSiO2_adjust-0.22)+3000/(8.3144*(T+273.15))+0.035*(P*10**3-1)/(8.3144*(T+273.15))-3000*2/(8.3144*(T+273.15))
    b = 3000*2/(8.3144*(T+273.15)*66.67)
    c = f_step/kdMgO_oll_cm
    d = res['MgO']+res['FeO']-66.67*c
    e = 66.67*c
    f = -66.67*res['FeO']
    x = sympy.Symbol('x')
    ol['FeOcm'] = sympy.nsolve(c*(x**2-x**2/sympy.exp(a+b*x))+d*x+e*x/sympy.exp(a+b*x)+f,0)
    kdFe2Mg_oll = math.exp(-6766/(8.3144*(T+273.15))-7.34/8.3144+math.log(0.036*clSiO2_adjust-0.22)+3000*(1-2*(66.67-ol['FeOcm'])/66.67)/(8.3144*(T+237.15))+0.035*(P*10**3-1)/(8.3144*(T+273.15)))
    return cl_molar,clSiO2_adjust,kdMgO_oll_cm,ol,kdFe2Mg_oll

# calculate MgO in the liquid and olivine as well as the FeO(Fe2+) in the liquid during polyfrac
# use the olivine stoichiometry and mass balance equation 
def MgOFeO_polyfrac(ol,kdMgO_oll_cm,f_step,kdFe2Mg_oll,res,cl_cm,cl_wt):  
    ol['MgOcm'] = 66.67-ol['FeOcm']
    cl_cm['MgO'] = ol['MgOcm']/kdMgO_oll_cm
    res['MgO'] = (res['MgO']-f_step*cl_cm['MgO'])/(1-f_step)
    kdFeO_oll_cm = kdFe2Mg_oll*kdMgO_oll_cm
    cl_cm['FeO'] = ol['FeOcm']/kdFeO_oll_cm  # Fe2+ in the liquid
    res['FeO'] = (res['FeO']-f_step*cl_cm['FeO'])/(1-f_step)
    res['mgnumber'] = 100*res['MgO']/(res['MgO']+res['FeO'])
    ol['Fo'] = 100*ol['MgOcm']/(ol['MgOcm']+ol['FeOcm'])
    cl_wt['MgO'] = cm_tot*cm_mass['MgO']*cl_cm['MgO']/100
    cl_wt['FeO'] = cm_tot*cm_mass['FeO']*cl_cm['FeO']/100
    return ol,cl_cm,res,kdFeO_oll_cm,cl_wt

# calculate MgO in the liquid and olivine as well as the FeO(Fe2+) in the liquid during isoequ
# use the olivine stoichiometry and mass balance equation 
def MgOFeO_isoequ(ol,kdMgO_oll_cm,f,kdFe2Mg_oll,res,cl_cm,cl_wt,source_cm):  
    ol['MgOcm'] = 66.67-ol['FeOcm']
    cl_cm['MgO'] = ol['MgOcm']/kdMgO_oll_cm
    res['MgO'] = (source_cm['MgO']*100-f*cl_cm['MgO'])/(1-f)
    kdFeO_oll_cm = kdFe2Mg_oll*kdMgO_oll_cm
    cl_cm['FeO'] = ol['FeOcm']/kdFeO_oll_cm  # Fe2+ in the liquid
    res['FeO'] = (source_cm['FeO']*100-f*cl_cm['FeO'])/(1-f)
    res['mgnumber'] = 100*res['MgO']/(res['MgO']+res['FeO'])
    ol['Fo'] = 100*ol['MgOcm']/(ol['MgOcm']+ol['FeOcm'])
    cl_wt['MgO'] = cm_tot*cm_mass['MgO']*cl_cm['MgO']/100
    cl_wt['FeO'] = cm_tot*cm_mass['FeO']*cl_cm['FeO']/100
    return ol,cl_cm,res,kdFeO_oll_cm,cl_wt

# calculate the extent of melting, pressure and temperature during polybaric fractional melting
def TPF_polyfrac(P,f,mgnumber_source,Po):  
    if round(f,2) < 0.22:  #  melting functions refer to Langmuir et al. 1992
        f = f+0.01*12/(6+6*(1-P/88))
        f_step = 0.01*12/(6+6*(1-P/88))
    else:
        f = f+0.01*12/(9+9*(1-P/88))
        f_step = 0.01*12/(9+9*(1-P/88))
    P = P-1  # polybaric melting, here we set the interval as 1 kbar
    T = 13*P+1140+600*(1-P/88)*f+20*(mgnumber_source-89) # mantle solidus modified after Langmuir et al. 1992 and Hirschmann 2000.
    crust_thickness = 0.5*f/(Po-P)*(Po-P)**2*10.2/(2.6212*Po**0.038)  # the thickness of the crust generated by the mantle melting, refer to Langmuir et al. 1992
    p_remain = crust_thickness/3-P  # the pressure difference between the depth of the bottom of the crust and the top of the melting column, refer to Langmuir et al. 1992
    return T, P, f, f_step, crust_thickness, p_remain

# calculate the extent of melting and temperature during isobaric equilibrium melting
def TPF_isoequ(P,f,mgnumber_source):  
    if round(f,2) < 0.22:  #  melting functions refer to Langmuir et al. 1992
        f = f+0.01*12/(6+6*(1-P/88))
        f_step = 0.01*12/(6+6*(1-P/88))
    else:
        f = f+0.01*12/(9+9*(1-P/88))
        f_step = 0.01*12/(9+9*(1-P/88))
    T = 13*P+1140+600*(1-P/88)*f+20*(mgnumber_source-89) # mantle solidus modified after Langmuir et al. 1992 and Hirschmann 2000.
    return T,f,f_step

# calculate mineral phase proportions during the polybaric fractional melting 
def mineral_phase_polyfrac(Po,P,f_step,f_mineral):
    if Po >= 30:  # here we assume that when pressure is higher than 30 kbar, garnet will converse to spinel in addition to the melting reactions
        if f_mineral['gt'] > 0:
            if P > 30:  # calculate the garnet-spinel conversion factor
                gt_factor = 0
            elif (f_mineral['gt']-45*f_step)/(1-f_step) >=1:
                gt_factor = 0.2
            else:
                gt_factor = 1  # when there is garnet remainning in the mineral, 45gt+12ol+137cpx=94opx+100melt (Walter 1998, 3 GPa grt-out) and 100gt+20ol=23sp+60opx+37cpx (garnet converse to spinel)
            if f_mineral['cpx'] > 0:
                f_mineral['sp'] = (f_mineral['sp']-0*f_step)/(1-f_step)+0.23*gt_factor*(f_mineral['gt']-45*f_step)/(1-f_step)
                f_mineral['ol'] = (f_mineral['ol']-12*f_step)/(1-f_step)-0.2*gt_factor*(f_mineral['gt']-45*f_step)/(1-f_step)
                f_mineral['opx'] = (f_mineral['opx']+94*f_step)/(1-f_step)+0.6*gt_factor*(f_mineral['gt']-45*f_step)/(1-f_step)
                f_mineral['cpx'] = (f_mineral['cpx']-137*f_step)/(1-f_step)+0.37*gt_factor*(f_mineral['gt']-45*f_step)/(1-f_step)
                f_mineral['gt'] = (1-gt_factor)*(f_mineral['gt']-45*f_step)/(1-f_step)
                if f_mineral['cpx'] < 0:
                    if f_mineral['gt'] < 0:
                        phase_tot = sum(f_mineral.values())
                        f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['cpx']-f_mineral['gt'])*100
                        f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['cpx']-f_mineral['gt'])*100
                        f_mineral['sp'] = f_mineral['sp']/(phase_tot-f_mineral['cpx']-f_mineral['gt'])*100
                        f_mineral['gt'] = 0
                        f_mineral['cpx'] = 0
                    else:
                        phase_tot = sum(f_mineral.values())
                        f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['cpx'])*100
                        f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['cpx'])*100
                        f_mineral['sp'] = f_mineral['sp']/(phase_tot-f_mineral['cpx'])*100
                        f_mineral['gt'] = f_mineral['gt']/(phase_tot-f_mineral['cpx'])*100
                        f_mineral['cpx'] = 0
                if f_mineral['gt'] < 0:
                    phase_tot = sum(f_mineral.values())
                    f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['gt'])*100
                    f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['gt'])*100
                    f_mineral['sp'] = f_mineral['sp']/(phase_tot-f_mineral['gt'])*100
                    f_mineral['cpx'] = f_mineral['cpx']/(phase_tot-f_mineral['gt'])*100
                    f_mineral['gt'] = 0     
            else:  # no cpx originally, so 100gt+20ol=23sp+60opx+37cpx (garnet converse to spinel) and 25gt+13ol+62opx = 100melt (Walter 1998, 7 GPa)
                if P > 30:
                    f_mineral['ol'] = (f_mineral['ol']-13*f_step)/(1-f_step)
                    f_mineral['opx'] = (f_mineral['opx']-62*f_step)/(1-f_step)
                    f_mineral['gt'] = (f_mineral['gt']-25*f_step)/(1-f_step)
                    f_mineral['cpx'] = 0
                    f_mineral['sp'] = 0
                    if f_mineral['gt'] < 0:
                        phase_tot = sum(f_mineral.values())
                        f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['gt'])*100
                        f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['gt'])*100
                        f_mineral['cpx'] = 0
                        f_mineral['sp'] = 0
                        f_mineral['gt'] = 0
                else:
                    f_mineral['sp'] = (f_mineral['sp']-0*f_step)/(1-f_step)+0.23*gt_factor*(f_mineral['gt']-45*f_step)/(1-f_step)
                    f_mineral['ol'] = (f_mineral['ol']-12*f_step)/(1-f_step)-0.2*gt_factor*(f_mineral['gt']-45*f_step)/(1-f_step)
                    f_mineral['opx'] = (f_mineral['opx']+94*f_step)/(1-f_step)+0.6*gt_factor*(f_mineral['gt']-45*f_step)/(1-f_step)
                    f_mineral['cpx'] = (f_mineral['cpx']-137*f_step)/(1-f_step)+0.37*gt_factor*(f_mineral['gt']-45*f_step)/(1-f_step)
                    f_mineral['gt'] = (1-gt_factor)*(f_mineral['gt']-45*f_step)/(1-f_step)
                    if f_mineral['cpx'] < 0:
                        if f_mineral['gt'] < 0:
                            phase_tot = sum(f_mineral.values())
                            f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['cpx']-f_mineral['gt'])*100
                            f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['cpx']-f_mineral['gt'])*100
                            f_mineral['sp'] = f_mineral['sp']/(phase_tot-f_mineral['cpx']-f_mineral['gt'])*100
                            f_mineral['cpx'] = 0
                            f_mineral['gt'] = 0 
                        else:
                            phase_tot = sum(f_mineral.values())
                            f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['cpx'])*100
                            f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['cpx'])*100
                            f_mineral['sp'] = f_mineral['sp']/(phase_tot-f_mineral['cpx'])*100
                            f_mineral['gt'] = f_mineral['gt']/(phase_tot-f_mineral['cpx'])*100
                            f_mineral['cpx'] = 0
                    if f_mineral['gt'] < 0:
                        phase_tot = sum(f_mineral.values())
                        f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['gt'])*100
                        f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['gt'])*100
                        f_mineral['sp'] = f_mineral['sp']/(phase_tot-f_mineral['gt'])*100
                        f_mineral['cpx'] = f_mineral['cpx']/(phase_tot-f_mineral['gt'])*100
                        f_mineral['gt'] = 0 
        else:
            f_mineral['gt'] = 0
            if f_mineral['cpx'] > 0:
                if f_mineral['sp'] > 0.1:  # when garnet is consumed and there is spinel remainning, 38opx+13sp+71cpx=100melt+22ol (Barker and Stolper 1994, 1 GPa sp-out)
                    f_mineral['ol'] = (f_mineral['ol']+22*f_step)/(1-f_step)
                    f_mineral['opx'] = (f_mineral['opx']-38*f_step)/(1-f_step)
                    f_mineral['cpx'] = (f_mineral['cpx']-71*f_step)/(1-f_step)
                    f_mineral['sp'] = (f_mineral['sp']-13*f_step)/(1-f_step)
                    if f_mineral['sp'] < 0.1:  # normalize to 100 when the calculated proportion of spinel is negative
                        phase_tot = sum(f_mineral.values())
                        f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['sp'])*100
                        f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['sp'])*100
                        f_mineral['cpx'] = f_mineral['cpx']/(phase_tot-f_mineral['sp'])*100
                        f_mineral['sp'] = 0                    
                else:
                    phase_tot = sum(f_mineral.values())
                    f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['sp'])*100
                    f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['sp'])*100
                    f_mineral['cpx'] = f_mineral['cpx']/(phase_tot-f_mineral['sp'])*100
                    f_mineral['sp'] = 0
                    if f_mineral['cpx'] > 0.5:  # when garnet and spinel are consumed, and there is clinopyroxene remainning, 147cpx+9ol=56opx+100melt (Walter 1998, 3 GPa cpx-out opx-max)
                        f_mineral['ol'] = (f_mineral['ol']-100*f_step*0.09)/(1-f_step)
                        f_mineral['opx'] = (f_mineral['opx']+100*f_step*0.56)/(1-f_step)
                        f_mineral['cpx'] = (f_mineral['cpx']-100*f_step*1.47)/(1-f_step)
                        if f_mineral['cpx'] < 0.5:  # normalize to 100 when the calculated proportion of clinopyroxene is negative
                            phase_tot = sum(f_mineral.values())
                            f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['cpx'])*100
                            f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['cpx'])*100
                            f_mineral['cpx'] = 0
                    else:
                        phase_tot = sum(f_mineral.values())
                        f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['cpx'])*100
                        f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['cpx'])*100
                        f_mineral['cpx'] = 0
                        if f_mineral['opx'] > 0.5:  # when there are only opx and ol remainning, 97opx+3ol=100melt (Walter 1998, 3 GPa opx-out)
                            f_mineral['ol'] = (f_mineral['ol']-3*f_step)/(1-f_step)
                            f_mineral['opx'] = (f_mineral['opx']-97*f_step)/(1-f_step)
                            if f_mineral['opx'] < 0:  # normalize to 100 when the calculated proportion of orthopyroxene is negative
                                f_mineral['ol'] = 100
                                f_mineral['opx'] = 0
                        else:
                            f_mineral['ol'] = 100
                            f_mineral['opx'] = 0
            else:
                f_mineral['cpx'] = 0
                if f_mineral['sp'] > 0.1:  # when there are only sp, opx and ol remainning, 109opx+20sp=100melt+29ol (Wasylenki et al. 2003, 1 GPa, sp-out)
                    f_mineral['ol'] = (f_mineral['ol']+29*f_step)/(1-f_step)
                    f_mineral['opx'] = (f_mineral['opx']-109*f_step)/(1-f_step)
                    f_mineral['sp'] = (f_mineral['sp']-20*f_step)/(1-f_step)
                    if f_mineral['sp'] < 0.1:
                        phase_tot = sum(f_mineral.values())
                        f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['sp'])*100
                        f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['sp'])*100
                        f_mineral['sp'] = 0
                else:
                    f_mineral['sp'] = 0
                    if f_mineral['opx'] > 0.5:
                        f_mineral['ol'] = (f_mineral['ol']-3*f_step)/(1-f_step)
                        f_mineral['opx'] = (f_mineral['opx']-97*f_step)/(1-f_step)
                        if f_mineral['opx'] < 0:  
                            f_mineral['ol'] = 100
                            f_mineral['opx'] = 0
                    else:
                        f_mineral['ol'] = 100
                        f_mineral['opx'] = 0                             
    else:  # when pressure is lower than 30 kbar, we assume there is no garnet in the source
        f_mineral['gt'] = 0
        if f_mineral['sp'] > 0.1:  # 38opx+13sp+71cpx=100melt+22ol (Barker and Stolper 1994, 1 GPa sp-out)
            f_mineral['ol'] = (f_mineral['ol']+22*f_step)/(1-f_step)
            f_mineral['opx'] = (f_mineral['opx']-38*f_step)/(1-f_step)
            f_mineral['cpx'] = (f_mineral['cpx']-71*f_step)/(1-f_step)
            f_mineral['sp'] = (f_mineral['sp']-13*f_step)/(1-f_step)
            if f_mineral['sp'] < 0.1:  # normalize to 100 when the calculated proportion of spinel is negative
                phase_tot = sum(f_mineral.values())
                f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['sp'])*100
                f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['sp'])*100
                f_mineral['cpx'] = f_mineral['cpx']/(phase_tot-f_mineral['sp'])*100
                f_mineral['sp'] = 0                
        else:
            phase_tot = sum(f_mineral.values())
            f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['sp'])*100
            f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['sp'])*100
            f_mineral['cpx'] = f_mineral['cpx']/(phase_tot-f_mineral['sp'])*100
            f_mineral['sp'] = 0    
            if f_mineral['cpx'] >= 0.5:  # 147cpx+9ol=56opx+100melt (Walter 1998, 3 GPa cpx-out opx-max)
                f_mineral['ol'] = (f_mineral['ol']-100*f_step*0.09)/(1-f_step)
                f_mineral['opx'] = (f_mineral['opx']+100*f_step*0.56)/(1-f_step)
                f_mineral['cpx'] = (f_mineral['cpx']-100*f_step*1.47)/(1-f_step)
                if f_mineral['cpx'] < 0.5:  # normalize to 100 when the calculated proportion of clinopyroxene is negative
                    phase_tot = sum(f_mineral.values())
                    f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['cpx'])*100
                    f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['cpx'])*100
                    f_mineral['cpx'] = 0                    
            else:
                phase_tot = sum(f_mineral.values())
                f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['cpx'])*100
                f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['cpx'])*100
                f_mineral['cpx'] = 0  
                if f_mineral['opx'] > 0.5:  # 124opx=24ol+100melt (Wasylenki etal 2003, 1 GPa opx out)
                    f_mineral['ol'] = (f_mineral['ol']+100*f_step*0.24)/(1-f_step)
                    f_mineral['opx'] = (f_mineral['opx']-100*f_step*1.24)/(1-f_step)
                    if f_mineral['opx'] < 0:  # normalize to 100 when the calculated proportion of orthopyroxene is negative
                        f_mineral['ol'] = 100
                        f_mineral['opx'] = 0
                else:
                    f_mineral['ol'] = 100
                    f_mineral['opx'] = 0
    phase_tot = sum(f_mineral.values())
    return f_mineral, phase_tot

# calculate mineral phase proportions during the isobaric equilibrium melting, refer to Baker and Stolper 1994, and Walter 1998
def mineral_phase_isoequ(Po,P,f,source_phase,source_phase2,f_gt0,source_phase3,source_phase4,f_cpx0,f_sp0,f_mineral):  
    if Po >= 30:  # when pressure is higher than 30 kbar, we assume there is garnet in the source but with no spinel
        if source_phase['gt'] > 0 and f_mineral['gt'] > 0:  # when there is garnet remainning, 45gt+12ol+137cpx=94opx+100melt (Walter 1998, 3 GPa grt-out)
            if f_mineral['cpx'] > 0.5:
                f_mineral['sp'] = 0
                f_mineral['ol'] = (source_phase['ol']-12*f)/(1-f)
                f_mineral['opx'] = (source_phase['opx']+94*f)/(1-f)
                f_mineral['cpx'] = (source_phase['cpx']-137*f)/(1-f)
                f_mineral['gt'] = (source_phase['gt']-45*f)/(1-f)
                if f_mineral['gt'] < 0:  # normalize to 100 when the calculated proportion of garnet is negative
                    if f_mineral['cpx'] < 0.5:  # normalize to 100 when the calculated proportion of cpx is negative
                        phase_tot = sum(f_mineral.values())
                        f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['cpx']-f_mineral['gt'])*100
                        f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['cpx']-f_mineral['gt'])*100
                        f_mineral['cpx'] = 0
                        f_mineral['gt'] = 0
                        source_phase2 = f_mineral.copy()
                        source_phase3 = f_mineral.copy()
                        f_gt0 = f
                        f_cpx0 = f
                    else:
                        phase_tot = sum(f_mineral.values())
                        f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['gt'])*100
                        f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['gt'])*100
                        f_mineral['cpx'] = f_mineral['cpx']/(phase_tot-f_mineral['gt'])*100
                        f_mineral['gt'] = 0
                        source_phase2 = f_mineral.copy()
                        f_gt0 = f
                else:
                    if f_mineral['cpx'] < 0.5:
                        phase_tot = sum(f_mineral.values())
                        f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['cpx'])*100
                        f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['cpx'])*100
                        f_mineral['gt'] = f_mineral['gt']/(phase_tot-f_mineral['cpx'])*100
                        f_mineral['cpx'] = 0
                        source_phase3 = f_mineral.copy()
                        f_cpx0 = f
            else:
                f_mineral['ol'] = (source_phase3['ol']-13*(f-f_cpx0))/(1-(f-f_cpx0))  # when there is garnet remainning but cpx consumed, 25gt+13ol+62opx=100melt (Walter 1998, 7 GPa)
                f_mineral['opx'] = (source_phase3['opx']-62*(f-f_cpx0))/(1-(f-f_cpx0))
                f_mineral['gt'] = (source_phase3['gt']-25*(f-f_cpx0))/(1-(f-f_cpx0))
                if f_mineral['gt'] < 0:  # normalize to 100 when the calculated proportion of garnet is negative
                    phase_tot = sum(f_mineral.values())
                    f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['gt'])*100
                    f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['gt'])*100
                    f_mineral['gt'] = 0
                    source_phase2 = f_mineral.copy()
                    source_phase3 = f_mineral.copy()
                    f_gt0 = f
                    f_cpx0 = f
        elif f_mineral['gt'] == 0:
            if f_mineral['cpx'] > 0.5:  # 147cpx+9ol=56opx+100melt (Walter 1998, 3 GPa cpx-out opx-max)
                f_mineral['ol'] = (source_phase2['ol']-9*(f-f_gt0))/(1-(f-f_gt0))
                f_mineral['opx'] = (source_phase2['opx']+56*(f-f_gt0))/(1-(f-f_gt0))
                f_mineral['cpx'] = (source_phase2['cpx']-147*(f-f_gt0))/(1-(f-f_gt0))
                if f_mineral['cpx'] < 0.5:  # normalize to 100 when the calculated proportion of clinopyroxene is negative
                    phase_tot = sum(f_mineral.values())
                    f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['cpx'])*100
                    f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['cpx'])*100
                    f_mineral['cpx'] = 0 
                    source_phase3 = f_mineral.copy()
                    f_cpx0 = f
            else:
                if f_mineral['opx'] > 0.5:  # when opx is remainning, 97opx+3ol=100melt (Walter 1998, 3 GPa opx-out)
                    f_mineral['ol'] = (source_phase3['ol']-3*(f-f_cpx0))/(1-(f-f_cpx0))
                    f_mineral['opx'] = (source_phase3['opx']-97*(f-f_cpx0))/(1-(f-f_cpx0))
                    if f_mineral['opx'] < 0.5:  # normalize to 100 when the calculated proportion of orthopyroxene is negative
                        f_mineral['ol'] = 100
                        f_mineral['opx'] = 0
                else:
                    f_mineral['ol'] = 100
                    f_mineral['opx'] = 0
    else:  # when the pressure is lower than 30 kbar, we assume there is spinel in the source but no garnet
        f_mineral['gt'] = 0
        if f_mineral['sp'] > 0:  # when spinel is remainning, 13sp+71cpx+38opx=22ol+100melt (Barker and Stolper 1994, 1 GPa sp-out)
            if f_mineral['cpx'] > 0.5:
                f_mineral['ol'] = (source_phase['ol']+22*f)/(1-f)
                f_mineral['opx'] = (source_phase['opx']-38*f)/(1-f)
                f_mineral['cpx'] = (source_phase['cpx']-71*f)/(1-f)
                f_mineral['sp'] = (source_phase['sp']-13*f)/(1-f)
                if f_mineral['sp'] < 0:  # normalize to 100 when the calculated proportion of spinel is negative
                    if f_mineral['cpx'] < 0.5:
                        phase_tot = sum(f_mineral.values())
                        f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['cpx']-f_mineral['sp'])*100
                        f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['cpx']-f_mineral['sp'])*100
                        f_mineral['cpx'] = 0
                        f_mineral['sp'] = 0
                        source_phase4 = f_mineral.copy()
                        source_phase3 = f_mineral.copy()
                        f_sp0 = f
                        f_cpx0 = f
                    else:
                        phase_tot = sum(f_mineral.values())
                        f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['sp'])*100
                        f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['sp'])*100
                        f_mineral['cpx'] = f_mineral['cpx']/(phase_tot-f_mineral['sp'])*100
                        f_mineral['sp'] = 0
                        source_phase4 = f_mineral.copy()
                        f_sp0 = f
                else:
                    if f_mineral['cpx'] < 0.5:  # normalize to 100 when the calculated proportion of clinopyroxene is negative
                        phase_tot = sum(f_mineral.values())
                        f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['cpx'])*100
                        f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['cpx'])*100
                        f_mineral['sp'] = f_mineral['sp']/(phase_tot-f_mineral['cpx'])*100
                        f_mineral['cpx'] = 0
                        source_phase3 = f_mineral.copy()
                        f_cpx0 = f
            else:  # when there are only sp, opx and ol remainning, 109opx+20sp=100melt+29ol (Wasylenki et al. 2003, 1 GPa, sp-out) 
                f_mineral['ol'] = (source_phase4['ol']+29*(f-f_cpx0))/(1-(f-f_cpx0))
                f_mineral['opx'] = (source_phase4['opx']-109*(f-f_cpx0))/(1-(f-f_cpx0))
                f_mineral['sp'] = (source_phase4['sp']-20*(f-f_cpx0))/(1-(f-f_cpx0))
                if f_mineral['sp'] < 0:
                    phase_tot = sum(f_mineral.values())
                    f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['sp'])*100
                    f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['sp'])*100
                    f_mineral['sp'] = 0
                    source_phase4 = f_mineral.copy()
                    source_phase3 = f_mineral.copy()
                    f_sp0 = f
                    f_cpx0 = f
        else:
            if f_mineral['cpx'] > 0.5:  # when cpx is remainning, 147cpx+9ol=56opx+100melt (Walter 1998, 3 GPa cpx-out opx-max)
                f_mineral['ol'] = (source_phase4['ol']-9*(f-f_sp0))/(1-(f-f_sp0))
                f_mineral['opx'] = (source_phase4['opx']+56*(f-f_sp0))/(1-(f-f_sp0))
                f_mineral['cpx'] = (source_phase4['cpx']-147*(f-f_sp0))/(1-(f-f_sp0))
                if f_mineral['cpx'] < 0.5:  # normalize to 100 when the calculated proportion of clinopyroxene is negative
                    phase_tot = sum(f_mineral.values())
                    f_mineral['ol'] = f_mineral['ol']/(phase_tot-f_mineral['cpx'])*100
                    f_mineral['opx'] = f_mineral['opx']/(phase_tot-f_mineral['cpx'])*100
                    f_mineral['cpx'] = 0
                    source_phase3 = f_mineral.copy()
                    f_cpx0 = f                    
            else:
                if f_mineral['opx'] > 0.5:  # when opx is remainning, 124opx=24ol+100melt (Wasylenki etal 2003, 1 GPa opx out)
                    f_mineral['ol'] = (source_phase3['ol']+24*(f-f_cpx0))/(1-(f-f_cpx0))
                    f_mineral['opx'] = (source_phase3['opx']-124*(f-f_cpx0))/(1-(f-f_cpx0))
                    if f_mineral['opx'] < 0.5:  # normalize to 100 when the calculated proportion of orthopyroxene is negative
                        f_mineral['ol'] = 100
                        f_mineral['opx'] = 0
                else:
                    f_mineral['ol'] = 100
                    f_mineral['opx'] = 0
    phase_tot = sum(f_mineral.values())
    return f_mineral, phase_tot,source_phase2,source_phase3,source_phase4,f_gt0,f_cpx0,f_sp0

# calculate Ni in the liquid and olivine during polyfrac melting
def Ni_polyfrac(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol):  
    keys = ['oll','opxl','cpxl','gtl','spl','opxol','cpxol','gtol','spol']
    kdNi_wt = {key:0 for key in keys}  
    kdNi_wt['oll'] = math.exp(4272/(T+273.15)+0.01582*cl_wt['SiO2']-2.7622)*(kdMgO_oll_cm*1.09) ## fitted by MPN+Hzb dataset (Eqn. 3 in the paper), *1.09 to convert from cmf to wt%, observed from Walter 1998 
    # Sobolev et al. (2005) Table S1 average KdNi value
    kdNi_wt['cpxol'] = 0.24 
    kdNi_wt['opxol'] = 0.4  
    kdNi_wt['gtol'] = 0.12
    kdNi_wt['spol'] = 1  # KdNi(sp/ol) refer to Righter et al. (2006) Chemical Geology and Li et al. (2008) GCA
    kdNi_wt['cpxl'] = kdNi_wt['oll']*kdNi_wt['cpxol']
    kdNi_wt['opxl'] = kdNi_wt['oll']*kdNi_wt['opxol']
    kdNi_wt['gtl'] = kdNi_wt['oll']*kdNi_wt['gtol']
    kdNi_wt['spl'] = kdNi_wt['oll']*kdNi_wt['spol']  
    bulkD['Ni'] = (f_mineral['ol']*kdNi_wt['oll']+f_mineral['opx']*kdNi_wt['opxl']+f_mineral['cpx']*kdNi_wt['cpxl']+f_mineral['gt']*kdNi_wt['gtl']+f_mineral['sp']*kdNi_wt['spl'])*0.01
    cl_wt['NiO'] = res['NiO']/(bulkD['Ni']*(1-f_step)+f_step)
    ol['NiOwt'] = cl_wt['NiO']*kdNi_wt['oll']
    res['NiO'] = (res['NiO']-f_step*cl_wt['NiO'])/(1-f_step)
    return kdNi_wt, bulkD, cl_wt, ol, res

# calculate Ni in the liquid and olivine during isoequ melting
def Ni_isoequ(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt):  
    keys = ['oll','opxl','cpxl','gtl','spl','opxol','cpxol','gtol','spol']
    kdNi_wt = {key:0 for key in keys}  
    kdNi_wt['oll'] = math.exp(4272/(T+273.15)+0.01582*cl_wt['SiO2']-2.7622)*(kdMgO_oll_cm*1.09) ## fitted by MPN+Hzb dataset (Eqn. 3 in the paper), *1.09 to convert from cmf to wt%, observed from Walter 1998 
    # Sobolev etal 2005 TableS1 average KdNi value
    kdNi_wt['cpxol'] = 0.24 
    kdNi_wt['opxol'] = 0.4  
    kdNi_wt['gtol'] = 0.12
    kdNi_wt['spol'] = 1  # KdNi(sp/ol) refer to Righter et al. (2006) Chemical Geology and Li et al. (2008) GCA
    kdNi_wt['cpxl'] = kdNi_wt['oll']*kdNi_wt['cpxol']
    kdNi_wt['opxl'] = kdNi_wt['oll']*kdNi_wt['opxol']
    kdNi_wt['gtl'] = kdNi_wt['oll']*kdNi_wt['gtol']
    kdNi_wt['spl'] = kdNi_wt['oll']*kdNi_wt['spol']
    bulkD['Ni'] = (f_mineral['ol']*kdNi_wt['oll']+f_mineral['opx']*kdNi_wt['opxl']+f_mineral['cpx']*kdNi_wt['cpxl']+f_mineral['gt']*kdNi_wt['gtl']+f_mineral['sp']*kdNi_wt['spl'])*0.01
    cl_wt['NiO'] = source_wt['NiO']/(bulkD['Ni']*(1-f)+f)
    ol['NiOwt'] = cl_wt['NiO']*kdNi_wt['oll']
    res['NiO'] = (source_wt['NiO']-f*cl_wt['NiO'])/(1-f)
    return kdNi_wt, bulkD, cl_wt, ol, res

# calculate Mn in the liquid and olivine during the polybaric fractional melting
def Mn_polyfrac(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol,Po,kdFeO_oll_cm):  
    keys = ['oll','opxl','cpxl','gtl','spl','opxol','cpxol','gtol','spol']
    kdMn_wt = {key:0 for key in keys}   
    ## Le Roux et al. (2011) Table 3 for low P-T opx and cpx, Davis et al. (2013) GCA Table 13 for high P-T opx, cpx, and grt, sp
    if Po < 30:
        kdMn_wt['oll'] = 0.79*kdFeO_oll_cm*1.09  # KDMnFe(ol/l) from Davis et al. (2013), *1.09 to convert from cmf to wt%, observed from Walter 1998  
        kdMn_wt['cpxl'] = 0.85 # low P-T melting
        kdMn_wt['opxl'] = 0.7 # low P-T melting
        kdMn_wt['gtl'] = 1.241 # low P-T melting
        kdMn_wt['spl'] = 0.46 # low P-T melting
    else:
        kdMn_wt['oll'] = 0.79*kdFeO_oll_cm*1.09  # KDMnFe(ol/l) from Davis et al. (2013), *1.09 to convert from cmf to wt%, observed from Walter 1998  
        kdMn_wt['cpxl'] = 0.768 # high P-T melting
        kdMn_wt['opxl'] = 0.640 # high P-T melting
        kdMn_wt['gtl'] = 1.241 # high P-T melting
        kdMn_wt['spl'] = 0.46 # high P-T melting 
    bulkD['Mn'] = (f_mineral['ol']*kdMn_wt['oll']+f_mineral['opx']*kdMn_wt['opxl']+f_mineral['cpx']*kdMn_wt['cpxl']+f_mineral['gt']*kdMn_wt['gtl']+f_mineral['sp']*kdMn_wt['spl'])*0.01
    cl_wt['MnO'] = res['MnO']/(bulkD['Mn']*(1-f_step)+f_step)
    ol['MnOwt'] = cl_wt['MnO']*kdMn_wt['oll']
    res['MnO'] = (res['MnO']-f_step*cl_wt['MnO'])/(1-f_step)
    return kdMn_wt, bulkD, cl_wt, ol, res

# calculate Mn in the liquid and olivine during the isobaric equilibrium melting
def Mn_isoequ(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt,Po,kdFeO_oll_cm):  
    keys = ['oll','opxl','cpxl','gtl','spl','opxol','cpxol','gtol','spol']
    kdMn_wt = {key:0 for key in keys}    
    ## Le Roux et al. (2011) Table 3 for low P-T opx and cpx, Davis et al. (2013) GCA Table 13 for high P-T opx, cpx, and grt, sp
    if Po < 30:
        kdMn_wt['oll'] = 0.79*kdFeO_oll_cm*1.09  # KDMnFe(ol/l) from Davis et al. (2013), *1.09 to convert from cmf to wt%, observed from Walter 1998  
        kdMn_wt['cpxl'] = 0.85 # low P-T melting
        kdMn_wt['opxl'] = 0.7 # low P-T melting
        kdMn_wt['gtl'] = 1.241 # low P-T melting
        kdMn_wt['spl'] = 0.46 # low P-T melting
    else:
        kdMn_wt['oll'] = 0.79*kdFeO_oll_cm*1.09  # KDMnFe(ol/l) from Davis et al. (2013), *1.09 to convert from cmf to wt%, observed from Walter 1998  
        kdMn_wt['cpxl'] = 0.768 # high P-T melting
        kdMn_wt['opxl'] = 0.640 # high P-T melting
        kdMn_wt['gtl'] = 1.241 # high P-T melting
        kdMn_wt['spl'] = 0.46 # high P-T melting 
    bulkD['Mn'] = (f_mineral['ol']*kdMn_wt['oll']+f_mineral['opx']*kdMn_wt['opxl']+f_mineral['cpx']*kdMn_wt['cpxl']+f_mineral['gt']*kdMn_wt['gtl']+f_mineral['sp']*kdMn_wt['spl'])*0.01
    cl_wt['MnO'] = source_wt['MnO']/(bulkD['Mn']*(1-f)+f)
    ol['MnOwt'] = cl_wt['MnO']*kdMn_wt['oll']
    res['MnO'] = (source_wt['MnO']-f*cl_wt['MnO'])/(1-f)
    return kdMn_wt, bulkD, cl_wt, ol, res

# calculate the accumulated fractional melt compositions based on the melting column concept from Langmuir et al. 1992
def itg(Cl_wt,F_step,F_melting,Po):  
    Cl_wt_itg1 = {element: [] for element in Cl_wt}
    Cl_wt_itg2 = {element: [] for element in Cl_wt}
    f_itg2_sum = 0
    F_melting_itg1 = F_melting
    F_melting_itg2 = []
    for element in Cl_wt:
        itg1_sum = 0
        itg2_sum = 0
        for i in range(0,len(F_melting)):
            itg1_sum = itg1_sum+Cl_wt[element][i]*F_step[i]
            itg1 = itg1_sum/F_melting[i]
            Cl_wt_itg1[element].append(itg1)
            if Po < 30:
                itg2_sum = itg2_sum+itg1*F_melting[i]
                itg2 = itg2_sum/sum(F_melting[0:i+1])
                Cl_wt_itg2[element].append(itg2)
    for i in range(0,len(F_melting)):
        f_itg2_sum = f_itg2_sum+F_melting[i]
        f_itg2 = f_itg2_sum/(i+1)
        F_melting_itg2.append(f_itg2)
    return Cl_wt_itg1,Cl_wt_itg2,F_melting_itg1,F_melting_itg2


# ## considering the influence of Sulfur during the melting on Ni contents in the melts
# # calculate fraction of sulfide and residual sulfur during polybaric fractional melting, refer to Zhao etal 2022
# def S_polyfrac(S_mantlesulfide,S_res,f_step,deltaQFM,SCSS_QFM_MORB):
#     S_melt = (1+10**(2*deltaQFM-2.1))*SCSS_QFM_MORB
#     S_res = (S_res-f_step*S_melt)/(1-f_step)
#     if S_res < 0:
#         S_res = 0
#     f_sulfide = S_res/S_mantlesulfide
#     return f_sulfide,S_res

# # calculate fraction of sulfide and residual sulfur during isobaric equilibrium melting, refer to Zhao etal 2022
# def S_isoequ(S_mantle,S_mantlesulfide,S_res,f,deltaQFM,SCSS_QFM_MORB):
#     S_melt = (1+10**(2*deltaQFM-2.1))*SCSS_QFM_MORB
#     S_res = (S_mantle-f*S_melt)/(1-f)
#     if S_res < 0:
#         S_res = 0
#     f_sulfide = S_res/S_mantlesulfide
#     return f_sulfide,S_res

# # recalculate mineral phase proportions when considering sulfur during melting
# def mineral_phase_S(f_sulfide,f_mineral):
#     f_mineral = {key:f_mineral[key]*(1-f_sulfide) for key in f_mineral}
#     f_sulfide = 100*f_sulfide
#     return f_mineral, f_sulfide 

# # calculate Ni in the liquid and olivine during polyfrac when considering sulfur during melting
# def Ni_polyfrac_S(T,kdMgO_oll_cm,cl_wt,f_mineral,res,f_step,bulkD,ol,f_sulfideliquid,f_sulfide):
#     kdNi_wt_sulfideliquid_melt = 503 # from Table3 in Li and Audetat 2012, the average of experiments with deltaQFM between 0-1, exclude run LY15 and LY17 which were excluded in Fig3 of their paper
#     kdNi_wt_mss_melt = 415 # from Table2 in Li and Audetat 2012, the average of experiments with deltaQFM between 0-1, exclude run LY15 and LY17 which were excluded in Fig3 of their paper
#     DNi_sulfide = f_sulfideliquid*kdNi_wt_sulfideliquid_melt+(1-f_sulfideliquid)*kdNi_wt_mss_melt
#     keys = ['oll','opxl','cpxl','gtl','spl','surfl','opxol','cpxol','gtol','spol']
#     kdNi_wt = {key:0 for key in keys} 
#     kdNi_wt['oll'] = math.exp(4272/(T+273.15)+0.01582*cl_wt['SiO2']-2.7622)*(kdMgO_oll_cm*1.09) ## fitted by MPN+Hzb dataset (Eqn. 3 in the paper), *1.09 to convert from cmf to wt%, observed from Walter 1998 
#     # Sobolev etal 2005 TableS1 average KdNi(cpx,opx,gt/ol) value
#     kdNi_wt['cpxol'] = 0.24 
#     kdNi_wt['opxol'] = 0.4  
#     kdNi_wt['gtol'] = 0.12
#     kdNi_wt['spol'] = 1  # no KdNisp in their table, but can refer to Righter etal 2006 Chemical Geology and Li etal 2008 GCA for sp
#     kdNi_wt['cpxl'] = kdNi_wt['oll']*kdNi_wt['cpxol']
#     kdNi_wt['opxl'] = kdNi_wt['oll']*kdNi_wt['opxol']
#     kdNi_wt['gtl'] = kdNi_wt['oll']*kdNi_wt['gtol']
#     kdNi_wt['spl'] = kdNi_wt['oll']*kdNi_wt['spol']
#     kdNi_wt['surfl'] = DNi_sulfide
#     bulkD['Ni'] = (f_mineral['ol']*kdNi_wt['oll']+f_mineral['opx']*kdNi_wt['opxl']+f_mineral['cpx']*kdNi_wt['cpxl']+f_mineral['gt']*kdNi_wt['gtl']+f_mineral['sp']*kdNi_wt['spl']+f_sulfide*kdNi_wt['surfl'])*0.01
#     cl_wt['NiO'] = res['NiO']/(bulkD['Ni']*(1-f_step)+f_step)
#     ol['NiOwt'] = cl_wt['NiO']*kdNi_wt['oll']
#     res['NiO'] = (res['NiO']-f_step*cl_wt['NiO'])/(1-f_step)
#     return kdNi_wt, bulkD, cl_wt, ol, res

# # calculate Ni in the liquid and olivine during isoequ when considering sulfur during melting
# def Ni_isoequ_S(T,kdMgO_oll_cm,cl_wt,f,f_mineral,res,bulkD,ol,source_wt,f_sulfideliquid,f_sulfide):  
#     kdNi_wt_sulfideliquid_melt = 503 # from Table3 in Li and Audetat 2012, the average of experiments with deltaQFM between 0-1, exclude run LY15 and LY17 which were excluded in Fig3 of their paper
#     kdNi_wt_mss_melt = 415 # from Table2 in Li and Audetat 2012, the average of experiments with deltaQFM between 0-1, exclude run LY15 and LY17 which were excluded in Fig3 of their paper
#     DNi_sulfide = f_sulfideliquid*kdNi_wt_sulfideliquid_melt+(1-f_sulfideliquid)*kdNi_wt_mss_melt
#     keys = ['oll','opxl','cpxl','gtl','spl','surfl','opxol','cpxol','gtol','spol']
#     kdNi_wt = {key:0 for key in keys}  
#     kdNi_wt['oll'] = math.exp(4272/(T+273.15)+0.01582*cl_wt['SiO2']-2.7622)*(kdMgO_oll_cm*1.09) ## fitted by MPN+Hzb dataset (Eqn. 3 in the paper), *1.09 to convert from cmf to wt%, observed from Walter 1998  
#     # Sobolev etal 2005 TableS1 average KdNi value
#     kdNi_wt['cpxol'] = 0.24 
#     kdNi_wt['opxol'] = 0.4  
#     kdNi_wt['gtol'] = 0.12
#     kdNi_wt['spol'] = 1  # no KdNisp in their table, but can refer to Righter etal 2006 Chemical Geology and Li etal 2008 GCA for sp
#     kdNi_wt['cpxl'] = kdNi_wt['oll']*kdNi_wt['cpxol']
#     kdNi_wt['opxl'] = kdNi_wt['oll']*kdNi_wt['opxol']
#     kdNi_wt['gtl'] = kdNi_wt['oll']*kdNi_wt['gtol']
#     kdNi_wt['spl'] = kdNi_wt['oll']*kdNi_wt['spol']
#     kdNi_wt['surfl'] = DNi_sulfide
#     bulkD['Ni'] = (f_mineral['ol']*kdNi_wt['oll']+f_mineral['opx']*kdNi_wt['opxl']+f_mineral['cpx']*kdNi_wt['cpxl']+f_mineral['gt']*kdNi_wt['gtl']+f_mineral['sp']*kdNi_wt['spl']+f_sulfide*kdNi_wt['surfl'])*0.01
#     cl_wt['NiO'] = source_wt['NiO']/(bulkD['Ni']*(1-f)+f)
#     ol['NiOwt'] = cl_wt['NiO']*kdNi_wt['oll']
#     res['NiO'] = (source_wt['NiO']-f*cl_wt['NiO'])/(1-f)
#     return kdNi_wt, bulkD, cl_wt, ol, res
