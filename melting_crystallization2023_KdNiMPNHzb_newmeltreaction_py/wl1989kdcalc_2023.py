###Jocelyn Fuentes 2016 - Based on WL1989
# Mingzhen Yu 2021 - add Ni and Mn, and update KD change with P-T and liquid compositions

from wl1989stoich_2023 import *
import numpy as np
import math


## from wl1989kdcalc2021.py
def kdCalc_langmuir1992(components, T, P, H = None):
    """
    This uses Langmuir Et Al. 1992 Calculations.
    P in bars, T in Kelvin.
    """
    keys = ['MgO', 'FeO', 'TiO2', 'PO52', 'MnO', 'CaAl2O4', 'NaAlO2', 'KAlO2', 'CaSiO3','NiO']
    cpx = {key:0. for key in keys}
    plg = {key:0. for key in keys}
    ol = {key:0. for key in keys}
    oxide_wt = cationFracToWeight(components)
    SiO2_molar = oxide_wt['SiO2']/60.083/1.625
    Na2O_molar = components['NaAlO2']/2*1.833*30.99/(30.99*2)/1.625
    K2O_molar = components['KAlO2']/2*1.833*47.1/(47.1*2)/1.625
    if SiO2_molar > 0.6:
        SiO2_adj = 100*SiO2_molar + 100*(Na2O_molar+K2O_molar)*(11-550/(100-100*SiO2_molar))*np.exp(-0.13*100*(Na2O_molar+K2O_molar))
    else:
        SiO2_adj = 100*SiO2_molar + 100*(Na2O_molar+K2O_molar)*((46/(100-100*SiO2_molar)-0.93)*100*(Na2O_molar+K2O_molar)-533/(100-100*SiO2_molar)+9.69)
    anorthite=components['CaAl2O4']/(components['CaAl2O4']+1.5*components['NaAlO2'])
    plg['CaAl2O4'] = np.power(10.,(2446./T) - (1.122  + 0.2562*anorthite))
    plg['NaAlO2'] = np.power(10.,((3195. + (3283.*anorthite) + (0.0506*P))/T) - (1.885*anorthite) -2.3715)
    plg['KAlO2'] = 0.15
    plg['MnO'] = 0.031
    plg['PO52'] = 0.1
    cpx['MgO'] = np.power(10.,(((3798. + (0.021*P))/T) - 2.28))
    cpx['FeO'] = cpx['MgO']*np.power(10.,-0.6198)
    cpx['CaSiO3'] = np.power(10.,((1783. + (0.0038*P))/T) -0.753)
    cpx['CaAl2O4'] = np.power(10.,((2418. + (0.068*P))/T) -2.3)
    cpx['NaAlO2'] = np.power(10.,((5087. + (0.073*P))/T) - 4.48)
    cpx['TiO2'] = np.power(10.,((1034. + (0.053*P))/T) - 1.27)
    cpx['KAlO2'] = 0.007
    cpx['PO52'] = 0.05
    ol['KAlO2'] = 0.001
    ol['PO52'] = 0.2
    ol['MgO'] = np.exp((6921./T) + (3.4*components['NaAlO2']/2.) + 
                (6.3*components['KAlO2']/2.) + (0.00001154*P) - 3.27)
    # ol['MgO'] = np.exp((6604./T) + (3.014*components['NaAlO2']/2.) + 
    #             (14.54*components['KAlO2']/2.) + (0.000010076*P) - 3.1174)
    KDFeMg = np.exp(-6766./(8.3144*T)-7.34/8.3144+math.log(0.036*SiO2_adj-0.22)+3000*(1-2*components['MgO']*100*ol['MgO']/66.67)/(8.3144*T))  # Toplis KDFe2Mg(ol/l)
    ol['FeO'] = ol['MgO']*KDFeMg  # cmKdFe2(ol/l)
    #ol['MnO'] = 0.78*0.98
    #ol['MnO'] = 0.259*ol['MgO']-0.049
    #ol['MnO'] = np.exp(-2.76+3583/(T-273.15))
    #ol['MnO'] = np.exp(0.0087796046628*oxide_wt['MgO']-1.50316580917181)*ol['MgO']  ## original Matzen equation
    ol['MnO'] = 0.79*ol['FeO']
    #ol['NiO'] = 3.346*ol['MgO']-3.665 ## original Beattie equation
    #ol['NiO'] = np.exp(4505/T-2.075)*ol['MgO']  ## original Matzen equation
    #ol['NiO'] = np.exp(4288/T+0.01804*oxide_wt['SiO2']-2.8799)*ol['MgO'] ## new equation fitted by MPN dataset
    #ol['NiO'] = np.exp(4449/T+0.01137*oxide_wt['SiO2']-2.6345)*ol['MgO'] ## new equation fitted by Hzb dataset
    ol['NiO'] = np.exp(4272/T+0.01582*oxide_wt['SiO2']-2.7622)*ol['MgO'] ## new equation fitted by MPN+Hzb dataset
    #ol['NiO'] = np.exp(4146/T+0.01559*oxide_wt['SiO2']-2.6742)*ol['MgO'] ## new equation fitted by MPN+Hzb dataset
    cpx['NiO'] = 0.24*1.08 * ol['NiO'] # Sobolev etal 2005 TableS1 average KdNi(cpx/ol), 1.08 is a factor to convert wtKdNi(cpx/ol) to cmKdNi(cpx/ol), observed from Walter 1998 data
    # cpx['MnO'] = 1.1*0.98  # Le Roux etal 2011 Table 3 for lowP-T KdMn(cpx/l), 0.98 is a factor to convert wtKdMn(cpx/l) to cmKdMn(cpx/l), observed from Walter 1998 data
    cpx['MnO'] = 0.85*0.98
    kd = {'cpx':cpx, 'ol':ol, 'plg':plg}
    return kd



