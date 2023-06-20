# calculate the liquidus temperature to start crystallization
# temperature decrease by 1 Celsius per step
# calculate the liquid fraction, phase fractions, liquid and phase compositions in wt.% per each step during crystallization
# calculte either fractional or equilibrium crystallization
# detailed algorithm is introduced in Weaver and Langmuir 1990
# originally written by Jocelyn Fuentes 2016
# modified by Mingzhen Yu 2021: add Ni and Mn in the system
# last modified: Jun 20, 2023

from wl1989stoich_2023 import *
from wl1989kdcalc_2023 import *
from wlState_2023 import *
import numpy as np
import math

# default values used to calculate the phase proportions called by function 'wlState_2023'
ta = {'cpx':1., 'plg':1., 'ol':2./3.}
uaj_plg = {'CaAl2O4':5./3., 'NaAlO2':2.5, 'MgO':0., 'FeO':0., 'CaSiO3':0., 'TiO2':0., 'KAlO2':0., 'PO52':0., 'MnO':0.,'NiO':0.} # CaAl2Si2O8, NaAlSi3O8
uaj_ol = {'CaAl2O4':0., 'NaAlO2':0., 'MgO':1, 'FeO':1, 'CaSiO3':0., 'TiO2':0., 'KAlO2':0., 'PO52':0., 'MnO':1., 'NiO':1.} # (Mg,Fe,Ni,Mn)2SiO4
uaj_cpx = {'CaAl2O4':4./3., 'NaAlO2':2., 'MgO':2., 'FeO':2., 'CaSiO3':1., 'TiO2':1., 'KAlO2':0., 'PO52':0., 'MnO':2., 'NiO':2.} # CaAl2SiO6, NaAlSi2O6, MgSiO3, FeSiO3, CaSiO3, MnSiO3, NiSiO3, CaTiO3
uaj = {'ol':uaj_ol, 'plg':uaj_plg, 'cpx':uaj_cpx}

# calculate the liquidus T (in Kelvin)
def get_first_T(system_components, P = 1., kdCalc = kdCalc_langmuir1992):
    firstT = 2000.  # a guess for liquidus T
    deltaT = 100.
    qa, fa, major_liquid_components, solid_phase_components, num_iter = state(system_components,firstT,uaj, ta, P=P, kdCalc= kdCalc)
    fl = 1-sum(fa.values()) # liquid fraction in the system
    if num_iter == 3000:
        print('MAX ITERATION!')
    while (fl == 1.) or (deltaT > 1.):
        if fl == 1.:
            firstT = firstT-deltaT
        elif (fl < 1.) and (deltaT > 1.): # fl<1 means minerals start to xtalized already, means that deltaT is too large now
            firstT = firstT+deltaT
            deltaT = deltaT/10.
            firstT=firstT-deltaT
        qa, fa, major_liquid_components, solid_phase_components, num_iter = state(system_components,firstT,uaj, ta, P=P, kdCalc= kdCalc)
        fl = 1-sum(fa.values())
        if num_iter == 3000:
            print('MAX ITERATION!')
            firstT = 2000.
    return firstT

# calculate fractional xtalization including liquid fraction, phase fractions, liquid and phase compositions in wt.%
def frac_model_trange(t_start, t_stop, major_start_comp, P=1., kdCalc = kdCalc_langmuir1992):
    tstep = 1.
    #bulk_d = {key:0. for key in trace_start_comp}
    trange = np.arange(t_stop,t_start, tstep)
    system_components = oxideToComponent(major_start_comp)  # input and output are dictionary
    major_liquid_components = system_components.copy()
    #trace_liquid_comp = trace_start_comp.copy()
    major_oxide_dict = {key:[] for key in major_start_comp}
    major_phase_oxide_dict = {phase:{key:[] for key in major_start_comp} for phase in ['ol','cpx','plg']}
    major_phase_oxides = {phase:[] for phase in ['ol','cpx','plg']}
    #trace_dict = {key:[] for key in trace_start_comp}
    fl = []
    fa_dict = {phase:[] for phase in ['plg', 'cpx', 'ol']}
    for i in range(len(trange)):
        ## Major Elements
        if i == 0:
            qa, fa, major_liquid_components, major_phase_components, num_iter = state(major_liquid_components,trange[-i-1],uaj, ta, P=P, kdCalc = kdCalc)
            for phase in fa:
                fa_dict[phase].append(fa[phase])
        else:
            major_liquid_components = oxideToComponent(major_oxides)
            qa, fa, major_liquid_components, major_phase_components, num_iter = state(major_liquid_components,trange[-i-1],uaj, ta, P=P, kdCalc = kdCalc)
            for phase in fa:
                solid_phase = fa[phase]*fl[-1]+fa_dict[phase][-1]
                fa_dict[phase].append(solid_phase)
        major_oxides = cationFracToWeight(major_liquid_components)
        for phase in major_phase_oxides:
            major_phase_oxides[phase] = cationFracToWeight(major_phase_components[phase])
            for key in major_phase_oxides[phase]:
                major_phase_oxide_dict[phase][key].append(major_phase_oxides[phase][key])
        liq = (1. - sum(fa.values()))
        if i == 0:
            fl.append(liq)
        else:
            fl.append(liq*fl[-1])
        fa_tot = sum(fa.values())
        for key in major_oxides:
            major_oxide_dict[key].append(major_oxides[key])
        ##Trace Elements
        #for elem in trace_start_comp:
            ##Calculate Bulk D
            #bulk_d[elem] = 0.
            #if fa_tot != 0.:
                #for phase in fa:
                    #bulk_d[elem] += (fa[phase]/fa_tot)*kd_dict[phase][elem]
            ##Add erupted composition to eruption dictionary
            #trace_liquid_comp[elem] = trace_liquid_comp[elem]/(liq +(1.-liq)*bulk_d[elem])
            #trace_dict[elem].append(trace_liquid_comp[elem])
    return fl, fa_dict, major_oxide_dict, major_phase_oxide_dict #, trace_dict   

# calculate equilibrium xtalization including liquid fraction, phase fractions, liquid and phase compositions in wt.%
def eq_model_trange(t_start, t_stop, major_start_comp, P = 1., kdCalc = kdCalc_langmuir1992):
    tstep = 1.
    #bulk_d = {key:0. for key in trace_start_comp}
    trange = np.arange(t_stop,t_start, tstep)
    system_components = oxideToComponent(major_start_comp)
    major_oxide_dict = {key:[] for key in major_start_comp}
    #trace_dict = {key:[] for key in trace_start_comp}
    fl = []
    fa_dict = {phase:[] for phase in ['plg', 'cpx', 'ol']}
    for i in range(len(trange)):
        ## Major Elements
        qa, fa, major_liquid_components, major_phase_components, num_iter = state(system_components,trange[-i-1],uaj, ta, P = P, kdCalc = kdCalc)
        for phase in fa:
            fa_dict[phase].append(fa[phase])
        major_oxides = cationFracToWeight(major_liquid_components)
        liq = 1.-sum(fa.values())
        fl.append(liq)
        fa_tot = sum(fa.values())
        for key in major_oxides:
            major_oxide_dict[key].append(major_oxides[key])
        # #Trace Elements
        # for elem in trace_start_comp:
        #     #Calculate Bulk D
        #     bulk_d[elem] = 0.
        #     if fa_tot != 0.:
        #         for phase in fa:
        #             bulk_d[elem] += (fa[phase]/fa_tot)*kd_dict[phase][elem]
        #     #Add erupted composition to eruption dictionary
        #     trace_dict[elem].append(trace_start_comp[elem]/(liq +(1.-liq)*bulk_d[elem]))
    return fl, fa_dict, major_oxide_dict, major_phase_oxide_dict #, trace_dict


