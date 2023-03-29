###Jocelyn Fuentes 2016 - Based on WL1989
# Mingzhen Yu 2021 - add Ni and Mn

from wl1989stoich_2023 import *
from wl1989kdcalc_2023 import *
from wl1989models_2023 import *
from wlState_2023 import *
import numpy as np
import math
import pandas as pd

#LLDinput = pd.read_csv('/Users/apple/Desktop/NiMnpaper/paper_draft/LLDinput.csv')
LLDinput = pd.read_csv('/Users/apple/Desktop/NiMnpaper/model parameters and results/LLDinput0517.csv')
row = len(LLDinput)
for k in range(55,57):
    magma = LLDinput.loc[k,['SiO2','TiO2','Al2O3','FeO','MgO','K2O','MnO','Na2O','P2O5','CaO','NiO']]
    system_components = magma.to_dict()
    T_system_components = oxideToComponent(system_components)
    t_start = get_first_T(T_system_components, P = 1., kdCalc = kdCalc_langmuir1992)
    t_stop = t_start -250
    fl,fa_dict,major_oxide_dict,major_phase_oxide_dict = frac_model_trange(t_start, t_stop,system_components,P=1.,kdCalc = kdCalc_langmuir1992) 

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
    LLD_df['liq_FeOt'] = LLD_df['liq_FeO']/0.9
    LLD_df['Fo'] = 100/(1+LLD_df['olFeO']/LLD_df['olMgO']*40.3/71.84)
    LLD_df['olNippm'] = LLD_df['olNiO']*58.6934/74.69*10**4
    LLD_df['olMnppm'] = LLD_df['olMnO']*54.938/70.94*10**4
    LLD_df['liq_Nippm'] = LLD_df['liq_NiO']*58.6934/74.69*10**4
    LLD_df['liq_FeOtMnO'] = LLD_df['liq_FeOt']/LLD_df['liq_MnO']
    #LLD_df.to_csv('/Users/apple/Desktop/KdNi/paper_draft/LLDpy_iso_PFK/{}.csv'.format(LLDinput.loc[k,['PFK_condition']][0]),sep=',',index=True,header=True)
    LLD_df.to_csv('/Users/apple/Desktop/NiMnpaper/model parameters and results/model_variation_LLDoutput0517/{}.csv'.format(LLDinput.loc[k,['PFK_condition']][0]),sep=',',index=True,header=True)
