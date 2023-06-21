# melt_xtal
Please read the README.md and LICENSE before downloading and using the codes.<br> 
Users must cite the original version in this format:<br> 
Codes and demo data are listed in folder 'mantle melting_crystallization2023'. Please download all the files.<br>
Codes here are used to calculated certain element concentrations in the liquid and olivine during mantle melting and crystallization with given mantle compositions and melting-crystallization conditions.<br>
This is a part of Supplementary Materials for paper "The origin of Ni and Mn variations in Hawaiian and MORB olivines and associated basalts" written by Mingzhen Yu (myu@g.harvard.edu) and Charles H. Langmuir (langmuir@eps.harvard.edu) being submitted to Journal (status will be updated). Correspondence to Mingzhen Yu (myu@g.harvard.edu, Department of Earth and Planetary Sciences, Harvard University, Cambridge, MA 02138, USA)<br>
Codes are written with Python.<br>

# Files Introduction
In the folder 'mantle melting_crystallization2023', there seven '.py' files and one '.csv' file.<br>
## data file
The '.csv' file named 'olivine_glass_data.csv' provides users with natural data for Hawaiian olivines and MORB olivines, Hawaiian basalts, and MORB glasses, which can be used to compared to the modeled crystallization results. After running code 'melting_cystallization2023.py', six figures will be plotted automatically.<br>
Olivine data are given by Sobolev, A. V. et al. The amount of recycled crust in sources of mantle-derived melts. science 316, 412-417 (2007). MORB glasses data are given by Jenner, F.E. and O'Neill, H.S.C., 2012. Analysis of 60 elements in 616 ocean floor basaltic glasses. Geochemistry, Geophysics, Geosystems, 13(2); Yang, S., Humayun, M. and Salters, V.J., 2018. Elemental systematics in MORB glasses from the Mid‐Atlantic Ridge. Geochemistry, Geophysics, Geosystems, 19(11), pp.4236-4259; Yang, A.Y., Langmuir, C.H., Cai, Y., Michael, P., Goldstein, S.L. and Chen, Z., 2021. A subduction influence on ocean ridge basalts outside the Pacific subduction shield. Nature communications, 12(1), p.4757. Hawaiian basalts data are compiled from Georoc (references listed in the .csv file).
## code files
### melting_function2023.py
This code defines functions used in the calculation of melt compositions for two types of mantle melting: polybaric fractional melting and isobaric equilibrium melting. Melt compositions calculated include SiO2, MgO, FeO, MnO, NiO, TiO2, Na2O and K2O.<br> 
Fundamental algorithms are given by Langmuir, C. H., Klein, E. M. & Plank, T. Petrological systematics of mid‐ocean ridge basalts: Constraints on melt generation beneath ocean ridges. Mantle flow and melt generation at mid‐ocean ridges 71, 183-280 (1992). Melting reactions and partition coefficients are commented in the code and explained in the paper "The origin of Ni and Mn variations in Hawaiian and MORB olivines and associated basalts" written by Mingzhen Yu (myu@g.harvard.edu) and Charles H. Langmuir (langmuir@eps.harvard.edu) being submitted to Journal (status will be updated).
This code will be called by 'melting_crystallization2023.py'.
### olonly_function2023.py
This code defines functions used in the calculation of melt and olivine compositions for twy types of olivine-only crystallization: fractional crystallization and equilibrium crystallization. Compositions calculated include MgO, FeO, SiO2, MnO and NiO.<br>
Fundamental algorithm is the olivine stoichiometry: MgO+FeO=66.67. Partition coefficients are commented in the code and explained in the paper "The origin of Ni and Mn variations in Hawaiian and MORB olivines and associated basalts" written by Mingzhen Yu (myu@g.harvard.edu) and Charles H. Langmuir (langmuir@eps.harvard.edu) being submitted to Journal (status will be updated).
This code will be called by 'melting_crystallization2023.py'.
### wl1990stoich_2023.py, wl1990kdcalc_2023.py, wl1990state_2023.py, wl1990models_2023.py
These codes define functions used in the calculation of melt and mineral (olivine, plagioclase, clinopyroxene) compositions for two types of crystallization: fractional crystallizationa nd equilibrium crystallization. Compositions calculated include SiO2, TiO2, Al2O3, FeO, MgO, K2O, MnO, Na2O, P2O5, CaO, NiO.<br>
Fundamental algorithms are given by Weaver, J.S. and Langmuir, C.H., 1990. Calculation of phase equilibrium in mineral-melt systems. Computers & Geosciences, 16(1), pp.1-19. The purpose is commented at the beginning of each code.
These codes will be called by 'melting_crystallization2023.py'.
### melting_crystallization2023.py







 
