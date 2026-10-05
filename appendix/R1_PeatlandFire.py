
import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats
from scipy.stats import linregress, pearsonr, t

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

startyear_use       = 2002;    # for calculating and plotting
endyear_use         = 2022;


# ============ MK trend test ===========
def calculate_trend_line(series):
    x=np.array(range(1,len(series)+1))
    y = series#.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    trend_line = slope * x + intercept
    slope_per = slope/np.mean(y)
    return trend_line, p_value

# =============== BA ===================
def calculate_Regional_Annual_Mean(var_name):
    dir_file = f"/scratch/yangbuw/Data/GFED5/GFED5_1/Output/GFED5_1_BA_m2_byBiomes_025x025_2002_2022_monthly.nc"
    ds2 = xr.open_dataset(dir_file)
    C = ds2[var_name] /1E6/1E6*100 # m to km to Mha
    region_C = C.sel(longitude=slice(63, 180), latitude=slice(58, 74), time=slice(str(startyear_use), str(endyear_use)))
    region_C_yearly = region_C.resample(time='YS').sum()
    region_C_sum = region_C_yearly.sum(dim=['latitude', 'longitude'])
    return region_C_sum

# the Var_name is in order, from largest to lowest
Var_name = ['TOTL', 'SAVA', 'AGRI', 'BORF', 'DEFO', 'TEMF', 'PEAT']
BA_data_use = []
for i in range(len(Var_name)):
    current_var_name = Var_name[i]
    BA_data_use.append(calculate_Regional_Annual_Mean(current_var_name))


# =============== Emission: CO2 ===================
def calculate_Regional_Annual_Mean(var_name):
    dir_file = f"/scratch/yangbuw/Data/GFED5/GFED5_1/Output/GFED5_1_CO2_by_gC_byBiomes_025x025_2002_2022_monthly.nc"
    ds2 = xr.open_dataset(dir_file)
    C = ds2[var_name] /1E12 # g C to Tg C
    region_C = C.sel(longitude=slice(63, 180), latitude=slice(58, 74), time=slice(str(startyear_use), str(endyear_use)))
    region_C_yearly = region_C.resample(time='YS').sum()
    region_C_sum = region_C_yearly.sum(dim=['latitude', 'longitude'])
    return region_C_sum

# the Var_name is in order, from largest to lowest
Var_name = ['TOTL', 'SAVA', 'AGRI', 'BORF', 'DEFO', 'TEMF', 'PEAT']
Emission_data_use = []
for i in range(len(Var_name)):
    current_var_name = Var_name[i]
    Emission_data_use.append(calculate_Regional_Annual_Mean(current_var_name))


Emission_data_use_co2 = Emission_data_use


# =============== Emission: CO ===================
def calculate_Regional_Annual_Mean(var_name):
    dir_file = f"/scratch/yangbuw/Data/GFED5/GFED5_1/Output/GFED5_1_CO_by_gC_byBiomes_025x025_2002_2022_monthly.nc"
    ds2 = xr.open_dataset(dir_file)
    C = ds2[var_name] /1E12 # g C to Tg C
    region_C = C.sel(longitude=slice(63, 180), latitude=slice(58, 74), time=slice(str(startyear_use), str(endyear_use)))
    region_C_yearly = region_C.resample(time='YS').sum()
    region_C_sum = region_C_yearly.sum(dim=['latitude', 'longitude'])
    return region_C_sum

# the Var_name is in order, from largest to lowest
Var_name = ['TOTL', 'SAVA', 'AGRI', 'BORF', 'DEFO', 'TEMF', 'PEAT']
Emission_data_use = []
for i in range(len(Var_name)):
    current_var_name = Var_name[i]
    Emission_data_use.append(calculate_Regional_Annual_Mean(current_var_name))


Emission_data_use_co = Emission_data_use


# =============== Emission: CH4 ===================
def calculate_Regional_Annual_Mean(var_name):
    dir_file = f"/scratch/yangbuw/Data/GFED5/GFED5_1/Output/GFED5_1_CH4_by_gC_byBiomes_025x025_2002_2022_monthly.nc"
    ds2 = xr.open_dataset(dir_file)
    C = ds2[var_name] /1E12 # g C to Tg C
    region_C = C.sel(longitude=slice(63, 180), latitude=slice(58, 74), time=slice(str(startyear_use), str(endyear_use)))
    region_C_yearly = region_C.resample(time='YS').sum()
    region_C_sum = region_C_yearly.sum(dim=['latitude', 'longitude'])
    return region_C_sum

# the Var_name is in order, from largest to lowest
Var_name = ['TOTL', 'SAVA', 'AGRI', 'BORF', 'DEFO', 'TEMF', 'PEAT']
Emission_data_use = []
for i in range(len(Var_name)):
    current_var_name = Var_name[i]
    Emission_data_use.append(calculate_Regional_Annual_Mean(current_var_name))


Emission_data_use_ch4 = Emission_data_use

Emission_data_use = Emission_data_use_co2 + Emission_data_use_co + Emission_data_use_ch4

# # == Trend ==
# data_use = BA_data_use
# trendline_use = []
# p_use = []
# for i in range(0, 7):  # Loop through indices 1 to 5
#     ratio = data_use[i]
#     slp,p= calculate_trend_line(ratio)
#     trendline_use.append(slp)
#     p_use.append(p)

# BA_trendline_data_use = np.array(trendline_use)

# data_use = EI_data_use
# trendline_use = []
# p_use = []
# for i in range(0, 7):  # Loop through indices 1 to 5
#     ratio = data_use[i]
#     slp,p= calculate_trend_line(ratio)
#     trendline_use.append(slp)
#     p_use.append(p)

# EI_trendline_data_use = np.array(trendline_use)

# # == Trend ==
# data_use = Emission_data_use
# trendline_use = []
# p_use = []
# for i in range(0, 7):  # Loop through indices 1 to 5
#     ratio = data_use[i]
#     slp,p= calculate_trend_line(ratio)
#     trendline_use.append(slp)
#     p_use.append(p)

# Emission_trendline_data_use = np.array(trendline_use)
