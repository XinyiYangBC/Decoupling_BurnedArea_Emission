
import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats
from scipy.stats import linregress, pearsonr, t

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

startyear_use       = 2002;    # for calculating and plotting
endyear_use         = 2022;

# Approximate Amazon/Brazil box
Lon_W = -90
Lon_E = -35
Lat_S = -30
Lat_N = 0

Lon_W = -180
Lon_E = 180
Lat_S = -30
Lat_N = 30

# Approximate core Arc of Deforestation
Lon_W = -65
Lon_E = -45
Lat_S = -18
Lat_N = -5
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
    region_C = C.sel(longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N), time=slice(str(startyear_use), str(endyear_use)))
    region_C_yearly = region_C.resample(time='YS').sum()
    region_C_sum = region_C_yearly.sum(dim=['latitude', 'longitude'])
    return region_C_sum

# the Var_name is in order, from largest to lowest
Var_name = ['TOTL', 'SAVA', 'AGRI', 'BORF', 'DEFO', 'TEMF', 'PEAT']
BA_data_use = []
for i in range(len(Var_name)):
    current_var_name = Var_name[i]
    BA_data_use.append(calculate_Regional_Annual_Mean(current_var_name))


# =============== Emission ===================
def calculate_Regional_Annual_Mean(var_name):
    dir_file = f"/scratch/yangbuw/Data/GFED5/GFED5_1/Output/GFED5_1_CO2_by_gC_byBiomes_025x025_2002_2022_monthly.nc"
    ds2 = xr.open_dataset(dir_file)
    C = ds2[var_name] /1E15 # g C to Pg C
    region_C = C.sel(longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N), time=slice(str(startyear_use), str(endyear_use)))
    region_C_yearly = region_C.resample(time='YS').sum()
    region_C_sum = region_C_yearly.sum(dim=['latitude', 'longitude'])
    return region_C_sum

# the Var_name is in order, from largest to lowest
Var_name = ['TOTL', 'SAVA', 'AGRI', 'BORF', 'DEFO', 'TEMF', 'PEAT']
Emission_data_use = []
for i in range(len(Var_name)):
    current_var_name = Var_name[i]
    Emission_data_use.append(calculate_Regional_Annual_Mean(current_var_name))


Emission_data_use = np.array(Emission_data_use)
BA_data_use = np.array(BA_data_use)

EI_data_use = Emission_data_use*1e3/BA_data_use  #Tg C/ Mha


# == Trend ==
data_use = BA_data_use
trendline_use = []
p_use = []
for i in range(0, 7):  # Loop through indices 1 to 5
    ratio = data_use[i]
    slp,p= calculate_trend_line(ratio)
    trendline_use.append(slp)
    p_use.append(p)

BA_trendline_data_use = np.array(trendline_use)

data_use = EI_data_use
trendline_use = []
p_use = []
for i in range(0, 7):  # Loop through indices 1 to 5
    ratio = data_use[i]
    slp,p= calculate_trend_line(ratio)
    trendline_use.append(slp)
    p_use.append(p)

EI_trendline_data_use = np.array(trendline_use)

# == Trend ==
data_use = Emission_data_use
trendline_use = []
p_use = []
for i in range(0, 7):  # Loop through indices 1 to 5
    ratio = data_use[i]
    slp,p= calculate_trend_line(ratio)
    trendline_use.append(slp)
    p_use.append(p)

Emission_trendline_data_use = np.array(trendline_use)




# ========================= Plotting EI only =========================

plot_time_period = pd.date_range(
    start=str(startyear_use),
    end=str(endyear_use),
    freq='YS'
)

# DEFO
biome_index = 0

EI_data_use_plot = EI_data_use[biome_index]
EI_trendline_data_use_plot = EI_trendline_data_use[biome_index]


# ========================= Figure setting =========================

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'

fig, ax = plt.subplots(figsize=(6, 5))

fontsize = 28
line_width = 4
line_width2 = 3

color_use = '#D67A13'


# ========================= EI =========================

ax.plot(
    plot_time_period,
    EI_data_use_plot,
    marker='o',
    markersize=10,
    markerfacecolor=color_use,
    markeredgewidth=2,
    linestyle='-',
    linewidth=line_width,
    color=color_use
)


# ========================= Y axis =========================

ax.set_ylabel(
    r'EI (Tg C/Mha)',
    fontsize=16,
    color='black'
)

ax.tick_params(
    axis='y',
    which='major',
    length=10,
    width=2,
    labelsize=16,
    color='black',
    labelcolor='black'
)


ax.spines['left'].set_linewidth(3)

ax.spines['bottom'].set_linewidth(3)


# ========================= X axis =========================

jump_num = 5
xticks = plot_time_period[::jump_num]

ax.set_xticks(xticks)
ax.set_xticklabels(
    [d.year for d in xticks],
    rotation=0,
    fontsize=fontsize,   # 28
    color='black'
)

ax.tick_params(
    axis='x',
    which='major',
    length=10,
    width=2,
    labelsize=16,  # 加这一句
    color='black',
    labelcolor='black'
)

# ========================= Spines =========================

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_facecolor('white')

ax.set_ylim([0, 20])

yticks = np.arange(0, 20.1, 5)
ax.set_yticks(yticks)

ax.set_yticklabels(
    [f'{y:.0f}' for y in yticks],
    fontsize=16,
    color='black'
)

# ========================= Title =========================

ax.set_title(
    'Brazilian Amazon',
    fontsize=20
)


plt.tight_layout()

# plt.savefig(
#     '/home/yangbuw/Program/EmissionIntensity/pics/DEFO_EI_Amazon.tif',
#     dpi=300,
#     bbox_inches='tight'
# )

plt.show()
