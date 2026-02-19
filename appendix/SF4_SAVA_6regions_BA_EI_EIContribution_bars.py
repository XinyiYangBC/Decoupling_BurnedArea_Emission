import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats
from scipy.stats import linregress, pearsonr, t

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

startyear_use       = 2002;    # for calculating and plotting
endyear_use         = 2022;

Var_name = ['TOTL', 'SAVA', 'AGRI', 'BORF', 'DEFO', 'TEMF', 'PEAT']
Var_name_idx = 1

#------------------------------ Mask ---------------------------------------
data_dir1 = f"/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_basis_regions_025x025.nc"
data_ds   = xr.open_dataset(data_dir1)
# data_ds = data_ds.rename({'lat': 'latitude', 'lon': 'longitude'})
mask_data = data_ds["basis_regions"] #14 regions
lat       = data_ds["latitude"]
lon       = data_ds["longitude"]

# ============ MK trend test ===========
def calculate_trend_line(series):
    x=np.array(range(1,len(series)+1))
    y = series#.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    trend_line = slope * x + intercept
    slope_per = slope/np.mean(y)
    return slope_per, p_value, slope

# =============== BA ===================
def calculate_Regional_Annual_Mean(var_name,biome_type):
    dir_file = f"/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_BA_m2_byBiomes_025x025_2002_2022_monthly.nc"
    dir_file = f"/scratch/yangbuw/Data/GFED5/GFED5_1/Output/GFED5_1_BA_m2_byBiomes_025x025_2002_2022_monthly.nc"
    ds2 = xr.open_dataset(dir_file)
    C = ds2[var_name] /1E6/1E6*100 # m to km to Mha
    region_C = C.sel(longitude=slice(-180, 180), latitude=slice(-90, 90), time=slice(str(startyear_use), str(endyear_use)))
    region_C_yearly = region_C.resample(time='YS').sum()

    biome_mask = mask_data == biome_type
    data_in_biome = region_C_yearly.where(biome_mask, drop=True)
    data_area_sum = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)
    return data_area_sum

# the Var_name is in order, from largest to lowest
Var_name = ['TOTL', 'SAVA', 'AGRI', 'BORF', 'DEFO', 'TEMF', 'PEAT']
BA_data_use = []
for i in range(1,15):
    current_var_name = Var_name[Var_name_idx]  # fix it to SAVA
    Biome_idx = i
    BA_data_use.append(calculate_Regional_Annual_Mean(current_var_name,Biome_idx))
    
BA_data_use = np.array(BA_data_use).round(2)

# =============== Emission ===================
def calculate_Regional_Annual_Mean(var_name,biome_type):
    dir_file = f"/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_CO2_by_gC_byBiomes_025x025_2002_2022_monthly.nc"
    dir_file = f"/scratch/yangbuw/Data/GFED5/GFED5_1/Output/GFED5_1_CO2_by_gC_byBiomes_025x025_2002_2022_monthly.nc"
    ds2 = xr.open_dataset(dir_file)
    C = ds2[var_name] /1E12 # g C to Pg C
    region_C = C.sel(longitude=slice(-180, 180), latitude=slice(-90, 90), time=slice(str(startyear_use), str(endyear_use)))
    region_C_yearly = region_C.resample(time='YS').sum()

    biome_mask = mask_data == biome_type
    data_in_biome = region_C_yearly.where(biome_mask, drop=True)
    data_area_sum = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)
    return data_area_sum


# the Var_name is in order, from largest to lowest
Var_name = ['TOTL', 'SAVA', 'AGRI', 'BORF', 'DEFO', 'TEMF', 'PEAT']
Emission_data_use = []
for i in range(1,15):
    current_var_name = Var_name[Var_name_idx]  # fix it to SAVA
    Biome_idx = i
    Emission_data_use.append(calculate_Regional_Annual_Mean(current_var_name,Biome_idx))

Emission_data_use = np.array(Emission_data_use).round(2)


# =============== DM ===================
def calculate_Regional_Annual_Mean(var_name,biome_type):
    dir_file = f"/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_DM_by_gDM_byBiomes_025x025_2002_2022_monthly.nc"
    ds2 = xr.open_dataset(dir_file)
    C = ds2[var_name] /1E15 # g C to Pg C
    region_C = C.sel(longitude=slice(-180, 180), latitude=slice(-90, 90), time=slice(str(startyear_use), str(endyear_use)))
    region_C_yearly = region_C.resample(time='YS').sum()

    biome_mask = mask_data == biome_type
    data_in_biome = region_C_yearly.where(biome_mask, drop=True)
    data_area_sum = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)
    return data_area_sum


# the Var_name is in order, from largest to lowest
Var_name = ['TOTL', 'SAVA', 'AGRI', 'BORF', 'DEFO', 'TEMF', 'PEAT']
DM_data_use = []
for i in range(1,15):
    current_var_name = Var_name[Var_name_idx]  # fix it to SAVA
    Biome_idx = i
    DM_data_use.append(calculate_Regional_Annual_Mean(current_var_name,Biome_idx))

DM_data_use = np.array(DM_data_use).round(2)


# =========== Emission Intensity ==============
EI_data_use = Emission_data_use/BA_data_use   #Tg C/ Mha

# Calculate EI trend across 14 regions
data_use = EI_data_use
trendline_use = []
p_use = []
slp_use = []
for i in range(0, 14):  # Loop through indices 1 to 5
    ratio = data_use[i]
    slp,p,slp2= calculate_trend_line(ratio)
    trendline_use.append(slp)
    p_use.append(p)
    slp_use.append(slp2)

EI_trendline_data_use = np.array(trendline_use)
EI_p_use = np.array(p_use)
EI_slp_use = np.array(slp_use)



Emission_biome = Emission_data_use.sum()/(2022-2002+1)  
#  Annual mean
Emission_data_use_ave = []
Emission_data_use_ratio = []
for i in range(0,14):
    y = Emission_data_use[i]
    Emission_data_use_ave.append(y.mean())
    Emission_data_use_ratio.append(y.mean()/Emission_biome)

Emission_data_use_ave = np.array(Emission_data_use_ave).round(4)
Emission_data_use_ratio = np.array(Emission_data_use_ratio).round(4)

BA_biome = BA_data_use.sum()/(2022-2002+1)
#  Annual mean
BA_data_use_ave = []
BA_data_use_ratio = []
for i in range(0,14):
    y = BA_data_use[i]
    BA_data_use_ave.append(y.mean())
    BA_data_use_ratio.append(y.mean()/BA_biome)
BA_data_use_ave = np.array(BA_data_use_ave).round(4)
BA_data_use_ratio = np.array(BA_data_use_ratio).round(4)


# User-defined
Region_idx_all =[ 4,7,8, 11,13]
Region_idx = Region_idx_all[0]  
Region_idx

# The rest regions as Others
other_Regions_idx_all =[ 0,1,2,3,5,6,9,10,12]
BA_other_Regions = sum(BA_data_use[idx] for idx in other_Regions_idx_all)
Emission_other_Regions = sum(Emission_data_use[idx] for idx in other_Regions_idx_all)
DM_other_Regions = sum(DM_data_use[idx] for idx in other_Regions_idx_all)
EI_other_Regions = Emission_other_Regions/BA_other_Regions

# trends
# BA
data_tran = BA_other_Regions
trendline_tran,p_tran,slp_tran= calculate_trend_line(data_tran)
BA_slp_other_Regions = slp_tran
BA_p_other_Regions = p_tran


# Emission
data_tran = Emission_other_Regions
trendline_tran,p_tran,slp_tran= calculate_trend_line(data_tran)
Emission_slp_other_Regions = slp_tran
Emission_p_other_Regions = p_tran

# EI
data_tran = EI_other_Regions
trendline_tran,p_tran,slp_tran= calculate_trend_line(data_tran)
EI_slp_other_Regions = slp_tran
EI_p_other_Regions = p_tran


# DM
data_tran = DM_other_Regions
trendline_tran,p_tran,slp_tran= calculate_trend_line(data_tran)
DM_slp_other_Regions = slp_tran
DM_p_other_Regions = p_tran



BA_data_use_plot = []
EI_data_use_plot = []
CO_data_use_plot = []
for i in range(0,5):
    Region_idx = Region_idx_all[i]
    BA_trans1 = BA_data_use_ave[Region_idx]
    EI_trans1 = EI_slp_use[Region_idx] 
    BA_data_use_plot.append(BA_trans1)
    EI_data_use_plot.append(EI_trans1)
    CO_data_use_plot.append(BA_trans1*EI_trans1)
BA_data_use_plot = np.array(BA_data_use_plot).round(4)   # annual average
EI_data_use_plot = np.array(EI_data_use_plot).round(4)
CO_data_use_plot = np.array(CO_data_use_plot).round(4)

# EI contribution in sum 
np.sum(CO_data_use_plot) 
CO_data_use_plot_other_regions=np.mean(BA_other_Regions)*EI_slp_other_Regions
CO_data_use_plot_total = CO_data_use_plot_other_regions + np.sum(CO_data_use_plot)   # 16.196503566058805


# Figure2C
ratio = CO_data_use_plot/CO_data_use_plot_total

#print(np.sum(CO_data_use_plot))                               # 14.165603896085084
#print(EI_slp_other_Regions * BA_other_Regions.mean())         # 2.03089966997372

# In summary, seperate into 6 region, which give sava EI-driven Contribution around 16.196503566058805, 
# which is very close to 16.43164045 calculated by whole globa from Figure 2 script. 
#                                                                 updated by Xinyi 05/16/2025

# Other regions (6th region)
SAVA_EI_Contribution = 16.43164045                                      # 5/16/2025 updated

BA_data_use_plot_total = np.append(BA_data_use_plot, BA_other_Regions.mean())
EI_data_use_plot_total = np.append(EI_data_use_plot, EI_slp_other_Regions)
CO_data_use_plot_total = np.append(CO_data_use_plot, CO_data_use_plot_other_regions)

ratios = CO_data_use_plot_total/np.sum(CO_data_use_plot_total)
ratios

# # ==================== old for backup ================
# EI_Contribution=BA_data_use_ave*EI_slp_use                              
# EI_CO_scaler = SAVA_EI_Contribution/CO_data_use_plot_total           # 1.2472885127929494
# #EI_CO_scaler = 1.187217601284326
# EI_Contribution_other = np.sum(EI_Contribution) - np.sum(CO_data_use_plot)
# BA_other = np.sum(BA_data_use_ave) - np.sum(BA_data_use_plot)           # 55.378662 Mha
# EI_trend_other = EI_Contribution_other/BA_other                         # 0.01233745512536817


# # Levep up EI trends to get consistent with EI Contribution calculated from WHOLE SAVA biome
# EI_data_use_plot      = EI_data_use_plot * EI_CO_scaler
# CO_data_use_plot      = CO_data_use_plot * EI_CO_scaler
# EI_trend_other        = EI_trend_other * EI_CO_scaler
# EI_Contribution_other = EI_Contribution_other * EI_CO_scaler

# BA_data_use_plot_total = np.append(BA_data_use_plot, BA_other)
# EI_data_use_plot_total = np.append(EI_data_use_plot, EI_trend_other)
# CO_data_use_plot_total = np.append(CO_data_use_plot, EI_Contribution_other)

# ratios = CO_data_use_plot_total/np.sum(CO_data_use_plot_total)
# ratios
# # ==================== old for backup ================


#---------------------- Plot -----------------

# Biome names
Name_of_biome = ["SHSA", "NHAF", "SHAF", "SEAS", "AUST","OTHERS"]

# Data for plotting
biomes = Name_of_biome
gfed5_ave_BA = BA_data_use_plot_total  
gfed5_ave_Emission = EI_data_use_plot_total *100
gfed5_ave_EI = CO_data_use_plot_total 

# X-axis positions
x = np.arange(len(biomes))

# Bar width
width = 0.25  

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(14, 6))

# Colors for bars
BA_color = 'silver'
Emission_color = 'orange'
EI_color = 'royalblue'

BA_color = 'gold'
Emission_color = 'orange'
EI_color = 'mediumseagreen'

# Plot Burned Area on left y-axis
bars1 = ax1.bar(x - width, gfed5_ave_BA, width, label='Burned Area', color=BA_color)

# Create second y-axis for Fire Emission
ax2 = ax1.twinx()
bars2 = ax2.bar(x, gfed5_ave_Emission, width, label='Emission Intensity Trend', color=Emission_color)

# Create third y-axis for Emission Intensity (EI) & shift it slightly right
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))  # Shift the third y-axis outward
bars3 = ax3.bar(x + width, gfed5_ave_EI, width, label='EI Contribution', color=EI_color)

# Formatting left y-axis (Burned Area)
#ax1.set_ylabel(r'Burned Area (Mha yr$^{-1}$)', fontsize=16, color=BA_color, fontweight='bold')
ax1.set_ylabel(r'Burned Area (Mha)', fontsize=16, color=BA_color, fontweight='bold')
ax1.set_ylim([0, 300.0])
yticks = np.arange(0, 301, 50)  # Define custom y-tick positions
ax1.set_yticks(yticks)
custom_yticklabels = [f'{float(y):.0f}' for y in yticks]
ax1.set_yticklabels(custom_yticklabels, fontsize=16, color='gray')
ax1.tick_params(axis='y', labelcolor='black')
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

# Formatting second right y-axis (Fire Emission)
ax2.set_ylabel(r'CO2 EI Trend (Tg C Mha$^{-1}$ yr$^{-1}$)', fontsize=16, color=Emission_color, fontweight='bold')
ax2.set_ylim([0, 6.0])
yticks = np.arange(0, 6.1, 1)  # Define custom y-tick positions
ax2.set_yticks(yticks)
custom_yticklabels = [f'{float(y):.0f}%' for y in yticks]
ax2.set_yticklabels(custom_yticklabels, fontsize=16, color='gray')
ax2.tick_params(axis='y', labelcolor=Emission_color)
ax2.yaxis.set_minor_locator(AutoMinorLocator(2))


# Formatting third right y-axis (Emission Intensity)
ax3.set_ylabel(r'CO2 Emission Trend (Tg C $\mathrm{yr}^{-1}$)', fontsize=16, color=EI_color, fontweight='bold')
ax3.set_ylim([0, 8.0])
yticks = np.arange(0, 8.1, 1)  # Define custom y-tick positions
ax3.set_yticks(yticks)
custom_yticklabels = [f'{float(y):.0f}' for y in yticks]
ax3.set_yticklabels(custom_yticklabels, fontsize=16, color='gray')
ax3.tick_params(axis='y', labelcolor=EI_color)
ax3.yaxis.set_minor_locator(AutoMinorLocator(2))

# X-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(biomes, rotation=0, ha='center', fontsize=16, color='black')

# Adding values on top of bars
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.0f}', 
             ha='center', va='bottom', fontsize=12, color=BA_color, rotation=0)

for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.1f}%', 
             ha='center', va='bottom', fontsize=12, color=Emission_color, rotation=0)

for bar in bars3:
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.1f}', 
             ha='center', va='bottom', fontsize=12, color=EI_color, rotation=0)

# Title
# ax1.set_title('Annual Mean Burned Area, Fire Emission, and Emission Intensity from Savanna Burning', 
#               fontsize=15, fontweight='bold')

# Add gridlines
ax1.grid(True, which='major', axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.2)
ax2.grid(False)
ax3.grid(False)  # Disable duplicate gridlines

# Adding legend
fig.legend( 
           loc='upper left', bbox_to_anchor=(0.2, 0.89), 
           fontsize=16, frameon=False, fancybox=True, ncol=3)

# Save figure
plt.savefig(f'/home/yangbuw/Program/EmissionIntensity/pics/SFigure4_BA_EI_EIContribution_6regions_bar.pdf', dpi=300)
plt.show()

