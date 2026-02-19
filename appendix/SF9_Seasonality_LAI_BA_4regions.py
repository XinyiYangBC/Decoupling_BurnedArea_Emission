import xarray as xr
import numpy as np
from scipy import stats
import pandas as pd
# plotting package
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

#------------------------------ Mask ---------------------------------------
data_dir1 = f"/scratch/yangbuw/Data/GFED5/BasisRegions/GFED5_BasisRegions_LandOnly_South2North_025x025.nc"
data_ds   = xr.open_dataset(data_dir1)
mask_data = data_ds["basisregions"] #14 regions
lat       = data_ds["latitude"]
lon       = data_ds["longitude"]

# Pre-setting
Lat_S = -90
Lat_N = 90
Lon_W = -180
Lon_E = 180
Time_start = "2002-01" #"1997-01"
Time_end   = "2022-12" #"2022-12"

#------------------------------ GFED5 ---------------------------------------
data_dir1 = f"/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_BA_m2_byBiomes_025x025_2002_2022_monthly.nc"
data_ds   = xr.open_dataset(data_dir1)
data      = data_ds['SAVA']/1E6/1E6*100 # m to km to Mha
lat       = data_ds["latitude"]
lon       = data_ds["longitude"]
time      = data_ds["time"]

data_selected = data.sel(longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N), time=slice(Time_start, Time_end))
monthly_totals = data_selected.groupby('time.month').mean(dim='time',skipna=True)

Name_of_biome = {
    "BONA": 1, "TENA": 2, "CEAM": 3, "NHSA": 4, "SHSA": 5, "EURO": 6,
    "MIDE": 7, "NHAF": 8, "SHAF": 9, "BOAS": 10, "CEAS": 11, "SEAS": 12,
    "EQAS": 13, "AUST": 14
}


burned_area_by_biome_absolute = {}
burned_area_by_biome = {}

for biome_zone in range(1, 15):
    biome_mask = mask_data == biome_zone

    data_in_biome = monthly_totals.where(biome_mask, drop=True)
    data_area_sum = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)
    
    burned_area_by_biome_absolute[biome_zone]      = data_area_sum
    burned_area_by_biome[biome_zone]               = burned_area_by_biome_absolute[biome_zone]/ (burned_area_by_biome_absolute[biome_zone].sum())


burned_area_values_by_biome = np.array(list(burned_area_by_biome.values()))
burned_area_by_biome_absolute = np.array(list(burned_area_by_biome_absolute.values()))
global_burned_area = np.sum(burned_area_by_biome_absolute,axis=0)

gfed5_global_burned_area      = global_burned_area/global_burned_area.sum()
gfed5_burned_area_by_biome    = burned_area_by_biome



#------------------------------ LAI ---------------------------------------
data_dir1 = f"/scratch/yangbuw/Data/LAI/MOD15A2H/MOD15A2H_LAI_025x025_2001_2024_monthly.nc"
data_dir1 = f"/scratch/yangbuw/Data/LAI/MYD15A2H/MYD15A2H_LAI_025x025_2003_2024_monthly.nc"
data_dir1 = f"/scratch/yangbuw/Data/LAI/GIMMS_LAI4g/GIMMS_LAI4g_V1.2_1982_2020_monthly_025x025.nc"
# data_dir1 = f"/scratch/yangbuw/Data/EVI/MOD13A2_EVI_025x025_2001_2024_monthly_025x025.nc"

data_ds   = xr.open_dataset(data_dir1)
data      = data_ds['LAI']
lat       = data_ds["latitude"]
lon       = data_ds["longitude"]
time      = data_ds["time"]

data_selected = data.sel(longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N), time=slice(Time_start, Time_end))
monthly_totals = data_selected.groupby('time.month').mean(dim='time',skipna=True)

Name_of_biome = {
    "BONA": 1, "TENA": 2, "CEAM": 3, "NHSA": 4, "SHSA": 5, "EURO": 6,
    "MIDE": 7, "NHAF": 8, "SHAF": 9, "BOAS": 10, "CEAS": 11, "SEAS": 12,
    "EQAS": 13, "AUST": 14
}


burned_area_by_biome_absolute = {}
burned_area_by_biome = {}

for biome_zone in range(1, 15):
    biome_mask = mask_data == biome_zone

    data_in_biome = monthly_totals.where(biome_mask, drop=True)
    data_area_sum = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)
    
    burned_area_by_biome_absolute[biome_zone]      = data_area_sum
    burned_area_by_biome[biome_zone]               = burned_area_by_biome_absolute[biome_zone]/ (burned_area_by_biome_absolute[biome_zone].sum())


burned_area_values_by_biome = np.array(list(burned_area_by_biome.values()))
burned_area_by_biome_absolute = np.array(list(burned_area_by_biome_absolute.values()))
global_burned_area = np.sum(burned_area_by_biome_absolute,axis=0)

LAI_global_burned_area      = global_burned_area/global_burned_area.sum()
LAI_burned_area_by_biome    = burned_area_by_biome


gfed5_burned_area_by_biome_use = [ LAI_burned_area_by_biome[8], LAI_burned_area_by_biome[9],LAI_burned_area_by_biome[5],LAI_burned_area_by_biome[14],gfed5_burned_area_by_biome[8], gfed5_burned_area_by_biome[9],gfed5_burned_area_by_biome[5], gfed5_burned_area_by_biome[14]]


# ======================================= Plotting ==============================================
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'

fig, axs = plt.subplots(2, 4, figsize=(11, 5))
plt.subplots_adjust(wspace=0.24, hspace=0.3)
axs = axs.ravel()

markeredgewidth = 1
linewidth = 3
fontsize = 9
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
titles = ['NHAF', 'SHAF', 'SHSA', 'AUST', 'NHAF', 'SHAF', 'SHSA', 'AUST']

for i in range(8):
    if ((i==0)|(i==1)|(i==2)|(i==3)):
        axs[i].plot(
            range(1, 13), 
            gfed5_burned_area_by_biome_use[i], 
            marker='o', 
            markersize=7, 
            markerfacecolor='green', 
            markeredgewidth=markeredgewidth, 
            linestyle='-', 
            linewidth=linewidth, 
            color='green'
        )
        peak_month = np.argmax(gfed5_burned_area_by_biome_use[i].values) + 1
        # axs[i].axvline(x=peak_month, linestyle='--', color='green', linewidth=2, alpha=0.5)
        axs[i].set_ylim([0.04, 0.12])
        axs[i].set_yticks(np.arange(0.04, 0.121, 0.02))
        axs[i].set_yticklabels([f'{y:.2f}' for y in np.arange(0.04, 0.121, 0.02)], fontsize=fontsize, color='gray')
    else:
        axs[i].plot(
            range(1, 13), 
            gfed5_burned_area_by_biome_use[i], 
            marker='o', 
            markersize=7, 
            markerfacecolor='orange', 
            markeredgewidth=markeredgewidth, 
            linestyle='-', 
            linewidth=linewidth, 
            color='orange'
        )        
        peak_month = np.argmax(gfed5_burned_area_by_biome_use[i].values) + 1
        # axs[i].axvline(x=peak_month, linestyle='--', color='orange', linewidth=2, alpha=0.5)
        axs[i].set_ylim([-0.02, 0.3])
        axs[i].set_yticks(np.arange(0.00, 0.31, 0.1))
        axs[i].set_yticklabels([f'{y:.2f}' for y in np.arange(0.00, 0.31, 0.1)], fontsize=fontsize, color='gray')
        
    axs[i].set_title(titles[i], fontsize=fontsize, color="black")
    if (i==0):
        axs[i].set_ylabel('Seasonality of LAI', fontsize=fontsize, color='black')
    if (i==4):
        axs[i].set_ylabel('Seasonality of BA', fontsize=fontsize, color='black')
        
    axs[i].grid(True, linestyle='--', color='gray', linewidth=0.5, alpha=0.2)
    axs[i].text(-0.1, 1.15, labels[i], transform=axs[i].transAxes, fontsize=fontsize + 3, fontweight='bold', va='top', ha='left')

    # X-axis labels for even-numbered months
    months = ['Feb', 'Apr', 'Jun', 'Aug', 'Oct', 'Dec']
    axs[i].set_xticks(np.arange(2, 13, 2))  # major ticks at months 2, 4, ..., 12
    axs[i].set_xticklabels(months, rotation=0, fontsize=fontsize, color='gray')
    
    # Add minor ticks at midpoints between major ticks (e.g., 1, 3, 5,...,11)
    midpoints = np.arange(1, 12, 2)
    axs[i].set_xticks(midpoints, minor=True)
    axs[i].tick_params(axis='x', which='minor', length=2, color='black')


for j in range(8, len(axs)):
    fig.delaxes(axs[j])

# Save and show
plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/SFigure9_GFED5_LAI_BA_4Regions_seasonality_fraction.tif', dpi=300, bbox_inches='tight')
plt.show()
