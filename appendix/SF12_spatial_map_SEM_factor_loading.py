import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats
from scipy.stats import linregress, pearsonr, t
from datetime import datetime
import os

# plotting package
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors
from cmap import Colormap

# ---------------------  Making Burned Fraction mask -------------------------------
startyear_use       = 2002;   
endyear_use         = 2022;
# =============== BF ===================
def calculate_Regional_Annual_Mean(var_name):
    dir_file = f"/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_BF_byBiomes_025x025_2002_2022_monthly.nc"
    ds2 = xr.open_dataset(dir_file)
    C = ds2[var_name]
    region_C = C.sel(longitude=slice(-180, 180), latitude=slice(-90, 90), time=slice(str(startyear_use), str(endyear_use)))
    region_C_yearly = region_C.resample(time='YS').sum()
    region_C_sum = region_C_yearly.sum(dim=['latitude', 'longitude'])
    return region_C_yearly

# the Var_name is in order, from largest to lowest
Var_name = ['TOTL', 'SAVA', 'AGRI', 'BORF', 'DEFO', 'TEMF', 'PEAT']
BA_data_use = []
for i in range(len(Var_name)):
    current_var_name = Var_name[i]
    BA_data_use.append(calculate_Regional_Annual_Mean(current_var_name))


data_use = BA_data_use[1]
data_area_sum = data_use.mean(dim=['time'], skipna=True)
BF_mask=data_area_sum
# ---------------------  Making Burned Fraction mask -------------------------------


dir_file = f"/home/yangbuw/Program/EmissionIntensity/code_version1/data_code/data/SEM_global_coeff_025x025_rmAC_2Path_VegIndex_Lag_3_month.nc"
ds2 = xr.open_dataset(dir_file)
C1 = ds2["FW_coeff1"]     # Surface Temperature
C2 = ds2["FW_coeff2"]     # VPD
C3 = ds2["FW_coeff3"]     # SM
C4 = ds2["FW_coeff4"]     # Precipitation
C5 = ds2["FW_coeff5"]     # Wind Speed 

P2 = ds2["FW_pvalue2"]     # VPD
P3 = ds2["FW_pvalue3"]     # SM
P4 = ds2["FW_pvalue4"]     # Precipitation
P5 = ds2["FW_pvalue5"]     # Wind Speed



# Mask 
C2 = C2.where(P2 <= 0.1, np.nan)
C3 = C3.where(P3 <= 0.1, np.nan)
C4 = C4.where(P4 <= 0.1, np.nan)
C5 = C5.where(P5 <= 0.1, np.nan)

# Mask 
C1 = C1.where(BF_mask >= 0.01, np.nan)
C2 = C2.where(BF_mask >= 0.01, np.nan)
C3 = C3.where(BF_mask >= 0.01, np.nan)
C4 = C4.where(BF_mask >= 0.01, np.nan)
C5 = C5.where(BF_mask >= 0.01, np.nan)


# ======================================  Plot ======================================
datasets = [C1,C2,C3,C4,C5]

# Map extents and latitude tick locations
lon_min, lon_max = -180, 180 
lat_min, lat_max = -70, 90    
panel_labels = ['a', 'b', 'c', 'd', 'e']  


titles = ['Surface Temperature', 'VPD','SM', 'Precipitation','Wind Speed']

# Create the figure and subplots
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'
fig, axes = plt.subplots(3, 2, figsize=(11, 9.5), subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(wspace=0.15)  # Decrease to bring columns closer together (default is ~0.2)
plt.subplots_adjust(hspace=0.39)  # vertical space between rows
axes_flat = axes.flat

for i in range(5):  # Only iterate over 5 datasets
    ax = axes_flat[i]

    if i ==0:
        bounds2 = np.arange(-2.0, 2.1, 0.2)
        cm = Colormap('matplotlib:seismic')  # case insensitive
        mpl_cmap = cm.to_mpl()
        
        white = np.array([[1, 1, 1, 1]])
        colors2 = mpl_cmap(np.linspace(0.1, 0.9, len(bounds2) - 1))  # Use the "bwr" colormap
        # colors2 = np.vstack((white, colors2))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
        mesh = ax.pcolormesh(
            datasets[i]['longitude'], datasets[i]['latitude'], datasets[i],
            transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto'
        )
    else:
        bounds2 = np.arange(-2.0, 2.1, 0.2)
        cm = Colormap('matplotlib:seismic')  # case insensitive
        mpl_cmap = cm.to_mpl()
        
        white = np.array([[1, 1, 1, 1]])
        colors2 = mpl_cmap(np.linspace(0.1, 0.9, len(bounds2) - 1))  # Use the "bwr" colormap
        # colors2 = np.vstack((white, colors2))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
        
        mesh = ax.pcolormesh(
            datasets[i]['longitude'], datasets[i]['latitude'], datasets[i],
            transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto'
        )
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.OCEAN, color='white')

    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--')#, linewidth=0.3)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9, 'color': 'gray'}
    gl.ylabel_style = {'size': 9, 'color': 'gray'}


    ax.set_title(titles[i], fontsize=9, fontweight='bold')

    # Colorbar
    cb_ax = inset_axes(ax, width="100%", height="8%", loc='lower center',
                       bbox_to_anchor=(0, -0.21, 1.0, 1),
                       bbox_transform=ax.transAxes, borderpad=0)
    
    cbar = plt.colorbar(mesh, cax=cb_ax, orientation='horizontal', ticks=bounds2)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.set_xticks(bounds2[::2])  # for horizontal colorbar


    ax.text(-0.1, 1.1, panel_labels[i], transform=ax.transAxes,fontsize=14, fontweight='bold', va='top', ha='left')

# Remove the unused subplot
fig.delaxes(axes[2, 1])


# Save
plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/SFigure12_Spatial_Map_FW_Factors_Loading.pdf',dpi=300, bbox_inches='tight')
plt.show()