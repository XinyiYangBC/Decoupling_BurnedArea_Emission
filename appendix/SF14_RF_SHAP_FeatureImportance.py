import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats
from scipy.stats import linregress, pearsonr, t
from datetime import datetime
import os

# plotting package
from cmap import Colormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors

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


dir_file = f"/home/yangbuw/Program/EmissionIntensity/code_version1/data_code/data/RF_SHAP_global_Output_025x025_rmAC_Lag_3_month.nc"
ds2 = xr.open_dataset(dir_file)
C1 = ds2["FI_Precip"]*100     # Precipitation
C2 = ds2["FI_SM"]*100         # SM
C3 = ds2["FI_WS"]*100         # Wind Speed 
C4 = ds2["FI_VPD"]*100        # VPD
C5 = ds2["FI_ST"]*100         # Surface Temperature


# ======================================  Plot ======================================
diff_use_list = [C1,C2,C3,C4,C5]

# Map extents and latitude tick locations
lon_min, lon_max = -180, 180 
lat_min, lat_max = -70, 90    
panel_labels = ['a', 'b', 'c', 'd', 'e']

titles = ['Precipitation','SM', 'VPD', 'Wind Speed','Surface Temperature']

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'
fig, axes = plt.subplots(3, 2, figsize=(11, 9.5), subplot_kw={'projection': ccrs.PlateCarree()})

plt.subplots_adjust(wspace=0.15)  # Decrease to bring columns closer together (default is ~0.2)
plt.subplots_adjust(hspace=0.39)  # vertical space between rows
axes_flat = axes.flat

cm = Colormap('cmasher:neon_r') 
cm = Colormap('bids:fake_parula_r')  # case insensitive
cm = Colormap('colorbrewer:GnBu')  # case insensitive
# Convert it to a native matplotlib colormap
mpl_cmap = cm.to_mpl()

# Loop through subplots
for i in range(len(diff_use_list)):
    ax = axes_flat[i]
    diff_use = diff_use_list[i]
    title = titles[i]

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.OCEAN, color='white')

    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9, 'color': 'gray'}
    gl.ylabel_style = {'size': 9, 'color': 'gray'}
    def custom_tick_format(x, pos):
        if x < 1:
            return f'{x:.1f}'  # One decimal place for values <1
        else:
            return f'{x:.1f}'  # No decimal place for values >=1

    if (i ==0) | (i==1):
        # Define shared colorbar range
        bounds2 = np.arange(0.0, 10.1, 1)
        # cmap_colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(bounds2) - 1))  # Use the "bwr" colormap
        cmap_colors = mpl_cmap(np.linspace(0., 1, len(bounds2) - 1))  # Use the "bwr" colormap
        custom_cmap2 = mcolors.ListedColormap(cmap_colors)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)

        mesh = ax.pcolormesh(diff_use['longitude'], diff_use['latitude'], diff_use,
        transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto')


    else:
        # Define shared colorbar range
        # bounds2 = np.arange(0.0, 30.1, 3)
        # cmap_colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(bounds2) - 1))  # Use the "bwr" colormap
        cmap_colors = mpl_cmap(np.linspace(0., 1, len(bounds2) - 1))  # Use the "bwr" colormap
        custom_cmap2 = mcolors.ListedColormap(cmap_colors)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
        mesh = ax.pcolormesh(diff_use['longitude'], diff_use['latitude'], diff_use,
        transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto')

        
    # Subplot title
    ax.set_title(f'{title}', fontsize=10, fontweight='bold')

    # Colorbar below each subplot using inset_axes
    cb_ax = inset_axes(ax, width="100%", height="8%", loc='lower center',
                       bbox_to_anchor=(0., -0.19, 1, 1),
                       bbox_transform=ax.transAxes, borderpad=0)
    
    cbar = plt.colorbar(mesh, cax=cb_ax, orientation='horizontal', ticks=bounds2)

    if (i ==0)|(i==1):
        cbar.set_label(r'Feature Importance (%)', fontsize=9, labelpad=1) 
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_format))
    else: 
        cbar.set_label(r'Feature Importance (%)', fontsize=9, labelpad=1) 
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_format))

    cbar.ax.tick_params(labelsize=9)
    ax.text(-0.1, 1.1, panel_labels[i], transform=ax.transAxes,fontsize=14, fontweight='bold', va='top', ha='left')

# Remove the unused subplot
fig.delaxes(axes[2, 1])


# Save
plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/SFigure14_RF_SHAP_Feature_Importance.pdf',dpi=300, bbox_inches='tight')
plt.show()