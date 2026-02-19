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


dir_file = f"/home/yangbuw/Program/EmissionIntensity/code_version1/data_code/data/SEM_global_coeff_025x025_rmAC_2Path_VegIndex_Lag_3_month.nc"
ds2 = xr.open_dataset(dir_file)
C1 = ds2["coeff1_sig"]    # Fuel Load
C2 = ds2["coeff2_sig"]    # Fire weather


# C1 = C1.where((C1 >= 0) & (C1 <= 2), np.nan)
# C2 = C2.where((C2 >= 0) & (C2 <= 2), np.nan)


# ======================================  Plot ======================================
diff_use_list = [C1,C2]

# Map extents and latitude tick locations
lon_min, lon_max = -180, 180 
lat_min, lat_max = -70, 90    


titles = ['Fuel Load', 'Fire Weather']
panel_labels = ['a', 'b']

# Setup figure and axes
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'
fig, axs = plt.subplots(1, 2, figsize=(11, 5), subplot_kw={'projection': ccrs.PlateCarree()})

plt.subplots_adjust(wspace=0.2)  # Decrease to bring columns closer together (default is ~0.2)
plt.subplots_adjust(hspace=0.45)  # vertical space between rows

# Flatten axes for easier indexing
axs = axs.flatten()

# Loop through subplots
for i, ax in enumerate(axs): 
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
            return f'{x:.2f}'  # One decimal place for values <1
        else:
            return f'{x:.2f}'  # No decimal place for values >=1

    bounds2 =[-1, 0.  , 0.08, 0.16, 0.24, 0.32, 0.4 , 0.48, 0.56, 0.64, 0.72]
    colors2 =['#ffffff','#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']
    custom_cmap2 = LinearSegmentedColormap.from_list("custom_rainbow", colors2, N=len(colors2))
    norm2 = BoundaryNorm(bounds2, len(colors2))

    mesh = ax.pcolormesh(diff_use['longitude'], diff_use['latitude'], diff_use,
    transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto')

        
    # Subplot title
    ax.set_title(f'{title}', fontsize=10, fontweight='bold')

    # Colorbar below each subplot using inset_axes
    cb_ax = inset_axes(ax, width="100%", height="8%", loc='lower center',
                       bbox_to_anchor=(0, -0.21, 1.0, 1),
                       bbox_transform=ax.transAxes, borderpad=0)
    
    cbar = plt.colorbar(mesh, cax=cb_ax, orientation='horizontal', ticks=bounds2)

    if i ==0:
        #cbar.set_label(r'Path Coefficients', fontsize=12, labelpad=2)
        ax.text(-0.125, 1.1, 'a', transform=ax.transAxes,fontsize=14, fontweight='bold', va='top', ha='left')
    else: 
        # cbar.set_label(r'Path Coefficients', fontsize=12, labelpad=2) 
        # cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_format))
        ax.text(-0.125, 1.1, 'b', transform=ax.transAxes,fontsize=14, fontweight='bold', va='top', ha='left')

    cbar.ax.tick_params(labelsize=8)


# Save
plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/Figure11Spatial_Map_Path_Coeffs_White.pdf', dpi=300, bbox_inches='tight')
plt.show()