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


# =========================
# Read latent correlation
# =========================
out_file = "/home/yangbuw/Program/EmissionIntensity/code_version1/data_code/data/SEM_global_LatentVar_colinearity_test_025x025_rmAC_2Path_VegIndex_Lag_3_month.nc"

ds = xr.open_dataset(out_file)

print(ds)

r = ds["coeff1"]   # Est. Std of Fire_Weather ~~ Fuel_Load


# ======================================  Plot ======================================
diff_use_list = [r]

# Map extents and latitude tick locations
lon_min, lon_max = -180, 180 
lat_min, lat_max = -70, 90    

titles = [r'Latent-variable correlation coefficient ($r$)']
panel_labels = ['a', 'b']

# Setup figure and axes
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'
fig, axs = plt.subplots(1, 1, figsize=(6, 5), subplot_kw={'projection': ccrs.PlateCarree()})

plt.subplots_adjust(wspace=0.2)  # Decrease to bring columns closer together (default is ~0.2)
plt.subplots_adjust(hspace=0.45)  # vertical space between rows

axs = np.atleast_1d(axs).flatten()

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
    gl.xlabel_style = {'size': 12, 'color': 'gray'}
    gl.ylabel_style = {'size': 12, 'color': 'gray'}

    def custom_tick_format(x, pos):
        if x < 1:
            return f'{x:.2f}'  # One decimal place for values <1
        else:
            return f'{x:.2f}'  # No decimal place for values >=1

    bounds2 = [-1.0, -0.7, -0.3,0.3, 0.7, 1.0]
    cm = Colormap('colorbrewer:Spectral_r')
    cm = Colormap('colorbrewer:RdYlBu_5_r')
    mpl_cmap = cm.to_mpl()
    
    white = np.array([[1, 1, 1, 1]])
    colors2 = mpl_cmap(np.linspace(0, 1, len(bounds2) - 1))  # Use the "bwr" colormap
    # colors2 = np.vstack((white, colors2))
    custom_cmap2 = mcolors.ListedColormap(colors2)
    norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)

    mesh = ax.pcolormesh(diff_use['longitude'], diff_use['latitude'], diff_use,
    transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto')

        
    # Subplot title
    ax.set_title(f'{title}', fontsize=12, fontweight='bold')

    # Colorbar below each subplot using inset_axes
    cb_ax = inset_axes(ax, width="100%", height="8%", loc='lower center',
                       bbox_to_anchor=(0, -0.21, 1.0, 1),
                       bbox_transform=ax.transAxes, borderpad=0)
    
    cbar = plt.colorbar(mesh, cax=cb_ax, orientation='horizontal', ticks=bounds2)

    # if i ==0:
    #     #cbar.set_label(r'Path Coefficients', fontsize=12, labelpad=2)
    #     ax.text(-0.125, 1.1, 'a', transform=ax.transAxes,fontsize=14, fontweight='bold', va='top', ha='left')
    # else: 
    #     # cbar.set_label(r'Path Coefficients', fontsize=12, labelpad=2) 
    #     # cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_format))
    #     ax.text(-0.125, 1.1, 'b', transform=ax.transAxes,fontsize=14, fontweight='bold', va='top', ha='left')

    cbar.ax.tick_params(labelsize=12)


    # =========================
    # Inset pie chart: |r| < 0.7
    # =========================
    
    # Clean r values
    r_valid = diff_use.values.flatten()
    r_valid = r_valid[np.isfinite(r_valid)]
    r_valid = r_valid[np.abs(r_valid) <= 1]
    
    # Count
    n_low  = np.sum(np.abs(r_valid) < 0.3)
    n_high = np.sum(np.abs(r_valid) >= 0.3)
    
    total = n_low + n_high
    
    pct_low  = n_low / total * 100
    pct_high = n_high / total * 100
    
    # -------------------------------------------------
    # inset location: [left, bottom, width, height]
    # adjust these four numbers if needed
    # -------------------------------------------------

    pie_ax = ax.inset_axes([-0.05, 0.07, 0.35, 0.35])
    
    # Pie
    wedges, texts = pie_ax.pie(
        [n_low, n_high],
        colors=['#FFFFBF', 'darkorange'],
        startangle=90,
        counterclock=False,
        wedgeprops={
            'edgecolor': 'none'
        }
    )
    
    # Main percentage in pie
    pie_ax.text(
        0,
        -0.4,
        f'{pct_low:.1f}%',
        ha='center',
        va='center',
        fontsize=9.5,
        fontweight='bold'
    )
    
    # Threshold label below the pie
    pie_ax.text(
        0,
        -1.39,
        r'$|r|<0.3$',
        ha='center',
        va='center',
        fontsize=12,
        fontweight='bold'
    )
    
    pie_ax.set_aspect('equal')
    pie_ax.set_xticks([])
    pie_ax.set_yticks([])
    
    # Remove frame
    for spine in pie_ax.spines.values():
        spine.set_visible(False)


# Save
plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/response_SEM_LatentVar_linearity_test.pdf', dpi=300, bbox_inches='tight')
plt.show()
