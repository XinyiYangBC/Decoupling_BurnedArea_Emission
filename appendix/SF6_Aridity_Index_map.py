import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats
from scipy.stats import linregress, pearsonr, t
from scipy.stats import kendalltau
from datetime import datetime

# plotting package
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm,ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker




dir_file = f"/scratch/yangbuw/Data/Aridity_Index/Aridity_Index_AnnualMean_025x025.nc"
ds2 = xr.open_dataset(dir_file)
AI = ds2["Aridity_Index"]


data_use_plot = AI

# Reclassification
AI_class = xr.full_like(AI, fill_value=np.nan)

mask = np.isnan(AI)
# Apply classification rules

# AI_class = AI_class.where(AI >= 0.2, 1)                              # < 0.2 → 1 (Hyper Arid + Arid)
AI_class = xr.where(AI < 0.2, 1, AI_class)   # Assign 1 to AI < 0.2
AI_class = xr.where((AI >= 0.2) & (AI < 0.5), 2, AI_class)           # 0.2–0.5 → 2 (Semi-Arid)
AI_class = xr.where((AI >= 0.5) & (AI <= 0.65), 3, AI_class)         # 0.5–0.65 → 3 (Dry Sub-humid)
AI_class = xr.where(AI > 0.65, 4, AI_class)                          # > 0.65 → 4 (Humid)
AI_class = AI_class.where(~mask)  


# ----------------------------- Two Subplots with Individual Colorbars --------------------------------
diff_use = data_use_plot
diff_use2 = diff_use  # Replace with your first dataset

bounds2 = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 10.0]
n_bins2 = len(bounds2) - 1
colors2 = plt.cm.jet_r(np.linspace(0, 1, n_bins2))
custom_cmap2 = LinearSegmentedColormap.from_list("custom_rainbow", colors2, N=len(colors2))
norm2 = BoundaryNorm(bounds2, len(colors2))

lon_min, lon_max = -180, 180 
lat_min, lat_max = -70, 90    
lat_ticks = range(-70, 91, 30)

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'

# Create figure and 2 axes
fig, ax0= plt.subplots( figsize=(11, 5),subplot_kw={'projection': ccrs.PlateCarree()})

plt.subplots_adjust(hspace=0.40)

# -------------------- ax0 --------------------
ax0.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax0.coastlines()
ax0.add_feature(cfeature.BORDERS, linestyle=':')
ax0.add_feature(cfeature.OCEAN, color='white')

gl0 = ax0.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
gl0.top_labels = False
gl0.right_labels = False
gl0.bottom_labels = True
gl0.xlabel_style = {'size': 16, 'color': 'gray'}
gl0.ylabel_style = {'size': 16, 'color': 'gray'}

mesh0 = ax0.pcolormesh(
    diff_use['longitude'], diff_use['latitude'], diff_use2,
    transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto'
)


def custom_tick_format(x, pos):
    if x ==0.:
        return f'{x:.0f}'
    elif x < 0.1:
        return f'{x:.2f}'
    elif (x < 1.76) & (x > 1.74):
        return f'{x:.2f}'
    elif x >= 2:
        return f'{x:.0f}'
    else:
        return f'{x:.1f}'

cb_ax0 = inset_axes(ax0, width="100%", height="8%", loc='lower center',
                   bbox_to_anchor=(0, -0.21, 1.0, 1),
                   bbox_transform=ax0.transAxes, borderpad=0)

cbar = plt.colorbar(mesh0, cax=cb_ax0, orientation='horizontal', ticks=bounds2)
cbar.set_label('Aridity Index', labelpad=2, fontsize=14)
cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_format))
cbar.ax.tick_params(labelsize=14)

plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/SFigure6_Aridity_Index_Map.pdf',dpi=300, bbox_inches='tight')
plt.show()
