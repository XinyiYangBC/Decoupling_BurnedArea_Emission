import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats
from scipy.stats import linregress, pearsonr, t
from datetime import datetime
import os
import seaborn as sns

# plotting package
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
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



# ===================================  RF + SHAP ===================================
# ===================================  1st data ===================================
dir_file = f"/home/yangbuw/Program/EmissionIntensity/code_version1/data_code/data/RF_SHAP_global_Output_025x025_rmAC_Lag_3_month.nc"
ds2 = xr.open_dataset(dir_file)
r2_score = ds2["r2_score"] 
r_square = ds2["r_square"] 
r2_score = r_square
p_value  = ds2["p_value"]

EVI     = ds2["FI_EVI"]
Precip  = ds2["FI_Precip"]
SM      = ds2["FI_SM"]
ST      = ds2["FI_ST"]
VPD     = ds2["FI_VPD"]
WS      = ds2["FI_WS"]


stacked = xr.concat([Precip, SM, WS, ST, VPD, EVI], dim='var')
stacked['var'] = [1, 2, 3, 4, 5, 6] 
all_nan_mask = stacked.isnull().all(dim='var')

stacked_filled = stacked.fillna(0)


dominant_var = stacked_filled.argmax(dim='var') + 1
dominant_var = dominant_var.where(~all_nan_mask)
dominant_var.name = "Dominant_Variable"



#---------------------------------------------- Plot ---------------------------------------------------------
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'
fig = plt.figure(figsize=(11, 6))
gs = gridspec.GridSpec(1, 1, height_ratios=[1], hspace=0.025, wspace=0.225)
ax5 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())

lon_min=-180
lon_max= 180
lat_min=-70
lat_max=90

# ========================== Plot 5: CFI map ==========================
r2_score_cal = r2_score.where(BF_mask>0.0010)
r2_score_cal = r2_score_cal.where(p_value<=0.050)
diff_use = r2_score_cal
ax = ax5  # Use ax2 for plot 2

ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.OCEAN, color='white')
gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 14, 'color': 'gray'}
gl.ylabel_style = {'size': 14, 'color': 'gray'}

# Color setup for CFI
bounds2 = np.arange(0.0, 1.01, 0.1)
colors2 = ['#ffffff','#fff7ec','#fee8c8','#fdd49e','#fdbb84','#fc8d59','#ef6548','#d7301f','#b30000','#7f0000']
colors2 = ['#ffffff','#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']
custom_cmap2 = LinearSegmentedColormap.from_list("custom_rainbow", colors2, N=len(colors2))
norm2 = BoundaryNorm(bounds2, len(colors2))

mesh = ax.pcolormesh(diff_use['longitude'], diff_use['latitude'], diff_use,
                     transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto')

# # Colorbar below the map
# cbar_ax = fig.add_axes([0.45, 0.69, 0.35, 0.015])
cbar_ax = fig.add_axes([0.13, 0.1, 0.75, 0.03])
cbar = plt.colorbar(mesh, cax=cbar_ax, orientation='horizontal', ticks=bounds2)
def custom_tick_format(x, pos):
    if np.isclose(x, 0.775, atol=1e-6):
        return f'{x:.2f}'
    elif np.isclose(x, 0.825, atol=1e-6):
        return f'{x:.2f}'
    elif np.isclose(x, 0.875, atol=1e-6):
        return f'{x:.2f}'
    elif np.isclose(x, 0.925, atol=1e-6):
        return f'{x:.2f}'
    elif np.isclose(x, 0.975, atol=1e-6):
        return f'{x:.2f}'
    else:
        return f'{x:.1f}'


cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_format))
# cbar.set_label('Aridity Index', labelpad=2, fontsize=14)
cbar.ax.tick_params(labelsize=13)

# Inset pie chart in lower right
pie_ax = inset_axes(ax, width="35%", height="35%", loc='lower right',
                    bbox_to_anchor=(-0.70, 0.1, 1, 1),
                    bbox_transform=ax.transAxes, borderpad=0)

# Pie data and plot
r2_score_cal = r2_score.where(BF_mask>0.0010)
r2_score_cal = r2_score_cal.where(p_value<=0.050)
CFI_flat = r2_score_cal.values.flatten()
CFI_clean = CFI_flat[~np.isnan(CFI_flat)]
per2 = np.sum(CFI_clean>=0.4)/np.sum(~np.isnan(CFI_clean))*100
per_rest2= 100-per2
sizes = [per2, per_rest2]
labels = ['CFI > 0.9', '']  # Optional: hide label for second slice
colors_pie = [colors2[5], colors2[3]]
explode = (0.1, 0)

def custom_autopct(pct):
    return f'{pct:.1f}%' if pct > 20 else ''  # Show only if pct > 20

pie_ax.pie(
    sizes,
    # labels=labels,  # Comment this out or leave for segment label
    colors=colors_pie,
    startangle=90,
    autopct=custom_autopct,
    textprops={'fontsize': 14, 'color': 'black'},
    explode=explode,
    pctdistance=0.37
)


pie_ax.set_aspect('equal')  # Equal aspect ratio ensures circle
fig.text(0.189, 0.225, 'R² >0.4', fontsize=14, color = 'Black',fontweight='bold', transform=fig.transFigure)


# plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/Figure3_SEM_Aridity_RF_SHAP.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/Figure15_RF_R2_Score_map.pdf', dpi=300, bbox_inches='tight')
plt.show()
