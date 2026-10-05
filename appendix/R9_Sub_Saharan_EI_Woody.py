import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats
from scipy.stats import linregress, pearsonr, t

# plotting packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from matplotlib.patches import ConnectionPatch, Patch
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from cmap import Colormap  # 你自己的 colormap 模块

# ========================= 公共设定 =========================
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'

startyear_use = 2002
endyear_use   = 2022

Var_name = ['TOTL', 'SAVA', 'AGRI', 'BORF', 'DEFO', 'TEMF', 'PEAT']

# ============================================================
# ====================== 2. BF & EI spatial maps =============
# ============================================================

# ---------- BF mask ----------
def calc_BF_annual(var_name):
    dir_file = "/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_BF_byBiomes_025x025_2002_2022_monthly.nc"
    ds2 = xr.open_dataset(dir_file)
    C = ds2[var_name]
    region_C = C.sel(longitude=slice(-180, 180),
                     latitude=slice(-90, 90),
                     time=slice(str(startyear_use), str(endyear_use)))
    region_C_yearly = region_C.resample(time='YS').sum()
    return region_C_yearly

BA_data_use_BF = [calc_BF_annual(v) for v in Var_name]
data_use_BF    = BA_data_use_BF[1]  # SAVA
data_area_sum  = data_use_BF.mean(dim=['time'], skipna=True)
BF_mask        = data_area_sum

# ---------- BF Trend ----------
dir_file = "/scratch/yangbuw/Data/GFED5/output_biome/GFED5_BF_SAVA_Trend.nc"
ds2 = xr.open_dataset(dir_file)
trend        = ds2["slope"]
p_values     = ds2["p_values"]
mk_p_values  = ds2["mk_significance"]
significance_mask = (p_values < 0.1) | (mk_p_values < 0.1)
trend_sig    = trend.where(significance_mask)
data_use_plot_BF_trend = trend_sig

# ---------- EI Trend ----------
dir_file = "/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_CO2_EI_Trend_by_Tg_per_Mha.nc"
ds2 = xr.open_dataset(dir_file)
trend        = ds2["slope"]
p_values     = ds2["p_values"]
mk_p_values  = ds2["mk_significance"]
significance_mask = (p_values < 0.1) | (mk_p_values < 0.1)
trend_sig    = trend.where(significance_mask)
data_use_plot_EI_trend = trend_sig


# ----------woody Trend ----------
dir_file = "/home/yangbuw/Program/EmissionIntensity/response/data/Venter2018_WoodyTrend_Africa_1km.nc"
ds2 = xr.open_dataset(dir_file)
trend        = ds2["woody_trend"]
trend_sig    = trend
data_use_plot_woody_trend = trend_sig

# 掩膜：只显示 BF 大于阈值的区域
data_use_plot_BF      = data_area_sum.where(BF_mask > 0.0010)
data_use_plot_EI_trend= data_use_plot_EI_trend.where(BF_mask > 0.0010)
# data_use_plot_woody_trend= data_use_plot_woody_trend.where(BF_mask > 0.0010)

diff_use_list = [
    data_use_plot_EI_trend * 100,   # c: BF trend (%/yr)
    data_use_plot_woody_trend    # d: EI trend
]



# ----------------------------- plot --------------------------------

# Map extents and latitude tick locations
lon_min, lon_max = -180, 180 
lat_min, lat_max = -70, 90    
lon_min, lon_max = -30, 68
lat_min, lat_max = -46, 43
panel_labels = ['a', 'b', 'c', 'd','e','f']

# Sample datasets (Replace these with your actual datasets)
datasets = diff_use_list


titles = [
    '',
    '',
    '',
    '',
    '',
    ''
]

labels = [
    r'Emission Intensity Trend (% Tg C $\text{Mha}^{-1}$ $\text{year}^{-1}$)',
    r'Woody Cover Trend (% $\text{year}^{-1}$)',

]

# Create the figure and subplots
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'
fig, axes = plt.subplots(1, 2, figsize=(11, 6),
                         subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(wspace=0.15)
plt.subplots_adjust(hspace=0.25)
axes_flat = axes.flat

# New plotting order:
# First subplot: wind (index 4)
# Second subplot: wind again (index 4)
# Third–Fifth: original 0–2 (ST, VPD, SM)
plot_order = [0,1]

for idx, data_index in enumerate(plot_order):
    ax = axes_flat[idx]
    i = data_index  # use original index logic for bounds/colors

    # Set color scale and colormap depending on variable
    if i == 0:
        bounds2 = np.arange(-4., 4.01, 0.8)
        cm = Colormap('colorbrewer:BrBG')
        cm = Colormap('colorbrewer:PuOr_r')
        mpl_cmap = cm.to_mpl()
        colors2 = mpl_cmap(np.linspace(0.1, 0.9, len(bounds2) - 1))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
 
    else:  #  i= 5
        bounds2 = np.arange(-2, 2.1, 0.4)
        cm = Colormap('colorbrewer:RdYlGn')  # case insensitive
        mpl_cmap = cm.to_mpl()
        colors2 = mpl_cmap(np.linspace(0., 1, len(bounds2) - 1))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)


    # Colorbar
    if (i ==0)|(i==1)|(i==2)|(i==3)|(i==4)|(i==5):
        mesh = ax.pcolormesh(
            datasets[i]['longitude'], datasets[i]['latitude'], datasets[i],
            transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto'
        )
    
        # Map details
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.OCEAN, color='white')
    
        gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12, 'color': 'gray'}
        gl.ylabel_style = {'size': 12, 'color': 'gray'}
    
        # Title
        ax.set_title(titles[i], fontsize=12, fontweight='bold')
    
        # Add region boundary
        # for geom in region_shape.geometry:
        #     ax.add_geometries([geom], crs=ccrs.PlateCarree(),
        #                       edgecolor='red', facecolor='none', linewidth=1.5)
        
        cb_ax = inset_axes(ax, width="100%", height="5%", loc='lower center',
                           bbox_to_anchor=(0, -0.17, 1.0, 1),
                           bbox_transform=ax.transAxes, borderpad=0)
        cbar = plt.colorbar(mesh, cax=cb_ax, orientation='horizontal', ticks=bounds2)
    
        cbar.set_label(labels[i], labelpad=-1, fontsize=12)
        cbar.ax.tick_params(labelsize=12)

    # Panel label
    ax.text(0.04, 0.99, panel_labels[idx], transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')

# # Remove the unused subplot (6th panel)
# fig.delaxes(axes[2, 1])

# plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/SFigure_12_Africa_Woddy.pdf',dpi=300, bbox_inches='tight')

plt.show()

