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


# ===================================  1st data ===================================
dir_file = f"/home/yangbuw/Program/EmissionIntensity/code_version1/data_code/data/SEM_Path_Coeffs_relative_importance.nc"
ds2 = xr.open_dataset(dir_file)
ratio_xr = ds2["ratio_xr"]    # Fuel Load
sig_mask = ds2["sig_mask"]    # CC

ratio_xr = ratio_xr.where(sig_mask)
plot1_data = ratio_xr

ratio_xr = ratio_xr.where(sig_mask)

mask_fl = ratio_xr <= 0.3                        # Fuel Load Dominated
mask_co = (ratio_xr > 0.3) & (ratio_xr <= 0.7)   # Co-Dominant
mask_fw = ratio_xr > 0.7                         # Fire Weather Dominated

num_use_fl = np.sum(mask_fl)
num_use_co = np.sum(mask_co)
num_use_fw = np.sum(mask_fw)
sem_total_use = [num_use_fl.values,num_use_co.values,num_use_fw.values]
ratio_SEM = sem_total_use/np.sum(sem_total_use)*100
# ===================================  1st data =================================== 



# ===================================  2nd data ===================================
dir_file = f"/home/yangbuw/Program/EmissionIntensity/code_version1/data_code/data/SEM_Path_Coeffs_relative_importance_4months.nc"
ds2 = xr.open_dataset(dir_file)
ratio_xr = ds2["ratio_xr"]    # Fuel Load
sig_mask = ds2["sig_mask"]    # CC

ratio_xr = ratio_xr.where(sig_mask)
plot2_data = ratio_xr

ratio_xr = ratio_xr.where(sig_mask)

mask_fl = ratio_xr <= 0.3                        # Fuel Load Dominated
mask_co = (ratio_xr > 0.3) & (ratio_xr <= 0.7)   # Co-Dominant
mask_fw = ratio_xr > 0.7                         # Fire Weather Dominated

num_use_fl = np.sum(mask_fl)
num_use_co = np.sum(mask_co)
num_use_fw = np.sum(mask_fw)
sem_total_use = [num_use_fl.values,num_use_co.values,num_use_fw.values]
ratio_SEM2 = sem_total_use/np.sum(sem_total_use)*100
# ===================================  2nd data =================================== 




#---------------------------------------------- Plot ---------------------------------------------------------
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'

fig = plt.figure(figsize=(11, 6))
gs = gridspec.GridSpec(1, 2, height_ratios=[1], hspace=0.01, wspace=0.15)

ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())


# ========================== Define 3-bin custom colormap ==========================
bounds2 = [0, 30, 70, 100]  # Fuel Load <30%, Co-Dominant 30–70%, Fire Weather >70%
colors2 = ['forestgreen', 'bisque', 'orange']

custom_cmap2 = ListedColormap(colors2)
norm2 = BoundaryNorm(bounds2, ncolors=custom_cmap2.N)


# ========================== Pie settings ==========================
colors_pie = ['forestgreen', 'bisque', 'orange']
explode = (0.1, 0, 0)

def custom_autopct(pct):
    return f'{pct:.0f}%' if pct > 20 else ''


# ========================== Data for two panels ==========================
plot_data_all = [plot1_data, plot2_data]
ratio_data_all = [ratio_SEM, ratio_SEM2]
axes_all = [ax1, ax2]


# ========================== Loop ==========================
for i in range(2):

    diff_use = plot_data_all[i] * 100
    ax = axes_all[i]

    lon_min = -180
    lon_max = 180
    lat_min = -70
    lat_max = 90

    ax.set_extent(
        [lon_min, lon_max, lat_min, lat_max],
        crs=ccrs.PlateCarree()
    )

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.OCEAN, color='white')

    # gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2,
    #                   linestyle='--', linewidth=0.4)
    gl = ax.gridlines(
        draw_labels=True,
        color='gray',
        alpha=0.2,
        linestyle='--'
    )

    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9, 'color': 'gray'}
    gl.ylabel_style = {'size': 9, 'color': 'gray'}

    # ------------------------------------------------------
    # Map
    # ------------------------------------------------------
    mesh = ax.pcolormesh(
        diff_use['longitude'],
        diff_use['latitude'],
        diff_use,
        transform=ccrs.PlateCarree(),
        cmap=custom_cmap2,
        norm=norm2,
        shading='auto'
    )

    # ------------------------------------------------------
    # Inset pie chart
    # ------------------------------------------------------
    pie_ax = inset_axes(
        ax,
        width="35%",
        height="35%",
        loc='lower right',
        bbox_to_anchor=(-0.70, 0.1, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    sizes = ratio_data_all[i]

    pie_ax.pie(
        sizes,
        colors=colors_pie,
        startangle=90,
        autopct=custom_autopct,
        textprops={'fontsize': 6, 'color': 'black'},
        explode=explode,
        pctdistance=0.57
    )

    pie_ax.set_aspect('equal')

    # ------------------------------------------------------
    # Panel label: a, b
    # ------------------------------------------------------
    ax.text(
        0.02, 0.98,
        ['a', 'b'][i],
        transform=ax.transAxes,
        fontsize=12,
        fontweight='bold',
        va='top',
        ha='left'
    )

    # ------------------------------------------------------
    # Bottom-center label: 3-month, 4-month
    # ------------------------------------------------------
    ax.text(
        0.5, 0.12,
        ['3-month', '4-month'][i],
        transform=ax.transAxes,
        fontsize=12,
        ha='center',
        va='top'
    )


# ========================== Colorbars ==========================
def custom_tick_format(x, pos):
    if x == 15:
        return 'Fuel Load Dominated'
    elif x == 50:
        return 'Co-Dominated'
    elif x == 85:
        return 'Fire Weather Dominated'
    else:
        return ''


# Left panel colorbar
cbar_ax1 = fig.add_axes([0.122, 0.3, 0.35, 0.015])

cbar1 = plt.colorbar(
    mesh,
    cax=cbar_ax1,
    orientation='horizontal',
    ticks=[15, 50, 85]
)

cbar1.ax.xaxis.set_major_formatter(
    FuncFormatter(custom_tick_format)
)

cbar1.ax.tick_params(labelsize=8.5)


# Right panel colorbar
cbar_ax2 = fig.add_axes([0.555, 0.3, 0.35, 0.015])

cbar2 = plt.colorbar(
    mesh,
    cax=cbar_ax2,
    orientation='horizontal',
    ticks=[15, 50, 85]
)

cbar2.ax.xaxis.set_major_formatter(
    FuncFormatter(custom_tick_format)
)


cbar2.ax.tick_params(labelsize=8.5)


plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/SFigure10_SEM_sensitive_test.pdf',  dpi=300, bbox_inches='tight')

plt.show()


