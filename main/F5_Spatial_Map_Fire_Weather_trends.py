import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import kendalltau
# import geopandas as gpd  
# plotting package
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
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
mask1 = BF_mask
# ---------------------  Making Burned Fraction mask -------------------------------

# mask2: 14 regions 
nc_file = "/scratch/yangbuw/Data/GFED5/BasisRegions/GFED5_BasisRegions_LandOnly_South2North_025x025.nc"
ds = xr.open_dataset(nc_file)
mask_data = ds["basisregions"]

# grid_area
nc_file = "/scratch/yangbuw/Data/GFED5/GridArea/GFED5_GridArea_m_square_masked_South2North.nc"
ds = xr.open_dataset(nc_file)
grid_area = ds["grid_area"]

# Pre-setting
Lat_S = -90
Lat_N = 90
Lon_W = -180
Lon_E = 180
Time_start = "2002-01" #"1997-01"
Time_end = "2022-12" #"2022-12"

time_span = pd.date_range(start=Time_start, end=Time_end, freq="YS")

CFI=0.05

#------------------------------ 1 satellite ---------------------------------------
data_dir1 = f"/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_skt_k_Trend_perYear.nc"
data_ds   = xr.open_dataset(data_dir1)
trends_da = data_ds["slope"]
p_values_da = data_ds["p_values"]
mk_significance_da = data_ds["mk_significance"]

lat       = data_ds["latitude"]
lon       = data_ds["longitude"]


p_values_da = p_values_da< CFI
mk_significance_da = mk_significance_da< CFI
significant_mask = np.logical_or(p_values_da, mk_significance_da)
trend_use = trends_da.where(significant_mask)  # remove non-significant grid cell

skt_use2 = trend_use


#------------------------------ 2 satellite ---------------------------------------
data_dir1 = f"/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_VPD_kpa_Trend_perYear.nc"
data_ds   = xr.open_dataset(data_dir1)
trends_da = data_ds["slope"]
p_values_da = data_ds["p_values"]
mk_significance_da = data_ds["mk_significance"]

lat       = data_ds["latitude"]
lon       = data_ds["longitude"]


p_values_da = p_values_da< CFI
mk_significance_da = mk_significance_da< CFI
significant_mask = np.logical_or(p_values_da, mk_significance_da)
trend_use = trends_da.where(significant_mask)  # remove non-significant grid cell

vpd_use2 = trend_use


#------------------------------ 3 satellite ---------------------------------------
data_dir1 = f"/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_swvl1_%_Trend_perYear.nc"
data_ds   = xr.open_dataset(data_dir1)
trends_da = data_ds["slope"]
p_values_da = data_ds["p_values"]
mk_significance_da = data_ds["mk_significance"]

lat       = data_ds["latitude"]
lon       = data_ds["longitude"]


p_values_da = p_values_da< CFI
mk_significance_da = mk_significance_da< CFI
significant_mask = np.logical_or(p_values_da, mk_significance_da)
trend_use = trends_da.where(significant_mask)  # remove non-significant grid cell

smv_use2 = trend_use


#------------------------------ 4 satellite ---------------------------------------
data_dir1 = f"/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_precip_cm_Trend_perYear.nc"
data_ds   = xr.open_dataset(data_dir1)
trends_da = data_ds["slope"]
p_values_da = data_ds["p_values"]
mk_significance_da = data_ds["mk_significance"]

lat       = data_ds["latitude"]
lon       = data_ds["longitude"]


p_values_da = p_values_da< CFI
mk_significance_da = mk_significance_da< CFI
significant_mask = np.logical_or(p_values_da, mk_significance_da)
trend_use = trends_da.where(significant_mask)  # remove non-significant grid cell

precip_use2 = trend_use


#------------------------------ 5 satellite ---------------------------------------
data_dir1 = f"/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_WindSpeed_mpers_Trend_perYear.nc"
data_ds   = xr.open_dataset(data_dir1)
trends_da = data_ds["slope"]
p_values_da = data_ds["p_values"]
mk_significance_da = data_ds["mk_significance"]

lat       = data_ds["latitude"]
lon       = data_ds["longitude"]


p_values_da = p_values_da< CFI
mk_significance_da = mk_significance_da< CFI
significant_mask = np.logical_or(p_values_da, mk_significance_da)
trend_use = trends_da.where(significant_mask)  # remove non-significant grid cell

wind_use2 = trend_use

# =====================change here====================================

min_BF = 0.00e-2
# Only show BF>0.15
skt_use3 = skt_use2.where(mask1 > min_BF)
vpd_use3 = vpd_use2.where(mask1 > min_BF)

smv_use3 = smv_use2.where(mask1 > min_BF)
precip_use3 = precip_use2.where(mask1 > min_BF)

wind_use3 = wind_use2.where(mask1 > min_BF)




# ===================================  SEM Path Coefff ===================================
dir_file = f"/home/yangbuw/Program/EmissionIntensity/code_version1/data_code/data/SEM_global_coeff_025x025_rmAC_2Path_VegIndex_Lag_3_month.nc"
ds2 = xr.open_dataset(dir_file)
C1 = ds2["coeff1_sig"]    # Fuel Load
C2 = ds2["coeff2_sig"]    # Fire weather

# ===================================  SEM Path Coefff ===================================



# ----------------------------- plot --------------------------------

# Map extents and latitude tick locations
lon_min, lon_max = -180, 180 
lat_min, lat_max = -70, 90    
panel_labels = ['a', 'b', 'c', 'd','e','f']

# Sample datasets (Replace these with your actual datasets)
datasets = [
    smv_use3*1e1, # % per year to % per decade
    precip_use3*10*1e1, # cm per month per year to mm per month per decade
    wind_use3*1e2, # m per s  per year to cm per s 
    C2,
    skt_use3*1e1, # K per year to K per decade
    vpd_use3*1e3, # kpa per year to pa per year
]


titles = [
    '',
    '',
    '',
    '',
    '',
    ''
]

labels = [
    r'SM Trend (%  $\text{decade}^{-1}$)',
    r'Precip Trend (mm $\text{month}^{-1}$ $\text{decade}^{-1}$)',
    r'Wind Speed Trend (cm $\text{s}^{-1}$ $\text{decade}^{-1}$)',
    '',
    r'Surf Temp Trend (K $\text{decade}^{-1}$)',
    r'VPD Trend (Pa $\text{year}^{-1}$)',
]

# Create the figure and subplots
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'
fig, axes = plt.subplots(3, 2, figsize=(11, 9),
                         subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(wspace=0.15)
plt.subplots_adjust(hspace=0.25)
axes_flat = axes.flat

# New plotting order:
# First subplot: wind (index 4)
# Second subplot: wind again (index 4)
# Third–Fifth: original 0–2 (ST, VPD, SM)
plot_order = [3, 2, 0, 1,4,5]

for idx, data_index in enumerate(plot_order):
    ax = axes_flat[idx]
    i = data_index  # use original index logic for bounds/colors

    # Set color scale and colormap depending on variable
    if i == 0:
        bounds2 = np.arange(-1.5, 1.51, 0.3)
        cm = Colormap('colorbrewer:BrBG')
        mpl_cmap = cm.to_mpl()
        colors2 = mpl_cmap(np.linspace(0.1, 1, len(bounds2) - 1))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
    elif i == 1:
        bounds2 = np.arange(-0.7, 0.71, 0.1)
        cm = Colormap('colorbrewer:BrBG')
        mpl_cmap = cm.to_mpl()
        colors2 = mpl_cmap(np.linspace(0.1, 1, len(bounds2) - 1))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
    elif i == 2:
        bounds2 = np.arange(-1.2, 1.21, 0.2)
        cm = Colormap('colorcet:CET_D2')
        mpl_cmap = cm.to_mpl()
        colors2 = mpl_cmap(np.linspace(0.1, 1, len(bounds2) - 1))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
    elif i == 3:
        bounds2 =[-1, 0.  , 0.2, 0.40, 0.6, 0.80]
        colors2 =['#ffffff','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']
        # colors2 =['#ffffff','#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']
        custom_cmap2 = LinearSegmentedColormap.from_list("custom_rainbow", colors2, N=len(colors2))
        norm2 = BoundaryNorm(bounds2, len(colors2))
    elif i == 4:
        bounds2 = np.arange(-1.2, 1.21, 0.2)
        cm = Colormap('colorbrewer:RdBu_r')
        mpl_cmap = cm.to_mpl()
        colors2 = mpl_cmap(np.linspace(0.1, 1, len(bounds2) - 1))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
    else:  #  i= 5
        bounds2 = np.arange(-24, 24.1, 4)
        cm = Colormap('colorbrewer:RdBu_r')
        mpl_cmap = cm.to_mpl()
        colors2 = mpl_cmap(np.linspace(0.1, 1, len(bounds2) - 1))
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
        gl.xlabel_style = {'size': 9, 'color': 'gray'}
        gl.ylabel_style = {'size': 9, 'color': 'gray'}
    
        # Title
        ax.set_title(titles[i], fontsize=9, fontweight='bold')
    
        # Add region boundary
        # for geom in region_shape.geometry:
        #     ax.add_geometries([geom], crs=ccrs.PlateCarree(),
        #                       edgecolor='red', facecolor='none', linewidth=1.5)
        
        cb_ax = inset_axes(ax, width="100%", height="5%", loc='lower center',
                           bbox_to_anchor=(0, -0.17, 1.0, 1),
                           bbox_transform=ax.transAxes, borderpad=0)
        cbar = plt.colorbar(mesh, cax=cb_ax, orientation='horizontal', ticks=bounds2)
    
        cbar.set_label(labels[i], labelpad=-1, fontsize=9)
        cbar.ax.tick_params(labelsize=9)

    # Panel label
    ax.text(-0.1, 1.1, panel_labels[idx], transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')

# # Remove the unused subplot (6th panel)
# fig.delaxes(axes[2, 1])

plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/Figure_5_Masked_FW_trends_Spatial_map_7subplots.pdf',dpi=300, bbox_inches='tight')

plt.show()
