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

#------------------------------ 6 SHAP results ---------------------------------------
dir_file = f"/home/yangbuw/Program/EmissionIntensity/code_version1/data_code/data/RF_SHAP_global_Output_025x025_rmAC_Lag_3_month.nc"
ds2 = xr.open_dataset(dir_file)
C1 = ds2["FI_Precip"]*100     # Precipitation
C2 = ds2["FI_SM"]*100         # SM
C3 = ds2["FI_WS"]*100         # Wind Speed 
C4 = ds2["FI_VPD"]*100        # VPD
C5 = ds2["FI_ST"]*100         # Surface Temperature
# Stack the 5 variables into a new dimension for comparison
min_BF = 0.2e-2
C1 = C1.where(mask1 > min_BF)
C2 = C2.where(mask1 > min_BF)
C3 = C3.where(mask1 > min_BF)
C4 = C4.where(mask1 > min_BF)
C5 = C5.where(mask1 > min_BF)

stacked = xr.concat([C1, C2, C3, C4, C5], dim="driver")

# Create a mask: True where **all** values are NaN across drivers
all_nan_mask = stacked.isnull().all(dim="driver")

# Temporarily fill NaNs with very low value so argmax can proceed
filled = stacked.fillna(-9999)

# Get index of max driver (1-based index for human readability)
dominant_driver = filled.argmax(dim="driver") + 1

# Mask the grid cells where all values were NaN
dominant_driver = dominant_driver.where(~all_nan_mask)

# Set variable name for clarity
dominant_driver.name = "Dominant_Factor"

# Calculate the ratio  
cate_5 = [np.sum(dominant_driver==1).values, np.sum(dominant_driver==2).values, np.sum(dominant_driver==3).values, np.sum(dominant_driver==4).values, np.sum(dominant_driver==5).values]
ratio_driver = cate_5/np.sum(cate_5)

# ----------------------------- plot --------------------------------
shapefile_path = "/scratch/yangbuw/Data/GFED5/14regions/GFED5_Region_8.shp"
# region_shape = gpd.read_file(shapefile_path)

# Map extents and latitude tick locations
lon_min, lon_max = -180, 180 
lat_min, lat_max = -70, 90    
panel_labels = ['a', 'b', 'c', 'd', 'e','f']

# Sample datasets (Replace these with your actual datasets)
datasets = [
    skt_use3*1e1, # K per year to K per decade
    vpd_use3*1e3, # kpa per year to pa per year
    smv_use3*1e1, # % per year to % per decade
    precip_use3*10*1e1, # cm per month per year to mm per month per decade
    wind_use3*1e2, # m per s  per year to cm per s 
    dominant_driver
]

titles = [
    'Surface Temperature',
    'VPD',
    'Soil Moisture',
    'Precipitation',
    'Wind Speed',
    ''
]

labels = [
    r'(K $\text{decade}^{-1}$)',
    r'(Pa $\text{year}^{-1}$)',
    r'(%  $\text{decade}^{-1}$)',
    r'(mm $\text{month}^{-1}$ $\text{decade}^{-1}$)',
    r'(cm $\text{s}^{-1}$ $\text{decade}^{-1}$)',
    ''
]

# Create the figure and subplots
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'
fig, axes = plt.subplots(3, 2, figsize=(11, 9.5),
                         subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(wspace=0.15)
plt.subplots_adjust(hspace=0.39)
axes_flat = axes.flat

# New plotting order:
# First subplot: wind (index 4)
# Second subplot: wind again (index 4)
# Third–Fifth: original 0–2 (ST, VPD, SM)
plot_order = [5, 4, 0, 1, 2, 3]

for idx, data_index in enumerate(plot_order):
    ax = axes_flat[idx]
    i = data_index  # use original index logic for bounds/colors

    # Set color scale and colormap depending on variable
    if i == 0:
        bounds2 = np.arange(-1.2, 1.21, 0.2)
        cm = Colormap('colorbrewer:RdBu_r')
        mpl_cmap = cm.to_mpl()
        colors2 = mpl_cmap(np.linspace(0.1, 1, len(bounds2) - 1))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
    elif i == 1:
        bounds2 = np.arange(-24, 24.1, 4)
        cm = Colormap('colorbrewer:RdBu_r')
        mpl_cmap = cm.to_mpl()
        colors2 = mpl_cmap(np.linspace(0.1, 1, len(bounds2) - 1))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
    elif i == 2:
        bounds2 = np.arange(-1.5, 1.51, 0.3)
        cm = Colormap('colorbrewer:BrBG')
        mpl_cmap = cm.to_mpl()
        colors2 = mpl_cmap(np.linspace(0.1, 1, len(bounds2) - 1))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
    elif i == 3:
        bounds2 = np.arange(-0.7, 0.71, 0.1)
        cm = Colormap('colorbrewer:BrBG')
        mpl_cmap = cm.to_mpl()
        colors2 = mpl_cmap(np.linspace(0.1, 1, len(bounds2) - 1))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
    elif i == 4:
        bounds2 = np.arange(-1.2, 1.21, 0.2)
        cm = Colormap('colorcet:CET_D2')
        mpl_cmap = cm.to_mpl()
        colors2 = mpl_cmap(np.linspace(0.1, 1, len(bounds2) - 1))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
    else:
        colors = ['#4A00C1', 'gold', '#83B767', '#FC6500', 'moccasin']
        colors = ['#4A00C1', 'gold', 'orange', 'darkgrey', 'lightgray']
        cat_labels = ['Precipitation', 'Soil Moisture', 'Wind Speed', 'VPD', 'Surface Temp']

        custom_cmap2 = mcolors.ListedColormap(colors)
        bounds2 = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)


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

    # # Add region boundary
    # for geom in region_shape.geometry:
    #     ax.add_geometries([geom], crs=ccrs.PlateCarree(),
    #                       edgecolor='red', facecolor='none', linewidth=1.5)

    # Colorbar
    if i ==5:
        cb_ax = inset_axes(ax, width="100%", height="8%", loc='lower center',
                           bbox_to_anchor=(0, -0.21, 1.0, 1),
                           bbox_transform=ax.transAxes, borderpad=0)
        
        cbar = plt.colorbar(
            mesh,
            cax=cb_ax,
            orientation='horizontal',
            boundaries=bounds2,
            ticks=[1, 2, 3, 4, 5],
            spacing='proportional'
        )
        
        formatter = FuncFormatter(lambda val, pos: cat_labels[int(val)-1] if 1 <= val <= 5 else '')
        cbar.ax.xaxis.set_major_formatter(formatter)
        cbar.ax.tick_params(labelsize=9)    
        
        # Create inset axes for bar chart
        categories = cat_labels
        ax_inset = inset_axes(
            ax,
            width="27%",   # relative to parent
            height="27%",  # relative to parent
            loc='upper right',  # position inside subplot
            bbox_to_anchor=(-0.622, -0.6, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=1
        )
    
        # Bar plot
        ax_inset.bar(
            x=np.arange(len(ratio_driver)),
            height=ratio_driver,
            color= colors
        )
        ax_inset.set_xticks(np.arange(len(ratio_driver)))
        ax_inset.set_xticklabels('', rotation=45, fontsize=7,color='gray')
        ax_inset.set_ylabel('', fontsize=7,color='gray')
        ax_inset.set_ylim(0, 0.5)
        ax_inset.tick_params(axis='both', labelsize=7,color='gray')
        # Remove top and right spines for a cleaner look
        ax_inset.spines['top'].set_visible(False)
        ax_inset.spines['right'].set_visible(False)
        ax_inset.patch.set_alpha(0)      # background fully transparent

    
    else:
        cb_ax = inset_axes(ax, width="100%", height="8%", loc='lower center',
                           bbox_to_anchor=(0, -0.21, 1.0, 1),
                           bbox_transform=ax.transAxes, borderpad=0)
        cbar = plt.colorbar(mesh, cax=cb_ax, orientation='horizontal', ticks=bounds2)
    
        cbar.set_label(labels[i], labelpad=-1, fontsize=9)
        cbar.ax.tick_params(labelsize=9)

    # Panel label
    ax.text(-0.1, 1.1, panel_labels[idx], transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')

# # Remove the unused subplot (6th panel)
# fig.delaxes(axes[2, 1])

plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/SFigure7_Masked_FW_trends_Spatial_map.pdf',dpi=300, bbox_inches='tight')

plt.show()
