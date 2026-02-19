import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import kendalltau
import geopandas as gpd  
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
dir_file = f"/home/yangbuw/Program/EmissionIntensity/code/data_code/data/RF_SHAP_global_Output_025x025_rmAC_Lag_3_month.nc"
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


all_nan_mask = stacked.isnull().all(dim="driver")
filled = stacked.fillna(-9999)
dominant_driver = filled.argmax(dim="driver") + 1
dominant_driver = dominant_driver.where(~all_nan_mask)
dominant_driver.name = "Dominant_Factor"

cate_5 = [np.sum(dominant_driver==1).values, np.sum(dominant_driver==2).values, np.sum(dominant_driver==3).values, np.sum(dominant_driver==4).values, np.sum(dominant_driver==5).values]
ratio_driver = cate_5/np.sum(cate_5)


# ===================================  Results: RF + SHAP ===================================
# ===================================  1st data ===================================
dir_file = f"/home/yangbuw/Program/EmissionIntensity/code/data_code/data/RF_SHAP_global_Output_025x025_rmAC_Lag_3_month.nc"
ds2 = xr.open_dataset(dir_file)
r2_score = ds2["r2_score"] 
r_square = ds2["r_square"] 
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

# ===================================  2nd data ===================================
dir_file = f"/home/yangbuw/Program/EmissionIntensity/code/data_code/data/RF_SHAP_global_Output_025x025_rmAC_Lag_3_month.nc"
ds2 = xr.open_dataset(dir_file)
C1 = ds2["FI_Precip"]    # Precipitation
C2 = ds2["FI_SM"]         # SM
C3 = ds2["FI_WS"]         # Wind Speed 
C4 = ds2["FI_VPD"]        # VPD
C5 = ds2["FI_ST"]        # Surface Temperature


lat= C1.latitude
lon= C1.longitude

C1_abs=abs(C1)
C2_abs=abs(C2)
C3_abs=abs(C3)

# Step 1: Replace NaNs with 0 (means "no contribution")
C1_clean = C1_abs.fillna(0)
C2_clean = C2_abs.fillna(0)
C3_clean = C3_abs.fillna(0)


# ------------------ Normalize to Relative Contributions -----------------------
total = C1_clean + C2_clean + C3_clean
total = total.where(total != 0)

W1 = (C1_clean / total).clip(0, 1)  # Surface Temp
W2 = (C2_clean / total).clip(0, 1)  # VPD
W3 = (C3_clean / total).clip(0, 1)  # Soil Moisture

# # Define custom RGB colors
color_ST = np.array([1.0, 0.0, 1.0])  # Magenta
color_VPD = np.array([0.0, 1.0, 1.0])  # VPD -> Cyan
color_SM = np.array([1.0, 1.0, 0.0])   # Soil Moisture -> Yellow



# Convert to DataArray with RGB stacking
R = W1 * color_ST[0] + W2 * color_VPD[0] + W3 * color_SM[0]
G = W1 * color_ST[1] + W2 * color_VPD[1] + W3 * color_SM[1]
B = W1 * color_ST[2] + W2 * color_VPD[2] + W3 * color_SM[2]

rgb_map = xr.concat([R, G, B], dim='band').transpose('latitude', 'longitude', 'band')
rgb_map_cal = rgb_map.where(BF_mask>0.0010)
rgb_map_cal = rgb_map_cal.where(p_value<=0.050)
rgb_map =rgb_map_cal
# ===================================  RF + SHAP ===================================



# ----------------------------- plot --------------------------------
shapefile_path = "/scratch/yangbuw/Data/GFED5/14regions/GFED5_Region_8.shp"
region_shape = gpd.read_file(shapefile_path)

# Map extents and latitude tick locations
lon_min, lon_max = -180, 180 
lat_min, lat_max = -70, 90    
panel_labels = ['a', 'b', 'c', 'd']

# Sample datasets (Replace these with your actual datasets)
datasets = [
    smv_use3*1e1, # % per year to % per decade
    precip_use3*10*1e1, # cm per month per year to mm per month per decade
    wind_use3*1e2, # m per s  per year to cm per s 
    dominant_driver
]

titles = [
    'Soil Moisture',
    'Precipitation',
    'Wind Speed',
    ''
]

titles = [
    '',
    '',
    '',
    ''
]

labels = [
    r'SM Trend (%  $\text{decade}^{-1}$)',
    r'Precip Trend (mm $\text{month}^{-1}$ $\text{decade}^{-1}$)',
    r'Wind Speed Trend (cm $\text{s}^{-1}$ $\text{decade}^{-1}$)',
    ''
]

# Create the figure and subplots
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'
fig, axes = plt.subplots(2, 2, figsize=(11, 6),
                         subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(wspace=0.15)
plt.subplots_adjust(hspace=0.16)
axes_flat = axes.flat

# New plotting order:
# First subplot: wind (index 4)
# Second subplot: wind again (index 4)
# Third–Fifth: original 0–2 (ST, VPD, SM)
plot_order = [3, 2, 0, 1]

for idx, data_index in enumerate(plot_order):
    ax = axes_flat[idx]
    i = data_index  

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
    else:
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.OCEAN, color='white')
        
        gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 9, 'color': 'gray'}
        gl.ylabel_style = {'size': 9, 'color': 'gray'}
        
        
        # ====== RGB map processing ======
        zero_mask = (rgb_map.sum(dim='band') == 0)
        rgb_plot_white_bg = rgb_map.copy()
        rgb_plot_white_bg.values[zero_mask.values] = [1.0, 1.0, 1.0]
        
        def make_custom_rgb_triangle(size=300):
            img = np.ones((size, size, 3))  # white background
        
            # Define triangle vertices
            top = (size // 2, 0)           # Soil Moisture (Yellow)
            left = (0, size - 1)           # VPD (Cyan)
            right = (size - 1, size - 1)   # Surface Temp (Purple)
        
        
            for i in range(size):
                for j in range(size):
                    px, py = i, j
        
                    # Compute barycentric coordinates
                    denom = ((left[1] - right[1]) * (top[0] - right[0]) +
                             (right[0] - left[0]) * (top[1] - right[1]))
                    if denom == 0:
                        continue
        
                    w1 = ((left[1] - right[1]) * (px - right[0]) +
                          (right[0] - left[0]) * (py - right[1])) / denom
                    w2 = ((right[1] - top[1]) * (px - right[0]) +
                          (top[0] - right[0]) * (py - right[1])) / denom
                    w3 = 1 - w1 - w2
        
                    if w1 >= 0 and w2 >= 0 and w3 >= 0:
                        # Normalize weights
                        weights = np.array([w1, w2, w3])
                        weights = weights / weights.sum()
        
                        # Map to custom RGB using weighted sum
                        # color = (
                        #     weights[0] * color_VPD +
                        #     weights[1] * color_SM +
                        #     weights[2] * color_ST
                        # )
                        color = (
                            weights[0] * color_ST +   # left  → Yellow
                            weights[1] * color_VPD +   # right → Magenta
                            weights[2] * color_SM    # top   → Cyan
                        )
        
                        img[py, px] = np.clip(color, 0, 1)
        
            return img
        
        
        def plot_rgb_triangle(ax, size=0.17, loc=(0.07, 0.15)):
            triangle_img = make_custom_rgb_triangle()
        
            inset_ax = inset_axes(ax,
                                  width=f"{size * 100}%",
                                  height=f"{size * 100}%",
                                  loc='lower left',
                                  bbox_to_anchor=(loc[0], loc[1], 1, 1),
                                  bbox_transform=ax.transAxes,
                                  borderpad=0)
        
            inset_ax.imshow(triangle_img, origin='lower')  # 保持 origin 为 lower
            inset_ax.axis('off')
        
            inset_ax.text(0.5, -0.05, 'Precipitation', transform=inset_ax.transAxes,  
                          fontsize=8.5, ha='center', va='top')
        
            inset_ax.text(-0.048, 1.02, 'Soil \nMoisture', transform=inset_ax.transAxes,       
                          fontsize=8.5, ha='right', va='bottom')
        
            inset_ax.text(1.08, 1.02, 'Wind \nSpeed', transform=inset_ax.transAxes, 
                          fontsize=8.5, ha='left', va='bottom')
        
        # Plot RGB map
        im = ax.imshow(
            rgb_plot_white_bg.values,
            origin='lower',
            transform=ccrs.PlateCarree(),
            extent=[lon.min(), lon.max(), lat.min(), lat.max()],
            interpolation='nearest'
        )
        
        # Add triangle RGB legend
        plot_rgb_triangle(ax)


    # Colorbar
    if (i ==0)|(i==1)|(i==2):
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
        for geom in region_shape.geometry:
            ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                              edgecolor='red', facecolor='none', linewidth=1.5)
        
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

# plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/Figure_5_Masked_FW_trends_Spatial_map.pdf',dpi=300, bbox_inches='tight')

plt.show()