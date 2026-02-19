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

#------------------------------ satellite ---------------------------------------
data_dir1 = f"/scratch/yangbuw/Data/NPP/data/MOD17A3HGF_NPP_025x025_Trend_by_gC_perM2_perYear.nc"
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

npp_use2 = trend_use


#------------------------------ satellite ---------------------------------------
data_dir1 = f"/scratch/yangbuw/Data/AGB/AGB_Trend_gC_perM2_perYear.nc"
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

agb_use2 = trend_use


#------------------------------ satellite ---------------------------------------
data_dir1 = f"/scratch/yangbuw/Data/VOD/VODCA/VODCA_CXKu_Trend_perYear.nc"
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

vod_use2 = trend_use


#------------------------------ satellite ---------------------------------------
data_dir1 = f"/scratch/yangbuw/Data/LAI/MOD15A2H/MOD15A2H_LAI_Trend_perYear.nc"
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

lai1_use2 = trend_use

#------------------------------ satellite ---------------------------------------
data_dir1 = f"/scratch/yangbuw/Data/LAI/MYD15A2H/MYD15A2H_LAI_Trend_perYear.nc"
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

lai2_use2 = trend_use


#------------------------------ satellite ---------------------------------------
data_dir1 = f"/scratch/yangbuw/Data/LAI/GIMMS_LAI4g/GIMMS_LAI4g_V1.2_Trend_perYear.nc"
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

lai3_use2 = trend_use

#------------------------------ satellite ---------------------------------------
data_dir1 = f"/scratch/yangbuw/Data/EVI/MOD13A2_EVI_Trend_perYear.nc"
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

evi_use2 = trend_use

#======================== path coeffs from SEM ===================================
dir_file = f"/home/yangbuw/Program/EmissionIntensity/code_version1/data_code/data/SEM_global_coeff_025x025_rmAC_2Path_VegIndex_Lag_3_month.nc"
ds2 = xr.open_dataset(dir_file)
C1 = ds2["coeff1_sig"]    # Fuel Load
C1 = ds2["coeff1"]    # Fuel Load
C2 = ds2["coeff2_sig"]    # Fire weather

# =====================change here====================================

min_BF = 0.01e-2
# Only show BF>0.15
npp_use3 = npp_use2.where(mask1 > min_BF)
agb_use3 = agb_use2.where(mask1 > min_BF)

vod_use3 = vod_use2.where(mask1 > min_BF)
lai1_use3 = lai1_use2.where(mask1 > min_BF)

lai2_use3 = lai2_use2.where(mask1 > min_BF)
lai3_use3 = lai3_use2.where(mask1 > min_BF)

evi_use3 = evi_use2.where(mask1 > min_BF)


C1_use = C1.where(mask1 > 0.2e-2)
C2_use = C2.where(mask1 > 0.2e-2)


# ----------------------------- plot --------------------------------
shapefile_path = "/scratch/yangbuw/Data/GFED5/14regions/GFED5_Region_8.shp"
# region_shape = gpd.read_file(shapefile_path) 

# Map extents and latitude tick locations
lon_min, lon_max = -180, 180 
lat_min, lat_max = -70, 90    
panel_labels = ['a', 'b','c','d']


# Sample datasets (Replace these with your actual datasets)
datasets = [
    C1_use ,  
    vod_use3 * 1e3,  # % per decade  
    lai1_use3 * 1e3,  # % per decade 
    evi_use3 * 1e3 
]

titles = [
    'Path Coefficients', 'AGB Trends (2002-2019)'
]
labels = [
    r'Path Coefficients',
    # r'AGB Trend (g C $\mathrm{m}^{-2}$ $\mathrm{year}^{-1}$)',
    r'VOD Trend (% $\mathrm{decade}^{-1}$)',
    r'LAI Trend (% $\mathrm{decade}^{-1}$)',
    r'EVI Trend (% $\mathrm{decade}^{-1}$)'
]



# Create the figure and subplots
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'
fig, axes = plt.subplots(2, 2, figsize=(11, 6), subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(wspace=0.15)  # Decrease to bring columns closer together (default is ~0.2)
plt.subplots_adjust(hspace=0.16)  # vertical space between rows

for i, ax in enumerate(axes.flat):
    if i ==0:
        bounds2 = np.arange(-35, 35.1, 5)
        diff_use =C1_use
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
        bounds2 =[-1, 0.  , 0.15, 0.30, 0.45, 0.60 , 0.75]
        colors2 =['#ffffff','#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']
        bounds2 =[-1, 0.  , 0.2, 0.40, 0.6, 0.80]
        colors2 =['#ffffff','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']
        custom_cmap2 = LinearSegmentedColormap.from_list("custom_rainbow", colors2, N=len(colors2))
        norm2 = BoundaryNorm(bounds2, len(colors2))
    
        mesh = ax.pcolormesh(diff_use['longitude'], diff_use['latitude'], diff_use,
        transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto')
    
    
        # # Colorbar below each subplot using inset_axes
        # cb_ax = inset_axes(ax, width="100%", height="5%", loc='lower center',
        #                    bbox_to_anchor=(0, -0.21, 1.0, 1),
        #                    bbox_transform=ax.transAxes, borderpad=0)
        
        # cbar = plt.colorbar(mesh, cax=cb_ax, orientation='horizontal', ticks=bounds2)

    elif i==1:
        bounds2 = np.arange(-10, 10.1, 1)
        cm = Colormap('colorbrewer:PiYG')  # case insensitive
        mpl_cmap = cm.to_mpl()
        
        white = np.array([[1, 1, 1, 1]])
        colors2 = mpl_cmap(np.linspace(0.1, 1, len(bounds2) - 1))  # Use the "bwr" colormap
        # colors2 = np.vstack((white, colors2))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
    
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.OCEAN, color='white')
    
        # Configure gridlines
        gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--')#,linewidth=0.3)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 9, 'color': 'gray'}
        gl.ylabel_style = {'size': 9, 'color': 'gray'}
    
        # Plot dataset
        mesh = ax.pcolormesh(
            datasets[i]['longitude'], datasets[i]['latitude'], datasets[i], 
            transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto'
        )

    elif i==2:
        bounds2 = np.arange(-21, 21.1, 3)
        cm = Colormap('colorbrewer:PiYG')  # case insensitive
        mpl_cmap = cm.to_mpl()
        
        white = np.array([[1, 1, 1, 1]])
        colors2 = mpl_cmap(np.linspace(0.1, 1, len(bounds2) - 1))  # Use the "bwr" colormap
        # colors2 = np.vstack((white, colors2))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
    
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.OCEAN, color='white')
    
        # Configure gridlines
        gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--')#,linewidth=0.3)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 9, 'color': 'gray'}
        gl.ylabel_style = {'size': 9, 'color': 'gray'}
    
        # Plot dataset
        mesh = ax.pcolormesh(
            datasets[i]['longitude'], datasets[i]['latitude'], datasets[i], 
            transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto'
        )


    else:
        bounds2 = np.arange(-3, 3.1, 0.5)
        cm = Colormap('colorbrewer:PiYG')  # case insensitive
        mpl_cmap = cm.to_mpl()
        
        white = np.array([[1, 1, 1, 1]])
        colors2 = mpl_cmap(np.linspace(0.1, 1, len(bounds2) - 1))  # Use the "bwr" colormap
        # colors2 = np.vstack((white, colors2))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
    
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.OCEAN, color='white')
    
        # Configure gridlines
        gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--')#,linewidth=0.3)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 9, 'color': 'gray'}
        gl.ylabel_style = {'size': 9, 'color': 'gray'}
    
        # Plot dataset
        mesh = ax.pcolormesh(
            datasets[i]['longitude'], datasets[i]['latitude'], datasets[i], 
            transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto'
        )

    # ax.set_title(titles[i], fontsize=9, fontweight='bold')

    # **Plot the shapefile (GFED5_Region_8.shp) in red**
    # for geom in region_shape.geometry:
    #     ax.add_geometries([geom], crs=ccrs.PlateCarree(), edgecolor='red', facecolor='none', linewidth=1.5)

    cb_ax = inset_axes(ax, width="100%", height="5%", loc='lower center',
                       bbox_to_anchor=(0, -0.17, 1.0, 1),
                       bbox_transform=ax.transAxes, borderpad=0)
    
    cbar = plt.colorbar(mesh, cax=cb_ax, orientation='horizontal', ticks=bounds2)
    cbar.set_label(labels[i], labelpad=-0, fontsize=9)
    cbar.ax.tick_params(labelsize=9)

    ax.text(-0.1, 1.1, panel_labels[i], transform=ax.transAxes,fontsize=14, fontweight='bold', va='top', ha='left')


plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/Figure_4_PathCoeff_Biomass_trends_Spatial_4maps.pdf',dpi=300, bbox_inches='tight')

plt.show()
