import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats
from scipy.stats import linregress, pearsonr, t

# plotting package
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
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



startyear_use       = 2002;    # for calculating and plotting
endyear_use         = 2022;

# ==============================   BF ==============================

# ============ MK trend test ===========
def calculate_trend_line(series):
    x=np.array(range(1,len(series)+1))
    y = series#.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    trend_line = slope * x + intercept
    slope_per = slope/np.mean(y)
    return slope_per, p_value

# =============== BA ===================
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
data_lat_sum = data_area_sum.mean(dim='longitude',skipna=True)

data_use_plot_BF = data_area_sum
data_use_BF = data_use





# ==============================   Emission Intensity Trend   ==============================
dir_file = f"/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_CO2_EI_Trend_by_Tg_per_Mha.nc"
ds2 = xr.open_dataset(dir_file)
trend = ds2["slope"]
#trend_sig = ds2["slope_sig"]
p_values = ds2["p_values"]
mk_p_values = ds2["mk_significance"]

significance_mask = (p_values < 0.1) | (mk_p_values < 0.1)
trend_sig = trend.where(significance_mask)

data_use_plot_EI_trend = trend_sig




# ======================================  Plot ======================================
from cmap import Colormap

data_use_plot_BF = data_use_plot_BF.where(BF_mask>0.0010)
data_use_plot_EI_trend = data_use_plot_EI_trend.where(BF_mask>0.0010)

diff_use_list = [data_use_plot_BF * 100, 
                 data_use_plot_EI_trend*100]

# Map extents and latitude tick locations
lon_min=-180
lon_max= 180
lat_min=-70
lat_max=90


titles = ['Burned Fraction', 'Emission Intensity Trend','Fire CO2 Emission', 'Emission Intensity']
panel_labels = ['a', 'b', 'c', 'd']

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'
# Setup figure and axes
fig, axs = plt.subplots(1, 2, figsize=(11, 5), subplot_kw={'projection': ccrs.PlateCarree()})


plt.subplots_adjust(wspace=0.15)  # Decrease to bring columns closer together (default is ~0.2)
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

    if i ==0:
        # Define shared colorbar range
        cm = Colormap('colorbrewer:Spectral_r')  # case insensitive
        # cm = Colormap('colorbrewer:RdYlGn_r')  # case insensitive
        cm = Colormap('yorick:rainbow_r')  # case insensitive
        cm = Colormap('imagej:fire_r')  # case insensitive
        mpl_cmap = cm.to_mpl()
        
        bounds2 = [0, 0.1, 0.5 ,1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        white = np.array([[1, 1, 1, 1]])
        colors2 = mpl_cmap(np.linspace(0, 0.9, len(bounds2) - 1))  # Use the "bwr" colormap
        # colors2 = np.vstack((white, colors2))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)

        def custom_tick_format(x, pos):
            if x ==0:
                return f'{x:.0f}'  # One decimal place for values <1
            elif (x >0) &(x<1):
                return f'{x:.1f}'  # One decimal place for values <1
            else:
                return f'{x:.0f}'  # No decimal place for values >=1

        mesh = ax.pcolormesh(diff_use['longitude'], diff_use['latitude'], diff_use,
        transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto')

    else:
        # Define shared colorbar range
        # bounds2 = np.arange(-4.0, 4.1, 0.5)
        # cm = Colormap('colorbrewer:RdYlGn')
        # mpl_cmap = cm.to_mpl()
        # white = np.array([[1, 1, 1, 1]])
        # colors2 = mpl_cmap(np.linspace(0, 1, len(bounds2) - 1))  # Use the "bwr" colormap
        # # colors2 = np.vstack((white, colors2))
        # custom_cmap2 = mcolors.ListedColormap(colors2)
        # norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
        
        cm = Colormap('colorbrewer:Spectral_r')  # case insensitive
        cm = Colormap('colorbrewer:PuOr_r')  # case insensitive
        cm = Colormap('colorbrewer:RdBu_r')  # case insensitive
        cm = Colormap('colorbrewer:RdYlGn_r')  # case insensitive
        cm = Colormap('colorbrewer:RdBu_r')  # case insensitive
        
        mpl_cmap = cm.to_mpl()
        
        bounds2 = np.arange(-4.0, 4.1, 0.5)
        white = np.array([[1, 1, 1, 1]])
        colors2 = mpl_cmap(np.linspace(0.1, 0.9, len(bounds2) - 1))  # Use the "bwr" colormap
        # colors2 = mpl_cmap(np.linspace(0, 1, len(bounds2) - 1))  # Use the "bwr" colormap
        # colors2 = np.vstack((white, colors2))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
        def custom_tick_format(x, pos):
            if (x ==-3.5)|(x ==3.5)|(x ==-2.5)|(x ==2.5)|(x ==-1.5)|(x ==1.5)|(x ==-0.5)|(x ==0.5):
                return f'{x:.1f}'  # One decimal place for values <1
            else:
                return f'{x:.0f}'  # No decimal place for values >=1
        mesh = ax.pcolormesh(diff_use['longitude'], diff_use['latitude'], diff_use,
        transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto')

        
    # Subplot title
    # ax.set_title(f'{title}', fontsize=20)#, fontweight='bold')

    # Colorbar below each subplot using inset_axes
    cb_ax = inset_axes(ax, width="100%", height="8%", loc='lower center',
                       bbox_to_anchor=(0, -0.21, 1.0, 1),
                       bbox_transform=ax.transAxes, borderpad=0)
    
    cbar = plt.colorbar(mesh, cax=cb_ax, orientation='horizontal', ticks=bounds2)

    if i ==0:
        cbar.set_label(r'Burned Fraction (%)', fontsize=9, labelpad=2)
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_format))
        cbar.ax.tick_params(labelsize=9)
        ax.text(-0.125, 1.1, 'd', transform=ax.transAxes,fontsize=14, fontweight='bold', va='top', ha='left')
    else: 
        cbar.set_label(r'Emission Intensity Trend (% Tg C $\mathrm{Mha}^{-1}$$\mathrm{yr}^{-1}$ )', fontsize=9, labelpad=2)
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_format))
        cbar.ax.tick_params(labelsize=9)
        ax.text(-0.125, 1.1, 'e', transform=ax.transAxes,fontsize=14, fontweight='bold', va='top', ha='left')

    # cbar.ax.tick_params(labelsize=6)


# Save
plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/SFigure5_Spatial_Map_BF.pdf',dpi=1000, bbox_inches='tight')
plt.show()