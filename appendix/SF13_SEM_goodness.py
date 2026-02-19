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

dir_file = f"/home/yangbuw/Program/EmissionIntensity/code_version1/data_code/data/SEM_Path_Coeffs_relative_importance.nc"
ds2 = xr.open_dataset(dir_file)
sig_mask = ds2["sig_mask"]  

dir_file = f"/home/yangbuw/Program/EmissionIntensity/code_version1/data_code/data/SEM_global_coeff_025x025_rmAC_2Path_VegIndex_Lag_3_month.nc"
ds2 = xr.open_dataset(dir_file)
C1 = ds2["GFI"]              # goodness-of-fit index
C2 = ds2["CFI"]              # comparative fit index
C3 = ds2["RMSEA"]            # root mean square error of approximation
C4 = ds2["chi2"]    #  χ2/d.f. represents the chi-square test result; Acceptable fit if <5 
C5 = ds2["DoF"]     #  χ2/d.f. represents the chi-square test result; Acceptable fit if <5 

# Mask 
C1 = C1.where(BF_mask >= 0.001, np.nan)
C2 = C2.where(BF_mask >= 0.001, np.nan)
C3 = C3.where(BF_mask >= 0.001, np.nan)
C4 = C4.where(BF_mask >= 0.001, np.nan)
C5 = C5.where(BF_mask >= 0.001, np.nan)

C4 = C4/C5

C4 = C4.where(BF_mask >= 0.01, np.nan)

# PIE chart 
GFI = C1
GFI =GFI.where(sig_mask)
GFI_flat = GFI.values.flatten()
GFI_clean = GFI_flat[~np.isnan(GFI_flat)]
per_GFI = np.sum(GFI_clean>=0.85)/np.sum(~np.isnan(GFI_clean))*100
ratio_GFI =[per_GFI, 100-per_GFI]

RMSE = C3
RMSE =RMSE.where(sig_mask)
RMSE_flat = RMSE.values.flatten()
RMSE_clean = RMSE_flat[~np.isnan(RMSE_flat)]
per_RMSE = np.sum(RMSE_clean<=0.15)/np.sum(~np.isnan(RMSE_clean))*100
per_RMSE
ratio_RMSE =[per_RMSE, 100-per_RMSE]

chi2 = C4
chi2 =chi2.where(sig_mask)
chi2_flat = chi2.values.flatten()
chi2_clean = chi2_flat[~np.isnan(chi2_flat)]
per_chi2 = np.sum(chi2_clean<=5)/np.sum(~np.isnan(chi2_clean))*100
per_chi2
ratio_chi2 =[per_chi2, 100-per_chi2]

CFI = C2
CFI =CFI.where(sig_mask)
CFI_flat = CFI.values.flatten()
CFI_clean = CFI_flat[~np.isnan(CFI_flat)]

per_CFI = np.sum(CFI_clean>=0.90)/np.sum(~np.isnan(CFI_clean))*100
ratio_CFI =[per_CFI, 100-per_CFI]

# ======================================  Plot ======================================
# Map extents and latitude tick locations
lon_min, lon_max = -180, 180 
lat_min, lat_max = -70, 90    
panel_labels = ['a', 'b', 'c','d']

datasets = [C1,C2,C3,C4]
titles = ['GFI','CFI','RMSE', 'χ2/DoF']



# Create the figure and subplots
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'
fig, axes = plt.subplots(2, 2, figsize=(11, 6), subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(wspace=0.15)  # Decrease to bring columns closer together (default is ~0.2)
plt.subplots_adjust(hspace=0.3)  # vertical space between rows
axes_flat = axes.flat

for i in range(4):  # Only iterate over 5 datasets
    ax = axes_flat[i]

    if i ==0:
        bounds2 = np.arange(0.75, 1.01, 0.025)
        cm = Colormap('colorbrewer:Greens')  # case insensitive
        mpl_cmap = cm.to_mpl()
        
        white = np.array([[1, 1, 1, 1]])
        colors2 = mpl_cmap(np.linspace(0.1, 1, len(bounds2) - 1))  # Use the "bwr" colormap
        colors2 = np.vstack((white, colors2))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
        mesh = ax.pcolormesh(
            datasets[i]['longitude'], datasets[i]['latitude'], datasets[i],
            transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto'
        )
    elif i == 1:
        bounds2 = np.arange(0.75, 1.01, 0.025)
        cm = Colormap('colorbrewer:Greens')  # case insensitive
        mpl_cmap = cm.to_mpl()
        
        white = np.array([[1, 1, 1, 1]])
        colors2 = mpl_cmap(np.linspace(0.1, 1, len(bounds2) - 1))  # Use the "bwr" colormap
        colors2 = np.vstack((white, colors2))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
        mesh = ax.pcolormesh(
            datasets[i]['longitude'], datasets[i]['latitude'], datasets[i],
            transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto'
        )
        
    elif i == 2:
        bounds2 = np.arange(0.0, 0.51, 0.05)

        colors2 =['#ffffff','#f7f4f9','#e7e1ef','#d4b9da','#c994c7','#df65b0','#e7298a','#ce1256','#980043','#67001f']
        colors2 = ['#f7f4f9', '#e7e1ef', '#d4b9da', '#cba0d4', '#c994c7', '#df65b0', '#e7298a','#ce1256', '#980043', '#67001f']
        custom_cmap2 = LinearSegmentedColormap.from_list("custom_rainbow", colors2, N=len(colors2))
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
        mesh = ax.pcolormesh(
            datasets[i]['longitude'], datasets[i]['latitude'], datasets[i],
            transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto'
        )

    else:
        bounds2 = np.arange(1.0, 10.01, 0.5)
        cm = Colormap('colorbrewer:GnBu')  # case insensitive
        mpl_cmap = cm.to_mpl()
        
        white = np.array([[1, 1, 1, 1]])
        colors2 = mpl_cmap(np.linspace(0, 1, len(bounds2) - 1))  # Use the "bwr" colormap
        # colors2 = np.vstack((white, colors2))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)
        
        mesh = ax.pcolormesh(
            datasets[i]['longitude'], datasets[i]['latitude'], datasets[i],
            transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto'
        )
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.OCEAN, color='white')

    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--')#, linewidth=0.3)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9, 'color': 'gray'}
    gl.ylabel_style = {'size': 9, 'color': 'gray'}


    ax.set_title(titles[i], fontsize=9, fontweight='bold')

    # Colorbar
    cb_ax = inset_axes(ax, width="100%", height="8%", loc='lower center',
                       bbox_to_anchor=(0, -0.21, 1.0, 1),
                       bbox_transform=ax.transAxes, borderpad=0)
    
    cbar = plt.colorbar(mesh, cax=cb_ax, orientation='horizontal', ticks=bounds2)
    cbar.ax.tick_params(labelsize=9)

    if i ==3:
        cbar.ax.set_xticks(bounds2[1::2]) 
    elif i==0:
        def custom_tick_format(x, pos):
            if np.isclose(x, 0.775, atol=1e-6):
                return f'{x:.3f}'
            elif np.isclose(x, 0.825, atol=1e-6):
                return f'{x:.3f}'
            elif np.isclose(x, 0.875, atol=1e-6):
                return f'{x:.3f}'
            elif np.isclose(x, 0.925, atol=1e-6):
                return f'{x:.3f}'
            elif np.isclose(x, 0.975, atol=1e-6):
                return f'{x:.3f}'
            else:
                return f'{x:.2f}'
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_format))
    else:
        cbar.ax.set_xticks(bounds2[::1]) 


    if i ==0:
        # Inset pie chart in lower right
        pie_ax = inset_axes(ax, width="35%", height="35%", loc='lower right',
                            bbox_to_anchor=(-0.70, 0.1, 1, 1),
                            bbox_transform=ax.transAxes, borderpad=0)
        
        # Pie data and plot
        per = per_GFI
        per_rest= 100-per
        sizes = [per, per_rest]
        colors_pie = [colors2[6], colors2[3]]
        explode = (0.1, 0)
        
        def custom_autopct(pct):
            return f'{pct:.1f}%' if pct > 20 else ''  # Show only if pct > 20
        
        pie_ax.pie(
            sizes,
            # labels=labels,  # Comment this out or leave for segment label
            colors=colors_pie,
            startangle=90,
            autopct=custom_autopct,
            textprops={'fontsize': 9, 'color': 'black'},
            explode=explode,
            pctdistance=0.35
        )
        
        
        pie_ax.set_aspect('equal')  # Equal aspect ratio ensures circle
        ax.text(0.07, 0.1, 'GFI>0.85', transform=ax.transAxes,color = 'Black',fontsize=8, fontweight='bold', va='top', ha='left')
    elif i==1:
        # Inset pie chart in lower right
        pie_ax = inset_axes(ax, width="35%", height="35%", loc='lower right',
                            bbox_to_anchor=(-0.70, 0.1, 1, 1),
                            bbox_transform=ax.transAxes, borderpad=0)
        
        # Pie data and plot
        per = per_CFI
        per_rest= 100-per
        sizes = [per, per_rest]
        colors_pie = [colors2[4], colors2[2]]
        explode = (0.1, 0)
        
        def custom_autopct(pct):
            return f'{pct:.1f}%' if pct > 20 else ''  # Show only if pct > 20
        
        pie_ax.pie(
            sizes,
            # labels=labels,  # Comment this out or leave for segment label
            colors=colors_pie,
            startangle=90,
            autopct=custom_autopct,
            textprops={'fontsize': 9, 'color': 'black'},
            explode=explode,
            pctdistance=0.35
        )
        
        
        pie_ax.set_aspect('equal')  # Equal aspect ratio ensures circle
        ax.text(0.06, 0.1, 'CFI>0.9', transform=ax.transAxes,color = 'Black',fontsize=8, fontweight='bold', va='top', ha='left')
        
    elif i==2:
        # Inset pie chart in lower right
        pie_ax = inset_axes(ax, width="35%", height="35%", loc='lower right',
                            bbox_to_anchor=(-0.70, 0.1, 1, 1),
                            bbox_transform=ax.transAxes, borderpad=0)
        
        # Pie data and plot
        per = per_RMSE
        per_rest= 100-per
        sizes = [per, per_rest]
        colors_pie = [colors2[4], colors2[2]]
        explode = (0.1, 0)
        
        def custom_autopct(pct):
            return f'{pct:.1f}%' if pct > 20 else ''  # Show only if pct > 20
        
        pie_ax.pie(
            sizes,
            # labels=labels,  # Comment this out or leave for segment label
            colors=colors_pie,
            startangle=90,
            autopct=custom_autopct,
            textprops={'fontsize': 9, 'color': 'black'},
            explode=explode,
            pctdistance=0.35
        )
        
        
        pie_ax.set_aspect('equal')  # Equal aspect ratio ensures circle
        ax.text(0.06, 0.1, 'RMSE<0.15', transform=ax.transAxes,color = 'Black',fontsize=8, fontweight='bold', va='top', ha='left')
        
    else:
        # Inset pie chart in lower right
        pie_ax = inset_axes(ax, width="35%", height="35%", loc='lower right',
                            bbox_to_anchor=(-0.70, 0.1, 1, 1),
                            bbox_transform=ax.transAxes, borderpad=0)
        
        # Pie data and plot
        per = per_chi2
        per_rest= 100-per
        sizes = [per, per_rest]
        colors_pie = [colors2[6], colors2[3]]
        explode = (0.1, 0)
        
        def custom_autopct(pct):
            return f'{pct:.1f}%' if pct > 20 else ''  # Show only if pct > 20
        
        pie_ax.pie(
            sizes,
            # labels=labels,  # Comment this out or leave for segment label
            colors=colors_pie,
            startangle=90,
            autopct=custom_autopct,
            textprops={'fontsize': 9, 'color': 'black'},
            explode=explode,
            pctdistance=0.35
        )
        
        
        pie_ax.set_aspect('equal')  # Equal aspect ratio ensures circle
        ax.text(0.07, 0.1, 'χ2/DoF<5', transform=ax.transAxes,color = 'Black',fontsize=8, fontweight='bold', va='top', ha='left')
    
    ax.text(-0.1, 1.1, panel_labels[i], transform=ax.transAxes,fontsize=14, fontweight='bold', va='top', ha='left')

# Remove the unused subplot
# fig.delaxes(axes[1, 1])

plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/SFigure13_Spatial_Map_SEM_Goodness_Fit.pdf',dpi=300, bbox_inches='tight')
plt.show()
