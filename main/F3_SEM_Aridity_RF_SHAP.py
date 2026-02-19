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
dir_file = f"/home/yangbuw/Program/EmissionIntensity/code/data_code/data/SEM_Path_Coeffs_relative_importance.nc"
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
dir_file = f"/home/yangbuw/Program/EmissionIntensity/code/data_code/data/SEM_global_coeff_025x025_rmAC_2Path_VegIndex_Lag_3_month.nc"
ds2 = xr.open_dataset(dir_file)
C1 = ds2["GFI"]              # goodness-of-fit index
C2 = ds2["CFI"]              # comparative fit index
C3 = ds2["RMSEA"]            # root mean square error of approximation

# Mask 
C1 = C1.where(BF_mask >= 0.001, np.nan)
C2 = C2.where(BF_mask >= 0.001, np.nan)
C3 = C3.where(BF_mask >= 0.001, np.nan)
CFI = C2

CFI =CFI.where(sig_mask)

CFI_flat = CFI.values.flatten()
CFI_clean = CFI_flat[~np.isnan(CFI_flat)]

per = np.sum(CFI_clean>=0.90)/np.sum(~np.isnan(CFI_clean))*100
# ===================================  2nd data ===================================



# ===================================  3rd data ===================================
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
# ===================================  3rd data ===================================



# ===================================  4th data ===================================
# =============== SEM Dominance =============
dir_file = f"/home/yangbuw/Program/EmissionIntensity/code/data_code/data/SEM_Path_Coeffs_relative_importance.nc"
ds2 = xr.open_dataset(dir_file)
ratio_xr = ds2["ratio_xr"]    # Fuel Load
sig_mask = ds2["sig_mask"]    # CC

ratio_xr = ratio_xr.where(sig_mask)

mask_fl = ratio_xr <= 0.3                        # Fuel Load Dominated
mask_co = (ratio_xr > 0.3) & (ratio_xr <= 0.7)   # Co-Dominant
mask_fw = ratio_xr > 0.7                         # Fire Weather Dominated





# Prepare Part-to-Whole 
region_idx =1
AI_mask = AI_class==region_idx
AI_class_use = AI_class.where(AI_mask)
AI_class_use_fl = np.sum(AI_class_use.where(mask_fl))
AI_class_use_co = np.sum(AI_class_use.where(mask_co))
AI_class_use_fw = np.sum(AI_class_use.where(mask_fw))
count_3type = [AI_class_use_fl.values , AI_class_use_co.values, AI_class_use_fw.values]
AI_class_use_total = np.sum(count_3type)
ratio_3type = count_3type/AI_class_use_total 
print(ratio_3type)
arid_use = ratio_3type



region_idx =2
AI_mask = AI_class==region_idx
AI_class_use = AI_class.where(AI_mask)
AI_class_use_fl = np.sum(AI_class_use.where(mask_fl))
AI_class_use_co = np.sum(AI_class_use.where(mask_co))
AI_class_use_fw = np.sum(AI_class_use.where(mask_fw))
count_3type = [AI_class_use_fl.values , AI_class_use_co.values, AI_class_use_fw.values]
AI_class_use_total = np.sum(count_3type)
ratio_3type = count_3type/AI_class_use_total 
print(ratio_3type)
semi_arid_use = ratio_3type


region_idx =3
AI_mask = AI_class==region_idx
AI_class_use = AI_class.where(AI_mask)
AI_class_use_fl = np.sum(AI_class_use.where(mask_fl))
AI_class_use_co = np.sum(AI_class_use.where(mask_co))
AI_class_use_fw = np.sum(AI_class_use.where(mask_fw))
count_3type = [AI_class_use_fl.values , AI_class_use_co.values, AI_class_use_fw.values]
AI_class_use_total = np.sum(count_3type)
ratio_3type = count_3type/AI_class_use_total 
print(ratio_3type)
dry_arid_use = ratio_3type


region_idx =4
AI_mask = AI_class==region_idx
AI_class_use = AI_class.where(AI_mask)
AI_class_use_fl = np.sum(AI_class_use.where(mask_fl))
AI_class_use_co = np.sum(AI_class_use.where(mask_co))
AI_class_use_fw = np.sum(AI_class_use.where(mask_fw))
count_3type = [AI_class_use_fl.values , AI_class_use_co.values, AI_class_use_fw.values]
AI_class_use_total = np.sum(count_3type)
ratio_3type = count_3type/AI_class_use_total 
print(ratio_3type)
humid_arid_use = ratio_3type

labels = ['Hyper Arid/Arid', 'Semi-Arid', 'Dry Sub-humid', 'Humid']
labels2 = ['Fuel Load Dominated', 'Co-Dominated', 'Fire Weather Dominated']
data = {
    'Humid': humid_arid_use*100,
    'Dry Sub-humid': dry_arid_use*100,
    'Semi-Arid': semi_arid_use*100,
    'Hyper Arid/Arid': arid_use*100
}


df = pd.DataFrame(data, index=labels2).T
# ===================================  4th data ===================================

# ===================================  RF + SHAP ===================================
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

#---------------------------------------------- Plot ---------------------------------------------------------
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'
fig = plt.figure(figsize=(6, 9.5))  # Narrower width for left column only
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)

ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())  # Top-left
ax3 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())  # Middle-left
ax4 = fig.add_subplot(gs[2])                                 # Bottom-left (bar chart)


# ========================== Plot 1: SEM Dominance ==========================
diff_use = plot1_data * 100  # Scale to percentage
ax = ax1  # Use ax1 for plot 1
lon_min=-180
lon_max= 180
lat_min=-70
lat_max=90
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.OCEAN, color='white')
# gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--', linewidth=0.4)
gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--')

gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 9, 'color': 'gray'}
gl.ylabel_style = {'size': 9, 'color': 'gray'}

# ---------- Define 3-bin custom colormap ----------
bounds2 = [0, 30, 70, 100]  # Categories: Fuel Load <30%, Co-Dominant 30–70%, Fire Weather >70%
colors2 = ['forestgreen', 'bisque', 'orange']  # Green, beige, orange
# colors2 = ['forestgreen', 'red', 'orange']  # Green, beige, orange
custom_cmap2 = ListedColormap(colors2)
norm2 = BoundaryNorm(bounds2, ncolors=custom_cmap2.N)

mesh = ax.pcolormesh(
    diff_use['longitude'], diff_use['latitude'], diff_use,
    transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto'
)

# Colorbar
def custom_tick_format(x, pos):
    if x == 15:
        return 'Fuel Load Dominated'
    elif x == 50:
        return 'Co-Dominated'
    elif x == 85:
        return 'Fire Weather Dominated'
    else:
        return ''
# cbar_ax = fig.add_axes([0.12, 0.62, 0.35, 0.015])
cbar_ax = ax.inset_axes([0.05, -0.18, 0.9, 0.07])
cbar = plt.colorbar(mesh, cax=cbar_ax, orientation='horizontal', ticks=[15, 50, 85])
cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_format))
cbar.ax.tick_params(labelsize=8.5)

# Inset pie chart in lower right
pie_ax = inset_axes(ax, width="35%", height="35%", loc='lower right',
                    bbox_to_anchor=(-0.70, 0.1, 1, 1),
                    bbox_transform=ax.transAxes, borderpad=0)

# Pie data and plot
sizes = ratio_SEM
labels = ['CFI > 0.9', '']  # Optional: hide label for second slice
colors_pie = ['forestgreen', 'bisque', 'orange'] 
explode = (0.1, 0,0)

def custom_autopct(pct):
    return f'{pct:.0f}%' if pct > 20 else ''  # Show only if pct > 20

pie_ax.pie(
    sizes,
    # labels=labels,  # Comment this out or leave for segment label
    colors=colors_pie,
    startangle=90,
    autopct=custom_autopct,
    textprops={'fontsize': 6, 'color': 'black'},
    explode=explode,
    pctdistance=0.57
)


pie_ax.set_aspect('equal')  # Equal aspect ratio ensures circle


# ========================== Plot 3: Aridity Class Map ==========================
ax = ax3  # Use ax3 for plot 3

ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.OCEAN, color='white')

gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 9, 'color': 'gray'}
gl.ylabel_style = {'size': 9, 'color': 'gray'}

# Categorical colormap setup for Aridity Index Classes
# 1: Hyper Arid + Arid, 2: Semi-Arid, 3: Dry Sub-humid, 4: Humid
bounds2 = [0.5, 1.5, 2.5, 3.5, 4.5]  # Category bins centered at 1, 2, 3, 4
colors2 = ['firebrick', 'orange', 'gold', 'forestgreen']
labels = ['Hyper Arid/Arid', 'Semi-Arid', 'Dry Sub-humid', 'Humid']
custom_cmap2 = ListedColormap(colors2)
norm2 = BoundaryNorm(bounds2, ncolors=custom_cmap2.N)

# Plot reclassified data
mesh1 = ax3.pcolormesh(
    AI_class['longitude'], AI_class['latitude'], AI_class,
    transform=ccrs.PlateCarree(), cmap=custom_cmap2, norm=norm2, shading='auto'
)

# Colorbar setup
# cbar_ax1 = fig.add_axes([0.12, 0.363,  0.35, 0.015])
cbar_ax1 = ax.inset_axes([0.05, -0.18, 0.9, 0.07])
cbar1 = plt.colorbar(mesh1, cax=cbar_ax1, orientation='horizontal', boundaries=bounds2, ticks=[1, 2, 3, 4])
cbar1.ax.set_xticklabels(labels)
# cbar1.set_label('Climate Classification', labelpad=2, fontsize=14)
cbar1.ax.tick_params(labelsize=8.5)


# ========================== Plot 4: Horizontal Bar Chart ==========================
ax = ax4  # Use ax4 for horizontal bar plot

df.index = ['Humid',
            'Dry\nSub-humid',
            'Semi-Arid',
            'Hyper Arid\n & Arid']
ax.set_aspect(10.5)  # Try values between 0.4–0.6 to visually match map proportions



# Custom stacked bar colors
colors = ['forestgreen', 'bisque', 'orange']
left = np.zeros(len(df))  # cumulative base for each bar

# Plot each component one-by-one for full control
for i, col in enumerate(df.columns):
    ax.barh(df.index, df[col], left=left, color=colors[i], label=col)
    left += df[col]

# Axis formatting
ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
ax.tick_params(axis='x', labelsize=9, colors='gray')


# Y-axis tick styling
y_label_colors = ['forestgreen', 'gold', 'orange', 'firebrick']
y_label_colors = ['gray', 'gray', 'gray', 'gray']
for tick_label, color in zip(ax.get_yticklabels(), y_label_colors):
    tick_label.set_color(color)
    tick_label.set_fontsize(9)
    # tick_label.set_fontweight('bold')
    
ax.legend(loc='lower center', bbox_to_anchor=(0.48, -0.35), ncol=2, frameon=False, fontsize=10)

# Optional: Add value text inside bars
for i, (idx, row) in enumerate(df.iterrows()):
    x_offset = 0
    for j, value in enumerate(row):
        if value > 5:  # only annotate meaningful segments
            ax.text(x_offset + value / 2, i, f'{value:.2f}%', va='center', ha='center', fontsize=8.5)
        x_offset += value


# Add subplot letters to each plot panel
letters = ['a', 'b', 'c']
axes_list = [ax1, ax3, ax4]

for i, ax in enumerate(axes_list):
    ax.text(-0.1, 1.1, letters[i], transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')


# plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/Figure3_SEM_Aridity_RF_SHAP.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/Figure3_SEM_Aridity.pdf', dpi=300, bbox_inches='tight')
plt.show()


