import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats
from scipy.stats import linregress, pearsonr, t

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator

startyear_use       = 2002;    # for calculating and plotting
endyear_use         = 2022;

# ============================= BA ===============================
def calculate_Regional_Annual_Mean(var_name):
    dir_file = f"/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_BA_m2_byBiomes_025x025_2002_2022_monthly.nc"
    ds2 = xr.open_dataset(dir_file)
    C = ds2[var_name] /1E6/1E6*100 # m to km to Mha
    region_C = C.sel(longitude=slice(-180, 180), latitude=slice(-90, 90), time=slice(str(startyear_use), str(endyear_use)))
    region_C_yearly = region_C.resample(time='YS').sum()
    region_C_sum = region_C_yearly.sum(dim=['latitude', 'longitude'])
    return region_C_sum

# the Var_name is in order, from largest to lowest
Var_name = ['TOTL', 'SAVA', 'AGRI', 'BORF', 'DEFO', 'TEMF', 'PEAT']
data_use = []
for i in range(len(Var_name)):
    current_var_name = Var_name[i]
    data_use.append(calculate_Regional_Annual_Mean(current_var_name))

ratio_use = []
for i in range(1, 7):  # Loop through indices 1 to 5
    #print(i)
    ratio = data_use[i].mean() / data_use[0].mean()  # Calculate ratio for the current biome
    ratio_use.append(ratio)

biome_names = Var_name[1:]  # Exclude "TOTL"
BA_annual_data = pd.DataFrame({biome: data_use[i].values for i, biome in enumerate(biome_names, start=1)},
                            index=data_use[1].time.values)

BA_ratio_use = ratio_use


# ============================= Emission ===============================
def calculate_Regional_Annual_Mean(var_name):
    dir_file = f"/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_CO2_by_gC_byBiomes_025x025_2002_2022_monthly.nc"
    ds2 = xr.open_dataset(dir_file)
    C = ds2[var_name] /1E15 # g C to Pg C
    region_C = C.sel(longitude=slice(-180, 180), latitude=slice(-90, 90), time=slice(str(startyear_use), str(endyear_use)))
    region_C_yearly = region_C.resample(time='YS').sum()
    region_C_sum = region_C_yearly.sum(dim=['latitude', 'longitude'])
    return region_C_sum

# the Var_name is in order, from largest to lowest
Var_name = ['TOTL', 'SAVA', 'AGRI', 'BORF', 'DEFO', 'TEMF', 'PEAT']
data_use = []
for i in range(len(Var_name)):
    current_var_name = Var_name[i]
    data_use.append(calculate_Regional_Annual_Mean(current_var_name))

ratio_use = []
for i in range(1, 7):  # Loop through indices 1 to 5
    #print(i)
    ratio = data_use[i].mean() / data_use[0].mean()  # Calculate ratio for the current biome
    ratio_use.append(ratio)

biome_names = Var_name[1:]  # Exclude "TOTL"
Emission_annual_data = pd.DataFrame({biome: data_use[i].values for i, biome in enumerate(biome_names, start=1)},
                            index=data_use[1].time.values)

Emission_ratio_use = ratio_use



# --------------------  Plot ---------------------------------
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 11), sharex=False)

# Define custom colors for biomes
custom_colors = {
    'SAVA': 'orange',
    'AGRI': 'green',
    'BORF': 'red',
    'DEFO': 'blue',
    'TEMF': 'pink',
    'PEAT': 'gray'
}

# ----------- Subplot A: Burned Area (BA) -------------------
bottom = np.zeros(len(BA_annual_data.index))
width = 150

for biome in biome_names:
    ax1.bar(BA_annual_data.index, BA_annual_data[biome], width=width, label=biome,
            bottom=bottom, color=custom_colors.get(biome, 'gray'))
    bottom += BA_annual_data[biome].values

#ax1.set_title("Annual Burned Area by Biome (2002–2022)", fontsize=18)
ax1.set_ylim([0, 1000])
ax1.set_yticks(np.arange(0, 1001, 200))
ax1.set_yticklabels([f'{y:.0f}' for y in np.arange(0, 1001, 200)], fontsize=16,color='black')
ax1.set_ylabel(r'Burned Area (Mha $\mathrm{yr}^{-1}$)', fontsize=16)
ax1.tick_params(axis='x', labelsize=16)

ax1.legend(handles=[mpatches.Patch(color=custom_colors[b], label=f"{b}: {r * 100:.1f}%") 
                    for b, r in zip(biome_names, BA_ratio_use)],
           loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=14, frameon=False, ncol=3)



# ----------- Subplot B: Emission (CO2) -------------------
bottom = np.zeros(len(Emission_annual_data.index))

for biome in biome_names:
    ax2.bar(Emission_annual_data.index, Emission_annual_data[biome], width=width, label=biome,
            bottom=bottom, color=custom_colors.get(biome, 'gray'))
    bottom += Emission_annual_data[biome].values

ax2.set_ylim([0, 4.0])
yticks = np.arange(0, 4.1, 0.5)
ax2.set_yticks(yticks)
ax2.set_yticklabels([f'{float(y):.1f}' for y in yticks], fontsize=16, color='black')
ax2.set_ylabel(r'Fire CO2 Emission (Pg C $\mathrm{yr}^{-1}$)', fontsize=16, color='black')
#ax2.set_xlabel("Year", fontsize=16)
ax2.tick_params(axis='x', labelsize=16)


ax2.legend(handles=[mpatches.Patch(color=custom_colors[b], label=f"{b}: {r * 100:.1f}%") 
                    for b, r in zip(biome_names, ratio_use)],
           loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=14, frameon=False, ncol=3)

# Label subplots A, B, C, D
letters = ['a', 'b']
axs = [ax1, ax2]
for i, ax in enumerate(axs):
    ax.text(-0.08, 1.1, letters[i], transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/SFigure_1_GFED5_Beta_BA_Emission_by_biome_Bar_TimeSeries.tif', dpi=300, bbox_inches='tight')
plt.show()