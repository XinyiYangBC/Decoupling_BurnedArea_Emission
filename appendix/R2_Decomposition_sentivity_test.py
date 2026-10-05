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

from cmap import Colormap  


plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'

startyear_use = 2002
endyear_use   = 2022


def calculate_trend_line(series):
    x = np.array(range(1, len(series) + 1))
    y = series
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    trend_line = slope * x + intercept
    slope_per  = slope / np.mean(y)
    return slope, p_value, std_err

# ============================================================
# ====================== 1. BA / Emission / EI / DM ==========
# ============================================================

# -------------- BA -----------------
def calculate_Regional_Annual_Mean_BA(var_name):
    dir_file = "/scratch/yangbuw/Data/GFED5/GFED5_1/Output/GFED5_1_BA_m2_byBiomes_025x025_2002_2022_monthly.nc"
    ds2 = xr.open_dataset(dir_file)
    C = ds2[var_name] / 1E6 / 1E6 * 100  # m2 -> km2 -> Mha
    region_C = C.sel(longitude=slice(-180, 180),
                     latitude=slice(-90, 90),
                     time=slice(str(startyear_use), str(endyear_use)))
    region_C_yearly = region_C.resample(time='YS').sum()
    region_C_sum    = region_C_yearly.sum(dim=['latitude', 'longitude'])
    return region_C_sum

# -------------- Emission -----------------
def calculate_Regional_Annual_Mean_Emission(var_name):
    dir_file = "/scratch/yangbuw/Data/GFED5/GFED5_1/Output/GFED5_1_CO2_by_gC_byBiomes_025x025_2002_2022_monthly.nc"
    ds2 = xr.open_dataset(dir_file)
    C = ds2[var_name] / 1E12  # g C -> Pg C
    region_C = C.sel(longitude=slice(-180, 180),
                     latitude=slice(-90, 90),
                     time=slice(str(startyear_use), str(endyear_use)))
    region_C_yearly = region_C.resample(time='YS').sum()
    region_C_sum    = region_C_yearly.sum(dim=['latitude', 'longitude'])
    return region_C_sum

# -------------- DM -----------------
def calculate_Regional_Annual_Mean_DM(var_name):
    dir_file = "/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_DM_by_gDM_byBiomes_025x025_2002_2022_monthly.nc"
    ds2 = xr.open_dataset(dir_file)
    C = ds2[var_name] / 1E15  # g DM -> Pg
    region_C = C.sel(longitude=slice(-180, 180),
                     latitude=slice(-90, 90),
                     time=slice(str(startyear_use), str(endyear_use)))
    region_C_yearly = region_C.resample(time='YS').sum()
    region_C_sum    = region_C_yearly.sum(dim=['latitude', 'longitude'])
    return region_C_sum

Var_name = ['TOTL', 'SAVA', 'AGRI', 'BORF', 'DEFO', 'TEMF', 'PEAT']

# ---- BA ----
BA_data_use = [calculate_Regional_Annual_Mean_BA(v) for v in Var_name]
BA_data_use = np.array(BA_data_use).round(2)

# ---- Emission ----
Emission_data_use = [calculate_Regional_Annual_Mean_Emission(v) for v in Var_name]
Emission_data_use = np.array(Emission_data_use).round(2)


Table_data = {
    'Emission': Emission_am,              #2
    'Trend2': Emission_slope,             #3
    'E std_err': Emission_std_err,             #6
    'P Value': Emission_p,                     #8
}
df       = pd.DataFrame(Table_data, index=Var_name)
df_array_emission = df.to_numpy()

# ---------- Slope ± MOE ----------
n    = endyear_use - startyear_use + 1
degf = n - 2
confidence = 0.95
t_value    = t.ppf((1 + confidence) / 2, degf)


trends_E_uncert   = []

for i in range(7):
    ratio = df_array_emission[i]
    # E
    moe_E  = t_value * ratio[2]

    trends_E_uncert.append(moe_E)

Emission_MOE_use    = np.array(trends_E_uncert)



# ---- DM ----
DM_data_use = [calculate_Regional_Annual_Mean_DM(v) for v in Var_name]
DM_data_use = np.array(DM_data_use).round(2)


def calc_mean_trend(data_array):
    am_use   = []
    slope_use = []
    p_use     = []
    std_err   = []
    for i in range(7):
        series = data_array[i]
        am_use.append(series.mean())
        slp, p_val, se = calculate_trend_line(series)
        slope_use.append(slp)
        p_use.append(p_val)
        std_err.append(se)
    return np.array(am_use), np.array(slope_use), np.array(p_use), np.array(std_err)

BA_am, BA_slope, BA_p, BA_std_err         = calc_mean_trend(BA_data_use)
Emission_am, Emission_slope, Emission_p, Emission_std_err = calc_mean_trend(Emission_data_use)

EI_data_use = Emission_data_use / BA_data_use  # Tg C / Mha
EI_am, EI_slope, EI_p, EI_std_err             = calc_mean_trend(EI_data_use)

DM_am, DM_slope, DM_p, DM_std_err             = calc_mean_trend(DM_data_use)


BA_slope_percentage        = BA_slope * 1.
Emission_slope_percentage  = Emission_slope * 1.
EI_slope_percentage        = EI_slope * 1.
DM_slope_percentage        = DM_slope * 1.

Table_data = {
    'BA': BA_am,                          #0
    'Trend1': BA_slope_percentage,        #1 
    'Emission': Emission_am,              #2
    'Trend2': Emission_slope_percentage,  #3
    'EI': EI_am,                          #4
    'Trend4': EI_slope_percentage,        #5
    'BA std_err': BA_std_err,             #6
    'EI std_err': EI_std_err,             #7
    'P Value1': BA_p,                     #8
    'P Value4': EI_p                      #9
}
df       = pd.DataFrame(Table_data, index=Var_name)
df_array = df.to_numpy()



# ---------- Slope ± MOE ----------
n    = endyear_use - startyear_use + 1
degf = n - 2
confidence = 0.95
t_value    = t.ppf((1 + confidence) / 2, degf)

trends_BA_use      = []
trends_BA_uncert   = []
trends_EI_use      = []
trends_EI_uncert   = []
trends_Total_use   = []

for i in range(7):
    ratio = df_array[i]
    # BA
    moe_BA  = t_value * ratio[6]
    BA_C    = ratio[1] * ratio[4]
    BA_uncert = moe_BA
    trends_BA_use.append(BA_C)
    trends_BA_uncert.append(BA_uncert)

    # EI
    moe_EI  = t_value * ratio[7]
    EI_C    = ratio[5] * ratio[0]
    EI_uncert = moe_EI
    trends_EI_use.append(EI_C)
    trends_EI_uncert.append(EI_uncert)

    Total_C = EI_C + BA_C
    trends_Total_use.append(Total_C)

trends_BA_use    = np.array(trends_BA_use)
trends_BA_uncert = np.array(trends_BA_uncert)
trends_EI_use    = np.array(trends_EI_use)
trends_EI_uncert = np.array(trends_EI_uncert)
trends_Total_use = np.array(trends_Total_use)


global_EI_C         = np.sum(trends_EI_use[1:])
trends_EI_use[0]    = global_EI_C
EI_C_Table          = trends_EI_use.copy()

global_BA_C         = np.sum(trends_BA_use[1:])
trends_BA_use[0]    = global_BA_C
BA_C_Table          = trends_BA_use.copy()

trends_EI_uncert_use   = np.sum(trends_EI_uncert[1:])
trends_EI_uncert[0]    = trends_EI_uncert_use
EI_uncert_Table        = trends_EI_uncert.copy()

trends_BA_uncert_use   = np.sum(trends_BA_uncert[1:])
trends_BA_uncert[0]    = trends_BA_uncert_use
BA_uncert_Table        = trends_BA_uncert.copy()

BA_C_Table_UP   = BA_C_Table + BA_uncert_Table
BA_C_Table_LOW  = BA_C_Table - BA_uncert_Table

net_trend_observed = np.array(Emission_slope_percentage)
Whole_uncert_Table = BA_uncert_Table + EI_uncert_Table

net_trend_mean     = EI_C_Table + BA_C_Table
net_trend_mean_up  = EI_C_Table + BA_C_Table_UP
net_trend_mean_low = EI_C_Table + BA_C_Table_LOW

Final_table_trend_contribution = np.array([
    EI_C_Table,
    EI_uncert_Table,
    BA_C_Table,
    BA_uncert_Table,
    net_trend_observed,
    net_trend_mean,
    net_trend_mean_up,
    net_trend_mean_low,
    Whole_uncert_Table
])


net_trend_table = Final_table_trend_contribution[4:].copy()
net_trend_table_4biome = np.array([
    net_trend_table[:, 0],  # Total
    net_trend_table[:, 1],  # SAVA
    net_trend_table[:, 3],  # BORF
    net_trend_table[:, 5]   # TEMF
]).T

observed   = net_trend_table_4biome[0, :].copy()
simulated1 = net_trend_table_4biome[1, :].copy()

EI_plot       = [Final_table_trend_contribution[0][1],
                 Final_table_trend_contribution[0][3],
                 Final_table_trend_contribution[0][5]]
EI_plot       = np.array(EI_plot)
EI_plot_sum   = EI_plot.sum()
EI_plot_ratio = EI_plot / EI_plot_sum


bar_trend_table_4biome = np.array([
    Final_table_trend_contribution[:, 0],
    Final_table_trend_contribution[:, 1],
    Final_table_trend_contribution[:, 3],
    Final_table_trend_contribution[:, 5]
]).T


bar_plot_use = np.array([
    bar_trend_table_4biome[0, :],  # EI
    bar_trend_table_4biome[2, :],  # BA
    bar_trend_table_4biome[4, :],  # observed
    bar_trend_table_4biome[5, :],  # reconstructed
    bar_trend_table_4biome[1, :],  # EI_uncert
    bar_trend_table_4biome[3, :],  # BA_uncert
    bar_trend_table_4biome[8, :]   # total_uncert
])


bar_trend_table_4biome_firstorder = bar_plot_use[3]


# ============================================================
# ============== Joint BA-EI interaction term ================
# ============================================================

# BA_data_use: shape = (7, 21)
# EI_data_use: shape = (7, 21)

# 1. anomalies relative to 2002–2022 mean
BA_anom = BA_data_use - BA_am[:, None]
EI_anom = EI_data_use - EI_am[:, None]

# 2. yearly joint term: BA'(t) * EI'(t)
Joint_data_use = BA_anom * EI_anom

# 3. long-term OLS trend of the joint term
Joint_am, Joint_slope, Joint_p, Joint_std_err = calc_mean_trend(Joint_data_use)

bar_trend_table_4biome_highorder = np.array([
    Joint_slope[0]+bar_plot_use[3,0],
    Joint_slope[1]+bar_plot_use[3,1],
    Joint_slope[3]+bar_plot_use[3,2],
    Joint_slope[5]+bar_plot_use[3,3],
]).T
bar_trend_table_4biome_highorder



Emission_MOE_use_4biome =  np.array([
    Emission_MOE_use[0],
    Emission_MOE_use[1],
    Emission_MOE_use[3],
    Emission_MOE_use[5]
]).T
Emission_MOE_use_4biome



# Calculate relative contribution of  High-order term to BA-/ EI-

EI_Contribution = bar_plot_use[0]
BA_Contribution = bar_plot_use[1]
Joint_Contribution = np.array([
    Joint_slope[0],
    Joint_slope[1],
    Joint_slope[3],
    Joint_slope[5]
]).T


R_joint = (
    np.abs(Joint_Contribution)
    /
    (np.abs(BA_Contribution) + np.abs(EI_Contribution))
    * 100
)

print("Relative joint contribution (%)")
for biome, value in zip(['Total', 'SAVA', 'BORF', 'TEMF'], R_joint):
    print(f"{biome}: {value:.2f}%")



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ==============================
# Data
# ==============================
biomes = ['Total', 'SAVA', 'BORF', 'TEMF']

observed = bar_plot_use[2]
first_order = bar_trend_table_4biome_firstorder
with_joint = bar_trend_table_4biome_highorder

# Relative joint contribution (%)
# R_joint should already be calculated as:
# |C_joint| / (|C_BA| + |C_EI|) * 100


# ==============================
# Figure settings
# ==============================
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'

fig, (ax_a, ax_b) = plt.subplots(
    1, 2,
    figsize=(12, 4.5)
)

x = np.arange(len(biomes))


# ============================================================
# Panel a
# Observed vs reconstructed trends
# ============================================================

# Reconstruction including joint BA-EI term
ax_a.plot(
    x,
    with_joint,
    marker='o',
    markersize=10,
    linestyle='none',
    markerfacecolor='none',
    markeredgecolor='blue',
    markeredgewidth=1.5,
    label='Reconstruction (including joint BA–EI term)',
    zorder=5
)

# First-order reconstruction
ax_a.plot(
    x,
    first_order,
    marker='^',
    markersize=12,
    linestyle='none',
    color='darkorange',
    label='Reconstruction (first-order)',
    zorder=4
)

# Observed trend
ax_a.bar(
    x,
    observed,
    width=0.45,
    color='lightgray',
    edgecolor='gray',
    linewidth=1.0,
    label='Observed',
    zorder=2
)

# Zero line
ax_a.axhline(
    0,
    color='0.7',
    linestyle='--',
    linewidth=0.8,
    zorder=0
)

# X-axis
ax_a.set_xticks(x)
ax_a.set_xticklabels(biomes)

# Y-axis
ax_a.set_ylabel(
    "CO$_2$ Trend (Tg C yr$^{-1}$)",
    fontsize=14
)
ax_a.set_ylim(-30, 20)

# Tick size
ax_a.tick_params(
    axis='both',
    labelsize=14
)

# Legend order
handles, labels = ax_a.get_legend_handles_labels()

# Current:
# 0 = including joint term
# 1 = first-order
# 2 = observed
order = [2, 1, 0]

ax_a.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    frameon=False,
    fontsize=12,
    loc='lower right'
)

# Panel label
ax_a.text(
    -0.10, 1.03,
    'a',
    transform=ax_a.transAxes,
    fontsize=18,
    fontweight='bold'
)


# ============================================================
# Panel b
# Relative joint contribution
# ============================================================

bars = ax_b.bar(
    x,
    R_joint,
    width=0.55,
    color='lightgray',
    edgecolor='gray',
    linewidth=1.0
)

# Percentage labels
for bar, value in zip(bars, R_joint):
    ax_b.text(
        bar.get_x() + bar.get_width()/2,
        value + 1.5,
        f'{value:.2f}%',
        ha='center',
        va='bottom',
        fontsize=12
    )

# X-axis
ax_b.set_xticks(x)
ax_b.set_xticklabels(biomes)

# Y-axis
ax_b.set_ylim(0, 100)
ax_b.set_yticks(np.arange(0, 101, 20))

# Add % to y-axis tick labels
ax_b.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

ax_b.set_ylabel(
    'Relative Magnitude of Joint BA–EI Term',
    fontsize=14
)
ax_b.tick_params(
    axis='both',
    labelsize=14
)

# Panel label
ax_b.text(
    -0.10, 1.03,
    'b',
    transform=ax_b.transAxes,
    fontsize=18,
    fontweight='bold'
)


# plt.tight_layout()
# plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/Response_1_4regions_Decomposition_sensetivetest.pdf', dpi=300, bbox_inches='tight')
plt.show()




# ======================================================

import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Data
# ==============================
biomes = ['Total', 'SAVA', 'BORF', 'TEMF']

observed = bar_plot_use[2]

first_order = bar_trend_table_4biome_firstorder
with_joint = bar_trend_table_4biome_highorder

# ==============================
# Plot
# ==============================
x = np.arange(len(biomes))
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'

fig, ax = plt.subplots(figsize=(6.5, 4.5))

ax.plot(
    x,
    with_joint,
    marker='o',
    markersize=10,
    linestyle='none',
    markerfacecolor='none',
    markeredgecolor='blue',
    markeredgewidth=1.5,
    label='Reconstruction (including second-order interaction)',
    zorder=5
)

ax.plot(
    x,
    first_order,
    marker='^',
    markersize=12,
    linestyle='none',
    color='darkorange',
    label='Reconstruction (first-order only)',
    zorder=3
)


ax.bar(
    x,
    observed,
    width=0.45,
    color='lightgray',
    edgecolor='gray',
    linewidth=1.0,

    capsize=4,
    ecolor='black',
    label='Observed',
    zorder=2
)



# zero line
ax.axhline(
    0,
    color='0.7',
    linestyle='--',
    linewidth=0.8,
    zorder=0
)

# x-axis
ax.set_xticks(x)
ax.set_xticklabels(biomes)



# Get legend handles and labels
handles, labels = ax.get_legend_handles_labels()

# Current order: with_joint, first_order, observed
# New order: observed, first_order, with_joint
order = [2, 1, 0]

ax.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    frameon=False,
    fontsize=14,
    loc='lower right',
    bbox_to_anchor=(0.98, 0.02)
)

# y-axis
ax.set_ylabel("CO$_2$ Trend (Tg C yr$^{-1}$)", fontsize=14)
ax.set_ylim(-30, 10)
ax.set_ylim(-30, 20)
# appearance
ax.tick_params(axis='both', labelsize=14)

# plt.tight_layout()
plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/Response_1_4regions_Decomposition_sensetivetest.pdf', dpi=300, bbox_inches='tight')
plt.show()



