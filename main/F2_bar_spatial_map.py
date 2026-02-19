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
    'BA': BA_am,
    'Trend1': BA_slope_percentage,
    'Emission': Emission_am,
    'Trend2': Emission_slope_percentage,
    'EI': EI_am,
    'Trend4': EI_slope_percentage,
    'BA std_err': BA_std_err,
    'EI std_err': EI_std_err,
    'P Value1': BA_p,
    'P Value4': EI_p
}
df       = pd.DataFrame(Table_data, index=Var_name)
df_array = df.to_numpy()


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

EI_contribution      = bar_plot_use[0, :]
BA_contribution      = bar_plot_use[1, :]
observed_bar         = bar_plot_use[2, :]
simulated_bar        = bar_plot_use[3, :]
EI_MOE_uncertainty   = bar_plot_use[4, :]
BA_MOE_uncertainty   = bar_plot_use[5, :]
TOTAL_MOE_uncertainty= bar_plot_use[6, :]
biomes               = ["Total", "SAVA", "BORF", "TEMF"]

# ------- Pie chart -------
EI_plot_ratio = EI_plot_ratio  # SAVA, BORF, TEMF
inner_sizes   = (EI_plot_ratio * 100).tolist()  # %


value       = inner_sizes[0]
percentages = np.array([(39 + 17), 44]) / 100.0
outer_sizes = value * percentages
outer_sizes = np.concatenate([outer_sizes, np.array(inner_sizes[1:])])

outer_sizes1 = np.concatenate([outer_sizes[0:2], [np.sum(outer_sizes[-2:])]])
outer_sizes2 = np.concatenate([
    outer_sizes1[0:1] * 39 / (39 + 17),
    outer_sizes1[0:1] * 17 / (39 + 17),
    [np.sum(outer_sizes1[-2:])]
])

# ============================================================
# ====================== 2. BF & EI spatial maps =============
# ============================================================

# ---------- BF mask ----------
def calc_BF_annual(var_name):
    dir_file = "/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_BF_byBiomes_025x025_2002_2022_monthly.nc"
    ds2 = xr.open_dataset(dir_file)
    C = ds2[var_name]
    region_C = C.sel(longitude=slice(-180, 180),
                     latitude=slice(-90, 90),
                     time=slice(str(startyear_use), str(endyear_use)))
    region_C_yearly = region_C.resample(time='YS').sum()
    return region_C_yearly

BA_data_use_BF = [calc_BF_annual(v) for v in Var_name]
data_use_BF    = BA_data_use_BF[1]  # SAVA
data_area_sum  = data_use_BF.mean(dim=['time'], skipna=True)
BF_mask        = data_area_sum

# ---------- BF Trend ----------
dir_file = "/scratch/yangbuw/Data/GFED5/output_biome/GFED5_BF_SAVA_Trend.nc"
ds2 = xr.open_dataset(dir_file)
trend        = ds2["slope"]
p_values     = ds2["p_values"]
mk_p_values  = ds2["mk_significance"]
significance_mask = (p_values < 0.1) | (mk_p_values < 0.1)
trend_sig    = trend.where(significance_mask)
data_use_plot_BF_trend = trend_sig

# ---------- EI Trend ----------
dir_file = "/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_CO2_EI_Trend_by_Tg_per_Mha.nc"
ds2 = xr.open_dataset(dir_file)
trend        = ds2["slope"]
p_values     = ds2["p_values"]
mk_p_values  = ds2["mk_significance"]
significance_mask = (p_values < 0.1) | (mk_p_values < 0.1)
trend_sig    = trend.where(significance_mask)
data_use_plot_EI_trend = trend_sig


data_use_plot_BF      = data_area_sum.where(BF_mask > 0.0010)
data_use_plot_EI_trend= data_use_plot_EI_trend.where(BF_mask > 0.0010)

diff_use_list = [
    data_use_plot_BF_trend * 100,   # c: BF trend (%/yr)
    data_use_plot_EI_trend * 100    # d: EI trend
]

lon_min, lon_max = -180, 180
lat_min, lat_max = -70, 90
titles = ['Burned Fraction Trend', 'Emission Intensity Trend']

#  ======================   Plotting ============================

fig = plt.figure(figsize=(11, 8))

gs = gridspec.GridSpec(
    2, 2,                      
    width_ratios=[1, 1],        
    height_ratios=[1, 1],      
    wspace=0.13,                
    hspace=0.01                 
)


# -------- (a) stacked bar --------
ax_a = fig.add_subplot(gs[0, 0])

for i, (value, err) in enumerate(zip(EI_contribution, EI_MOE_uncertainty)):
    ax_a.bar(
        biomes[i], value,
        color="bisque",
        label="EI-Driven" if i == 0 else "",
        yerr=err if err != 0 else None,
        capsize=5, ecolor="gray", alpha=1,
        error_kw={'elinewidth': 1, 'capsize': 5} if err != 0 else None
    )

for i, (value, err) in enumerate(zip(BA_contribution, BA_MOE_uncertainty)):
    ax_a.bar(
        biomes[i], value,
        color="forestgreen",
        label="BA-Driven" if i == 0 else "",
        yerr=err if err != 0 else None,
        capsize=5, ecolor="gray", alpha=1,
        error_kw={'elinewidth': 1, 'capsize': 5} if err != 0 else None
    )

color_recon = 'red'


ax_a.plot(
    biomes, simulated_bar,
    label='Reconstructed',
    marker='s', markersize=4,
    markerfacecolor=color_recon, markeredgewidth=2,
    linestyle='none', linewidth=1, color=color_recon
)

ax_a.errorbar(
    biomes, simulated_bar, yerr=TOTAL_MOE_uncertainty + 0.7,
    marker='s', markersize=4, markerfacecolor=color_recon, markeredgewidth=1,
    linestyle='none', linewidth=2, color=color_recon,
    capsize=5, elinewidth=1, zorder=1
)

ax_a.plot(
    biomes, observed_bar,
    label='Observed',
    marker='^', markersize=4,
    markerfacecolor='black', markeredgewidth=2,
    linestyle='none', linewidth=2, color='black'
)

ax_a.axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
ax_a.set_ylim([-45, 45])
ax_a.set_yticks(np.arange(-40, 50, 20))
ax_a.set_yticklabels([f'{y:.0f}' for y in np.arange(-40, 50, 20)],
                     fontsize=10, color='gray')
ax_a.set_ylabel("CO$_2$ Trend (Tg C yr$^{-1}$)", fontsize=12)
ax_a.tick_params(axis='x', labelsize=12, colors='gray')
ax_a.tick_params(axis='y', labelsize=12, colors='gray')
ax_a.legend(bbox_to_anchor=(0.8, 1.0), loc='upper center',
            fontsize=12, frameon=False, ncol=1)

# panel label a
ax_a.text(-0.1, 1.12, 'a', transform=ax_a.transAxes,
          fontsize=18, fontweight='bold', va='top', ha='left')

# -------- (b) donut pie --------
ax_b = fig.add_subplot(gs[0, 1])

inner_colors  = ["orange", "forestgreen", "yellowgreen"]
inner_colors  = ["#F4A657", "forestgreen", "yellowgreen"]
inner_labels  = ['SAVA', 'BORF', 'TEMF']

outer_colors1 = ["sandybrown", "peachpuff"]      # Africa / Others
outer_colors1 = ["#A79DCF", "#D3D0E5"]
outer_labels1 = ['Africa', 'Others']

outer_colors2 = ["darkgray", "lightgray"]        # NHAF / SHAF
outer_colors2 = ["#F7CCA1", "#F5E6D3"]   
outer_labels2 = ['Northern Hemisphere Africa', 'Southern Hemisphere Africa']

startangle = 90

# ============ donut sizes ============
width_inner = 0.4
width_mid   = 0.4
width_outer = 0.4

radius_inner = 0.6
radius_mid   = radius_inner + width_inner + 0.0    # small spacing
radius_outer = radius_mid   + width_mid   + 0.0

# =================== inner donut ===================
wedges, texts, autotexts = ax_b.pie(
    inner_sizes,
    autopct='%1.1f%%',
    startangle=startangle,
    colors=inner_colors,
    radius=radius_inner,
    counterclock=False,
    wedgeprops={'width': width_inner, 'edgecolor': 'white'}
)

# formatting percentage labels
for a in autotexts:
    x, y = a.get_position()
    a.set_position((x * 1.13, y * 1.2))
    a.set_fontsize(10)
    a.set_color('white')
    a.set_weight('bold')

# =================== middle donut (Africa / Others) ===================
wedges2, _ = ax_b.pie(
    outer_sizes1,
    radius=radius_mid,
    colors=outer_colors1,
    labels=None,
    autopct=None,
    startangle=startangle,
    counterclock=False,
    wedgeprops={'width': width_mid, 'edgecolor': 'white'}
)

# Only show first two wedges
for i, w in enumerate(wedges2):
    if i >= 2:
        w.set_alpha(0)

total_outer1 = np.sum(outer_sizes1)
for i, (size, w) in enumerate(zip(outer_sizes1, wedges2)):
    if i >= 2:
        break

    theta1, theta2 = w.theta1, w.theta2
    mid_angle = (theta1 + theta2) * 0.5
    rad = np.deg2rad(mid_angle)

    r = radius_mid - width_mid * 0.5
    x = r * np.cos(rad)
    y = r * np.sin(rad)

    percent = size / inner_sizes[0] * 100
    ax_b.text(
        x, y, f"{percent:.1f}%",
        ha='center', va='center',
        fontsize=10, fontweight='bold', color='white'
    )

# =================== outer donut (NHAF / SHAF) ===================
wedges3, _ = ax_b.pie(
    outer_sizes2,
    radius=radius_outer,
    colors=outer_colors2,
    labels=None,
    autopct=None,
    startangle=startangle,
    counterclock=False,
    wedgeprops={'width': width_outer, 'edgecolor': 'white'}
)

# Only first two wedges
for i, w in enumerate(wedges3):
    if i >= 2:
        w.set_alpha(0)

total_outer2 = np.sum(outer_sizes2)
for i, (size, w) in enumerate(zip(outer_sizes2, wedges3)):
    if i >= 2:
        break

    theta1, theta2 = w.theta1, w.theta2
    mid_angle = (theta1 + theta2) * 0.5
    rad = np.deg2rad(mid_angle)

    r = radius_outer - width_outer * 0.5
    x = r * np.cos(rad)
    y = r * np.sin(rad)

    percent = size / outer_sizes1[0] * 100
    ax_b.text(
        x, y, f"{percent:.1f}%",
        ha='center', va='center',
        fontsize=10, fontweight='bold', color='white'
    )

ax_b.set(aspect="equal")

# =================== Legends ===================
from matplotlib.patches import Patch

# inner
handles_inner = [Patch(facecolor=c) for c in inner_colors]
legend1 = ax_b.legend(
    handles_inner, inner_labels,
    loc='lower left', bbox_to_anchor=(-0.27, -0.12),
    fontsize=10, frameon=False
)
ax_b.add_artist(legend1)

# middle (Africa / Others)
handles_outer1 = [Patch(facecolor=c) for c in outer_colors1]
legend2 = ax_b.legend(
    handles_outer1, outer_labels1,
    loc='lower left', bbox_to_anchor=(0.10, -0.12),
    fontsize=10, frameon=False
)
ax_b.add_artist(legend2)

# outer (NHAF / SHAF)
handles_outer2 = [Patch(facecolor=c) for c in outer_colors2]
legend3 = ax_b.legend(
    handles_outer2, outer_labels2,
    loc='lower left', bbox_to_anchor=(0.45, -0.12),
    fontsize=10, frameon=False
)
ax_b.add_artist(legend3)

# panel label
ax_b.text(
    -0.28, 1.12, 'b', transform=ax_b.transAxes,
    fontsize=18, fontweight='bold', va='top', ha='left'
)



# -------- (c, d) maps --------
ax_c = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
ax_d = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
axs_map = [ax_c, ax_d]

# plt.subplots_adjust(wspace=0.3, hspace=0.15)

for i, ax in enumerate(axs_map):
    diff_use = diff_use_list[i]

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.OCEAN, color='white')

    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--')
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12, 'color': 'gray'}
    gl.ylabel_style = {'size': 12, 'color': 'gray'}

    if i == 0:
        cm = Colormap('colorbrewer:PuOr_r')
        # cm = Colormap('colorbrewer:RdBu_r')

        mpl_cmap = cm.to_mpl()
        bounds2  = np.arange(-1.0, 1.1, 0.2)
        colors2  = mpl_cmap(np.linspace(0.1, 0.9, len(bounds2) - 1))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2    = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)

        def custom_tick_format(x, pos):
            return f'{x:.1f}'

        mesh = ax.pcolormesh(
            diff_use['longitude'], diff_use['latitude'], diff_use,
            transform=ccrs.PlateCarree(),
            cmap=custom_cmap2, norm=norm2, shading='auto'
        )

        cb_ax = inset_axes(ax, width="100%", height="8%", loc='lower center',
                           bbox_to_anchor=(0, -0.21, 1.0, 1),
                           bbox_transform=ax.transAxes, borderpad=0)
        cbar = plt.colorbar(mesh, cax=cb_ax, orientation='horizontal', ticks=bounds2)
        cbar.set_label(r'Burned Fraction Trend (% yr$^{-1}$)',
                       fontsize=12, labelpad=2)
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_format))
        cbar.ax.tick_params(labelsize=12)

        ax.text(-0.1, 1.12, 'c', transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='left')

    else:
        mpl_cmap = cm.to_mpl()
        bounds2  = np.arange(-4.0, 4.1, 0.8)
        colors2  = mpl_cmap(np.linspace(0.1, 0.9, len(bounds2) - 1))
        custom_cmap2 = mcolors.ListedColormap(colors2)
        norm2    = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)

        def custom_tick_format(x, pos):
            return f'{x:.1f}'

        mesh = ax.pcolormesh(
            diff_use['longitude'], diff_use['latitude'], diff_use,
            transform=ccrs.PlateCarree(),
            cmap=custom_cmap2, norm=norm2, shading='auto'
        )

        cb_ax = inset_axes(ax, width="100%", height="8%", loc='lower center',
                           bbox_to_anchor=(0, -0.21, 1.0, 1),
                           bbox_transform=ax.transAxes, borderpad=0)
        cbar = plt.colorbar(mesh, cax=cb_ax, orientation='horizontal', ticks=bounds2)
        cbar.set_label(r'Emission Intensity Trend (% Tg C Mha$^{-1}$ yr$^{-1}$)',
                       fontsize=12, labelpad=2)
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_tick_format))
        cbar.ax.tick_params(labelsize=12)

        ax.text(-0.1, 1.12, 'd', transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='left')

plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/Figure2_combined_whole.pdf',
            dpi=600, bbox_inches='tight')
plt.show()
