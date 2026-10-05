import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import os
from multiprocessing import Pool, cpu_count

from scipy import stats
from scipy.stats import linregress, pearsonr, t
from scipy.stats import kendalltau

# plotting package
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from cmap import Colormap


# ============================================================
# Settings
# ============================================================

startyear_use = 2001
endyear_use   = 2019

bin_width = 10
nbin = int(100 / bin_width)

p_threshold = 0.05
BA_threshold = 0.01      # Mha yr-1


# ============================================================
# Land cover fraction
# ============================================================

dir_file = "/scratch/yangbuw/Data/ESA_CCI/WeiLi/ESA_CCI_PFT_Tree_Shrub_Grass_1992_2022_025x025.nc"
ds_lc = xr.open_dataset(dir_file)

tree = (ds_lc["TREE"] * 1E-2).sel(longitude=slice(-180, 180), latitude=slice(-90, 90), time=slice(str(startyear_use), str(endyear_use)))
shrub = (ds_lc["SHRUB"] * 1E-2).sel(longitude=slice(-180, 180), latitude=slice(-90, 90), time=slice(str(startyear_use), str(endyear_use)))
grass = (ds_lc["GRASS"] * 1E-2).sel(longitude=slice(-180, 180), latitude=slice(-90, 90), time=slice(str(startyear_use), str(endyear_use)))


# ============================================================
# Normalize to vegetated area
# ============================================================

veg_total = tree + shrub + grass

tree_frac = tree / veg_total
shrub_frac = shrub / veg_total
grass_frac = grass / veg_total

tree_frac = tree_frac.where(veg_total > 0)
shrub_frac = shrub_frac.where(veg_total > 0)
grass_frac = grass_frac.where(veg_total > 0)


# ============================================================
# Mean vegetation composition
# ============================================================

tree_mean = tree_frac.mean(dim="time", skipna=True)
shrub_mean = shrub_frac.mean(dim="time", skipna=True)
grass_mean = grass_frac.mean(dim="time", skipna=True)

veg_mean = tree_mean + shrub_mean + grass_mean

tree_mean = tree_mean / veg_mean
shrub_mean = shrub_mean / veg_mean
grass_mean = grass_mean / veg_mean


# ============================================================
# Ternary bin function
# ============================================================

def ternary_bin(tree_value, shrub_value, grass_value, nbin=10):

    frac = np.column_stack([tree_value, shrub_value, grass_value])

    scaled = frac * nbin
    base = np.floor(scaled).astype(int)
    remainder = scaled - base

    need = nbin - base.sum(axis=1)

    result = base.copy()

    order = np.argsort(-remainder, axis=1)
    rows = np.arange(len(result))

    mask = need >= 1
    result[rows[mask], order[mask, 0]] += 1

    mask = need >= 2
    result[rows[mask], order[mask, 1]] += 1

    return result


# ============================================================
# Flatten vegetation
# ============================================================

tree_temp = tree_mean.values.flatten()
shrub_temp = shrub_mean.values.flatten()
grass_temp = grass_mean.values.flatten()


# ============================================================
# BA data
# ============================================================

dir_file_BA = "/scratch/yangbuw/Data/GFED5/GFED5_1/Output/GFED5_1_BA_m2_byBiomes_025x025_2002_2022_monthly.nc"
ds_BA = xr.open_dataset(dir_file_BA)

# m2 -> Mha
BA_monthly = ds_BA["TOTL"] / 1E10

BA_monthly = BA_monthly.sel(
    longitude=slice(-180, 180),
    latitude=slice(-90, 90),
    time=slice(str(startyear_use), str(endyear_use))
)

BA = BA_monthly.resample(time="YS").sum()


# ============================================================
# CO2 emission data
# ============================================================

dir_file_Emission = "/scratch/yangbuw/Data/GFED5/GFED5_1/Output/GFED5_1_CO2_by_gC_byBiomes_025x025_2002_2022_monthly.nc"
ds_Emission = xr.open_dataset(dir_file_Emission)

# g C -> Tg C
Emission_monthly = ds_Emission["TOTL"] / 1E12

Emission_monthly = Emission_monthly.sel(
    longitude=slice(-180, 180),
    latitude=slice(-90, 90),
    time=slice(str(startyear_use), str(endyear_use))
)

Emission = Emission_monthly.resample(time="YS").sum()


# ============================================================
# Make sure BA and emission have same years
# ============================================================

BA, Emission = xr.align(BA, Emission, join="inner")

years = BA.time.dt.year.values

print("Years used:", years[0], "-", years[-1])


# ============================================================
# Valid spatial mask
# ============================================================

BA_mean_grid = BA.mean(dim="time", skipna=True)
Emission_mean_grid = Emission.mean(dim="time", skipna=True)

ba_temp_mean = BA_mean_grid.values.flatten()
emission_temp_mean = Emission_mean_grid.values.flatten()

valid = (
    np.isfinite(tree_temp) &
    np.isfinite(shrub_temp) &
    np.isfinite(grass_temp) &
    np.isfinite(ba_temp_mean) &
    np.isfinite(emission_temp_mean)
)

tree_temp = tree_temp[valid]
shrub_temp = shrub_temp[valid]
grass_temp = grass_temp[valid]


# ============================================================
# Assign ternary bins
# ============================================================

bins = ternary_bin(tree_temp, shrub_temp, grass_temp, nbin=nbin)

tree_bin = bins[:, 0]
shrub_bin = bins[:, 1]
grass_bin = bins[:, 2]


# ============================================================
# Annual BA and emission for each ternary bin
# ============================================================

BA_bin_yearly = np.full((len(years), nbin + 1, nbin + 1, nbin + 1), np.nan)
Emission_bin_yearly = np.full((len(years), nbin + 1, nbin + 1, nbin + 1), np.nan)

for tt in range(len(years)):

    ba_temp_year = BA.isel(time=tt).values.flatten()
    emission_temp_year = Emission.isel(time=tt).values.flatten()

    ba_temp_year = ba_temp_year[valid]
    emission_temp_year = emission_temp_year[valid]

    for i in range(nbin + 1):
        for j in range(nbin + 1):

            k = nbin - i - j

            if k < 0:
                continue

            mask_bin = (tree_bin == i) & (shrub_bin == j) & (grass_bin == k)

            ba_vals = ba_temp_year[mask_bin]
            emission_vals = emission_temp_year[mask_bin]

            if np.any(np.isfinite(ba_vals)):
                BA_bin_yearly[tt, i, j, k] = np.nansum(ba_vals)

            if np.any(np.isfinite(emission_vals)):
                Emission_bin_yearly[tt, i, j, k] = np.nansum(emission_vals)


# ============================================================
# Mean annual BA
# ============================================================

BA_bin_mean = np.nanmean(BA_bin_yearly, axis=0)


# ============================================================
# Annual EI
# ============================================================

EI_bin_yearly = np.full((len(years), nbin + 1, nbin + 1, nbin + 1), np.nan)

for tt in range(len(years)):
    for i in range(nbin + 1):
        for j in range(nbin + 1):

            k = nbin - i - j

            if k < 0:
                continue

            BA_temp = BA_bin_yearly[tt, i, j, k]
            Emission_temp = Emission_bin_yearly[tt, i, j, k]

            if np.isfinite(BA_temp) and np.isfinite(Emission_temp) and BA_temp > 0:
                EI_bin_yearly[tt, i, j, k] = Emission_temp / BA_temp


# ============================================================
# Mean EI
# ============================================================

EI_bin_mean = np.nanmean(EI_bin_yearly, axis=0)


# ============================================================
# EI trend
# ============================================================

EI_bin_trend = np.full((nbin + 1, nbin + 1, nbin + 1), np.nan)
EI_bin_relative_trend = np.full((nbin + 1, nbin + 1, nbin + 1), np.nan)
EI_bin_p = np.full((nbin + 1, nbin + 1, nbin + 1), np.nan)

for i in range(nbin + 1):
    for j in range(nbin + 1):

        k = nbin - i - j

        if k < 0:
            continue

        EI_ts = EI_bin_yearly[:, i, j, k]

        valid_year = np.isfinite(EI_ts)

        if np.sum(valid_year) >= 5:

            slope, intercept, r, p, se = linregress(years[valid_year], EI_ts[valid_year])

            EI_bin_trend[i, j, k] = slope
            EI_bin_p[i, j, k] = p

            if np.isfinite(EI_bin_mean[i, j, k]) and EI_bin_mean[i, j, k] > 0:
                EI_bin_relative_trend[i, j, k] = slope / EI_bin_mean[i, j, k] * 100


# ============================================================
# Mask low-fire bins
# ============================================================

EI_bin_trend[BA_bin_mean < BA_threshold] = np.nan
EI_bin_relative_trend[BA_bin_mean < BA_threshold] = np.nan


# ============================================================
# Print range
# ============================================================

print("Absolute EI trend min =", np.nanmin(EI_bin_trend))
print("Absolute EI trend max =", np.nanmax(EI_bin_trend))

print("Relative EI trend min =", np.nanmin(EI_bin_relative_trend))
print("Relative EI trend max =", np.nanmax(EI_bin_relative_trend))


# ============================================================
# Plot relative EI trend
# ============================================================

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'

fig, ax = plt.subplots(figsize=(6, 5))

height = np.sqrt(3) / 2

patches = []
values = []
gray_patches = []


for i in range(nbin):
    for j in range(nbin - i):

        k = nbin - i - j

        # ====================================================
        # Upward triangle
        # ====================================================

        tree1 = i / nbin
        shrub1 = j / nbin

        tree2 = (i + 1) / nbin
        shrub2 = j / nbin

        tree3 = i / nbin
        shrub3 = (j + 1) / nbin

        x1 = tree1 + 0.5 * shrub1
        y1 = height * shrub1

        x2 = tree2 + 0.5 * shrub2
        y2 = height * shrub2

        x3 = tree3 + 0.5 * shrub3
        y3 = height * shrub3

        polygon = Polygon([[x1, y1], [x2, y2], [x3, y3]], closed=True)

        value_use = EI_bin_relative_trend[i, j, k]
        p_use = EI_bin_p[i, j, k]

        if np.isfinite(value_use):

            if (not np.isfinite(p_use)) or (p_use >= p_threshold):
                gray_patches.append(polygon)

            else:
                patches.append(polygon)
                values.append(value_use)


        # ====================================================
        # Downward triangle
        # ====================================================

        if (i + j) < (nbin - 1):

            tree1 = (i + 1) / nbin
            shrub1 = j / nbin

            tree2 = (i + 1) / nbin
            shrub2 = (j + 1) / nbin

            tree3 = i / nbin
            shrub3 = (j + 1) / nbin

            x1 = tree1 + 0.5 * shrub1
            y1 = height * shrub1

            x2 = tree2 + 0.5 * shrub2
            y2 = height * shrub2

            x3 = tree3 + 0.5 * shrub3
            y3 = height * shrub3

            polygon = Polygon([[x1, y1], [x2, y2], [x3, y3]], closed=True)

            ii = i + 1
            jj = j
            kk = nbin - (i + 1) - j

            value_use = EI_bin_relative_trend[ii, jj, kk]
            p_use = EI_bin_p[ii, jj, kk]

            if np.isfinite(value_use):

                if (not np.isfinite(p_use)) or (p_use >= p_threshold):
                    gray_patches.append(polygon)

                else:
                    patches.append(polygon)
                    values.append(value_use)


values = np.array(values)

print("Significant EI relative trend min =", np.nanmin(values))
print("Significant EI relative trend max =", np.nanmax(values))


# ============================================================
# Color scale
# ============================================================

levels = np.arange(-4, 4.1, 0.1)

cm = Colormap('colorbrewer:RdYlBu')
# cm = Colormap('colorbrewer:RdBu')
cmap = cm.reversed().to_mpl()

norm = mcolors.BoundaryNorm(
    boundaries=levels,
    ncolors=cmap.N,
    clip=True
)


# ============================================================
# Gray nonsignificant patches
# ============================================================

pc_gray = PatchCollection(
    gray_patches,
    facecolor="lightgray",
    edgecolor="none"
)

ax.add_collection(pc_gray)


# ============================================================
# Significant colored patches
# ============================================================

pc = PatchCollection(
    patches,
    cmap=cmap,
    norm=norm,
    edgecolor="none"
)

pc.set_array(values)
ax.add_collection(pc)


# ============================================================
# Triangle boundary
# ============================================================

ax.plot([0, 1, 0.5, 0], [0, 0, height, 0], color="black", linewidth=1.5)


# ============================================================
# Tree fraction ticks
# ============================================================

for value in range(0, 101, 20):
    frac = value / 100
    ax.text(frac, -0.055, str(value), ha="center", va="top", fontsize=13)

ax.text(0.5, -0.1, "Tree fraction (%)", ha="center", va="top", fontsize=14)


# ============================================================
# Grass fraction ticks
# ============================================================

for value in range(0, 101, 20):

    grass_p = value / 100
    shrub_p = 1 - grass_p

    x = 0.5 * shrub_p
    y = height * shrub_p

    ax.text(x - 0.01, y, str(value), ha="right", va="center", rotation=60, fontsize=13)

ax.text(0.12, height * 0.53, "Grass fraction (%)", rotation=60, ha="center", va="center", fontsize=14)


# ============================================================
# Shrub fraction ticks
# ============================================================

for value in range(0, 101, 20):

    shrub_p = value / 100
    tree_p = 1 - shrub_p

    x = tree_p + 0.5 * shrub_p
    y = height * shrub_p

    ax.text(x, y, str(value), ha="left", va="center", rotation=-60, fontsize=13)

ax.text(0.88, height * 0.53, "Shrub fraction (%)", rotation=-60, ha="center", va="center", fontsize=14)


# ============================================================
# Colorbar
# ============================================================

cbar = plt.colorbar(pc, ax=ax, orientation="horizontal", fraction=0.055, pad=0.05)

cbar.set_ticks([-4,  0,  4])
cbar.ax.tick_params(labelsize=13)
cbar.ax.minorticks_off()

cbar.set_label(r"CO$_2$ Emission Intensity Trend (% yr$^{-1}$)", fontsize=14)


# ============================================================
# Panel label
# ============================================================

ax.text(-0.08, height, "f", fontsize=19, fontweight="bold")


# ============================================================
# Figure settings
# ============================================================

ax.set_xlim(-0.08, 1.08)
ax.set_ylim(-0.13, height + 0.04)
ax.set_aspect("equal")
ax.axis("off")

plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/response_Ternary_EI_relative_trend_GFED5_025.pdf', dpi=300, bbox_inches='tight')

plt.show()
