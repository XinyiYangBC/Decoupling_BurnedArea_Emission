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


startyear_use = 2001
endyear_use   = 2019


# ========================== Land cover fraction ======================

dir_file = "/scratch/yangbuw/Data/ESA_CCI/WeiLi/ESA_CCI_PFT_Tree_Shrub_Grass_1992_2022_025x025.nc"
ds2 = xr.open_dataset(dir_file)
data = ds2["TREE"] * 1E-2
tree = data.sel(longitude=slice(-180, 180), latitude=slice(-90, 90), time=slice(str(startyear_use), str(endyear_use)))
data = ds2["SHRUB"] * 1E-2
shrub = data.sel(longitude=slice(-180, 180), latitude=slice(-90, 90), time=slice(str(startyear_use), str(endyear_use)))
data = ds2["GRASS"] * 1E-2
grass = data.sel(longitude=slice(-180, 180), latitude=slice(-90, 90), time=slice(str(startyear_use), str(endyear_use)))


# ==================== Normalize to vegetated area ====================

veg_total = tree + shrub + grass

tree_frac  = tree / veg_total
shrub_frac = shrub / veg_total
grass_frac = grass / veg_total

tree_frac  = tree_frac.where(veg_total > 0)
shrub_frac = shrub_frac.where(veg_total > 0)
grass_frac = grass_frac.where(veg_total > 0)


# ==================== Mean vegetation composition ====================

tree_mean  = tree_frac.mean(dim="time", skipna=True)
shrub_mean = shrub_frac.mean(dim="time", skipna=True)
grass_mean = grass_frac.mean(dim="time", skipna=True)

veg_mean = tree_mean + shrub_mean + grass_mean

tree_mean  = tree_mean / veg_mean
shrub_mean = shrub_mean / veg_mean
grass_mean = grass_mean / veg_mean

# ====================== mask; do not change========================================
dir_file = "/scratch/yangbuw/Data/GFED5/GFED5_1/Output/GFED5_1_BA_m2_byBiomes_025x025_2002_2022_monthly.nc"
ds2 = xr.open_dataset(dir_file)

data = ds2["TOTL"]/ 1E6 / 1E4       # km2 -> Mha

data_selected = data.sel(
    longitude=slice(-180, 180),
    latitude=slice(-90, 90),
    time=slice(str(startyear_use), str(endyear_use))
)

BA = data_selected.resample(time="YS").sum()

BA_mean = BA.mean(dim="time", skipna=True)
bin_width = 10
nbin = int(100 / bin_width)


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


tree_temp  = tree_mean.values.flatten()
shrub_temp = shrub_mean.values.flatten()
grass_temp = grass_mean.values.flatten()
ba_temp    = BA_mean.values.flatten()


valid = np.isfinite(tree_temp) & np.isfinite(shrub_temp) & np.isfinite(grass_temp) & np.isfinite(ba_temp)

tree_temp  = tree_temp[valid]
shrub_temp = shrub_temp[valid]
grass_temp = grass_temp[valid]
ba_temp    = ba_temp[valid]


bins = ternary_bin(tree_temp, shrub_temp, grass_temp, nbin=nbin)

tree_bin  = bins[:, 0]
shrub_bin = bins[:, 1]
grass_bin = bins[:, 2]


# BA_bin_mean = np.zeros((nbin + 1, nbin + 1, nbin + 1))
mask_bin_mean = np.full(
    (nbin + 1, nbin + 1, nbin + 1),
    np.nan
)

for i in range(nbin + 1):
    for j in range(nbin + 1):

        k = nbin - i - j

        if k < 0:
            continue

        mask_bin = (tree_bin == i) & (shrub_bin == j) & (grass_bin == k)

        mask_bin_mean[i, j, k] = np.nansum(ba_temp[mask_bin])
        if np.any(mask_bin):
            mask_bin_mean[i, j, k] = np.nansum(ba_temp[mask_bin])





# ================================= BA ============================

dir_file = "/scratch/yangbuw/Data/GFED5/GFED5_1/Output/GFED5_1_BA_m2_byBiomes_025x025_2002_2022_monthly.nc"
ds2 = xr.open_dataset(dir_file)

data = ds2["TOTL"] / 1E6 / 1E4       # m2 -> Mha

data_selected = data.sel(
    longitude=slice(-180, 180),
    latitude=slice(-90, 90),
    time=slice(str(startyear_use), str(endyear_use))
)

BA = data_selected.resample(time="YS").sum()

BA_mean_2 = BA.mean(dim="time", skipna=True)   # Mha yr-1



# ==================== Annual BA for each ternary bin ====================

years = BA.time.dt.year.values

BA_bin_yearly = np.full(
    (len(years), nbin + 1, nbin + 1, nbin + 1),
    np.nan
)

for tt in range(len(years)):

    ba_temp_year = BA.isel(time=tt).values.flatten()

    # use exactly the same spatial valid mask used for ternary bin assignment
    ba_temp_year = ba_temp_year[valid]

    for i in range(nbin + 1):
        for j in range(nbin + 1):

            k = nbin - i - j

            if k < 0:
                continue

            mask_bin = (
                (tree_bin == i) &
                (shrub_bin == j) &
                (grass_bin == k)
            )

            vals = ba_temp_year[mask_bin]

            if np.any(np.isfinite(vals)):
                BA_bin_yearly[tt, i, j, k] = np.nansum(vals)


# ==================== Mean annual BA for each ternary bin ====================

BA_bin_mean = np.nanmean(
    BA_bin_yearly,
    axis=0
)


# ==================== Trend for each ternary bin ====================

BA_bin_trend = np.full(
    (nbin + 1, nbin + 1, nbin + 1),
    np.nan
)

BA_bin_relative_trend = np.full(
    (nbin + 1, nbin + 1, nbin + 1),
    np.nan
)

BA_bin_p = np.full(
    (nbin + 1, nbin + 1, nbin + 1),
    np.nan
)


for i in range(nbin + 1):
    for j in range(nbin + 1):

        k = nbin - i - j

        if k < 0:
            continue

        ba_ts = BA_bin_yearly[:, i, j, k]

        valid_year = np.isfinite(ba_ts)

        if np.sum(valid_year) >= 5:

            slope, intercept, r, p, se = linregress(
                years[valid_year],
                ba_ts[valid_year]
            )

            BA_bin_trend[i, j, k] = slope
            BA_bin_p[i, j, k] = p

            if BA_bin_mean[i, j, k] > 0:
                BA_bin_relative_trend[i, j, k] = (
                    slope /
                    BA_bin_mean[i, j, k] *
                    100
                )


# -------------------- mask ----------------------
BA_bin_relative_trend[mask_bin_mean < 0.01] = np.nan
BA_bin_trend[mask_bin_mean < 0.01] = np.nan

# ==================== Convert ternary coordinate to x-y ====================

tree_plot = []
shrub_plot = []
grass_plot = []
value_plot = []

x_plot = []
y_plot = []

height = np.sqrt(3) / 2


for i in range(nbin + 1):
    for j in range(nbin + 1):

        k = nbin - i - j

        if k < 0:
            continue

        tree_p  = i / nbin
        shrub_p = j / nbin
        grass_p = k / nbin

        x = tree_p + 0.5 * shrub_p
        y = height * shrub_p

        tree_plot.append(tree_p)
        shrub_plot.append(shrub_p)
        grass_plot.append(grass_p)

        x_plot.append(x)
        y_plot.append(y)

        value_plot.append(
            BA_bin_relative_trend[i, j, k]
        )


x_plot = np.array(x_plot)
y_plot = np.array(y_plot)
value_plot = np.array(value_plot)


# ==================== Plot ====================

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'

fig, ax = plt.subplots(figsize=(6, 5))

height = np.sqrt(3) / 2

p_threshold = 0.1


# ---------------------------------------------------------
# Color-filled ternary cells
# ---------------------------------------------------------

patches = []
values = []

gray_patches = []


for i in range(nbin):
    for j in range(nbin - i):

        k = nbin - i - j


        # =====================================================
        # upward triangle
        # =====================================================

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


        polygon = Polygon(
            [[x1, y1], [x2, y2], [x3, y3]],
            closed=True
        )


        value_use = BA_bin_relative_trend[i, j, k]
        p_use = BA_bin_p[i, j, k]


        # value_use is NaN -> very little burned area -> leave white
        if np.isfinite(value_use):

            # nonsignificant trend -> gray
            if np.isfinite(p_use) and p_use >= p_threshold:
                gray_patches.append(polygon)

            # significant trend -> blue/red
            elif np.isfinite(p_use) and p_use < p_threshold:
                patches.append(polygon)
                values.append(value_use)


        # =====================================================
        # downward triangle
        # =====================================================

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


            polygon = Polygon(
                [[x1, y1], [x2, y2], [x3, y3]],
                closed=True
            )


            ii = i + 1
            jj = j
            kk = nbin - (i + 1) - j


            value_use = BA_bin_relative_trend[ii, jj, kk]
            p_use = BA_bin_p[ii, jj, kk]


            # value_use is NaN -> very little burned area -> leave white
            if np.isfinite(value_use):

                # nonsignificant trend -> gray
                if np.isfinite(p_use) and p_use >= p_threshold:
                    gray_patches.append(polygon)

                # significant trend -> blue/red
                elif np.isfinite(p_use) and p_use < p_threshold:
                    patches.append(polygon)
                    values.append(value_use)


values = np.array(values)


# ---------------------------------------------------------
# Check range
# ---------------------------------------------------------

print("Significant BA relative trend min =", np.nanmin(values))
print("Significant BA relative trend max =", np.nanmax(values))


# ---------------------------------------------------------
# Color scale
# ---------------------------------------------------------

levels = np.arange(-4, 4.1, 0.1)

from cmap import Colormap

cm = Colormap('colorbrewer:RdBu')
cmap = cm.reversed().to_mpl()

norm = mcolors.BoundaryNorm(
    boundaries=levels,
    ncolors=cmap.N,
    clip=True
)


# ---------------------------------------------------------
# Significant trend patches
# ---------------------------------------------------------

pc = PatchCollection(
    patches,
    cmap=cmap,
    norm=norm,
    edgecolor="none"
)

pc.set_array(values)

ax.add_collection(pc)


# ---------------------------------------------------------
# Nonsignificant trend patches
# ---------------------------------------------------------

pc_gray = PatchCollection(
    gray_patches,
    facecolor="lightgray",
    edgecolor="none"
)

ax.add_collection(pc_gray)


# ---------------------------------------------------------
# Triangle boundary
# ---------------------------------------------------------

ax.plot(
    [0, 1, 0.5, 0],
    [0, 0, height, 0],
    color="black",
    linewidth=1.5
)


# ---------------------------------------------------------
# Tree fraction ticks - bottom
# ---------------------------------------------------------

for value in range(0, 101, 20):

    frac = value / 100

    ax.text(
        frac,
        -0.055,
        str(value),
        ha="center",
        va="top",
        fontsize=13
    )


ax.text(
    0.5,
    -0.1,
    "Tree fraction (%)",
    ha="center",
    va="top",
    fontsize=14
)


# ---------------------------------------------------------
# Grass fraction ticks - left edge
# ---------------------------------------------------------

for value in range(0, 101, 20):

    grass_p = value / 100
    shrub_p = 1 - grass_p

    x = 0.5 * shrub_p
    y = height * shrub_p

    ax.text(
        x - 0.01,
        y,
        str(value),
        ha="right",
        va="center",
        rotation=60,
        fontsize=13
    )


ax.text(
    0.12,
    height * 0.53,
    "Grass fraction (%)",
    rotation=60,
    ha="center",
    va="center",
    fontsize=14
)


# ---------------------------------------------------------
# Shrub fraction ticks - right edge
# ---------------------------------------------------------

for value in range(0, 101, 20):

    shrub_p = value / 100
    tree_p = 1 - shrub_p

    x = tree_p + 0.5 * shrub_p
    y = height * shrub_p

    ax.text(
        x,
        y,
        str(value),
        ha="left",
        va="center",
        rotation=-60,
        fontsize=13
    )


ax.text(
    0.88,
    height * 0.53,
    "Shrub fraction (%)",
    rotation=-60,
    ha="center",
    va="center",
    fontsize=14
)


# ---------------------------------------------------------
# Colorbar
# ---------------------------------------------------------

cbar = plt.colorbar(
    pc,
    ax=ax,
    orientation="horizontal",
    fraction=0.055,
    pad=0.05
)


cbar.set_ticks(
    [-4, 0,  4]
)

cbar.ax.tick_params(
    labelsize=13
)

cbar.ax.minorticks_off()


cbar.set_label(
    r"Burned Area Trend (% yr$^{-1}$)",
    fontsize=14
)


# ---------------------------------------------------------
# Panel label
# ---------------------------------------------------------

ax.text(
    -0.08,
    height,
    "d",
    fontsize=19,
    fontweight="bold"
)


# ---------------------------------------------------------
# Figure setting
# ---------------------------------------------------------

ax.set_xlim(-0.08, 1.08)
ax.set_ylim(-0.13, height + 0.04)

ax.set_aspect("equal")
ax.axis("off")
plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/response_Tenary_BA_Trend_GFED5_025.svg', dpi=300, bbox_inches='tight')
plt.show()
