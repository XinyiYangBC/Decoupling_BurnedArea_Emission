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

dir_file = "/scratch/yangbuw/Data/GFED5/GFED5_1/Output/GFED5_1_CO2_by_gC_byBiomes_025x025_2002_2022_monthly.nc"
ds2 = xr.open_dataset(dir_file)

data = ds2["TOTL"]/1e15   #Pg

data_selected = data.sel(
    longitude=slice(-180, 180),
    latitude=slice(-90, 90),
    time=slice(str(startyear_use), str(endyear_use))
)

BA = data_selected.resample(time="YS").sum()

BA_mean = BA.mean(dim="time", skipna=True)


# ==================== Check grids ====================

print("tree_mean shape :", tree_mean.shape)
print("shrub_mean shape:", shrub_mean.shape)
print("grass_mean shape:", grass_mean.shape)
print("BA_mean shape   :", BA_mean.shape)


# ==================== Ternary binning ====================

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


# ==================== Flatten mean data ====================

tree_temp  = tree_mean.values.flatten()
shrub_temp = shrub_mean.values.flatten()
grass_temp = grass_mean.values.flatten()
ba_temp    = BA_mean.values.flatten()


# ==================== Valid grids ====================

valid = np.isfinite(tree_temp) & np.isfinite(shrub_temp) & np.isfinite(grass_temp) & np.isfinite(ba_temp)

tree_temp  = tree_temp[valid]
shrub_temp = shrub_temp[valid]
grass_temp = grass_temp[valid]
ba_temp    = ba_temp[valid]


# ==================== Assign ternary bins ====================

bins = ternary_bin(tree_temp, shrub_temp, grass_temp, nbin=nbin)

tree_bin  = bins[:, 0]
shrub_bin = bins[:, 1]
grass_bin = bins[:, 2]


# ==================== Aggregate mean BA in each ternary bin ====================

# BA_bin_mean = np.zeros((nbin + 1, nbin + 1, nbin + 1))
BA_bin_mean = np.full(
    (nbin + 1, nbin + 1, nbin + 1),
    np.nan
)

for i in range(nbin + 1):
    for j in range(nbin + 1):

        k = nbin - i - j

        if k < 0:
            continue

        mask_bin = (tree_bin == i) & (shrub_bin == j) & (grass_bin == k)

        BA_bin_mean[i, j, k] = np.nansum(ba_temp[mask_bin])
        if np.any(mask_bin):
            BA_bin_mean[i, j, k] = np.nansum(ba_temp[mask_bin])


BA_bin_mean[mask_bin_mean < 0.01] = np.nan

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

        value_plot.append(BA_bin_mean[i, j, k])


x_plot = np.array(x_plot)
y_plot = np.array(y_plot)
value_plot = np.array(value_plot)



# ==================== Plot ====================
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'

fig, ax = plt.subplots(figsize=(6, 5))

height = np.sqrt(3) / 2


# ---------------------------------------------------------
# Color-filled ternary cells
# ---------------------------------------------------------

patches = []
values = []

for i in range(nbin):
    for j in range(nbin - i):

        k = nbin - i - j

        # upward triangle
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

        patches.append(Polygon([[x1, y1], [x2, y2], [x3, y3]], closed=True))
        values.append(BA_bin_mean[i, j, k])

        # downward triangle
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

            patches.append(Polygon([[x1, y1], [x2, y2], [x3, y3]], closed=True))
            values.append(BA_bin_mean[i + 1, j, nbin - (i + 1) - j])



from matplotlib.colors import LinearSegmentedColormap\

# 0–40, every 2 Mha as one color level
levels = np.arange(0, 0.3001, 0.002)

cm = Colormap('colorbrewer:YlOrRd_r')
cmap = cm.reversed().to_mpl()

cmap.set_bad("white")

norm = mcolors.BoundaryNorm(
    boundaries=levels,
    ncolors=cmap.N,
    clip=True
)

pc = PatchCollection(
    patches,
    cmap=cmap,
    norm=norm,
    edgecolor="none"
)

pc.set_array(np.array(values))
ax.add_collection(pc)


# ---------------------------------------------------------
# Triangle boundary
# ---------------------------------------------------------

ax.plot([0, 1, 0.5, 0], [0, 0, height, 0], color="black", linewidth=1.5)


# ---------------------------------------------------------
# Tree fraction ticks - bottom
# ---------------------------------------------------------

for value in range(0, 101, 20):
    frac = value / 100
    ax.text(frac, -0.055, str(value), ha="center", va="top", fontsize=13)

ax.text(0.5, -0.1, "Tree fraction (%)", ha="center", va="top", fontsize=14)


# ---------------------------------------------------------
# Grass fraction ticks - left edge
# ---------------------------------------------------------

for value in range(0, 101, 20):

    grass_p = value / 100
    shrub_p = 1 - grass_p

    x = 0.5 * shrub_p
    y = height * shrub_p

    ax.text(x - 0.01, y, str(value), ha="right", va="center", rotation=60, fontsize=13)

ax.text(0.12, height * 0.53, "Grass fraction (%)", rotation=60, ha="center", va="center", fontsize=14)


# ---------------------------------------------------------
# Shrub fraction ticks - right edge
# ---------------------------------------------------------

for value in range(0, 101, 20):

    shrub_p = value / 100
    tree_p = 1 - shrub_p

    x = tree_p + 0.5 * shrub_p
    y = height * shrub_p

    ax.text(x + 0.0, y, str(value), ha="left", va="center", rotation=-60, fontsize=13)

ax.text(0.88, height * 0.53, "Shrub fraction (%)", rotation=-60, ha="center", va="center", fontsize=14)


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

cbar.set_ticks([0,0.1,0.2, 0.3])

cbar.ax.tick_params(labelsize=13)

# remove minor ticks
cbar.ax.minorticks_off()

cbar.set_label(
    r"CO$_2$ Emission (Pg C year$^{-1}$)",
    fontsize=14
)


# ---------------------------------------------------------
# Panel label
# ---------------------------------------------------------

ax.text(-0.08, height + 0.0, "b", fontsize=19, fontweight="bold")


# ---------------------------------------------------------
# Figure setting
# ---------------------------------------------------------

ax.set_xlim(-0.08, 1.08)
ax.set_ylim(-0.13, height + 0.04)
ax.set_aspect("equal")
ax.axis("off")
plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/response_Tenary_Emission_GFED5_025.pdf', dpi=300, bbox_inches='tight')
plt.show()
