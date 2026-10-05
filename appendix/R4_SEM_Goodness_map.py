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


# --------------------- Making Burned Fraction mask -------------------------------
startyear_use = 2002
endyear_use   = 2022


# =============== BF ===================
def calculate_Regional_Annual_Mean(var_name):

    dir_file = f"/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_BF_byBiomes_025x025_2002_2022_monthly.nc"

    ds2 = xr.open_dataset(dir_file)

    C = ds2[var_name]

    region_C = C.sel(
        longitude=slice(-180, 180),
        latitude=slice(-90, 90),
        time=slice(str(startyear_use), str(endyear_use))
    )

    region_C_yearly = region_C.resample(time='YS').sum()

    region_C_sum = region_C_yearly.sum(
        dim=['latitude', 'longitude']
    )

    return region_C_yearly


# the Var_name is in order, from largest to lowest
Var_name = ['TOTL', 'SAVA', 'AGRI', 'BORF', 'DEFO', 'TEMF', 'PEAT']

BA_data_use = []

for i in range(len(Var_name)):

    current_var_name = Var_name[i]

    BA_data_use.append(
        calculate_Regional_Annual_Mean(current_var_name)
    )


data_use = BA_data_use[1]

data_area_sum = data_use.mean(
    dim=['time'],
    skipna=True
)

BF_mask = data_area_sum

# --------------------- Making Burned Fraction mask -------------------------------



# =================================================================================
#                               3-MONTH DATA
# =================================================================================

# ========================== sig_mask: 3-month ==========================
dir_file = f"/home/yangbuw/Program/EmissionIntensity/code_version1/data_code/data/SEM_Path_Coeffs_relative_importance.nc"

ds2 = xr.open_dataset(dir_file)

sig_mask_3month = ds2["sig_mask"]


# ========================== SEM fit: 3-month ==========================
dir_file = f"/home/yangbuw/Program/EmissionIntensity/code_version1/data_code/data/SEM_global_coeff_025x025_rmAC_2Path_VegIndex_Lag_3_month.nc"

ds2 = xr.open_dataset(dir_file)

C1_3month = ds2["GFI"]       # goodness-of-fit index
C2_3month = ds2["CFI"]       # comparative fit index
C3_3month = ds2["RMSEA"]     # root mean square error of approximation
C4_3month = ds2["chi2"]
C5_3month = ds2["DoF"]


# ========================== Mask ==========================
C1_3month = C1_3month.where(BF_mask >= 0.001, np.nan)
C2_3month = C2_3month.where(BF_mask >= 0.001, np.nan)
C3_3month = C3_3month.where(BF_mask >= 0.001, np.nan)
C4_3month = C4_3month.where(BF_mask >= 0.001, np.nan)
C5_3month = C5_3month.where(BF_mask >= 0.001, np.nan)


# chi2 / DoF
C4_3month = C4_3month / C5_3month


# Your original additional mask for chi2/DoF
C4_3month = C4_3month.where(
    BF_mask >= 0.01,
    np.nan
)


# ========================== PIE chart ==========================

# ---------- GFI ----------
GFI = C1_3month

GFI = GFI.where(sig_mask_3month)

GFI_flat = GFI.values.flatten()

GFI_clean = GFI_flat[
    ~np.isnan(GFI_flat)
]

per_GFI_3month = (
    np.sum(GFI_clean >= 0.85)
    /
    np.sum(~np.isnan(GFI_clean))
    * 100
)

ratio_GFI_3month = [
    per_GFI_3month,
    100 - per_GFI_3month
]


# ---------- RMSE ----------
RMSE = C3_3month

RMSE = RMSE.where(sig_mask_3month)

RMSE_flat = RMSE.values.flatten()

RMSE_clean = RMSE_flat[
    ~np.isnan(RMSE_flat)
]

per_RMSE_3month = (
    np.sum(RMSE_clean <= 0.15)
    /
    np.sum(~np.isnan(RMSE_clean))
    * 100
)

ratio_RMSE_3month = [
    per_RMSE_3month,
    100 - per_RMSE_3month
]


# ---------- chi2 / DoF ----------
chi2 = C4_3month

chi2 = chi2.where(sig_mask_3month)

chi2_flat = chi2.values.flatten()

chi2_clean = chi2_flat[
    ~np.isnan(chi2_flat)
]

per_chi2_3month = (
    np.sum(chi2_clean <= 5)
    /
    np.sum(~np.isnan(chi2_clean))
    * 100
)

ratio_chi2_3month = [
    per_chi2_3month,
    100 - per_chi2_3month
]


# ---------- CFI ----------
CFI = C2_3month

CFI = CFI.where(sig_mask_3month)

CFI_flat = CFI.values.flatten()

CFI_clean = CFI_flat[
    ~np.isnan(CFI_flat)
]

per_CFI_3month = (
    np.sum(CFI_clean >= 0.90)
    /
    np.sum(~np.isnan(CFI_clean))
    * 100
)

ratio_CFI_3month = [
    per_CFI_3month,
    100 - per_CFI_3month
]



# =================================================================================
#                               4-MONTH DATA
# =================================================================================

# ========================== sig_mask: 4-month ==========================
dir_file = f"/home/yangbuw/Program/EmissionIntensity/code_version1/data_code/data/SEM_Path_Coeffs_relative_importance_4months.nc"

ds2 = xr.open_dataset(dir_file)

sig_mask_4month = ds2["sig_mask"]


# ========================== SEM fit: 4-month ==========================
dir_file = f"/home/yangbuw/Program/EmissionIntensity/code_version1/data_code/data/SEM_global_coeff_025x025_rmAC_2Path_VegIndex_Lag_4_month.nc"

ds2 = xr.open_dataset(dir_file)

C1_4month = ds2["GFI"]       # goodness-of-fit index
C2_4month = ds2["CFI"]       # comparative fit index
C3_4month = ds2["RMSEA"]     # root mean square error of approximation
C4_4month = ds2["chi2"]
C5_4month = ds2["DoF"]


# ========================== Mask ==========================
C1_4month = C1_4month.where(BF_mask >= 0.001, np.nan)
C2_4month = C2_4month.where(BF_mask >= 0.001, np.nan)
C3_4month = C3_4month.where(BF_mask >= 0.001, np.nan)
C4_4month = C4_4month.where(BF_mask >= 0.001, np.nan)
C5_4month = C5_4month.where(BF_mask >= 0.001, np.nan)


# chi2 / DoF
C4_4month = C4_4month / C5_4month


# Your original additional mask for chi2/DoF
C4_4month = C4_4month.where(
    BF_mask >= 0.01,
    np.nan
)


# ========================== PIE chart ==========================

# ---------- GFI ----------
GFI = C1_4month

GFI = GFI.where(sig_mask_4month)

GFI_flat = GFI.values.flatten()

GFI_clean = GFI_flat[
    ~np.isnan(GFI_flat)
]

per_GFI_4month = (
    np.sum(GFI_clean >= 0.85)
    /
    np.sum(~np.isnan(GFI_clean))
    * 100
)

ratio_GFI_4month = [
    per_GFI_4month,
    100 - per_GFI_4month
]


# ---------- RMSE ----------
RMSE = C3_4month

RMSE = RMSE.where(sig_mask_4month)

RMSE_flat = RMSE.values.flatten()

RMSE_clean = RMSE_flat[
    ~np.isnan(RMSE_flat)
]

per_RMSE_4month = (
    np.sum(RMSE_clean <= 0.15)
    /
    np.sum(~np.isnan(RMSE_clean))
    * 100
)

ratio_RMSE_4month = [
    per_RMSE_4month,
    100 - per_RMSE_4month
]


# ---------- chi2 / DoF ----------
chi2 = C4_4month

chi2 = chi2.where(sig_mask_4month)

chi2_flat = chi2.values.flatten()

chi2_clean = chi2_flat[
    ~np.isnan(chi2_flat)
]

per_chi2_4month = (
    np.sum(chi2_clean <= 5)
    /
    np.sum(~np.isnan(chi2_clean))
    * 100
)

ratio_chi2_4month = [
    per_chi2_4month,
    100 - per_chi2_4month
]


# ---------- CFI ----------
CFI = C2_4month

CFI = CFI.where(sig_mask_4month)

CFI_flat = CFI.values.flatten()

CFI_clean = CFI_flat[
    ~np.isnan(CFI_flat)
]

per_CFI_4month = (
    np.sum(CFI_clean >= 0.90)
    /
    np.sum(~np.isnan(CFI_clean))
    * 100
)

ratio_CFI_4month = [
    per_CFI_4month,
    100 - per_CFI_4month
]



# =================================================================================
#                                      Plot
# =================================================================================

# Map extents and latitude tick locations
lon_min, lon_max = -180, 180
lat_min, lat_max = -70, 90


# panel labels: row by row
panel_labels = [
    'a', 'b',
    'c', 'd',
    'e', 'f',
    'g', 'h'
]


# ========================== datasets ==========================
# rows:
# 0 = GFI
# 1 = CFI
# 2 = RMSE
# 3 = chi2/DoF
#
# columns:
# 0 = 3-month
# 1 = 4-month

datasets = [
    [C1_3month, C1_4month],
    [C2_3month, C2_4month],
    [C3_3month, C3_4month],
    [C4_3month, C4_4month]
]


titles = [
    'GFI',
    'CFI',
    'RMSE',
    'χ2/DoF'
]


# PIE percentages
per_datasets = [
    [per_GFI_3month,  per_GFI_4month],
    [per_CFI_3month,  per_CFI_4month],
    [per_RMSE_3month, per_RMSE_4month],
    [per_chi2_3month, per_chi2_4month]
]


# Labels used beside pie charts
pie_labels = [
    'GFI>0.85',
    'CFI>0.9',
    'RMSE<0.15',
    'χ2/DoF<5'
]


# =================================================================================
# Create the figure and subplots
# =================================================================================

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'


fig, axes = plt.subplots(
    4,
    2,
    figsize=(11, 11),
    subplot_kw={
        'projection': ccrs.PlateCarree()
    }
)


plt.subplots_adjust(
    wspace=0.1
)

plt.subplots_adjust(
    hspace=0.4
)


# =================================================================================
# LOOP
# =================================================================================

for row in range(4):

    for col in range(2):

        ax = axes[row, col]

        data_use = datasets[row][col]


        # =====================================================================
        # GFI
        # =====================================================================
        if row == 0:

            bounds2 = np.arange(
                0.75,
                1.01,
                0.025
            )

            cm = Colormap(
                'colorbrewer:Greens'
            )

            mpl_cmap = cm.to_mpl()

            white = np.array(
                [[1, 1, 1, 1]]
            )

            colors2 = mpl_cmap(
                np.linspace(
                    0.1,
                    1,
                    len(bounds2) - 1
                )
            )

            colors2 = np.vstack(
                (white, colors2)
            )

            custom_cmap2 = mcolors.ListedColormap(
                colors2
            )

            norm2 = mcolors.BoundaryNorm(
                bounds2,
                custom_cmap2.N
            )

            mesh = ax.pcolormesh(
                data_use['longitude'],
                data_use['latitude'],
                data_use,
                transform=ccrs.PlateCarree(),
                cmap=custom_cmap2,
                norm=norm2,
                shading='auto'
            )


        # =====================================================================
        # CFI
        # =====================================================================
        elif row == 1:

            bounds2 = np.arange(
                0.75,
                1.01,
                0.025
            )

            cm = Colormap(
                'colorbrewer:Greens'
            )

            mpl_cmap = cm.to_mpl()

            white = np.array(
                [[1, 1, 1, 1]]
            )

            colors2 = mpl_cmap(
                np.linspace(
                    0.1,
                    1,
                    len(bounds2) - 1
                )
            )

            colors2 = np.vstack(
                (white, colors2)
            )

            custom_cmap2 = mcolors.ListedColormap(
                colors2
            )

            norm2 = mcolors.BoundaryNorm(
                bounds2,
                custom_cmap2.N
            )

            mesh = ax.pcolormesh(
                data_use['longitude'],
                data_use['latitude'],
                data_use,
                transform=ccrs.PlateCarree(),
                cmap=custom_cmap2,
                norm=norm2,
                shading='auto'
            )


        # =====================================================================
        # RMSE
        # =====================================================================
        elif row == 2:

            bounds2 = np.arange(
                0.0,
                0.51,
                0.05
            )

            colors2 = [
                '#f7f4f9',
                '#e7e1ef',
                '#d4b9da',
                '#cba0d4',
                '#c994c7',
                '#df65b0',
                '#e7298a',
                '#ce1256',
                '#980043',
                '#67001f'
            ]

            custom_cmap2 = LinearSegmentedColormap.from_list(
                "custom_rainbow",
                colors2,
                N=len(colors2)
            )

            norm2 = mcolors.BoundaryNorm(
                bounds2,
                custom_cmap2.N
            )

            mesh = ax.pcolormesh(
                data_use['longitude'],
                data_use['latitude'],
                data_use,
                transform=ccrs.PlateCarree(),
                cmap=custom_cmap2,
                norm=norm2,
                shading='auto'
            )


        # =====================================================================
        # chi2 / DoF
        # =====================================================================
        else:

            bounds2 = np.arange(
                1.0,
                10.01,
                0.5
            )

            cm = Colormap(
                'colorbrewer:GnBu'
            )

            mpl_cmap = cm.to_mpl()

            white = np.array(
                [[1, 1, 1, 1]]
            )

            colors2 = mpl_cmap(
                np.linspace(
                    0,
                    1,
                    len(bounds2) - 1
                )
            )

            # colors2 = np.vstack((white, colors2))

            custom_cmap2 = mcolors.ListedColormap(
                colors2
            )

            norm2 = mcolors.BoundaryNorm(
                bounds2,
                custom_cmap2.N
            )

            mesh = ax.pcolormesh(
                data_use['longitude'],
                data_use['latitude'],
                data_use,
                transform=ccrs.PlateCarree(),
                cmap=custom_cmap2,
                norm=norm2,
                shading='auto'
            )


        # =====================================================================
        # MAP SETTINGS
        # =====================================================================

        ax.set_extent(
            [
                lon_min,
                lon_max,
                lat_min,
                lat_max
            ],
            crs=ccrs.PlateCarree()
        )


        ax.coastlines()

        ax.add_feature(
            cfeature.BORDERS,
            linestyle=':'
        )

        ax.add_feature(
            cfeature.OCEAN,
            color='white'
        )


        gl = ax.gridlines(
            draw_labels=True,
            color='gray',
            alpha=0.2,
            linestyle='--'
        )

        gl.top_labels = False
        gl.right_labels = False

        gl.xlabel_style = {
            'size': 9,
            'color': 'gray'
        }

        gl.ylabel_style = {
            'size': 9,
            'color': 'gray'
        }


        # Optional:
        # remove repeated latitude labels from right column
        #
        # if col == 1:
        #     gl.left_labels = False


        # =====================================================================
        # TITLE
        # =====================================================================

        # ax.set_title(
        #     titles[row],
        #     fontsize=9,
        #     fontweight='bold'
        # )


        # =====================================================================
        # COLORBAR
        # =====================================================================

        cb_ax = inset_axes(
            ax,
            width="100%",
            height="8%",
            loc='lower center',
            bbox_to_anchor=(
                0,
                -0.21,
                1.0,
                1
            ),
            bbox_transform=ax.transAxes,
            borderpad=0
        )


        cbar = plt.colorbar(
            mesh,
            cax=cb_ax,
            orientation='horizontal',
            ticks=bounds2
        )

        cbar.ax.tick_params(
            labelsize=9
        )


        # chi2 / DoF
        if row == 3:

            cbar.ax.set_xticks(
                bounds2[1::2]
            )


        # GFI
        elif row == 0:

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


            cbar.ax.xaxis.set_major_formatter(
                FuncFormatter(
                    custom_tick_format
                )
            )


        else:

            cbar.ax.set_xticks(
                bounds2[::1]
            )


        # =====================================================================
        # PIE CHART
        # =====================================================================

        pie_ax = inset_axes(
            ax,
            width="35%",
            height="35%",
            loc='lower right',
            bbox_to_anchor=(
                -0.70,
                0.1,
                1,
                1
            ),
            bbox_transform=ax.transAxes,
            borderpad=0
        )


        per = per_datasets[row][col]

        per_rest = 100 - per

        sizes = [
            per,
            per_rest
        ]


        # ---------------------------------------------------------
        # Your original pie colors
        # ---------------------------------------------------------
        if row == 0:

            colors_pie = [
                colors2[6],
                colors2[3]
            ]

        elif row == 1:

            colors_pie = [
                colors2[4],
                colors2[2]
            ]

        elif row == 2:

            colors_pie = [
                colors2[4],
                colors2[2]
            ]

        else:

            colors_pie = [
                colors2[6],
                colors2[3]
            ]


        explode = (
            0.1,
            0
        )


        def custom_autopct(pct):

            return (
                f'{pct:.1f}%'
                if pct > 20
                else ''
            )


        pie_ax.pie(
            sizes,
            colors=colors_pie,
            startangle=90,
            autopct=custom_autopct,
            textprops={
                'fontsize': 9,
                'color': 'black'
            },
            explode=explode,
            pctdistance=0.35
        )


        pie_ax.set_aspect(
            'equal'
        )


        # =====================================================================
        # PIE TEXT
        # =====================================================================

        if row == 0:

            ax.text(
                0.07,
                0.1,
                'GFI>0.85',
                transform=ax.transAxes,
                color='Black',
                fontsize=8,
                fontweight='bold',
                va='top',
                ha='left'
            )


        elif row == 1:

            ax.text(
                0.06,
                0.1,
                'CFI>0.9',
                transform=ax.transAxes,
                color='Black',
                fontsize=8,
                fontweight='bold',
                va='top',
                ha='left'
            )


        elif row == 2:

            ax.text(
                0.06,
                0.1,
                'RMSE<0.15',
                transform=ax.transAxes,
                color='Black',
                fontsize=8,
                fontweight='bold',
                va='top',
                ha='left'
            )


        else:

            ax.text(
                0.07,
                0.1,
                'χ2/DoF<5',
                transform=ax.transAxes,
                color='Black',
                fontsize=8,
                fontweight='bold',
                va='top',
                ha='left'
            )


        # =====================================================================
        # PANEL LABELS: a-h
        # =====================================================================

        panel_index = row * 2 + col

        ax.text(
            0.03,
            0.98,
            panel_labels[panel_index],
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            va='top',
            ha='left'
        )



# =================================================================================
# 3-month / 4-month column labels
# =================================================================================

# Put them above the two columns so the reader immediately knows
# which sensitivity analysis each column represents.

axes[0, 0].text(
    0.5,
    1.0,
    '3-month',
    transform=axes[0, 0].transAxes,
    fontsize=13,
    fontweight='bold',
    ha='center',
    va='bottom'
)


axes[0, 1].text(
    0.5,
    1.0,
    '4-month',
    transform=axes[0, 1].transAxes,
    fontsize=13,
    fontweight='bold',
    ha='center',
    va='bottom'
)



# =================================================================================
# SAVE
# =================================================================================

plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/SFigure13_Spatial_Map_SEM_Goodness_Fit_8subplots.pdf', dpi=300,bbox_inches='tight')

plt.show()

