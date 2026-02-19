import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats
from scipy.stats import linregress, pearsonr, t
# import pymannkendall as mk

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator 
import matplotlib.gridspec as gridspec


# BA, Emission, EI
global_trend_per = [-1.21, -0.26, 0.95]
global_trend_per_CI = [0.28, 0.47, 0.40]
global_trend_per_p = [1, 0, 1]

# Zheng et al (2021)
global_trend_per_ref = [-1.6, -0.5, 0.9]
global_trend_per_CI_ref = [0.4, 0.8, 0.90]
global_trend_per_p_ref = [1, 0, 1]

sava_trend_per = [-1.07, -0.35, 0.72]
sava_trend_per_CI = [0.30, 0.34, 0.20]
sava_trend_per_p = [1, 1, 1]


borf_trend_per = [-1.59, 1.06, 2.31]
borf_trend_per_CI = [2.53, 2.53, 1.06]
borf_trend_per_p = [0, 0, 1]


temf_trend_per = [-1.19, 3.79, 4.32]
temf_trend_per_CI = [1.71, 3.91, 2.06]
temf_trend_per_p = [0, 0.5, 1] 


# ====================================== Plot =============================
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'

labels = ['BA', 'Emission', 'EI']
colors = ['green', 'bisque', 'orange']
x = np.arange(3)

# =================== Grid Layout ===================
# fig = plt.figure(figsize=(13, 6), dpi=300)
# gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

# # Define axes manually
# ax1 = fig.add_subplot(gs[0, 0])  # Global 1
# ax2 = fig.add_subplot(gs[0, 1])  # Global 2
# ax3 = fig.add_subplot(gs[1, 0])  # SAVA
# ax4 = fig.add_subplot(gs[1, 1])  # BORF
# ax5 = fig.add_subplot(gs[1, 2])  # TEMF


# Set up 2-row, 6-column layout (to allow 2 top wide, 3 bottom narrow subplots)
fig = plt.figure(figsize=(12, 6), dpi=300)
gs = gridspec.GridSpec(2, 12, height_ratios=[1, 1], width_ratios=[1]*12)

# Top row (each = 6 columns wide = half of 12)
ax1 = fig.add_subplot(gs[0, 0:4])   # Global 1
ax2 = fig.add_subplot(gs[0, 5:9])  # Global 2

# Bottom row (each = 4 columns = 1/3 of 12)
ax3 = fig.add_subplot(gs[1, 0:3])   # SAVA
ax4 = fig.add_subplot(gs[1, 3:6])   # BORF
ax5 = fig.add_subplot(gs[1, 6:9])  # TEMF




# =================== Plot 1: Global ===================
ax = ax1
trend_per_use = global_trend_per
trend_per_CI_use = global_trend_per_CI
trend_per_P_use = global_trend_per_p

bars = ax.bar(x, trend_per_use, yerr=trend_per_CI_use,
              color=colors, edgecolor='black',
              error_kw={'elinewidth': 1, 'capsize': 4, 'ecolor': 'gray', 'alpha': 1},
              label=labels)

for i, (bar, pval) in enumerate(zip(bars, trend_per_P_use)):
    trend_value = trend_per_use[i]
    CI_value = trend_per_CI_use[i]
    x_pos = bar.get_x() + bar.get_width() / 2
    if pval == 1:
        ax.text(x_pos, trend_value + CI_value + 0.05 if trend_value >= 0 else trend_value - CI_value - 0.05,
                '*', ha='center', va='bottom' if trend_value >= 0 else 'top', fontsize=14)
    elif pval == 0.5:
        ax.text(x_pos, trend_value + 2 if trend_value >= 0 else trend_value - 2,
                '+', ha='center', va='bottom' if trend_value >= 0 else 'top', fontsize=14)

ax.set_xticks(x)
ax.set_xticklabels(['', '', ''], fontsize=12)
ax.set_ylabel('Relative Trend (% yr$^{-1}$)', fontsize=13)
ax.axhline(0, color='black', linewidth=1)
ax.set_ylim([-2.3, 2.3])
ax.set_yticks(np.arange(-2, 2.1, 1))
ax.set_yticklabels([f'{y:.1f}' for y in np.arange(-2, 2.1, 1)], fontsize=13, color='gray')
ax.legend(loc='upper left', bbox_to_anchor=(-0.04, 1.04), frameon=False, fontsize=13)
ax.set_title("Global", fontsize=13)
# ax.text(0.5, 0.93, 'GFED5',
#         transform=ax.transAxes,
#         fontsize=13,
#         fontweight='bold',
#         ha='center', va='center')

# =================== Plot 2: Global again ===================
ax = ax2
trend_per_use = global_trend_per_ref
trend_per_CI_use = global_trend_per_CI_ref
trend_per_P_use = global_trend_per_p_ref

bars = ax.bar(x, trend_per_use, yerr=trend_per_CI_use,
              color=colors, edgecolor='black',
              error_kw={'elinewidth': 1, 'capsize': 4, 'ecolor': 'gray', 'alpha': 1})

hatches = ['//', '//', '//']
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)


for i, (bar, pval) in enumerate(zip(bars, trend_per_P_use)):
    trend_value = trend_per_use[i]
    CI_value = trend_per_CI_use[i]
    x_pos = bar.get_x() + bar.get_width() / 2
    if pval == 1:
        ax.text(x_pos, trend_value + CI_value + 0.05 if trend_value >= 0 else trend_value - CI_value - 0.05,
                '*', ha='center', va='bottom' if trend_value >= 0 else 'top', fontsize=14)
    elif pval == 0.5:
        ax.text(x_pos, trend_value + CI_value + 0.05 if trend_value >= 0 else trend_value - 2,
                '+', ha='center', va='bottom' if trend_value >= 0 else 'top', fontsize=14)

ax.set_xticks(x)
ax.set_xticklabels(['', '', ''], fontsize=12)

ax.axhline(0, color='black', linewidth=1)
ax.set_ylim([-2.3, 2.3])
ax.set_yticks(np.arange(-2, 2.1, 1))
ax.set_yticklabels([f'{y:.1f}' for y in np.arange(-2, 2.1, 1)], fontsize=13, color='gray')
# ax.set_ylabel('Relative Trend (% yr$^{-1}$)', fontsize=13)
ax.set_title("Global", fontsize=13)
# ax.tick_params(labelleft=False)

# Add legend including the hatch
hatch_patch = plt.Rectangle((0, 0), 1, 1, facecolor='bisque', edgecolor='black', hatch='//', label='Emission (hatched)')

ax.legend(handles=[bars[0], hatch_patch, bars[2]],
          labels=['BA', 'Emission', 'EI'],
          loc='lower right', bbox_to_anchor=(1.04, -0.05), frameon=False, fontsize=13)

ax.text(0.5, 0.93, 'MOPITT-based',
        transform=ax.transAxes,
        fontsize=13,
        fontweight='bold',
        ha='center', va='center')


# =================== Plot 3: SAVA ===================
ax = ax3
trend_per_use = sava_trend_per
trend_per_CI_use = sava_trend_per_CI
trend_per_P_use = sava_trend_per_p

bars = ax.bar(x, trend_per_use, yerr=trend_per_CI_use,
              color=colors, edgecolor='black',
              error_kw={'elinewidth': 1, 'capsize': 4, 'ecolor': 'gray', 'alpha': 1})

for i, (bar, pval) in enumerate(zip(bars, trend_per_P_use)):
    trend_value = trend_per_use[i]
    CI_value = trend_per_CI_use[i]
    x_pos = bar.get_x() + bar.get_width() / 2
    if pval == 1:
        ax.text(x_pos, trend_value + CI_value + 0.05 if trend_value >= 0 else trend_value - CI_value - 0.05,
                '*', ha='center', va='bottom' if trend_value >= 0 else 'top', fontsize=14)

ax.set_xticks(x)
ax.set_xticklabels(['', '', ''], fontsize=12)

ax.axhline(0, color='black', linewidth=1)
ax.set_ylim([-2, 2])
ax.set_yticks(np.arange(-1.5, 2.1, 1))
ax.set_yticklabels([f'{y:.1f}' for y in np.arange(-1.5, 2.1, 1)], fontsize=13, color='gray')
ax.set_ylabel('Relative Trend (% yr$^{-1}$)', fontsize=13)
ax.set_title("SAVA", fontsize=13)

# =================== Plot 4: BORF ===================
ax = ax4
trend_per_use = borf_trend_per
trend_per_CI_use = borf_trend_per_CI
trend_per_P_use = borf_trend_per_p

# pos = ax.get_position()
# ax.set_position([pos.x0 - 0.02, pos.y0, pos.width, pos.height])

bars = ax.bar(x, trend_per_use, yerr=trend_per_CI_use,
              color=colors, edgecolor='black',
              error_kw={'elinewidth': 1, 'capsize': 4, 'ecolor': 'gray', 'alpha': 1})

for i, (bar, pval) in enumerate(zip(bars, trend_per_P_use)):
    trend_value = trend_per_use[i]
    CI_value = trend_per_CI_use[i]
    x_pos = bar.get_x() + bar.get_width() / 2
    if pval == 1:
        ax.text(x_pos, trend_value + CI_value + 0.05 if trend_value >= 0 else trend_value - CI_value - 0.05,
                '*', ha='center', va='bottom' if trend_value >= 0 else 'top', fontsize=14)

ax.set_xticks(x)
ax.set_xticklabels(['', '', ''], fontsize=12)
ax.axhline(0, color='black', linewidth=1)
ax.set_ylim([-6, 8])
ax.set_yticks(np.arange(-5, 8.1, 2.5))
ax.set_yticklabels([f'{y:.1f}' for y in np.arange(-5, 8.1, 2.5)], fontsize=13, color='gray')
ax.set_title("BORF", fontsize=13)
ax.tick_params(labelleft=True)

# =================== Plot 5: TEMF ===================
ax = ax5
trend_per_use = temf_trend_per
trend_per_CI_use = temf_trend_per_CI
trend_per_P_use = temf_trend_per_p

bars = ax.bar(x, trend_per_use, yerr=trend_per_CI_use,
              color=colors, edgecolor='black',
              error_kw={'elinewidth': 1, 'capsize': 4, 'ecolor': 'gray', 'alpha': 1})

for i, (bar, pval) in enumerate(zip(bars, trend_per_P_use)):
    trend_value = trend_per_use[i]
    CI_value = trend_per_CI_use[i]
    x_pos = bar.get_x() + bar.get_width() / 2
    if pval == 1:
        ax.text(x_pos, trend_value + CI_value + 0.05 if trend_value >= 0 else trend_value - CI_value - 0.05,
                '*', ha='center', va='bottom' if trend_value >= 0 else 'top', fontsize=14)
    elif pval == 0.5:
        ax.text(x_pos, trend_value + CI_value + 0.05 if trend_value >= 0 else trend_value - 2,
                '+', ha='center', va='bottom' if trend_value >= 0 else 'top', fontsize=14)

ax.set_xticks(x)
ax.set_xticklabels(['', '', ''], fontsize=12)
ax.axhline(0, color='black', linewidth=1)
ax.set_ylim([-4, 10])
ax.set_yticks(np.arange(-3, 10.1, 3))
ax.set_yticklabels([f'{y:.1f}' for y in np.arange(-3, 10.1, 3)], fontsize=14, color='gray')
ax.set_title("TEMF", fontsize=13)
ax.tick_params(labelleft=True)


letters = ['a', 'b', 'c', 'd',' e']
axes_list = [ax1, ax2, ax3, ax4, ax5]

for i, ax in enumerate(axes_list):
    ax.text(-0.14, 1.1, letters[i], transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='top', ha='left')

# =================== Save and Show ===================
plt.tight_layout()
plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/Figure2_Bar_4_biome_trends_5subplots.tif', dpi=300, bbox_inches='tight')
plt.show()
