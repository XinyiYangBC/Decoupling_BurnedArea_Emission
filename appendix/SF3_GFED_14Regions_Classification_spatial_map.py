import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats
from scipy.stats import linregress, pearsonr, t

# plotting package
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from cmap import Colormap

# ============================= Data =========================
dir_file = f"/scratch/yangbuw/Data/GFED5/BasisRegions/GFED5_BasisRegions_LandOnly_South2North_025x025.nc"
ds2 = xr.open_dataset(dir_file)
mask = ds2['basisregions']
data_use_plot = mask

# ============================= Plot Setup =========================
lon_min, lon_max = -180, 180
lat_min, lat_max = -90, 90

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'

fig, ax = plt.subplots(figsize=(11, 5), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.OCEAN, color='white')

gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12, 'color': 'gray'}
gl.ylabel_style = {'size': 12, 'color': 'gray'}

# ============================= Colormap =========================
bounds2 = np.arange(1, 15)
region_names = [
    'BONA: Boreal North America',
    'TENA: Temperate North America',
    'CEAM: Central America',
    'NHSA: Northern Hemisphere South America',
    'SHSA: Southern Hemisphere South America',
    'EURO: Europe',
    'MIDE: Middle East',
    'NHAF: Northern Hemisphere Africa',
    'SHAF: Southern Hemisphere Africa',
    'BOAS: Boreal Asia',
    'CEAS: Central Asia',
    'SEAS: Southeast Asia',
    'EQAS: Equatorial Asia',
    'AUST: Australia and New Zealand'
]

cm = Colormap('tol:rainbow_discrete_14')  # case insensitive

mpl_cmap = cm.to_mpl()
bounds2 = np.arange(1, 15)  # Regions 1 to 14
white = np.array([[1, 1, 1, 1]])
colors2 = mpl_cmap(np.linspace(0, 1, len(bounds2)))  # Use the "bwr" colormap
# colors2 = np.vstack((white, colors2))
custom_cmap2 = mcolors.ListedColormap(colors2)
norm2 = mcolors.BoundaryNorm(bounds2, custom_cmap2.N)

# ============================= Map =========================
mesh = ax.pcolormesh(
    data_use_plot['longitude'],
    data_use_plot['latitude'],
    data_use_plot,
    transform=ccrs.PlateCarree(),
    cmap=custom_cmap2,
    norm=norm2,
    shading='auto'
)

# ============================= Custom Legend =========================
patches = [mpatches.Patch(color=colors2[i], label=region_names[i]) for i in range(len(bounds2))]
legend = ax.legend(
    handles=patches,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.38),
    ncol=3,
    fontsize=10,
    frameon=False,
    handlelength=1.2,
    columnspacing=1.5
)

# ============================= Save/Show =========================
plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/SFigure3_Spatial_Map_GFED_14_regions.tif', dpi=300, bbox_inches='tight')
plt.show()
