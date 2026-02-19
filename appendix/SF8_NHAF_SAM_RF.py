# -*- coding: utf-8 -*-
"""
Multiprocessing-enabled SEM run across spatial grids.
"""

import numpy as np
import pandas as pd
import xarray as xr
import semopy
import os
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count

# Suppress warnings from semopy
logging.getLogger('semopy').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


#------------------------------ 14 regions Mask ---------------------------------------
data_dir1 = f"/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_basis_regions_025x025.nc"
data_ds   = xr.open_dataset(data_dir1)
# data_ds = data_ds.rename({'lat': 'latitude', 'lon': 'longitude'})
mask_data = data_ds["basis_regions"] #14 regions
lat       = data_ds["latitude"]
lon       = data_ds["longitude"]
region_id = 8
biome_mask = mask_data == region_id
#------------------------------ 14 regions Mask ---------------------------------------

# ---- Model config ----
startyear_use = 2003
endyear_use = 2021
Lat_S =  -90;  # -90
Lat_N =   90;  #  90
Lon_W = -180;  # -180
Lon_E =  180;  # 180


# ---- Load all datasets into memory ----
print("Loading data starts...", datetime.now())


# ==================== Lagged Vegeation Index ==========================
startyear_use1 = 2003-1
endyear_use1 = 2021 +1
months_lag = 3

region_VOD_yearly = xr.open_dataset("/scratch/yangbuw/Data/VOD/VODCA/VODCA_CXKu_1988-2021_monthly_025x025.nc")["VODCA_CXKu"]
region_VOD_yearly = region_VOD_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use1), str(endyear_use1))
)#.resample(time='MS').sum()
region_VOD_yearly = region_VOD_yearly.groupby("time.month") - region_VOD_yearly.groupby("time.month").mean("time")
region_VOD_yearly_lag =region_VOD_yearly.shift(time=months_lag)
region_VOD_yearly_lag =region_VOD_yearly_lag.sel(time=slice(str(startyear_use), str(endyear_use)))
data_in_biome = region_VOD_yearly_lag.where(biome_mask, drop=True)
region_VOD_yearly_lag = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

region_LAI_yearly = xr.open_dataset("/scratch/yangbuw/Data/LAI/MOD15A2H/MOD15A2H_LAI_025x025_2001_2024_monthly.nc")["LAI"]
region_LAI_yearly = region_LAI_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use1), str(endyear_use1))
)#.resample(time='MS').sum()
region_LAI_yearly = region_LAI_yearly.groupby("time.month") - region_LAI_yearly.groupby("time.month").mean("time")
region_LAI_yearly_lag = region_LAI_yearly.shift(time=months_lag)
region_LAI_yearly_lag = region_LAI_yearly_lag.sel(time=slice(str(startyear_use), str(endyear_use)))
data_in_biome = region_LAI_yearly_lag.where(biome_mask, drop=True)
region_LAI_yearly_lag = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

region_EVI_yearly = xr.open_dataset("/scratch/yangbuw/Data/EVI/MOD13A2_EVI_025x025_2001_2024_monthly_025x025.nc")["EVI"]
region_EVI_yearly = region_EVI_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use1), str(endyear_use1))
)#.resample(time='MS').sum()
region_EVI_yearly = region_EVI_yearly.groupby("time.month") - region_EVI_yearly.groupby("time.month").mean("time")
region_EVI_yearly_lag = region_EVI_yearly.shift(time=months_lag)
region_EVI_yearly_lag = region_EVI_yearly_lag.sel(time=slice(str(startyear_use), str(endyear_use)))
data_in_biome = region_EVI_yearly_lag.where(biome_mask, drop=True)
region_EVI_yearly_lag = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)
# ==================== Lagged Vegeation Index ==========================

region_EI_yearly = xr.open_dataset("/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_EI_SAVA_025x025_2002_2022_monthly.nc")
# region_EI_yearly = region_EI_yearly["EI"].sel(
#     longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
#     time=slice(str(startyear_use), str(endyear_use))
# ).resample(time='MS').sum()
region_EI_yearly = region_EI_yearly["EI"].sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use)))
region_EI_yearly = region_EI_yearly.groupby("time.month") - region_EI_yearly.groupby("time.month").mean("time")
data_in_biome = region_EI_yearly.where(biome_mask, drop=True)
region_EI_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

region_Dead_Foliage_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_FUEL_MAP_kg_per_m2_2003_2021_monthly_025x025.nc")["Dead_Foliage"]
region_Dead_Foliage_yearly = region_Dead_Foliage_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
)#.resample(time='MS').sum()
region_Dead_Foliage_yearly = region_Dead_Foliage_yearly.groupby("time.month") - region_Dead_Foliage_yearly.groupby("time.month").mean("time")
data_in_biome = region_Dead_Foliage_yearly.where(biome_mask, drop=True)
region_Dead_Foliage_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

region_Dead_Wood_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_FUEL_MAP_kg_per_m2_2003_2021_monthly_025x025.nc")["Dead_Wood"]
region_Dead_Wood_yearly = region_Dead_Wood_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
)#.resample(time='MS').sum() 
region_Dead_Wood_yearly = region_Dead_Wood_yearly.groupby("time.month") - region_Dead_Wood_yearly.groupby("time.month").mean("time")
data_in_biome = region_Dead_Wood_yearly.where(biome_mask, drop=True)
region_Dead_Wood_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

region_Live_Leaf_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_FUEL_MAP_kg_per_m2_2003_2021_monthly_025x025.nc")["Live_Leaf"]
region_Live_Leaf_yearly = region_Live_Leaf_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
)#.resample(time='MS').sum() 
region_Live_Leaf_yearly = region_Live_Leaf_yearly.groupby("time.month") - region_Live_Leaf_yearly.groupby("time.month").mean("time")
data_in_biome = region_Live_Leaf_yearly.where(biome_mask, drop=True)
region_Live_Leaf_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

region_Live_Wood_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_FUEL_MAP_kg_per_m2_2003_2021_monthly_025x025.nc")["Live_Wood"]
region_Live_Wood_yearly = region_Live_Wood_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
)#.resample(time='MS').sum() 
region_Live_Wood_yearly = region_Live_Wood_yearly.groupby("time.month") - region_Live_Wood_yearly.groupby("time.month").mean("time")
data_in_biome = region_Live_Wood_yearly.where(biome_mask, drop=True)
region_Live_Wood_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

region_Live_Fuel_Load_yearly  = region_Live_Wood_yearly + region_Live_Leaf_yearly
region_Dead_Fuel_Load_yearly  = region_Dead_Wood_yearly + region_Dead_Foliage_yearly

region_Total_Fuel_Load_yearly = region_Dead_Fuel_Load_yearly + region_Live_Fuel_Load_yearly

region_Live_Fuel_Load_yearly = region_Live_Fuel_Load_yearly.groupby("time.month") - region_Live_Fuel_Load_yearly.groupby("time.month").mean("time")
region_Dead_Fuel_Load_yearly = region_Dead_Fuel_Load_yearly.groupby("time.month") - region_Dead_Fuel_Load_yearly.groupby("time.month").mean("time")
region_Total_Fuel_Load_yearly = region_Total_Fuel_Load_yearly.groupby("time.month") - region_Total_Fuel_Load_yearly.groupby("time.month").mean("time")



# # ================= DLEM estimated Fuel Load =================
# region_FL_fine_rec_yearly = xr.open_dataset("/scratch/yangbuw/DLEM/1_ExperimentOutput/PostProcess/Experiment/fire/fire_on_GFED5_prescribed/FL_fine_rec_Gridpool_fire_on_GFED5_prescribed_monthly_025x025_1997_2023_g_per_m2_S2N.nc")["FL_fine_rec"]
# region_FL_fine_rec_yearly = region_FL_fine_rec_yearly.sel(
#     longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
#     time=slice(str(startyear_use), str(endyear_use))
# )#.resample(time='MS').sum()
# region_FL_fine_rec_yearly = region_FL_fine_rec_yearly.groupby("time.month") - region_FL_fine_rec_yearly.groupby("time.month").mean("time")
# data_in_biome = region_FL_fine_rec_yearly.where(biome_mask, drop=True)
# region_FL_fine_rec_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)


# region_FL_coarse_rec_yearly = xr.open_dataset("/scratch/yangbuw/DLEM/1_ExperimentOutput/PostProcess/Experiment/fire/fire_on_GFED5_prescribed/FL_coarse_rec_Gridpool_fire_on_GFED5_prescribed_monthly_025x025_1997_2023_g_per_m2_S2N.nc")["FL_coarse_rec"]
# region_FL_coarse_rec_yearly = region_FL_coarse_rec_yearly.sel(
#     longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
#     time=slice(str(startyear_use), str(endyear_use))
# )#.resample(time='MS').sum()
# region_FL_coarse_rec_yearly = region_FL_coarse_rec_yearly.groupby("time.month") - region_FL_coarse_rec_yearly.groupby("time.month").mean("time")
# data_in_biome = region_FL_coarse_rec_yearly.where(biome_mask, drop=True)
# region_FL_coarse_rec_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)


# region_FL_liveherb_rec_yearly = xr.open_dataset("/scratch/yangbuw/DLEM/1_ExperimentOutput/PostProcess/Experiment/fire/fire_on_GFED5_prescribed/FL_liveherb_rec_Gridpool_fire_on_GFED5_prescribed_monthly_025x025_1997_2023_g_per_m2_S2N.nc")["FL_liveherb_rec"]
# region_FL_liveherb_rec_yearly = region_FL_liveherb_rec_yearly.sel(
#     longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
#     time=slice(str(startyear_use), str(endyear_use))
# )#.resample(time='MS').sum()
# region_FL_liveherb_rec_yearly = region_FL_liveherb_rec_yearly.groupby("time.month") - region_FL_liveherb_rec_yearly.groupby("time.month").mean("time")
# data_in_biome = region_FL_liveherb_rec_yearly.where(biome_mask, drop=True)
# region_FL_liveherb_rec_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)


# region_FL_livewood_rec_yearly = xr.open_dataset("/scratch/yangbuw/DLEM/1_ExperimentOutput/PostProcess/Experiment/fire/fire_on_GFED5_prescribed/FL_livewood_rec_Gridpool_fire_on_GFED5_prescribed_monthly_025x025_1997_2023_g_per_m2_S2N.nc")["FL_livewood_rec"]
# region_FL_livewood_rec_yearly = region_FL_livewood_rec_yearly.sel(
#     longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
#     time=slice(str(startyear_use), str(endyear_use))
# )#.resample(time='MS').sum()
# region_FL_livewood_rec_yearly = region_FL_livewood_rec_yearly.groupby("time.month") - region_FL_livewood_rec_yearly.groupby("time.month").mean("time")
# data_in_biome = region_FL_livewood_rec_yearly.where(biome_mask, drop=True)
# region_FL_livewood_rec_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

# # ================= DLEM estimated Fuel Load =================


region_VOD_yearly = xr.open_dataset("/scratch/yangbuw/Data/VOD/VODCA/VODCA_CXKu_1988-2021_monthly_025x025.nc")["VODCA_CXKu"]
region_VOD_yearly = region_VOD_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
)#.resample(time='MS').sum()
region_VOD_yearly = region_VOD_yearly.groupby("time.month") - region_VOD_yearly.groupby("time.month").mean("time")
data_in_biome = region_VOD_yearly.where(biome_mask, drop=True)
region_VOD_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

region_LAI_yearly = xr.open_dataset("/scratch/yangbuw/Data/LAI/MOD15A2H/MOD15A2H_LAI_025x025_2001_2024_monthly.nc")["LAI"]
region_LAI_yearly = region_LAI_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
)#.resample(time='MS').sum()
region_LAI_yearly = region_LAI_yearly.groupby("time.month") - region_LAI_yearly.groupby("time.month").mean("time")
data_in_biome = region_LAI_yearly.where(biome_mask, drop=True)
region_LAI_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

region_EVI_yearly = xr.open_dataset("/scratch/yangbuw/Data/EVI/MOD13A2_EVI_025x025_2001_2024_monthly_025x025.nc")["EVI"]
region_EVI_yearly = region_EVI_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
)#.resample(time='MS').sum()
region_EVI_yearly = region_EVI_yearly.groupby("time.month") - region_EVI_yearly.groupby("time.month").mean("time")
data_in_biome = region_EVI_yearly.where(biome_mask, drop=True)
region_EVI_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)


region_SM_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_swvl1_percentage_1950_2024_monthly_025x025.nc")["swvl1"]
region_SM_yearly = region_SM_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
)#.resample(time='MS').sum()
region_SM_yearly = region_SM_yearly.groupby("time.month") - region_SM_yearly.groupby("time.month").mean("time")
data_in_biome = region_SM_yearly.where(biome_mask, drop=True)
region_SM_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

region_VPD_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_VPD_kpa_1950_2024_Monthly_025x025.nc")["vpd"]
region_VPD_yearly = region_VPD_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
)#.resample(time='MS').sum()
region_VPD_yearly = region_VPD_yearly.groupby("time.month") - region_VPD_yearly.groupby("time.month").mean("time")
data_in_biome = region_VPD_yearly.where(biome_mask, drop=True)
region_VPD_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

region_Precip_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_precip_m_1950_2024_monthly_025x025.nc")["precip"] * 1e2
region_Precip_yearly = region_Precip_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
)#.resample(time='MS').sum()
region_Precip_yearly = region_Precip_yearly.groupby("time.month") - region_Precip_yearly.groupby("time.month").mean("time")
data_in_biome = region_Precip_yearly.where(biome_mask, drop=True)
region_Precip_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

region_Wind_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_WindSpeed_mpers_1950_2024_Monthly_025x025.nc")["uv"]
region_Wind_yearly = region_Wind_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
)#.resample(time='MS').sum()
region_Wind_yearly = region_Wind_yearly.groupby("time.month") - region_Wind_yearly.groupby("time.month").mean("time")
data_in_biome = region_Wind_yearly.where(biome_mask, drop=True)
region_Wind_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

region_SKT_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_skt_k_1950_2024_monthly_025x025.nc")["skt"]
region_SKT_yearly = region_SKT_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
)#.resample(time='MS').sum()
region_SKT_yearly = region_SKT_yearly.groupby("time.month") - region_SKT_yearly.groupby("time.month").mean("time")
data_in_biome = region_SKT_yearly.where(biome_mask, drop=True)
region_SKT_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

print("Loading data ends...", datetime.now())


# ---- SEM model specification ----
model_spec = """
# Measurement model
# Fuel_load: potential combustible biomass
# Fuel_Load =~ Dead_Fuel_Load + Live_Fuel_Load
Fuel_Load =~ VOD + EVI + LAI
Fire_Weather =~ Surface_Temperature + VPD + SM + Precipitation + Wind_Speed
Emission_Intensity =~ EI

# Structural model
Emission_Intensity ~ Fuel_Load + Fire_Weather

# Correlations
SM ~~  Surface_Temperature + VPD + Precipitation + Wind_Speed
"""

data = pd.DataFrame({
                'VOD': region_VOD_yearly_lag,
                'LAI': region_LAI_yearly_lag,
                'EVI': region_EVI_yearly_lag,
                'SM': region_SM_yearly,
                'VPD': region_VPD_yearly,
                'Precipitation': region_Precip_yearly,
                'Wind_Speed': region_Wind_yearly,
                'Surface_Temperature': region_SKT_yearly,
                'EI': region_EI_yearly
            })

# Standardize
scaler = StandardScaler()
scaled_array = scaler.fit_transform(data)
data_scaled = pd.DataFrame(scaled_array, columns=data.columns, index=data.index)

# SEM model
model = semopy.Model(model_spec)
model.fit(data_scaled)

# Inspect output
inspect_df = model.inspect(se_robust=True)
top11 = inspect_df.head(11)

print(inspect_df)


#================================================================================================
#================================================================================================
#================================================================================================


# -*- coding: utf-8 -*-
"""
Multiprocessing-enabled SEM run across spatial grids.
"""

import numpy as np
np.bool = bool 

import shap

import pandas as pd
import xarray as xr
import semopy
import os
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
from scipy.stats import pearsonr

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
import shap
import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')




#------------------------------ 14 regions Mask ---------------------------------------
data_dir1 = f"/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_basis_regions_025x025.nc"
data_ds   = xr.open_dataset(data_dir1)
# data_ds = data_ds.rename({'lat': 'latitude', 'lon': 'longitude'})
mask_data = data_ds["basis_regions"] #14 regions
lat       = data_ds["latitude"]
lon       = data_ds["longitude"]
region_id = 8
biome_mask = mask_data == region_id
#------------------------------ 14 regions Mask ---------------------------------------

# ---- Model config ----
startyear_use = 2003
endyear_use = 2021
Lat_S =  -90;  # -90
Lat_N =   90;  #  90
Lon_W = -180;  # -180
Lon_E =  180;  # 180


# ---- Load all datasets into memory ----
print("Loading data starts...", datetime.now())


# ==================== Lagged Vegeation Index ==========================
startyear_use1 = 2003-1
endyear_use1 = 2021 +1
months_lag = 3

region_VOD_yearly = xr.open_dataset("/scratch/yangbuw/Data/VOD/VODCA/VODCA_CXKu_1988-2021_monthly_025x025.nc")["VODCA_CXKu"]
region_VOD_yearly = region_VOD_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use1), str(endyear_use1))
).resample(time='MS').sum()
region_VOD_yearly = region_VOD_yearly.groupby("time.month") - region_VOD_yearly.groupby("time.month").mean("time")
region_VOD_yearly_lag =region_VOD_yearly.shift(time=months_lag)
region_VOD_yearly_lag =region_VOD_yearly_lag.sel(time=slice(str(startyear_use), str(endyear_use)))
data_in_biome = region_VOD_yearly_lag.where(biome_mask, drop=True)
region_VOD_yearly_lag = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

region_LAI_yearly = xr.open_dataset("/scratch/yangbuw/Data/LAI/MOD15A2H/MOD15A2H_LAI_025x025_2001_2024_monthly.nc")["LAI"]
region_LAI_yearly = region_LAI_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use1), str(endyear_use1))
).resample(time='MS').sum()
region_LAI_yearly = region_LAI_yearly.groupby("time.month") - region_LAI_yearly.groupby("time.month").mean("time")
region_LAI_yearly_lag = region_LAI_yearly.shift(time=months_lag)
region_LAI_yearly_lag = region_LAI_yearly_lag.sel(time=slice(str(startyear_use), str(endyear_use)))
data_in_biome = region_LAI_yearly_lag.where(biome_mask, drop=True)
region_LAI_yearly_lag = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

region_EVI_yearly = xr.open_dataset("/scratch/yangbuw/Data/EVI/MOD13A2_EVI_025x025_2001_2024_monthly_025x025.nc")["EVI"]
region_EVI_yearly = region_EVI_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use1), str(endyear_use1))
).resample(time='MS').sum()
region_EVI_yearly = region_EVI_yearly.groupby("time.month") - region_EVI_yearly.groupby("time.month").mean("time")
region_EVI_yearly_lag = region_EVI_yearly.shift(time=months_lag)
region_EVI_yearly_lag = region_EVI_yearly_lag.sel(time=slice(str(startyear_use), str(endyear_use)))
data_in_biome = region_EVI_yearly_lag.where(biome_mask, drop=True)
region_EVI_yearly_lag = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)
# ==================== Lagged Vegeation Index ==========================

region_EI_yearly = xr.open_dataset("/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_EI_SAVA_025x025_2002_2022_monthly.nc")
#region_EI_yearly = region_EI_yearly["EI"].sel(longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),time=slice(str(startyear_use), str(endyear_use)))
region_EI_yearly = region_EI_yearly["EI"].sel(longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),time=slice(str(startyear_use), str(endyear_use))).resample(time='MS').sum()
region_EI_yearly = region_EI_yearly.groupby("time.month") - region_EI_yearly.groupby("time.month").mean("time")
data_in_biome = region_EI_yearly.where(biome_mask, drop=True)
region_EI_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)


region_SM_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_swvl1_percentage_1950_2024_monthly_025x025.nc")["swvl1"]
region_SM_yearly = region_SM_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
).resample(time='MS').sum()
region_SM_yearly = region_SM_yearly.groupby("time.month") - region_SM_yearly.groupby("time.month").mean("time")
data_in_biome = region_SM_yearly.where(biome_mask, drop=True)
region_SM_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

region_VPD_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_VPD_kpa_1950_2024_Monthly_025x025.nc")["vpd"]
region_VPD_yearly = region_VPD_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
).resample(time='MS').sum()
region_VPD_yearly = region_VPD_yearly.groupby("time.month") - region_VPD_yearly.groupby("time.month").mean("time")
data_in_biome = region_VPD_yearly.where(biome_mask, drop=True)
region_VPD_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

region_Precip_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_precip_m_1950_2024_monthly_025x025.nc")["precip"] * 1e2
region_Precip_yearly = region_Precip_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
).resample(time='MS').sum()
region_Precip_yearly = region_Precip_yearly.groupby("time.month") - region_Precip_yearly.groupby("time.month").mean("time")
data_in_biome = region_Precip_yearly.where(biome_mask, drop=True)
region_Precip_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

region_Wind_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_WindSpeed_mpers_1950_2024_Monthly_025x025.nc")["uv"]
region_Wind_yearly = region_Wind_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
).resample(time='MS').sum()
region_Wind_yearly = region_Wind_yearly.groupby("time.month") - region_Wind_yearly.groupby("time.month").mean("time")
data_in_biome = region_Wind_yearly.where(biome_mask, drop=True)
region_Wind_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

region_SKT_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_skt_k_1950_2024_monthly_025x025.nc")["skt"]
region_SKT_yearly = region_SKT_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
).resample(time='MS').sum()
region_SKT_yearly = region_SKT_yearly.groupby("time.month") - region_SKT_yearly.groupby("time.month").mean("time")
data_in_biome = region_SKT_yearly.where(biome_mask, drop=True)
region_SKT_yearly = data_in_biome.sum(dim=['longitude','latitude'], skipna=True)

print("Loading data ends...", datetime.now())



#############  Random Forest ############
data = pd.DataFrame({
                'EVI': region_EVI_yearly_lag,
                'SM': region_SM_yearly,
                'VPD': region_VPD_yearly,
                'Precipitation': region_Precip_yearly,
                'Wind_Speed': region_Wind_yearly,
                'Surface_Temperature': region_SKT_yearly,
                'EI': region_EI_yearly
            })

data
scaler = StandardScaler()
scaled_array = scaler.fit_transform(data)
data_scaled = pd.DataFrame(scaled_array, columns=data.columns, index=data.index)
data_scaled


y = data_scaled['EI']
X = data_scaled.drop(['EI'], axis = 1)
X.head()

# Splitting the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30, shuffle = True)


model = RandomForestRegressor(n_estimators = 50, max_depth = 12, min_samples_leaf = 4, min_samples_split = 2, max_leaf_nodes = 5, random_state = 30)


# Fitting the model using the training data
rf = model.fit(X_train, y_train)

# Training model evaluation
y_predtr = rf.predict(X_train)
print('The training r-sq is:', r2_score(y_train, y_predtr))
print('The training MAE is:', mean_absolute_error(y_train, y_predtr))
print('The training MSE is:', mean_squared_error(y_train, y_predtr))

#############  SHAP analysis ############
shap.initjs()

# Initializing an explainer on the model
explainer = shap.TreeExplainer(rf)

# Calculating shap values for the training dataset
shap_valuestr = explainer.shap_values(X_train)

# shap.summary_plot(shap_valuestr, X_train, feature_names = X_train.columns, plot_type = 'bar')

r, p_value = pearsonr(y_train, y_predtr)
r_squre = r**2
r2_score1 = r2_score(y_train, y_predtr)

print(r2_score1)


# feature importance
EVI_FeaureImportance = np.mean(abs(shap_valuestr[:,0]))
SM_FeaureImportance = np.mean(abs(shap_valuestr[:,1]))
VPD_FeaureImportance = np.mean(abs(shap_valuestr[:,2]))
Precip_FeaureImportance = np.mean(abs(shap_valuestr[:,3]))
WS_FeaureImportance = np.mean(abs(shap_valuestr[:,4]))
ST_FeaureImportance = np.mean(abs(shap_valuestr[:,5]))


print(EVI_FeaureImportance)
print(SM_FeaureImportance)
print(VPD_FeaureImportance)
print(Precip_FeaureImportance)
print(WS_FeaureImportance)
print(ST_FeaureImportance)

# ============================= Plotting ================================
features = ['Precipitation', 'Soil \nMoisture', 'Wind \nSpeed', 'VPD', 'Temperature']
shap_values = [Precip_FeaureImportance, SM_FeaureImportance, WS_FeaureImportance, VPD_FeaureImportance, ST_FeaureImportance]

# Sort values (optional: for descending order)
sorted_idx = np.argsort(shap_values)[::-1]
features = [features[i] for i in sorted_idx]
shap_values = [shap_values[i] for i in sorted_idx]

# Figure setup
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

# Bar plot
bars = ax.barh(features, shap_values, color='bisque', edgecolor='bisque', height=0.6,label='Mean Absolute SHAP Value')

# Reverse y-axis for descending feature importance
ax.invert_yaxis()

# Annotate values
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
            f'{width:.3f}', va='center', fontsize=8, color='black')

# Labels and styling
# ax.set_xlabel(r'Mean Absolute SHAP Value', fontsize=11)
ax.set_xlabel(r'Feature Importance', fontsize=11)
ax.set_xlim(0, max(shap_values) + 0.07)
# ax.xaxis.grid(True, linestyle='--', alpha=0.2)
ax.set_axisbelow(True)
ax.tick_params(axis='both', labelsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.text(0.94, 0.1, f'$R^2$ Score = {r2_score1:.2f}', 
        transform=ax.transAxes, fontsize=10,  
        ha='right', va='bottom')

ax.legend(loc='lower right', fontsize=9, frameon=False)

plt.subplots_adjust(left=0.2)
# plt.savefig('/home/yangbuw/Program/EmissionIntensity/pics/SFigure8_RF_SHAP_NHAF.pdf', dpi=300)
plt.tight_layout()
plt.show()

