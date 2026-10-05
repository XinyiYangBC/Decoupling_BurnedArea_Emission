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

region_LAI_yearly = xr.open_dataset("/scratch/yangbuw/Data/LAI/MOD15A2H/MOD15A2H_LAI_025x025_2001_2024_monthly.nc")["LAI"]
region_LAI_yearly = region_LAI_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use1), str(endyear_use1))
).resample(time='MS').sum()
region_LAI_yearly = region_LAI_yearly.groupby("time.month") - region_LAI_yearly.groupby("time.month").mean("time")
region_LAI_yearly_lag = region_LAI_yearly.shift(time=months_lag)
region_LAI_yearly_lag = region_LAI_yearly_lag.sel(time=slice(str(startyear_use), str(endyear_use)))

region_EVI_yearly = xr.open_dataset("/scratch/yangbuw/Data/EVI/MOD13A2_EVI_025x025_2001_2024_monthly_025x025.nc")["EVI"]
region_EVI_yearly = region_EVI_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use1), str(endyear_use1))
).resample(time='MS').sum()
region_EVI_yearly = region_EVI_yearly.groupby("time.month") - region_EVI_yearly.groupby("time.month").mean("time")
region_EVI_yearly_lag = region_EVI_yearly.shift(time=months_lag)
region_EVI_yearly_lag = region_EVI_yearly_lag.sel(time=slice(str(startyear_use), str(endyear_use)))
# ==================== Lagged Vegeation Index ==========================
region_Emission_yearly = xr.open_dataset("/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_EI_SAVA_025x025_2002_2022_monthly.nc")
region_Emission_yearly = region_Emission_yearly["EI"].sel(longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),time=slice(str(startyear_use), str(endyear_use)))


region_EI_yearly = xr.open_dataset("/scratch/yangbuw/Data/GFED5/output_biome/GFED5_Beta_EI_SAVA_025x025_2002_2022_monthly.nc")
region_EI_yearly = region_EI_yearly["EI"].sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
).resample(time='MS').sum()
region_EI_yearly = region_EI_yearly.groupby("time.month") - region_EI_yearly.groupby("time.month").mean("time")

region_Dead_Foliage_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_FUEL_MAP_kg_per_m2_2003_2021_monthly_025x025.nc")["Dead_Foliage"]
region_Dead_Foliage_yearly = region_Dead_Foliage_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
).resample(time='MS').sum()
region_Dead_Foliage_yearly = region_Dead_Foliage_yearly.groupby("time.month") - region_Dead_Foliage_yearly.groupby("time.month").mean("time")

region_Dead_Wood_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_FUEL_MAP_kg_per_m2_2003_2021_monthly_025x025.nc")["Dead_Wood"]
region_Dead_Wood_yearly = region_Dead_Wood_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
).resample(time='MS').sum() 
region_Dead_Wood_yearly = region_Dead_Wood_yearly.groupby("time.month") - region_Dead_Wood_yearly.groupby("time.month").mean("time")

region_Live_Leaf_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_FUEL_MAP_kg_per_m2_2003_2021_monthly_025x025.nc")["Live_Leaf"]
region_Live_Leaf_yearly = region_Live_Leaf_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
).resample(time='MS').sum() 
region_Live_Leaf_yearly = region_Live_Leaf_yearly.groupby("time.month") - region_Live_Leaf_yearly.groupby("time.month").mean("time")

region_Live_Wood_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_FUEL_MAP_kg_per_m2_2003_2021_monthly_025x025.nc")["Live_Wood"]
region_Live_Wood_yearly = region_Live_Wood_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
).resample(time='MS').sum() 
region_Live_Wood_yearly = region_Live_Wood_yearly.groupby("time.month") - region_Live_Wood_yearly.groupby("time.month").mean("time")

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
# ).resample(time='MS').sum()
# region_FL_fine_rec_yearly = region_FL_fine_rec_yearly.groupby("time.month") - region_FL_fine_rec_yearly.groupby("time.month").mean("time")


# region_FL_coarse_rec_yearly = xr.open_dataset("/scratch/yangbuw/DLEM/1_ExperimentOutput/PostProcess/Experiment/fire/fire_on_GFED5_prescribed/FL_coarse_rec_Gridpool_fire_on_GFED5_prescribed_monthly_025x025_1997_2023_g_per_m2_S2N.nc")["FL_coarse_rec"]
# region_FL_coarse_rec_yearly = region_FL_coarse_rec_yearly.sel(
#     longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
#     time=slice(str(startyear_use), str(endyear_use))
# ).resample(time='MS').sum()
# region_FL_coarse_rec_yearly = region_FL_coarse_rec_yearly.groupby("time.month") - region_FL_coarse_rec_yearly.groupby("time.month").mean("time")


# region_FL_liveherb_rec_yearly = xr.open_dataset("/scratch/yangbuw/DLEM/1_ExperimentOutput/PostProcess/Experiment/fire/fire_on_GFED5_prescribed/FL_liveherb_rec_Gridpool_fire_on_GFED5_prescribed_monthly_025x025_1997_2023_g_per_m2_S2N.nc")["FL_liveherb_rec"]
# region_FL_liveherb_rec_yearly = region_FL_liveherb_rec_yearly.sel(
#     longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
#     time=slice(str(startyear_use), str(endyear_use))
# ).resample(time='MS').sum()
# region_FL_liveherb_rec_yearly = region_FL_liveherb_rec_yearly.groupby("time.month") - region_FL_liveherb_rec_yearly.groupby("time.month").mean("time")


# region_FL_livewood_rec_yearly = xr.open_dataset("/scratch/yangbuw/DLEM/1_ExperimentOutput/PostProcess/Experiment/fire/fire_on_GFED5_prescribed/FL_livewood_rec_Gridpool_fire_on_GFED5_prescribed_monthly_025x025_1997_2023_g_per_m2_S2N.nc")["FL_livewood_rec"]
# region_FL_livewood_rec_yearly = region_FL_livewood_rec_yearly.sel(
#     longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
#     time=slice(str(startyear_use), str(endyear_use))
# ).resample(time='MS').sum()
# region_FL_livewood_rec_yearly = region_FL_livewood_rec_yearly.groupby("time.month") - region_FL_livewood_rec_yearly.groupby("time.month").mean("time")

# # ================= DLEM estimated Fuel Load =================


region_VOD_yearly = xr.open_dataset("/scratch/yangbuw/Data/VOD/VODCA/VODCA_CXKu_1988-2021_monthly_025x025.nc")["VODCA_CXKu"]
region_VOD_yearly = region_VOD_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
).resample(time='MS').sum()
region_VOD_yearly = region_VOD_yearly.groupby("time.month") - region_VOD_yearly.groupby("time.month").mean("time")

region_LAI_yearly = xr.open_dataset("/scratch/yangbuw/Data/LAI/MOD15A2H/MOD15A2H_LAI_025x025_2001_2024_monthly.nc")["LAI"]
region_LAI_yearly = region_LAI_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
).resample(time='MS').sum()
region_LAI_yearly = region_LAI_yearly.groupby("time.month") - region_LAI_yearly.groupby("time.month").mean("time")

region_EVI_yearly = xr.open_dataset("/scratch/yangbuw/Data/EVI/MOD13A2_EVI_025x025_2001_2024_monthly_025x025.nc")["EVI"]
region_EVI_yearly = region_EVI_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
).resample(time='MS').sum()
region_EVI_yearly = region_EVI_yearly.groupby("time.month") - region_EVI_yearly.groupby("time.month").mean("time")


region_SM_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_swvl1_percentage_1950_2024_monthly_025x025.nc")["swvl1"]
region_SM_yearly = region_SM_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
).resample(time='MS').sum()
region_SM_yearly = region_SM_yearly.groupby("time.month") - region_SM_yearly.groupby("time.month").mean("time")

region_VPD_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_VPD_kpa_1950_2024_Monthly_025x025.nc")["vpd"]
region_VPD_yearly = region_VPD_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
).resample(time='MS').sum()
region_VPD_yearly = region_VPD_yearly.groupby("time.month") - region_VPD_yearly.groupby("time.month").mean("time")

region_Precip_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_precip_m_1950_2024_monthly_025x025.nc")["precip"] * 1e2
region_Precip_yearly = region_Precip_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
).resample(time='MS').sum()
region_Precip_yearly = region_Precip_yearly.groupby("time.month") - region_Precip_yearly.groupby("time.month").mean("time")

region_Wind_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_WindSpeed_mpers_1950_2024_Monthly_025x025.nc")["uv"]
region_Wind_yearly = region_Wind_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
).resample(time='MS').sum()
region_Wind_yearly = region_Wind_yearly.groupby("time.month") - region_Wind_yearly.groupby("time.month").mean("time")

region_SKT_yearly = xr.open_dataset("/scratch/yangbuw/Data/ERA5_Land/ERA5_Land_skt_k_1950_2024_monthly_025x025.nc")["skt"]
region_SKT_yearly = region_SKT_yearly.sel(
    longitude=slice(Lon_W, Lon_E), latitude=slice(Lat_S, Lat_N),
    time=slice(str(startyear_use), str(endyear_use))
).resample(time='MS').sum()
region_SKT_yearly = region_SKT_yearly.groupby("time.month") - region_SKT_yearly.groupby("time.month").mean("time")

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

# DoF + chi2 + CFI + GFI + RMSEA
num_var_output= 2 #20+ 5
# ---- Multiprocessing SEM function ----
def process_grid(args):
    i, j = args
    EI_use = region_EI_yearly[:, i, j]
    mask = (EI_use != 0) & (~np.isnan(EI_use))

    if np.sum(mask) < 5:
        return (i, j) + (np.nan,) * num_var_output
    try:
        data = pd.DataFrame({
            'VOD': region_VOD_yearly_lag[:, i, j][mask],
            'LAI': region_LAI_yearly_lag[:, i, j][mask],
            'EVI': region_EVI_yearly_lag[:, i, j][mask],
            'SM': region_SM_yearly[:, i, j][mask],
            'VPD': region_VPD_yearly[:, i, j][mask],
            'Precipitation': region_Precip_yearly[:, i, j][mask],
            'Wind_Speed': region_Wind_yearly[:, i, j][mask],
            'Surface_Temperature': region_SKT_yearly[:, i, j][mask],
            'EI': region_EI_yearly[:, i, j][mask]
        })

        if data.isnull().values.any() or np.any(data.std() < 1e-6):
            return (i, j) + (np.nan,) * num_var_output

        scaler = StandardScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

        model = semopy.Model(model_spec)
        model.fit(data_scaled)
        stats = semopy.calc_stats(model)

        # inspect_df = model.inspect(se_robust=True)
        inspect_df = model.inspect(std_est=True,se_robust=True)
        return (
            i, j,
            inspect_df['Est. Std'].values[14],
            inspect_df['p-value'].values[14]
        )

    except:
        return (i, j) + (np.nan,) *num_var_output

# ---- Run Multiprocessing ----
print("Multiprocessing starts...", datetime.now())
nlat = region_EI_yearly.sizes['latitude']
nlon = region_EI_yearly.sizes['longitude']
indices = [(i, j) for i in range(nlat) for j in range(nlon)]

with Pool(processes=min(65, cpu_count())) as pool:
    results = pool.map(process_grid, indices)

# ---- Save outputs ----
# Path Coefficients
sem_output_coeff1 = xr.full_like(region_EI_yearly[0,:,:], fill_value=np.nan)
sem_output_p_value1 = xr.full_like(region_EI_yearly[0,:,:], fill_value=np.nan)


print("Multiprocessing ends...", datetime.now())

for result in results:
    if len(result) != (num_var_output+2):
        continue

    (i, j,
     c1, p1,
    ) = result

    sem_output_coeff1[i, j] = c1
    sem_output_p_value1[i, j] = p1


# Keep Statistical Significant values
sem_output_coeff1_sig = sem_output_coeff1.where(sem_output_p_value1 < 0.1, np.nan)


# Save to NetCDF
print("Saving results...", datetime.now())
dataset = xr.Dataset({
    # Path coefficients
    "coeff1": sem_output_coeff1,
    "coeff1_pvalue": sem_output_p_value1,

})

dataset["longitude"].attrs["units"] = "degrees_east"
dataset["latitude"].attrs["units"] = "degrees_north"
dataset.attrs["title"] = "SEM Output in NetCDF (Fuel Load + Fire Weather Indicators)"
dataset.attrs["institution"] = "Department of Earth and Environmental Sciences, Boston College"
dataset.attrs["source"] = "Data post-processed by Xinyi Yang"
dataset.attrs["history"] = f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

out_file = "/home/yangbuw/Program/EmissionIntensity/code_version1/data_code/data/SEM_global_LatentVar_colinearity_test_025x025_rmAC_2Path_VegIndex_Lag_3_month.nc"
if os.path.exists(out_file):
    os.remove(out_file)
dataset.to_netcdf(out_file)
print("Done.", datetime.now())
