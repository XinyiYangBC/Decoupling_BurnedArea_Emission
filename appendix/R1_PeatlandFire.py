import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats
from scipy.stats import linregress, pearsonr, t

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator



# ============== GFED5 ==================
startyear_use       = 2002;    
endyear_use         = 2022;
var_name = ['TOTL', 'SAVA', 'AGRI', 'BORF', 'DEFO', 'TEMF', 'PEAT']

#   === BA ===
dir_file = f"/scratch/yangbuw/Data/GFED5/GFED5_1/Output/GFED5_1_BA_m2_byBiomes_025x025_2002_2022_monthly.nc"
ds2 = xr.open_dataset(dir_file)
C = ds2[var_name[6]] /1E6/1E6*100 # m to km to Mha
region_C = C.sel(longitude=slice(63, 180), latitude=slice(58, 74), time=slice(str(startyear_use), str(endyear_use)))
region_C_yearly = region_C.resample(time='YS').sum()
region_C_sum = region_C_yearly.sum(dim=['latitude', 'longitude'])
BA_region = region_C_sum

#   === Emission ===
dir_file = f"/scratch/yangbuw/Data/GFED5/GFED5_1/Output/GFED5_1_CO2_by_gC_byBiomes_025x025_2002_2022_monthly.nc"
ds2 = xr.open_dataset(dir_file)
C = ds2[var_name[6]] /1E12 # g C to Tg C
region_C = C.sel(longitude=slice(63, 180), latitude=slice(58, 74), time=slice(str(startyear_use), str(endyear_use)))
region_C_yearly = region_C.resample(time='YS').sum()
region_C_sum = region_C_yearly.sum(dim=['latitude', 'longitude'])
EM_region = region_C_sum

EI_region = EM_region/BA_region

# ============== Siberia PeatFire ==================
file = '/home/yangbuw/Program/EmissionIntensity/response/data/Siberia_PeatFire_FullTables_2001-2023.csv'

df = pd.read_csv(file)

df = df.rename(columns={'Unnamed: 0': 'Year'})

df['PeatFire_Emission_Intensity (Tg C/Mha)'] = (
    df['PeatFire_Emissions (Tg C)'] /
    df['PeatFire Area (Mha)']
)
df = df.set_index('Year')

# print(df.head())



peat_emission = df.loc[2002:2022, 'PeatFire_Emission_Intensity (Tg C/Mha)']

result = linregress(peat_emission.index, peat_emission.values)

print(f"Slope = {result.slope:.4f} Tg C/year")
print(f"P-value = {result.pvalue:.4f}")





#====================== GFED5 PEAT EI Trend ==============================

peat_ei_gfed5 = EI_region.sel(time=slice('2002', '2022'))

years = peat_ei_gfed5.time.dt.year.values
values = peat_ei_gfed5.values

result = linregress(years, values)

print(f"Slope = {result.slope:.4f} Tg C/Mha/year")
print(f"P-value = {result.pvalue:.4f}")

