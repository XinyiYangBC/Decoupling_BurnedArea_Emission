# Non-forest fires drive the global decoupling of burned area and fire emissions

This repository contains the analysis and visualization code used in the study

Non-forest fires drive the global decoupling of burned area and fire emissions

The code was used to process publicly available fire, vegetation, and meteorological datasets, perform statistical and structural equation modeling analyses, and generate the figures presented in the manuscript and Supplementary Information.

## System requirements

The analyses were conducted in a Conda environment using Python 3.9.19 on a Linux-based computing system.

The main Python packages used in the analyses include

- NumPy 1.26.4
- Pandas 2.2.2
- xarray 2023.6.0
- SciPy 1.13.1
- scikit-learn 1.5.2
- Matplotlib 3.9.2
- Cartopy 0.22.0
- Dask 2024.5.0
- netCDF4 1.6.2
- semopy 2.3.11
- statsmodels 0.14.5

The complete Conda environment, including all package dependencies and version numbers, is provided in `geophysics.yml`.

No non-standard hardware is required to run the analysis code.

## Installation

The analysis environment can be recreated using Conda

    conda env create -f geophysics.yml
    conda activate geophysics

Installation time will depend on the computing system and network connection.

## Input data

All datasets used in this study are publicly available from their original data providers. The main datasets include

- Global Fire Emissions Database version 5 (GFED5): burned area and fire emissions (https://www.globalfiredata.org/). 
- MODIS MOD13A2 vegetation indices (https://www.earthdata.nasa.gov/data/instruments/modis).
- MODIS MOD15A2H leaf area index (https://www.earthdata.nasa.gov/data/instruments/modis).
- Vegetation Optical Depth (VOD) vegetation biomass-related information (https://researchdata.tuwien.at/records/t74ty-tcx62).
- ERA5-Land meteorological variables used in the structural equation modeling analysis (https://www.ecmwf.int/).

Detailed descriptions and data sources are provided in the manuscript. Because the original datasets are publicly available and some are large global gridded datasets, the complete raw datasets are not redistributed in this repository.

## Code organization

The repository contains the source code used for data processing, statistical analysis, structural equation modeling (SEM), and figure generation.

- `main/`: Code used to perform the analyses and generate the results and figures presented in the main manuscript.
- `appendix/`: Code used to perform the additional analyses and generate the results and figures presented in the Supplementary Information.


## Usage

Before running the analysis scripts, download the required input datasets from the sources described in the manuscript and specify the corresponding input and output paths in the scripts.

The scripts can then be executed using Python, for example

    python script_name.py

Specific input requirements and analysis steps are documented within the corresponding scripts.

## Output

The analysis code produces processed data, statistical results, and figures used in the manuscript and Supplementary Information.

## Code availability

The source code associated with this study is provided in this repository for research reproducibility and reviewer assessment.