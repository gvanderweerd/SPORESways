import numpy as np
import pandas as pd

# Technology sets
PRIMARY_ENERGY_SOURCES = {
    "waste_supply": "Waste",
    "biofuel_supply": "Biofuels",
    "hydro_reservoir": "Hydro",
    "hydro_run_of_river": "Hydro",
    "nuclear": "Nuclear heat",
    "open_field_pv": "PV",
    "roof_mounted_pv": "PV",
    "wind_offshore": "Offshore wind",
    "wind_onshore": "Onshore wind",
    "natural_gas_supply": "Natural gas",
    "oil_supply": "Oil",
}

# FIXME: base these numbers on research
ELECTRICITY_PRODUCERS_LIFE = {
    "PV": 25,
    "Offshore wind": 20,
    "Onshore wind": 25,
    "CCGT": 30,
    "CHP": 30,
    "Hydro": 30,
    "Nuclear": 30,
    "Bio to liquids": 30,
}

ELECTRICITY_PRODUCERS_SPORES = {
    "open_field_pv": "PV",
    "roof_mounted_pv": "PV",
    "wind_offshore": "Offshore wind",
    "wind_onshore": "Onshore wind",
    "ccgt": "CCGT",
    "chp_biofuel_extraction": "CHP",
    "chp_methane_extraction": "CHP",
    "chp_wte_back_pressure": "CHP",
    "hydro_reservoir": "Hydro",
    "hydro_run_of_river": "Hydro",
    "nuclear": "Nuclear",
}
ELECTRICITY_PRODUCERS_IRENASTAT = {
    "Solar photovoltaic": "PV",
    "Onshore wind energy": "Onshore wind",
    "Offshore wind energy": "Offshore wind",
    "Mixed Hydro Plants": "Hydro",
    "Renewable hydropower": "Hydro",
    "Pumped storage": "Hydro",
    "Nuclear": "Nuclear",
    "Natural gas": "CCGT",
}


vRES_PRODUCERS = {
    "open_field_pv": "vRES",
    "roof_mounted_pv": "vRES",
    "wind_offshore": "vRES",
    "wind_onshore": "vRES",
}
HEAT_PRODUCERS = {
    "biofuel_boiler": "Boiler",
    "chp_biofuel_extraction": "CHP",
    "chp_methane_extraction": "CHP",
    "chp_wte_back_pressure": "CHP",
    "electric_heater": "Electric heater",
    "hp": "Heat pump",
    "methane_boiler": "Boiler",
}
EL_HEAT_PRODUCERS = {
    # "biofuel_boiler": "Fuel",
    # "chp_biofuel_extraction": "Fuel",
    # "chp_methane_extraction": "Fuel",
    # "chp_wte_back_pressure": "Fuel",
    "electric_heater": "Electric",
    "hp": "Electric",
    # "methane_boiler": "Fuel",
}
HYDROGEN_PRODUCERS = {
    "electrolysis": "Electrolysis",
}
STORAGE_TECHNOLOGIES = {
    "battery": "Battery storage",
    "heat_storage_big": "Heat storage",
    "heat_storage_small": "Heat storage",
    "hydro_reservoir": "Hydro storage",
    "pumped_hydro": "Hydro storage",
    "hydrogen_storage": "Hydrogen storage",
    "methane_storage": "Methane storage",
}

# GAGR estimated by EU Market Outlook for Solar Power (2022-2026) converted to growth factor
pv_growth_rates_exponential = {
    "Belgium": 1.13,
    "France": 1.21,
    "Germany": 1.18,
    "Greece": 1.3,
    "Hungary": 1.23,
    "Ireland": 1.9,
    "Italy": 1.17,
    "Netherlands": 1.2,
    "Poland": 1.29,
    "Portugal": 1.36,
    "Romania": 1.44,
    "Spain": 1.31,
}

NUCLEAR_HEAT_MULTIPLIER = 1 / 0.4  # our model uses an efficiency of 40% for nuclear

# Locations
REGION_MAPPING = {
    "ALB_1": "Albania",
    "AUT_1": "Austria",
    "AUT_2": "Austria",
    "AUT_3": "Austria",
    "BEL_1": "Belgium",
    "BGR_1": "Bulgaria",
    "BIH_1": "Bosnia Herzegovina",  #'Bosniaand\nHerzegovina'
    "CHE_1": "Switzerland",
    "CHE_2": "Switzerland",
    "CYP_1": "Cyprus",
    "CZE_1": "Czechia",
    "CZE_2": "Czechia",
    "DEU_1": "Germany",
    "DEU_2": "Germany",
    "DEU_3": "Germany",
    "DEU_4": "Germany",
    "DEU_5": "Germany",
    "DEU_6": "Germany",
    "DEU_7": "Germany",
    "DNK_1": "Denmark",
    "DNK_2": "Denmark",
    "ESP_1": "Spain",
    "ESP_10": "Spain",
    "ESP_11": "Spain",
    "ESP_2": "Spain",
    "ESP_3": "Spain",
    "ESP_4": "Spain",
    "ESP_5": "Spain",
    "ESP_6": "Spain",
    "ESP_7": "Spain",
    "ESP_8": "Spain",
    "ESP_9": "Spain",
    "EST_1": "Estonia",
    "FIN_1": "Finland",
    "FIN_2": "Finland",
    "FRA_1": "France",
    "FRA_10": "France",
    "FRA_11": "France",
    "FRA_12": "France",
    "FRA_13": "France",
    "FRA_14": "France",
    "FRA_15": "France",
    "FRA_2": "France",
    "FRA_3": "France",
    "FRA_4": "France",
    "FRA_5": "France",
    "FRA_6": "France",
    "FRA_7": "France",
    "FRA_8": "France",
    "FRA_9": "France",
    "GBR_1": "United Kingdom",
    "GBR_2": "United Kingdom",
    "GBR_3": "United Kingdom",
    "GBR_4": "United Kingdom",
    "GBR_5": "United Kingdom",
    "GBR_6": "United Kingdom",  # Northern Ireland
    "GRC_1": "Greece",
    "GRC_2": "Greece",
    "HRV_1": "Croatia",
    "HUN_1": "Hungary",
    "IRL_1": "Ireland",
    "ISL_1": "Iceland",
    "ITA_1": "Italy",
    "ITA_2": "Italy",
    "ITA_3": "Italy",
    "ITA_4": "Italy",
    "ITA_5": "Italy",
    "ITA_6": "Italy",
    "LTU_1": "Lithuania",
    "LUX_1": "Luxembourg",
    "LVA_1": "Latvia",
    "MKD_1": "North Macedonia",  #'North Macedonia'
    "MNE_1": "Montenegro",
    "NLD_1": "Netherlands",
    "NOR_1": "Norway",
    "NOR_2": "Norway",
    "NOR_3": "Norway",
    "NOR_4": "Norway",
    "NOR_5": "Norway",
    "NOR_6": "Norway",
    "NOR_7": "Norway",
    "POL_1": "Poland",
    "POL_2": "Poland",
    "POL_3": "Poland",
    "POL_4": "Poland",
    "POL_5": "Poland",
    "PRT_1": "Portugal",
    "PRT_2": "Portugal",
    "ROU_1": "Romania",
    "ROU_2": "Romania",
    "ROU_3": "Romania",
    "SRB_1": "Serbia",
    "SVK_1": "Slovakia",
    "SVN_1": "Slovenia",
    "SWE_1": "Sweden",
    "SWE_2": "Sweden",
    "SWE_3": "Sweden",
    "SWE_4": "Sweden",
}
COUNTRIES = np.unique(list(REGION_MAPPING.values()))

# Define x-axis for past-, projected-, and spores-capacity data
years_2015_2021 = np.arange(2015, 2022)
years_2021_2030 = np.arange(2021, 2031)
years_2030_2050 = np.arange(2030, 2051)
years_2021_2050 = np.arange(2021, 2051)

# FIXME: move to processing functions.py
calculate_growth_factor = lambda start_capacity, end_capacity, start_year, end_year: (
    end_capacity / start_capacity
) ** (1 / (end_year - start_year))


"""Create a simulated series for power and heat capacities"""
# # Create the first level of the multi-index
# technology_p = ['PV', 'wind']
# technology_h = ['HP', 'boiler']
# # Create the first level of the multi-index
# country_p = ["Spain"]
# country_h = ["France", "Spain"]
# # Create the second level of the multi-index
# spore = list(range(6))
# # Create the multi-index
# index_p = pd.MultiIndex.from_product([country_p, technology_p, spore], names=['country', 'technology', 'spore'])
# index_h = pd.MultiIndex.from_product([country_h, technology_h, spore], names=['country', 'technology', 'spore'])
# # Create a random array of values between 0 and 1
# values_p = range(12)
# values_h = range(24)
# # Create the Series with the multi-index and values
# series_p = pd.Series(values_p, index=index_p)
# series_h = pd.Series(values_h, index=index_h)
# series = pd.concat([series_p, series_h]).groupby(level=series.index.names).first()
# print(series)
