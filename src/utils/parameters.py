import numpy as np
import pandas as pd

"""
TECHNOLOGY SETS
"""
# Primary energy sources
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
ENERGY_PRODUCERS = {
    "waste_supply": "Waste",
    "biofuel_supply": "Biofuels",
    "hydro_reservoir": "Renewable electricity",
    "hydro_run_of_river": "Renewable electricity",
    "nuclear": "Nuclear electricity",
    "open_field_pv": "Renewable electricity",
    "roof_mounted_pv": "Renewable electricity",
    "wind_offshore": "Renewable electricity",
    "wind_onshore": "Renewable electricity",
    "coal_supply": "Coal",
}

# Heat sector
HEAT_TECHS_BUILDING = ["biofuel_boiler", "electric_heater", "hp", "methane_boiler"]
HEAT_TECHS_DISTRICT = [
    "chp_biofuel_extraction",
    "chp_methane_extraction",
    "chp_wte_back_pressure",
    "chp_hydrogen",
]
COOKING_TECHS = ["electric_hob", "gas_hob"]
HEAT_PRODUCERS = {
    # the full capacity of CHP is assigned in ELECTRICITY_PRODUCERS_SPORES (therefore this capacity is considered in power data and not in heat)
    "biofuel_boiler": "Boiler",
    "electric_heater": "Electric heater",
    "hp": "Heat pump",
    "methane_boiler": "Boiler",
}

# Electricity sector

# Storage
STORAGE_DISCHARGE_TECHS = [
    "battery_storage",
    "heat_storage_big",
    "heat_storage_small",
    "hydro_storage",
    "hydrogen_storage",
    "ccgt",
]


GRID_TECHS_SPORES = {"ac_transmission": "Power grid", "dc_transmission": "Power grid"}

ELECTRICITY_PRODUCERS_SPORES = {
    "open_field_pv": "PV",
    "roof_mounted_pv": "PV",
    "wind_offshore": "Offshore wind",
    "wind_onshore": "Onshore wind",
    "coal_power_plant": "Coal",
    "ccgt": "Gas turbines",
    "chp_biofuel_extraction": "Gas turbines",
    "chp_methane_extraction": "Gas turbines",
    "chp_wte_back_pressure": "Gas turbines",
    "chp_hydrogen": "Gas turbines",
    "hydro_reservoir": "Hydro",
    "hydro_run_of_river": "Hydro",
    "nuclear": "Nuclear",
    "battery": "Battery",
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

EL_HEAT_PRODUCERS = {
    # "biofuel_boiler": "Fuel",
    # "chp_biofuel_extraction": "Fuel",
    # "chp_methane_extraction": "Fuel",
    # "chp_wte_back_pressure": "Fuel",
    "electric_heater": "Electric",
    "hp": "Electric",
    # "methane_boiler": "Fuel",
}
FUEL_PRODUCERS = {
    "electrolysis": "Hydrogen",
    # FIXME: find the other fuel producers and summarise them as "Conventional fuels" (think, diesel, methane, etc.)
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

TECH_UNITS = {
    "Electric heating": "GW",
    "Boiler": "GW",
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

COUNTRY_MAPPING_IRENASTAT = {
    "Netherlands (Kingdom of the)": "Netherlands",
    "United Kingdom of Great Britain and Northern Ireland (the)": "United Kingdom",
}
TECH_MAPPING_IRENASTAT = {
    "Onshore wind energy": "Onshore wind",
    "Offshore wind energy": "Offshore wind",
}

TECH_MAPPING_EMBER = {
    "Gas": "CCGT",
    "Solar": "PV",
}

# Metrics
RENAME_METRICS = {
    "transport_electrification": "Transport electrification",
    "heat_electrification": "Heat electrification",
    "electricity_production_gini": "Electricity production gini",
    "storage_discharge_capacity": "Storage discharge capacity",
    "average_national_import": "Avg. national import",
    "biofuel_utilisation": "Biofuel utilisation",
}

# Locations
REGION_MAPPING = {
    "ALB": "Albania",
    "ALB_1": "Albania",
    "AUT": "Austria",
    "AUT_1": "Austria",
    "AUT_2": "Austria",
    "AUT_3": "Austria",
    "BEL": "Belgium",
    "BEL_1": "Belgium",
    "BGR": "Bulgaria",
    "BGR_1": "Bulgaria",
    "BIH": "Bosnia Herzegovina",  # 'Bosniaand\nHerzegovina'
    "BIH_1": "Bosnia Herzegovina",  #'Bosniaand\nHerzegovina'
    "CHE": "Switzerland",
    "CHE_1": "Switzerland",
    "CHE_2": "Switzerland",
    "CYP": "Cyprus",
    "CYP_1": "Cyprus",
    "CZE": "Czechia",
    "CZE_1": "Czechia",
    "CZE_2": "Czechia",
    "DEU": "Germany",
    "DEU_1": "Germany",
    "DEU_2": "Germany",
    "DEU_3": "Germany",
    "DEU_4": "Germany",
    "DEU_5": "Germany",
    "DEU_6": "Germany",
    "DEU_7": "Germany",
    "DNK": "Denmark",
    "DNK_1": "Denmark",
    "DNK_2": "Denmark",
    "ESP": "Spain",
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
    "EST": "Estonia",
    "EST_1": "Estonia",
    "FIN": "Finland",
    "FIN_1": "Finland",
    "FIN_2": "Finland",
    "FRA": "France",
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
    "GBR": "United Kingdom",
    "GBR_1": "United Kingdom",
    "GBR_2": "United Kingdom",
    "GBR_3": "United Kingdom",
    "GBR_4": "United Kingdom",
    "GBR_5": "United Kingdom",
    "GBR_6": "United Kingdom",  # Northern Ireland
    "GRC": "Greece",
    "GRC_1": "Greece",
    "GRC_2": "Greece",
    "HRV": "Croatia",
    "HRV_1": "Croatia",
    "HUN": "Hungary",
    "HUN_1": "Hungary",
    "IRL": "Ireland",
    "IRL_1": "Ireland",
    "ISL": "Iceland",
    "ISL_1": "Iceland",
    "ITA": "Italy",
    "ITA_1": "Italy",
    "ITA_2": "Italy",
    "ITA_3": "Italy",
    "ITA_4": "Italy",
    "ITA_5": "Italy",
    "ITA_6": "Italy",
    "LTU": "Lithuania",
    "LTU_1": "Lithuania",
    "LUX": "Luxembourg",
    "LUX_1": "Luxembourg",
    "LVA": "Latvia",
    "LVA_1": "Latvia",
    "MKD": "North Macedonia",  # 'North Macedonia'
    "MKD_1": "North Macedonia",  #'North Macedonia'
    "MNE": "Montenegro",
    "MNE_1": "Montenegro",
    "NLD": "Netherlands",
    "NLD_1": "Netherlands",
    "NOR": "Norway",
    "NOR_1": "Norway",
    "NOR_2": "Norway",
    "NOR_3": "Norway",
    "NOR_4": "Norway",
    "NOR_5": "Norway",
    "NOR_6": "Norway",
    "NOR_7": "Norway",
    "POL": "Poland",
    "POL_1": "Poland",
    "POL_2": "Poland",
    "POL_3": "Poland",
    "POL_4": "Poland",
    "POL_5": "Poland",
    "PRT": "Portugal",
    "PRT_1": "Portugal",
    "PRT_2": "Portugal",
    "ROU": "Romania",
    "ROU_1": "Romania",
    "ROU_2": "Romania",
    "ROU_3": "Romania",
    "SRB": "Serbia",
    "SRB_1": "Serbia",
    "SVK": "Slovakia",
    "SVK_1": "Slovakia",
    "SVN": "Slovenia",
    "SVN_1": "Slovenia",
    "SWE": "Sweden",
    "SWE_1": "Sweden",
    "SWE_2": "Sweden",
    "SWE_3": "Sweden",
    "SWE_4": "Sweden",
}
COUNTRIES = np.unique(list(REGION_MAPPING.values()))

YEARS = range(2010, 2051)
POWER_TECH_ORDER = [
    "PV",
    "Onshore wind",
    "Offshore wind",
    "CHP",
    "CCGT",
    "Nuclear",
    "Hydro",
    "Bio to liquids",
]

# Define x-axis for past-, projected-, and spores-capacity data
years_2000_2050 = np.arange(2000, 2051)
years_2015_2021 = np.arange(2015, 2022)
years_2021_2030 = np.arange(2021, 2031)
years_2030_2050 = np.arange(2030, 2051)
years_2021_2050 = np.arange(2021, 2051)

s_curve_params_power_sector = {
    "PV": {"x0": 2035, "K_min": 0.2, "K_max": 0.3},
    "Onshore wind": {"x0": 2035, "K_min": 0.2, "K_max": 0.3},
    "Offshore wind": {"x0": 2035, "K_min": 0.2, "K_max": 0.3},
    "Hydro": {"x0": 2035, "K_min": 0.2, "K_max": 0.3},
    "Nuclear": {"x0": 2035, "K_min": 0.2, "K_max": 0.3},
    "CHP": {"x0": 2035, "K_min": 0.2, "K_max": 0.3},
    "CCGT": {"x0": 2035, "K_min": 0.2, "K_max": 0.3},
}

# FIXME: move to processing functions.py
calculate_growth_factor = lambda start_capacity, end_capacity, start_year, end_year: (
    end_capacity / start_capacity
) ** (1 / (end_year - start_year))


"""Create a simulated series for power and heat capacities"""
# # Create the first level of the multi-index
# technology_p = ['PV', 'wind']
# technology_h = ['Heat pump', 'boiler']
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

# """
# Define historical data
# """
# # Power sector
# power_technologies = ["PV", "Onshore wind", "Offshore wind", "Hydro", "Nuclear", "CCGT", "CHP"]
# years = ["2020", "2021"]
# power_present_gw = [
#     # FIXME: make real estimate of PV capacity 2020-2021:
#     100,
#     110,
#     # FIXME: make real estimate of PV capacity 2020-2021:
#     100,
#     110,
#     # FIXME: make real estimate of PV capacity 2020-2021:
#     100,
#     110,
#     # FIXME: make real estimate of PV capacity 2020-2021:
#     100,
#     110,
#     # FIXME: make real estimate of PV capacity 2020-2021:
#     100,
#     110,
#     # FIXME: make real estimate of PV capacity 2020-2021:
#     100,
#     110,
#     # FIXME: make real estimate of PV capacity 2020-2021:
#     100,
#     110,
# ]
# index = pd.MultiIndex.from_product([power_technologies, years], names=["technology", "year"])
# power_present_gw = pd.Series(power_present_gw, index=index)
# # Heat sector
# heat_technologies = ["Boiler", "Electrical heating"]
# years = ["2020", "2021"]
# heat_present_gw = [
#     # FIXME: make real estimate of boiler capacity 2020:
#     100,
#     105,
#     # Electrical heating capacity 2020: estimated as heat pump capacity obtained from eurostat database: https://ec.europa.eu/eurostat/databrowser/view/NRG_INF_HPTC__custom_4864488/default/table?lang=en
#     253.191469,
#     271.288156
# ]
# index = pd.MultiIndex.from_product([heat_technologies, years], names=["technology", "year"])
# heat_present_gw = pd.Series(heat_present_gw, index=index)
