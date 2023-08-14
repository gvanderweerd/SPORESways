import string
import yaml
import os
import numpy as np
import pandas as pd
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


from friendly_data.converters import to_df
from frictionless.resource import Resource
from frictionless.package import Package
from sklearn.preprocessing import MinMaxScaler

# Define the order of the index for the capacity data series
INDEX_NAME_ORDER = ["year", "region", "technology", "spore"]
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
# Define mapping dictionaries that sum values for similar technologies under the same name (e.g. values for roof_mounted_pv and open_field_pv are summed under the name PV)
HEAT_PRODUCERS = {
    "biofuel_boiler": "Boiler",
    "electric_heater": "Electric heater",
    "hp": "Heat pump",
    "methane_boiler": "Boiler",
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
GRID_TECHS_SPORES = {"ac_transmission": "AC grid", "dc_transmission": "DC grid"}
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

# Define open circles for plotting 'All other SPORES'
pts = np.linspace(0, np.pi * 2, 24)
circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
vert = np.r_[circ, circ[::-1] * 0.7]
open_circle = mpl.path.Path(vert)


def read_spores_data(path_to_spores, slack="slack-10", file_names=None):
    """
    This function reads the SPORES dataset.
    :param path_to_spores:      The path to the SPORES dataset.
    :param slack:               The name of the directory that corresponds to the desired cost relaxation.
                                Default is a cost relaxation of 10% (slack-10).
    :param file_names:          A list of filenames if you only want to read specific .csv files instead of all the files.
                                If the parameter is not set, the function reads all files in the dataset.
    :return data:               A dictionary that contains the names of the .csv files as keys and the corresponding data as dataframe.
                                The dataframes can be accessed as data[filename].
    """

    print(f"Loading files in directory {path_to_spores}/{slack}")
    dpkg = Package(os.path.join(path_to_spores, slack, "datapackage.json"))
    resources = dpkg["resources"]
    if file_names is not None:
        print(f"Reading .csv files: {file_names}")
        resources = list(
            filter(lambda resource: resource["name"] in file_names, dpkg["resources"])
        )
    else:
        print("Reading all .csv files")

    data = {resource["name"]: to_df(resource).squeeze() for resource in resources}
    return data


def get_power_capacity(spores_data, save_to_csv=False):
    """

    :param spores_data:     A dictionary that contains SPORES data for 2020, categorised_spores and 2050.
    :param save_to_csv:     Set to True if you want to save the capacity data to a .csv file called "power_capacity.csv"
    :return:                A Series containing the capacities of electricity producting technologies for each country and for Europe as a whole
    """
    file_name = "nameplate_capacity"

    power_capacity = pd.Series(dtype="float64")
    for year in spores_data.keys():
        capacity_data = spores_data.get(year).get(file_name)
        capacity_national = (
            capacity_data.xs("tw", level="unit")
            .xs("electricity", level="carriers")
            .unstack("spore")
            .groupby(
                [REGION_MAPPING, ELECTRICITY_PRODUCERS_SPORES],
                level=["region", "technology"],
            )
            .sum()
            .stack("spore")
        )
        # # Filter capacity on carrier to avoid double counting of CHP capacity (that outputs multiple carriers)
        # carriers = capacity_national.index.get_level_values("carriers")
        # capacity_national = capacity_national[carriers == "electricity"]

        # Add the year as an index
        capacity_national = pd.concat({year: capacity_national}, names=["year"])

        # Calculate continental capacity
        capacity_eu = capacity_national.groupby(["year", "technology", "spore"]).sum()

        # Add "Europe" as an index with name "region" and reorder the index levels
        index_names = ["year", "region", "technology", "spore"]
        capacity_eu = pd.concat({"Europe": capacity_eu}, names=["region"])
        capacity_eu = capacity_eu.reorder_levels(index_names)

        # Concatenate national and continental values in one Series
        s = pd.concat([capacity_national, capacity_eu])
        power_capacity = power_capacity.append(s)

    # Concatenating series changes the MultiIndex to a tuple. These lines changes the Series back to a MultiIndex
    index = pd.MultiIndex.from_tuples(power_capacity.index, names=index_names)
    power_capacity = pd.Series(power_capacity.array, index=index)

    power_capacity *= 1000
    power_capacity.name = "capacity_gw"

    if save_to_csv:
        power_capacity.to_csv(f"data/power_capacity.csv")
    else:
        return power_capacity


def get_heat_capacity(spores_data, save_to_csv=False):
    """

    :param spores_data:     A dictionary that contains SPORES data for 2020, categorised_spores and 2050.
    :param save_to_csv:     Set to True if you want to save the capacity data to a .csv file called "heat_capacity.csv"
    :return:                A Series containing the capacities of heat producting technologies for each country and for Europe as a whole
    """

    file_name = "nameplate_capacity"

    heat_capacity = pd.Series(dtype="float64")
    for year in spores_data.keys():
        capacity_data = spores_data.get(year).get(file_name)
        capacity_national = (
            capacity_data.xs("tw", level="unit")
            .xs("heat", level="carriers")
            .unstack("spore")
            .groupby(
                [REGION_MAPPING, HEAT_PRODUCERS],
                level=["region", "technology"],
            )
            .sum()
            .stack("spore")
        )
        # # Filter capacity on carrier to avoid double counting of CHP capacity (that outputs multiple carriers)
        # carriers = capacity_national.index.get_level_values("carriers")
        # capacity_national = capacity_national[carriers == "heat"]

        # Add the year as an index
        capacity_national = pd.concat({year: capacity_national}, names=["year"])

        # Calculate continental capacity
        capacity_eu = capacity_national.groupby(["year", "technology", "spore"]).sum()

        # Add "Europe" as an index with name "region" and reorder the index levels
        index_names = ["year", "region", "technology", "spore"]
        capacity_eu = pd.concat({"Europe": capacity_eu}, names=["region"])
        capacity_eu = capacity_eu.reorder_levels(index_names)

        # Concatenate national and continental values in one Series
        s = pd.concat([capacity_national, capacity_eu])
        heat_capacity = heat_capacity.append(s)

    # Concatenating series changes the MultiIndex to a tuple. These lines changes the Series back to a MultiIndex
    index = pd.MultiIndex.from_tuples(heat_capacity.index, names=index_names)
    heat_capacity = pd.Series(heat_capacity.array, index=index)

    heat_capacity *= 1000
    heat_capacity.name = "capacity_gw"

    if save_to_csv:
        heat_capacity.to_csv(f"data/heat_capacity.csv")
    else:
        return heat_capacity


def get_storage_capacity(spores_data, save_to_csv=False):
    """

    :param spores_data:     A dictionary that contains SPORES data for 2020, categorised_spores and 2050.
    :param save_to_csv:     Set to True if you want to save the capacity data to a .csv file called "storage_capacity.csv"
    :return:                A Series containing the capacities of storage technologies for each country and for Europe as a whole
    """

    file_name = "storage_capacity"

    storage_capacity = pd.Series(dtype="float64")
    for year in spores_data.keys():
        capacity_data = spores_data.get(year).get(file_name)
        capacity_national = (
            capacity_data.xs("twh", level="unit")
            .unstack("spore")
            .groupby(
                [REGION_MAPPING, STORAGE_TECHNOLOGIES],
                level=["region", "technology"],
            )
            .sum()
            .stack("spore")
        )
        # Add the year as an index
        capacity_national = pd.concat({year: capacity_national}, names=["year"])
        # Calculate continental capacity
        capacity_eu = capacity_national.groupby(["year", "technology", "spore"]).sum()

        # Add "Europe" as an index with name "region" and reorder the index levels
        index_names = INDEX_NAME_ORDER
        capacity_eu = pd.concat({"Europe": capacity_eu}, names=["region"])
        capacity_eu = capacity_eu.reorder_levels(index_names)

        # Concatenate national and continental values in one Series
        s = pd.concat([capacity_national, capacity_eu])
        storage_capacity = storage_capacity.append(s)

    # Concatenating series changes the MultiIndex to a tuple. These lines changes the Series back to a MultiIndex
    index = pd.MultiIndex.from_tuples(storage_capacity.index, names=index_names)
    storage_capacity = pd.Series(storage_capacity.array, index=index)
    storage_capacity.name = "capacity_twh"

    if save_to_csv:
        storage_capacity.to_csv(f"data/{file_name}.csv")
    else:
        return storage_capacity


def get_grid_capacity(spores_data, expansion_only=False, save_to_csv=False):
    """

    :param spores_data:     A dictionary that contains SPORES data for 2020, categorised_spores and 2050.
    :param save_to_csv:     Set to True if you want to save the capacity data to a .csv file called "grid_capacity.csv"
    :return:                A Series containing the capacities of all power lines
    """

    if expansion_only:
        file_name = "grid_capacity_expansion"
    else:
        file_name = "grid_transfer_capacity"

    grid_capacity = pd.Series(dtype="float64")
    for year in spores_data.keys():
        capacity_data = spores_data.get(year).get(file_name)
        capacity = (
            capacity_data.unstack("spore")
            .groupby(
                ["importing_region", "exporting_region", GRID_TECHS_SPORES],
                level=["importing_region", "exporting_region", "technology"],
            )
            .sum()
            .stack("spore")
        )
        # Add the year as an index with name "year"
        capacity = pd.concat({year: capacity}, names=["year"])

        # Make sure the index names are in the correct order
        index_names = [
            "year",
            "importing_region",
            "exporting_region",
            "technology",
            "spore",
        ]
        capacity = capacity.reorder_levels(index_names)

        grid_capacity = grid_capacity.append(capacity)
        index = pd.MultiIndex.from_tuples(grid_capacity.index, names=index_names)
        grid_capacity = pd.Series(grid_capacity.array, index=index)

    grid_capacity *= 1000
    grid_capacity.name = "capacity_gw"

    if save_to_csv:
        grid_capacity.to_csv(f"data/{file_name}.csv")
    else:
        return grid_capacity


def plot_capacity_distribution(ax, capacity, year, country):
    scaler = MinMaxScaler()
    all_techs = capacity.index.get_level_values("technology").unique()
    capacity_normalised = pd.DataFrame(
        scaler.fit_transform(capacity.unstack("technology")), columns=all_techs
    )
    capacity_ranges = capacity.groupby("technology").agg(["min", "max"])

    sns.stripplot(
        ax=ax,
        data=capacity_normalised,
        marker=open_circle,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Normalised capacity")
    ax.set_xticks(range(len(all_techs)))
    ax.set_xticklabels(all_techs, fontsize=10)
    xticklabels = []
    for ticklabel in ax.get_xticklabels():
        technology = ticklabel.get_text()

        if technology in capacity.index.get_level_values("technology").unique():
            xticklabels.append(
                f"{technology}\n{capacity_ranges.loc[technology, 'min'].round(2)} - {capacity_ranges.loc[technology, 'max'].round(2)} [GW]"
            )
        else:
            xticklabels.append(f"{technology}\n0.0 - 0.0 [GW]")
    ax.set_xticklabels(xticklabels, fontsize=10)


if __name__ == "__main__":

    # Define paths to data
    paths = {
        "2050": os.path.join(os.getcwd(), "data", "euro-spores-results-v2022-05-13")
    }
    # Define for which cost relaxation we want to read the data
    slack = "slack-10"
    # Define which files we want to read
    files = [
        "nameplate_capacity",
        "flow_out_sum",
        "grid_capacity_expansion",
        "grid_transfer_capacity",
        "storage_capacity",
    ]

    data = {"2050": read_spores_data(paths["2050"], slack, files)}

    save = False
    power = get_power_capacity(spores_data=data, save_to_csv=save)
    heat = get_heat_capacity(spores_data=data, save_to_csv=save)
    storage = get_storage_capacity(spores_data=data, save_to_csv=save)
    # Grid is the total grid capacity (grid = existing grid + planned grid + grid_expansion)
    grid = get_grid_capacity(spores_data=data, save_to_csv=save)
    # Grid expansion is the extra expansion of the grid on top of the already existing and planned grid capacity for 2050.
    grid_expansion = get_grid_capacity(
        spores_data=data, expansion_only=True, save_to_csv=save
    )

    # Prepare data to plot
    country_to_plot = "Europe"
    year_to_plot = "2050"
    print(power)
    print(heat)
    print(storage)
    print(grid)
    power_Europe = power.loc[year_to_plot, country_to_plot, :, :]
    heat_Europe = heat.loc[year_to_plot, country_to_plot, :, :]
    storage_Europe = storage.loc[year_to_plot, country_to_plot, :, :]
    # The power grid data is not aggregated for each nation since this is a 'cross border' technology, we need to sum the values of all importing-exporting regions for each technology to obtain the grid capacity for Europe
    grid_Europe = grid.groupby(level=["technology", "spore"]).sum()
    print(power_Europe)
    print(heat_Europe)
    print(storage_Europe)
    print(grid_Europe)

    fig, axs = plt.subplots(nrows=4)
    plot_capacity_distribution(
        ax=axs[0], capacity=power_Europe, year=2050, country=country_to_plot
    )
    plot_capacity_distribution(
        ax=axs[1], capacity=heat_Europe, year=2050, country=country_to_plot
    )
    plot_capacity_distribution(
        ax=axs[2], capacity=storage_Europe, year=2050, country=country_to_plot
    )
    plot_capacity_distribution(
        ax=axs[3], capacity=grid_Europe, year=2050, country=country_to_plot
    )

    plt.show()
