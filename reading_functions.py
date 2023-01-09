import string
import yaml
import os
import numpy as np
import pandas as pd
import csv

from friendly_data.converters import to_df
from frictionless.resource import Resource
from frictionless.package import Package

from variables import *
from processing_functions import *


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

def generate_sim_data(data_2050):
    """
    This function simulates a smaller dataset that contains data for the years 2020, 2030, and 2050 that is easier to handle and speeds up testing code.

    :param data_2050:       SPORES data from 2050.
    :return:                A dictionary that contains simulated SPORES data for 2020, 2030 and 2050, based on 11 SPORES.
    """

    spores = [0, 21, 32, 77, 100, 206, 255, 263, 328, 345, 431]
    """
    Select 11 SPORES for simulated data
    0 - ?
    21 - High onshore
    32 - No biofuel utilisation (0%)
    77 - Low PV
    100 - Low heat electrification
    206 - High heat electrification
    255 - Low storage capacity
    263 - High PV
    328 - High grid capacity expansion
    345 - Low grid capacity expansion
    431 - High storage capacity
    """

    sim_data = {"2020": {}, "2030": {}, "2050": {}}
    for filename in data_2050.keys():
        # Simulate a 2020 and 2030 dataset based on the 2050 dataset and a random factor if the file contains numbers as data, else just copy the file
        if data_2050[filename].dtype == np.float64:
            # Multiply selected spores data for 2050 with a randomising factor between 0 and 0.4
            sim_data["2030"][filename] = (
                data_2050[filename][spores]
                * 0.4
                * np.random.random(len(data_2050[filename][spores]))
            )
            # Multiply spore 0 data for 2050 with a randomising factor between 0 and 0.1
            sim_data["2020"][filename] = (
                data_2050[filename][0]
                * 0.1
                * np.random.random(len(data_2050[filename][0]))
            )
        else:
            sim_data["2030"][filename] = data_2050[filename]
            sim_data["2020"][filename] = data_2050[filename]

        if "spore" in data["2050"][filename].index.names:
            # Filter all data that contains the column "spore" on the selected spores
            sim_data["2050"][filename] = data_2050[filename][spores]
            # Add a column "spore" to the simulated 2020 data and set "spore" to 0 (needed because functions rely on a column with the spore number)
            sim_data["2020"][filename] = pd.concat(
                {0: sim_data["2020"][filename]}, names=["spore"]
            )
    return sim_data

def get_power_capacity(spores_data, save_to_csv=False):
    """

    :param spores_data:     A dictionary that contains SPORES data for 2020, 2030 and 2050.
    :param save_to_csv:     Set to True if you want to save the capacity data to a .csv file called "power_capacity.csv"
    :return:                A Series containing the capacities of electricity producting technologies for each country and for Europe as a whole
    """
    file_name = "nameplate_capacity"

    power_capacity = pd.Series(dtype="float64")
    for year in spores_data.keys():
        capacity_data = spores_data.get(year).get(file_name)
        capacity_national = (
            capacity_data.xs("tw", level="unit")
            .unstack("spore")
            .groupby(
                [REGION_MAPPING, ELECTRICITY_PRODUCERS, "carriers"],
                level=["region", "technology", "carriers"],
            )
            .sum()
            .stack("spore")
        )
        # Filter capacity on carrier to avoid double counting of CHP capacity (that outputs multiple carriers)
        carriers = capacity_national.index.get_level_values("carriers")
        capacity_national = capacity_national[carriers == "electricity"]

        # Add the year as an index
        capacity_national = pd.concat({year: capacity_national}, names=["year"])

        # Calculate continental capacity
        capacity_eu = capacity_national.groupby(
            ["year", "technology", "carriers", "spore"]
        ).sum()

        # Add "Europe" as an index with name "region" and reorder the index levels
        index_names = ["year", "region", "technology", "carriers", "spore"]
        capacity_eu = pd.concat({"Europe": capacity_eu}, names=["region"])
        capacity_eu = capacity_eu.reorder_levels(index_names)

        # Concatenate national and continental values in one Series
        s = pd.concat([capacity_national, capacity_eu])
        power_capacity = power_capacity.append(s)

    # Concatenating series changes the MultiIndex to a tuple. These lines changes the Series back to a MultiIndex
    index = pd.MultiIndex.from_tuples(power_capacity.index, names=index_names)
    power_capacity = pd.Series(power_capacity.array, index=index)
    power_capacity.name = "capacity_tw"

    if save_to_csv:
        power_capacity.to_csv(f"data/power_capacity.csv")
    else:
        return power_capacity

def get_heat_capacity(spores_data, save_to_csv):
    """

    :param spores_data:     A dictionary that contains SPORES data for 2020, 2030 and 2050.
    :param save_to_csv:     Set to True if you want to save the capacity data to a .csv file called "heat_capacity.csv"
    :return:                A Series containing the capacities of heat producting technologies for each country and for Europe as a whole
    """

    file_name = "nameplate_capacity"

    heat_capacity = pd.Series(dtype="float64")
    for year in spores_data.keys():
        capacity_data = spores_data.get(year).get(file_name)
        capacity_national = (
            capacity_data.xs("tw", level="unit")
            .unstack("spore")
            .groupby(
                [REGION_MAPPING, HEAT_PRODUCERS, "carriers"],
                level=["region", "technology", "carriers"],
            )
            .sum()
            .stack("spore")
        )
        # Filter capacity on carrier to avoid double counting of CHP capacity (that outputs multiple carriers)
        carriers = capacity_national.index.get_level_values("carriers")
        capacity_national = capacity_national[carriers == "heat"]

        # Add the year as an index
        capacity_national = pd.concat({year: capacity_national}, names=["year"])

        # Calculate continental capacity
        capacity_eu = capacity_national.groupby(
            ["year", "technology", "carriers", "spore"]
        ).sum()

        # Add "Europe" as an index with name "region" and reorder the index levels
        index_names = ["year", "region", "technology", "carriers", "spore"]
        capacity_eu = pd.concat({"Europe": capacity_eu}, names=["region"])
        capacity_eu = capacity_eu.reorder_levels(index_names)

        # Concatenate national and continental values in one Series
        s = pd.concat([capacity_national, capacity_eu])
        heat_capacity = heat_capacity.append(s)

    # Concatenating series changes the MultiIndex to a tuple. These lines changes the Series back to a MultiIndex
    index = pd.MultiIndex.from_tuples(heat_capacity.index, names=index_names)
    heat_capacity = pd.Series(heat_capacity.array, index=index)
    heat_capacity.name = "capacity_tw"

    if save_to_csv:
        heat_capacity.to_csv(f"data/heat_capacity.csv")
    else:
        return heat_capacity

def get_storage_capacity(spores_data, save_to_csv=False):
    """

    :param spores_data:     A dictionary that contains SPORES data for 2020, 2030 and 2050.
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
                [REGION_MAPPING, STORAGE_TECHNOLOGIES, "carriers"],
                level=["region", "technology", "carriers"],
            )
            .sum()
            .stack("spore")
        )
        # Add the year as an index
        capacity_national = pd.concat({year: capacity_national}, names=["year"])
        # Calculate continental capacity
        capacity_eu = capacity_national.groupby(
            ["year", "technology", "carriers", "spore"]
        ).sum()

        # Add "Europe" as an index with name "region" and reorder the index levels
        index_names = ["year", "region", "technology", "carriers", "spore"]
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

    :param spores_data:     A dictionary that contains SPORES data for 2020, 2030 and 2050.
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
            .groupby(level=["importing_region", "exporting_region"])
            .sum()
            .stack("spore")
        )
        # Add the year as an index with name "year"
        capacity = pd.concat({year: capacity}, names=["year"])

        # Make sure the index names are in the correct order
        index_names = ["year", "importing_region", "exporting_region", "spore"]
        capacity = capacity.reorder_levels(index_names)

        grid_capacity = grid_capacity.append(capacity)
        index = pd.MultiIndex.from_tuples(grid_capacity.index, names=index_names)
        grid_capacity = pd.Series(grid_capacity.array, index=index)

    grid_capacity.name = "capacity_tw"

    if save_to_csv:
        grid_capacity.to_csv(f"data/{file_name}.csv")
    else:
        return grid_capacity


if __name__ == "__main__":

    # Define paths to data
    paths = {
        "2050": os.path.join(os.getcwd(), "data", "euro-spores-results-v2022-05-13"),
        "ember_electricity_data": "data/ember_data/ember_electricitydata_yearly_full_release_long_format-1.csv",
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
    # Simulating a smaller dataset for 2020, 2030 and 2050
    data = generate_sim_data(data["2050"])

    power = get_power_capacity(spores_data=data, save_to_csv=False)
    heat = get_heat_capacity(spores_data=data, save_to_csv=False)
    storage = get_storage_capacity(spores_data=data, save_to_csv=False)
    grid = get_grid_capacity(spores_data=data, save_to_csv=False)




    df = pd.read_csv(paths["ember_electricity_data"])
    ember_countries = list(df["Area"].unique())

    print(df)
    print(df.columns)

    # Check if the dataset from ember-climate.org contains all countries
    for country in COUNTRIES:
        if country not in ember_countries:
            raise Exception(f"{country} does not exist in the ember data")

    # Filter on countries
    mask = df["Area"].isin(COUNTRIES)
    df = df[mask]

    # Filter on capacity and check if unit is in [GW]
    df = df[df["Category"] == "Capacity"]
    print(df["Unit"].unique())
    print(df["Subcategory"].unique())
    print(df["Variable"].unique())

    # TEST case for the Netherlands
    df_NL = df[df["Area"] == "Netherlands"]
    df_NL = df_NL[df_NL["Year"] == 2020]
    pd.set_option('display.max_columns', 100)
    print(df_NL)
    # & df["Year"] == 2020