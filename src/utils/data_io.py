import string
import yaml
import os
import shutil
import numpy as np
import pandas as pd
import csv

from friendly_data.converters import to_df
from frictionless.resource import Resource
from frictionless.package import Package

from src.utils.parameters import *
# from processing_functions import *

INDEX_NAME_ORDER = ["year", "region", "technology", "spore"]


def compare_technologies_2030_vs_2050(spores_data, filename):
    techs_2030 = set(spores_data.get("2030").get(filename).index.unique(level="technology"))
    techs_2050 = set(spores_data.get("2050").get(filename).index.unique(level="technology"))

    common_techs = techs_2030.intersection(techs_2050)
    unique_techs_in_2030 = techs_2030.difference(techs_2050)
    unique_techs_in_2050 = techs_2050.difference(techs_2030)

    print(f"Technologies that exist in 2030 & 2050 (file: {filename})")
    print(common_techs)
    print("")
    print(f"Technologies that exist only in 2030 (file: {filename})")
    print(unique_techs_in_2030)
    print("")
    print(f"Technologies that exist only in 2050 (file: {filename})")
    print(unique_techs_in_2050)
    print("")


def rename_spores_to_spore(path_to_csv_file):
    df = pd.read_csv(path_to_csv_file)
    if "spores" in df.columns:
        print(f"Renaming 'spores' to 'spore' in column names of file \n {path_to_csv_file}")
        df.rename(columns={"spores": "spore"}, inplace=True)
        df.to_csv(path_to_csv_file, index=False)


def match_column_name_with_index_file(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                rename_spores_to_spore(file_path)


def aggregate_categorised_spores(path_to_spores, path_to_result):
    """
    This function aggregates a categorised set of SPORES that has .csv files in multiple folders for each subset of SPORES into a set of SPORES that contains files with all SPORES that exist within the directory.
    :param path_to_spores: path to the directory where the categorised SPORES are placed
    :param path_to_result: path to the directory where the aggregated SPORES should be saved
    :return:
    """
    # Make new folder for aggregated spores if it does not exist yet
    if not os.path.exists(path_to_result):
        os.makedirs(path_to_result)
    # Get all unique filenames in the directory
    file_names = set()
    for root, dirs, files in os.walk(path_to_spores):
        for file in files:
            if file != '.DS_Store':
                file_names.add(file)
    # Process each filename
    for file_name in file_names:
        print(f"Aggregating {file_name}")
        # Create a new filename in the result directory
        result_file_path = os.path.join(path_to_result, file_name)
        result_file = open(result_file_path, "w")
        # Flag to indicate if top row with column names has been written
        top_row_written = False

        # Iterate of each subdirectory
        for root, dirs, files in os.walk(path_to_spores):
            # Check if the file exists
            if file_name in files:
                # Read the content of the file and write it to the result file
                file_path = os.path.join(root, file_name)
                with open(file_path, "r") as file:
                    # Skip first row if with column names if it has been written
                    if top_row_written:
                        next(file)
                    else:
                        top_row_written = True
                    # Write remaining rows to the resultfile
                    shutil.copyfileobj(file, result_file)
        # Close the result file
        result_file.close()


def read_spores_data(path_to_spores, file_names=None):
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

    print(f"Loading files in directory {path_to_spores}")
    dpkg = Package(os.path.join(path_to_spores, "datapackage.json"))
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
    spores = range(441)
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

    sim_data = {"2030": {}, "2050": {}}
    for filename in data_2050.keys():
        # Simulate a 2020 and 2030 dataset based on the 2050 dataset and a random factor if the file contains numbers as data, else just copy the file
        if data_2050[filename].dtype == np.float64:
            # Multiply selected spores data for 2050 with a randomising factor between 0 and 0.4
            sim_data["2030"][filename] = (
                data_2050[filename][spores]
                * 0.4
                * np.random.random(len(data_2050[filename][spores]))
            )
            # # Multiply spore 0 data for 2050 with a randomising factor between 0 and 0.1
            # sim_data["2020"][filename] = (
            #     data_2050[filename][0]
            #     * 0.1
            #     * np.random.random(len(data_2050[filename][0]))
            # )
        else:
            sim_data["2030"][filename] = data_2050[filename]
            # sim_data["2020"][filename] = data_2050[filename]

        if "spore" in data_2050["2050"][filename].index.names:
            # Filter all data that contains the column "spore" on the selected spores
            sim_data["2050"][filename] = data_2050[filename][spores]
            # # Add a column "spore" to the simulated 2020 data and set "spore" to 0 (needed because functions rely on a column with the spore number)
            # sim_data["2020"][filename] = pd.concat(
            #     {0: sim_data["2020"][filename]}, names=["spore"]
            # )
    return sim_data


def add_transport_electrification(data_dict):
    # Calculate transport electrification
    electric_transport = (
        data_dict.get("flow_out_sum")
        .unstack("technology")
        .reindex(["light_transport_ev", "heavy_transport_ev"], axis=1)
        .sum(axis=1, min_count=1)
        .groupby("spore")
        .sum()
    )
    all_transport = (
        data_dict.get("flow_out_sum")
        .xs("transport", level="carriers")
        .groupby("spore")
        .sum()
    )
    transport_electrification = 100 * electric_transport / all_transport
    # Set indices and name
    transport_electrification.index = pd.MultiIndex.from_product([transport_electrification.index, ["percentage"]], names=["spore", "unit"])
    transport_electrification.name = "transport_electrification"

    return transport_electrification


def add_heat_electrification(data_dict):
    # Calculate heat electrification
    electric_heat = (
        data_dict["flow_out_sum"]
        .unstack("technology")
        .reindex(["hp", "electric_heater", "electric_hob"], axis=1)
        .sum(axis=1, min_count=1)
        .groupby("spore")
        .sum()
    )
    all_heat = (
        data_dict["flow_out_sum"]
        .unstack("technology")
        .reindex(HEAT_TECHS_BUILDING + HEAT_TECHS_DISTRICT + COOKING_TECHS , axis=1)
        .sum(axis=1)
        .unstack("carriers")
        .loc[:, ["heat", "cooking"]]
        .sum(axis=1)
        .groupby("spore")
        .sum()
    )
    heat_electrification = 100 * electric_heat / all_heat
    # Set indices and name
    heat_electrification.index = pd.MultiIndex.from_product([heat_electrification.index, ["percentage"]], names=["spore", "unit"])
    heat_electrification.name = "heat_electrification"

    return heat_electrification


def add_electricity_production_gini(data_dict):
    electricity_production_gini = (
        data_dict.get("flow_out_sum")
        .xs("electricity", level="carriers")
        .unstack("technology")
        .reindex(list(ENERGY_PRODUCERS.keys()) + HEAT_TECHS_DISTRICT, axis=1)
        .sum(axis=1)
        .groupby(level=["spore", "region"])
        .sum()
        .groupby(level="spore").apply(get_gini)
    )
    # Set indices and name
    electricity_production_gini.index = pd.MultiIndex.from_product([electricity_production_gini.index, ["fraction"]], names=["spore", "unit"])
    electricity_production_gini.name = "electricity_production_gini"

    return electricity_production_gini


def add_storage_discharge_capacity(data_dict):
    storage_discharge_capacity = (
        data_dict.get("nameplate_capacity")
        .unstack("technology")
        .reindex(STORAGE_DISCHARGE_TECHS, axis=1)
        .sum(axis=1)
        .groupby("spore")
        .sum()
    )
    # Set indices and name
    storage_discharge_capacity.index = pd.MultiIndex.from_product([storage_discharge_capacity.index, ["tw"]], names=["spore", "unit"])
    storage_discharge_capacity.name = "storage_discharge_capacity"

    return storage_discharge_capacity


def add_average_national_imports(data_dict):
    average_national_import = (
        data_dict.get("net_import_sum").unstack(["spore", "unit"])
        .groupby(
            [_region_to_country, _region_to_country],
            level=["importing_region", "exporting_region"]
        ).sum().where(lambda x: x > 0).mean()
    )
    # Set name
    average_national_import.name = "average_national_import"

    return average_national_import


def get_paper_metrics(data_dict, result_path, save_to_csv=False):
    # Add metrics
    metrics = pd.concat([
        add_transport_electrification(data_dict),
        add_heat_electrification(data_dict),
        add_electricity_production_gini(data_dict),
        add_storage_discharge_capacity(data_dict),
        #FIXME: average_national_imports does not produce the same result as we find in "average_national_imports.csv" in 2050 euro-spores-results
        add_average_national_imports(data_dict)
    ], axis=1).rename_axis("metric", axis=1).stack()
    # Reorder indices and name data
    metrics.index = metrics.index.reorder_levels(["spore", "metric", "unit"])
    metrics.name = "paper_metrics"

    # Save to .csv or return dataframe
    if save_to_csv:
        metrics.to_csv(os.path.join(result_path, "paper_metrics.csv"))
    else:
        return metrics


def get_gini(metric):
    """
    Get the gini index for a particular metric.
    This is used to give an indication of the spatial 'equity' of a metric.
    """

    results = []
    vals = metric.values
    for i in vals:
        for j in vals:
            results.append(abs(i - j))
    return sum(results) / (2 * len(vals)**2 * vals.mean())


def _region_to_country(region):
    return region.split("_")[0]


def get_power_capacity(spores_data, result_path, save_to_csv=False):
    """

    :param spores_data:     A dictionary that contains SPORES data for 2020, 2030 and 2050.
    :param save_to_csv:     Set to True if you want to save the capacity data to a .csv file called "power_capacity.csv"
    :return:                A Series containing the capacities of electricity producting technologies for each country and for Europe as a whole
    """
    file_name = "nameplate_capacity"

    power_capacity = pd.Series(dtype="float64")
    for year in spores_data.keys():
        # print(f"YEAR: {year}")
        # print(spores_data.get(year).get(file_name))
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
        power_capacity.to_csv(os.path.join(result_path, "power_capacity.csv"))
    else:
        return power_capacity


def get_heat_capacity(spores_data, result_path, save_to_csv=False):
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
        heat_capacity.to_csv(os.path.join(result_path, "heat_capacity.csv"))
    else:
        return heat_capacity


def get_storage_capacity(spores_data, result_path, save_to_csv=False):
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
        storage_capacity.to_csv(os.path.join(result_path, f"{file_name}.csv"))
    else:
        return storage_capacity





def get_grid_capacity(
    spores_data, result_path, expansion_only=False, save_to_csv=False
):
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
            .groupby(
                [REGION_MAPPING, REGION_MAPPING, GRID_TECHS_SPORES],
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
        grid_capacity.to_csv(os.path.join(result_path, f"{file_name}.csv"))
    else:
        return grid_capacity


def get_power_capacity_irenastat(path, save_to_csv=False):
    # FIXME: make this function compatible with the .csv download (remove first rows and deal with column title difference)
    s = pd.read_csv(path, index_col=[0, 1, 2, 3], squeeze=True)
    s = s.rename(
        {
            "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
            "Republic of North Macedonia": "North Macedonia",
        }
    )
    s = s.rename({"Country/area": "region", "Technology": "technology", "Year": "year"})
    s = s.replace("..", 0)
    s = s.astype(float)

    power_capacity = s.groupby(
        ["region", ELECTRICITY_PRODUCERS_IRENASTAT, "year"],
        level=["region", "technology", "year"],
    ).sum()

    power_capacity /= 1000
    power_capacity.name = "capacity_gw"

    if save_to_csv:
        power_capacity.to_csv("data/power_capacity_irenastat.csv")
    else:
        return power_capacity