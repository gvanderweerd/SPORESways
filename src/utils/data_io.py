import string
import yaml
import json
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


def find_years(path_to_processed_data):
    for root, dirs, files in os.walk(path_to_processed_data):
        if root == path_to_processed_data:
            return dirs


def compare_technologies_2030_vs_2050(spores_data, filename):
    techs_2030 = set(
        spores_data.get("2030").get(filename).index.unique(level="technology")
    )
    techs_2050 = set(
        spores_data.get("2050").get(filename).index.unique(level="technology")
    )

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
        print(
            f"Renaming 'spores' to 'spore' in column names of file \n {path_to_csv_file}"
        )
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
            if file != ".DS_Store":
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

    data = {}
    for resource in resources:
        df = to_df(resource).squeeze()
        df.name = resource["name"]
        data[resource["name"]] = df
    return data


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
    transport_electrification.index = pd.MultiIndex.from_product(
        [transport_electrification.index, ["percentage"]], names=["spore", "unit"]
    )
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
        .reindex(HEAT_TECHS_BUILDING + HEAT_TECHS_DISTRICT + COOKING_TECHS, axis=1)
        .sum(axis=1)
        .unstack("carriers")
        .loc[:, ["heat", "cooking"]]
        .sum(axis=1)
        .groupby("spore")
        .sum()
    )
    heat_electrification = 100 * electric_heat / all_heat
    # Set indices and name
    heat_electrification.index = pd.MultiIndex.from_product(
        [heat_electrification.index, ["percentage"]], names=["spore", "unit"]
    )
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
        .groupby(level="spore")
        .apply(get_gini)
    )
    # Set indices and name
    electricity_production_gini.index = pd.MultiIndex.from_product(
        [electricity_production_gini.index, ["fraction"]], names=["spore", "unit"]
    )
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
    storage_discharge_capacity.index = pd.MultiIndex.from_product(
        [storage_discharge_capacity.index, ["tw"]], names=["spore", "unit"]
    )
    storage_discharge_capacity.name = "storage_discharge_capacity"

    return storage_discharge_capacity


def add_average_national_imports(data_dict):
    average_national_import = (
        data_dict.get("net_import_sum")
        .unstack(["spore", "unit"])
        .groupby(
            [_region_to_country, _region_to_country],
            level=["importing_region", "exporting_region"],
        )
        .sum()
        .where(lambda x: x > 0)
        .mean()
    )
    # Set name
    average_national_import.name = "average_national_import"

    return average_national_import


def process_paper_metrics(spores_data, result_path, save_to_csv=False):
    # Add metrics
    metrics = (
        pd.concat(
            [
                add_transport_electrification(spores_data),
                add_heat_electrification(spores_data),
                add_electricity_production_gini(spores_data),
                add_storage_discharge_capacity(spores_data),
                # FIXME: average_national_imports does not produce the same result as we find in "average_national_imports.csv" in 2050 euro-spores-results
                add_average_national_imports(spores_data),
            ],
            axis=1,
        )
        .rename_axis("metric", axis=1)
        .stack()
    )
    # Reorder indices and name data
    metrics.index = metrics.index.reorder_levels(["spore", "metric", "unit"])
    metrics.name = "paper_metrics"

    # Save to .csv and return dataframe
    if save_to_csv:
        metrics.to_csv(os.path.join(result_path, "paper_metrics.csv"))

    return metrics


def process_grid_transfer_capacity(spores_data, result_path, save_to_csv=False):
    grid_transfer_capacity = (
        spores_data.get("grid_transfer_capacity")
        .groupby(
            ["spore", REGION_MAPPING, REGION_MAPPING],
            level=["spore", "importing_region", "exporting_region"],
        )
        .sum()
    )

    # Drop all subnational links where importing_region == exporting_region
    grid_transfer_capacity = grid_transfer_capacity.loc[
        grid_transfer_capacity.index.get_level_values("importing_region")
        != grid_transfer_capacity.index.get_level_values("exporting_region")
    ]

    if save_to_csv:
        grid_transfer_capacity.to_csv(
            os.path.join(result_path, "grid_transfer_capacity.csv")
        )

    return grid_transfer_capacity


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
    return sum(results) / (2 * len(vals) ** 2 * vals.mean())


def _region_to_country(region):
    return region.split("_")[0]


def filter_power_capacity(df, spatial_resolution):
    if spatial_resolution == "national":
        df = df.drop(index="Europe", level="region")
    else:
        df = df.xs(spatial_resolution, level="region", drop_level=False)
    return df


def process_power_capacity(spores_data, result_path, save_to_csv=False):
    """

    :param spores_data:     A dictionary that contains SPORES data for 2020, 2030 and 2050.
    :param save_to_csv:     Set to True if you want to save the capacity data to a .csv file called "power_capacity.csv"
    :return:                A Series containing the capacities of electricity producting technologies for each country and for Europe as a whole
    """

    power_capacity_national = (
        spores_data.get("nameplate_capacity")
        .xs("tw", level="unit")
        .xs("electricity", level="carriers")
        .unstack("spore")
        .groupby(
            [REGION_MAPPING, ELECTRICITY_PRODUCERS_SPORES],
            level=["region", "technology"],
        )
        .sum()
        .stack("spore")
    )
    # Calculate continental capacity
    power_capacity_eu = power_capacity_national.groupby(["technology", "spore"]).sum()

    # Add "Europe" as an index with name "region" and reorder the index levels
    index_names = ["region", "technology", "spore"]
    power_capacity_eu = pd.concat({"Europe": power_capacity_eu}, names=["region"])
    power_capacity_eu = power_capacity_eu.reorder_levels(index_names)

    # Concatenate national and continental values in one Series
    power_capacity = pd.concat([power_capacity_national, power_capacity_eu])

    # Transform values from TW to GW
    power_capacity *= 1000
    power_capacity.name = "capacity_gw"

    if save_to_csv:
        power_capacity.to_csv(os.path.join(result_path, "power_capacity.csv"))
    else:
        return power_capacity


def process_storage_capacity(spores_data, result_path, save_to_csv=False):
    """

    :param spores_data:     A dictionary that contains SPORES data for 2020, 2030 and 2050.
    :param save_to_csv:     Set to True if you want to save the capacity data to a .csv file called "power_capacity.csv"
    :return:                A Series containing the capacities of electricity producting technologies for each country and for Europe as a whole
    """

    storage_capacity_national = (
        spores_data.get("storage_capacity")
        .xs("twh", level="unit")
        .xs("electricity", level="carriers")
        .unstack("spore")
        .groupby(
            [REGION_MAPPING, "technology"],
            level=["region", "technology"],
        )
        .sum()
        .stack("spore")
    )
    # Calculate continental capacity
    storage_capacity_eu = storage_capacity_national.groupby(
        ["technology", "spore"]
    ).sum()

    # Add "Europe" as an index with name "region" and reorder the index levels
    index_names = ["region", "technology", "spore"]
    storage_capacity_eu = pd.concat({"Europe": storage_capacity_eu}, names=["region"])
    storage_capacity_eu = storage_capacity_eu.reorder_levels(index_names)

    # Concatenate national and continental values in one Series
    storage_capacity = pd.concat([storage_capacity_national, storage_capacity_eu])

    storage_capacity.name = "capacity_twh"

    if save_to_csv:
        storage_capacity.to_csv(os.path.join(result_path, "storage_capacity.csv"))
    else:
        return storage_capacity


def process_final_consumption(spores_data, result_path, save_to_csv=False):
    """

    :param spores_data:     A dictionary that contains SPORES data for 2020, 2030 and 2050.
    :param save_to_csv:     Set to True if you want to save the capacity data to a .csv file called "power_capacity.csv"
    :return:                A Series containing the capacities of electricity producting technologies for each country and for Europe as a whole
    """

    final_consumption_national = (
        spores_data.get("final_consumption")
        .groupby(
            ["spore", REGION_MAPPING, "carriers"],
            level=["spore", "region", "carriers"],
        )
        .sum()
    )
    # Calculate continental capacity
    final_consumption_eu = final_consumption_national.groupby(
        ["carriers", "spore"]
    ).sum()

    # Add "Europe" as an index with name "region" and reorder the index levels
    index_names = ["region", "carriers", "spore"]
    final_consumption_eu = pd.concat({"Europe": final_consumption_eu}, names=["region"])
    final_consumption_eu = final_consumption_eu.reorder_levels(index_names)

    # Concatenate national and continental values in one Series
    final_consumption = pd.concat([final_consumption_national, final_consumption_eu])

    final_consumption.name = "final_consumption_twh"

    if save_to_csv:
        final_consumption.to_csv(os.path.join(result_path, "final_consumption.csv"))
    else:
        return final_consumption


def process_primary_energy_supply(spores_data, result_path, save_to_csv=False):
    """

    :param spores_data:     A dictionary that contains SPORES data for 2020, 2030 and 2050.
    :param save_to_csv:     Set to True if you want to save the capacity data to a .csv file called "power_capacity.csv"
    :return:                A Series containing the capacities of electricity producting technologies for each country and for Europe as a whole
    """

    tpes_national = (
        spores_data.get("primary_energy_supply")
        .groupby(
            ["spore", REGION_MAPPING, "carriers"],
            level=["spore", "region", "carriers"],
        )
        .sum()
    )
    # Calculate continental supply
    tpes_eu = tpes_national.groupby(["carriers", "spore"]).sum()

    # Add "Europe" as an index with name "region" and reorder the index levels
    index_names = ["region", "carriers", "spore"]
    tpes_eu = pd.concat({"Europe": tpes_eu}, names=["region"])
    tpes_eu = tpes_eu.reorder_levels(index_names)

    # Concatenate national and continental values in one Series
    tpes = pd.concat([tpes_national, tpes_eu])

    tpes.name = "primary_energy_supply_twh"

    if save_to_csv:
        tpes.to_csv(os.path.join(result_path, "primary_energy_supply.csv"))
    else:
        return tpes


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


def add_cluster_index_to_series(data, cluster_mapper):
    # Get name of the data
    data_name = data.name

    # Add cluster using cluster mapper
    data = data.reset_index()
    data["cluster"] = data["spore"].map(cluster_mapper)

    # Get index names
    index_names = list(data.columns)
    index_names.remove(data_name)

    # Return clustered data as a multi index series
    return data.set_index(index_names)[data_name]


def get_spore_to_scenario_maps(path_to_processed_data, years, resolution="Europe"):
    spore_to_scenario_maps = {}

    for year in years:
        # Load SPORE to scenario map
        with open(
            os.path.join(
                "../data/processed",
                year,
                f"spore_to_scenario_{resolution}.json",
            ),
            "r",
        ) as json_file:
            spore_to_scenario_maps[year] = json.load(json_file)
            spore_to_scenario_maps[year] = {
                int(spore): cluster
                for spore, cluster in spore_to_scenario_maps[year].items()
            }
    return spore_to_scenario_maps


def load_processed_paper_metrics(path_to_processed_data, years):
    metrics = {}

    for year in years:
        # Paper metrics
        metrics[year] = pd.read_csv(
            os.path.join(path_to_processed_data, year, "paper_metrics.csv"),
            index_col=["spore", "metric", "unit"],
        ).squeeze()

    return metrics


def load_raw_final_consumption(path_to_raw_data):
    consumption = {}

    consumption["2030"] = pd.read_csv(
        os.path.join(
            path_to_raw_data,
            "euro-spores-results-2030",
            "aggregated",
            "data",
            "final_consumption.csv",
        ),
        index_col=["spore", "sector", "subsector", "carriers", "locs", "unit"],
    ).squeeze()
    consumption["2050"] = pd.read_csv(
        os.path.join(
            path_to_raw_data,
            "euro-spores-results-2050",
            "aggregated-slack-10",
            "data",
            "final_consumption.csv",
        ),
        index_col=["spore", "sector", "subsector", "carriers", "locs", "unit"],
    ).squeeze()

    return consumption


def load_raw_primary_energy_supply(path_to_raw_data):
    tpes = {}

    tpes["2030"] = (
        pd.read_csv(
            os.path.join(
                path_to_raw_data,
                "euro-spores-results-2030",
                "aggregated",
                "data",
                "primary_energy_supply.csv",
            ),
            index_col=["spore", "locs", "carriers", "unit"],
        )
        .squeeze()
        .groupby(
            ["spore", REGION_MAPPING, "carriers", "unit"],
            level=["spore", "locs", "carriers", "unit"],
        )
        .sum()
    )
    tpes["2050"] = (
        pd.read_csv(
            os.path.join(
                path_to_raw_data,
                "euro-spores-results-2050",
                "aggregated-slack-10",
                "data",
                "primary_energy_supply.csv",
            ),
            index_col=["spore", "locs", "carriers", "unit"],
        )
        .squeeze()
        .groupby(
            ["spore", REGION_MAPPING, "carriers", "unit"],
            level=[
                "spore",
                "locs",
                "carriers",
                "unit",
            ],
        )
        .sum()
    )

    return tpes


def load_processed_power_capacity(path_to_processed_data, years):
    power = {}

    for year in years:
        # Get power capacity
        power[year] = pd.read_csv(
            os.path.join(path_to_processed_data, year, "power_capacity.csv"),
            index_col=["region", "technology", "spore"],
        ).squeeze()

    return power


def load_processed_grid_transfer_capacity(path_to_processed_data, years):
    grid = {}

    for year in years:
        # Get power capacity
        grid[year] = pd.read_csv(
            os.path.join(path_to_processed_data, year, "grid_transfer_capacity.csv"),
            index_col=["spore", "importing_region", "exporting_region"],
        ).squeeze()

    return grid


def get_processed_data(path_to_processed_data, years):
    power = {}
    metrics = {}

    for year in years:
        # Get power capacity
        power[year] = pd.read_csv(
            os.path.join(path_to_processed_data, year, "power_capacity.csv"),
            index_col=["region", "technology", "spore"],
        ).squeeze()
        # power_capacity[year].name = os.path.basename(file_path)

        # Paper metrics
        metrics[year] = pd.read_csv(
            os.path.join(path_to_processed_data, year, "paper_metrics.csv"),
            index_col=["spore", "metric", "unit"],
        ).squeeze()

    return power, metrics


# def get_processed_data(path_to_processed_data,"):
#     years = find_years(path_to_processed_data)
#
#     power_capacity = {}
#     paper_metrics = {}
#
#     for year in years:
#         # Get power capacity
#         power_capacity[year] = pd.read_csv(
#             os.path.join(path_to_processed_data, year, "power_capacity.csv"),
#             index_col=["region", "technology", "spore"],
#         ).squeeze()
#         # power_capacity[year].name = os.path.basename(file_path)
#
#         # Filter power capacity on the right spatial resolution
#         # if resolution == "continental":
#         #     power_capacity[year] = power_capacity.get(year).xs(
#         #         "Europe", level="region", drop_level=False
#         #     )
#         if resolution == "national":
#             power_capacity[year] = power_capacity.get(year).drop(
#                 index="Europe", level="region"
#             )
#         else:
#             power_capacity[year] = power_capacity.get(year).xs(
#                 resolution, level="region", drop_level=False
#             )
#
#         # Paper metrics
#         paper_metrics[year] = pd.read_csv(
#             os.path.join(path_to_processed_data, year, "paper_metrics.csv"),
#             index_col=["spore", "metric", "unit"],
#         ).squeeze()
#
#     return power_capacity, paper_metrics
#


def count_spores_per_cluster(clustered_data):
    return clustered_data.reset_index().groupby("cluster")["spore"].nunique()


def describe_scenario(power_capacity):
    return (
        power_capacity.groupby(["cluster", "technology"])
        .agg(["min", "mean", "median", "max"])
        .reset_index()
    )


def find_feasible_scenarios(spores, power_capacity):
    scenarios = power_capacity.index.unique(level="cluster")
    feasible_scenarios = power_capacity.loc[:, :, spores, :].index.unique(
        level="cluster"
    )
    infeasible_scenarios = scenarios.difference(feasible_scenarios)
    return list(feasible_scenarios), list(infeasible_scenarios)
