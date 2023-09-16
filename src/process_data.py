import os
from utils.data_io import *


def get_raw_data(paths_to_raw_data, years):
    # Define which files we want to read
    files = [
        "nameplate_capacity",
        "grid_transfer_capacity",
        "flow_out_sum",
        "net_import_sum",
        "storage_capacity",
        "primary_energy_supply",
        "final_consumption",
    ]

    # Read spores results for the years that were defined
    data = {}
    for year in years:
        data[year] = read_spores_data(
            path_to_spores=paths_to_raw_data[year], file_names=files
        )

    return data


def convert_spore_names_to_integers(input_data_dict):
    output_data_dict = {}

    # Get mapping dictionary to change spore names from strings to integers
    string_to_integer_map = get_spore_string_to_integer_map(
        input_data_dict.get("nameplate_capacity")
    )
    for filename, data in input_data_dict.items():
        index_names = data.index.names

        if "spore" in index_names:
            # Change spore names from strings to integers
            data = data.reset_index()
            data["spore"] = data["spore"].map(string_to_integer_map)
            data = data.set_index(index_names)[filename]

        output_data_dict[filename] = data

    return output_data_dict


def get_spore_string_to_integer_map(spores_series):
    return {
        spore: i for i, spore in enumerate(spores_series.index.unique(level="spore"))
    }


def process_paper_metrics(spores_data):
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

    return metrics


def process_grid_transfer_capacity(spores_data):
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

    return grid_transfer_capacity


def process_power_capacity(spores_data):
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

    return power_capacity


def process_storage_capacity(spores_data):
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

    return storage_capacity


def process_final_consumption(spores_data):
    """

    :param spores_data:     A dictionary that contains SPORES data for 2020, 2030 and 2050.
    :param save_to_csv:     Set to True if you want to save the capacity data to a .csv file called "power_capacity.csv"
    :return:                A Series containing the capacities of electricity producting technologies for each country and for Europe as a whole
    """

    final_consumption_national = (
        spores_data.get("final_consumption")
        .groupby(
            [REGION_MAPPING, "carriers", "spore"],
            level=["region", "carriers", "spore"],
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

    return final_consumption


def process_primary_energy_supply(spores_data):
    """

    :param spores_data:     A dictionary that contains SPORES data for 2020, 2030 and 2050.
    :param save_to_csv:     Set to True if you want to save the capacity data to a .csv file called "power_capacity.csv"
    :return:                A Series containing the capacities of electricity producting technologies for each country and for Europe as a whole
    """

    tpes_national = (
        spores_data.get("primary_energy_supply")
        .groupby(
            [REGION_MAPPING, "carriers", "spore"],
            level=["region", "carriers", "spore"],
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

    return tpes


def add_internation_transmission_to_power_capacity(power_data, grid_data):
    # Compute international grid capacity per country
    international_transmission = grid_data.groupby(["importing_region", "spore"]).sum()
    international_transmission.index = pd.MultiIndex.from_tuples(
        [
            (index[0], "International transmission", index[1])
            for index in international_transmission.index
        ]
    ).set_names(["region", "technology", "spore"])

    # Compute total international grid capacity in europe
    df = grid_data.reset_index()
    df["unique_link"] = df.apply(
        lambda row: frozenset([row["importing_region"], row["exporting_region"]]),
        axis=1,
    )
    df_unique = df.drop_duplicates(subset=["spore", "unique_link"], keep="first")
    international_transmission_eu = df_unique.groupby("spore")[
        "grid_transfer_capacity"
    ].sum()
    international_transmission_eu.index = pd.MultiIndex.from_tuples(
        [
            ("Europe", "International transmission", spore)
            for spore in international_transmission_eu.index
        ]
    ).set_names(["region", "technology", "spore"])

    # Add international transmission capacity data to power capacity data
    return pd.concat(
        [power_data, international_transmission, international_transmission_eu]
    ).sort_index(level=["region", "technology", "spore"])


def save_processed_data(spores_data, path_to_processed_data, save=False):
    for year in spores_data.keys():
        # If the directory data/processed/{year} does not exist make directory
        path_to_result = os.path.join(path_to_processed_data, year)
        if not os.path.exists(path_to_result):
            os.makedirs(path_to_result)

        # Calculate metrics (used in Pickering et al. 2022)
        paper_metrics = process_paper_metrics(spores_data.get(year))

        # Process spores results:
        #   - to a national level (and include total values for the whole continent under region "Europe")
        #   - combine the results for all years in that are defined in one file
        power = process_power_capacity(spores_data.get(year))
        grid_transfer_capacity = process_grid_transfer_capacity(spores_data.get(year))
        storage_capacity = process_storage_capacity(spores_data.get(year))
        final_consumption = process_final_consumption(spores_data.get(year))
        tpes = process_primary_energy_supply(spores_data.get(year))

        # Add international transmission to power_capacity
        power = add_internation_transmission_to_power_capacity(
            power, grid_transfer_capacity
        )

        if save:
            paper_metrics.to_csv(os.path.join(path_to_result, "paper_metrics.csv"))
            power.to_csv(os.path.join(path_to_result, "power_capacity.csv"))
            grid_transfer_capacity.to_csv(
                os.path.join(path_to_result, "grid_transfer_capacity.csv")
            )
            storage_capacity.to_csv(
                os.path.join(path_to_result, "storage_capacity.csv")
            )
            tpes.to_csv(os.path.join(path_to_result, "primary_energy_supply.csv"))


if __name__ == "__main__":
    """
    0. SET PARAMETERS
    """

    # Years for which we want to process spores results
    years = ["2030", "2050"]

    # Set to True if you want processed data to be saved as new csv files
    save = False

    # Set this to a list of years (like ["2030"]) if the spores results are provided in different folders for different categories of spores results
    # Set this to None if spores results are provided in one folder containing all spores
    categorised_spores_years = None

    # Define path to where the raw euro-spores-results can be found
    paths_to_raw_spores = {
        "2030": os.path.join(
            os.getcwd(), "../data/raw/euro-spores-results-2030/aggregated"
        ),
        "2050": os.path.join(
            os.getcwd(),
            "../data/raw/euro-spores-results-2050/aggregated-slack-10",
        ),
    }

    # Define path to where the script will save the processed spores results
    path_to_processed_spores = os.path.join(os.getcwd(), "..", "data", "processed")

    """
    1. PROCESS DATA
    """

    # Change column names from "spores" to "spore" in the .csv files that have this column
    match_column_name_with_index_file(paths_to_raw_spores.get("2030"))

    # Aggregate spores results that are provided in different folders for different categories of spores results instead of having all spores in each file
    if categorised_spores_years is not None:
        for year in categorised_spores_years:
            aggregate_categorised_spores(
                path_to_spores=os.path.join(
                    os.getcwd(),
                    f"../data/raw/euro-spores-results-{year}/categorised",
                ),
                path_to_result=os.path.join(
                    os.getcwd(),
                    f"../data/raw/euro-spores-results-{year}/aggregated/data",
                ),
            )
        # FIXME: we need to manually change the title name of the grid capacity from "grid_transfer_capacity.csv" to "grid_transfer_capacity.csv"

    # Get raw data
    data = get_raw_data(paths_to_raw_data=paths_to_raw_spores, years=years)

    # Change "spore" to integer values for 2030 SPORES results
    data["2030"] = convert_spore_names_to_integers(data.get("2030"))

    # Process spores and save to "processed data"
    save_processed_data(
        spores_data=data,
        path_to_processed_data=path_to_processed_spores,
        save=save,
    )

    """
    COMPARE PAPER METRICS CALCULATION
    """
    # Get paper metrics of 2050 for comparison
    paper_metrics_2050 = read_spores_data(
        path_to_spores=paths_to_raw_spores["2050"], file_names=["paper_metrics"]
    ).get("paper_metrics")
    # Get paper metrics from own calculation
    paper_metrics_2050_calculated = process_paper_metrics(data.get("2050"))

    # Filter out 1 metric for comparison
    metric = "electricity_production_gini"

    # Compare metric calculatino result
    difference = (
        paper_metrics_2050_calculated.loc[:, metric, :, :]
        - paper_metrics_2050.loc[:, metric, :, :]
    )
    # print(difference.describe())
