import os
from utils.data_io import *


def get_raw_data(paths_to_raw_data, years):
    # Define which files we want to read
    files = [
        "nameplate_capacity",
        "grid_transfer_capacity",
        "storage_capacity",
        "flow_out_sum",
        "net_import_sum",
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


def save_processed_data(spores_data, path_to_processed_data, save=False):
    for year in spores_data.keys():
        # If the directory data/processed/{year} does not exist make directory
        path_to_result = os.path.join(path_to_processed_data, year)
        if not os.path.exists(path_to_result):
            os.makedirs(path_to_result)

        # Calculate metrics (used in Pickering et al. 2022)
        paper_metrics = get_paper_metrics(
            data_dict=spores_data.get(year),
            result_path=path_to_result,
            save_to_csv=save,
        )

        # Process spores results:
        #   - to a national level (and include total values for the whole continent under region "Europe")
        #   - combine the results for all years in that are defined in one file
        power = get_power_capacity2(
            spores_data=spores_data.get(year),
            result_path=path_to_result,
            save_to_csv=save,
        )
    return power, paper_metrics


if __name__ == "__main__":
    """
    0. SET PARAMETERS
    """

    # Years for which we want to process spores results
    years = ["2030", "2050"]

    # Set to True if you want processed data to be saved as new csv files
    save = True

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
                    "../data/raw/euro-spores-results-{year}/aggregated/data",
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
    paper_metrics_2050_calculated = get_paper_metrics(
        data_dict=data.get("2050"),
        result_path=os.path.join(path_to_processed_spores, "2050"),
        save_to_csv=False,
    )

    # Filter out 1 metric for comparison
    metric = "electricity_production_gini"

    # Compare metric calculatino result
    difference = (
        paper_metrics_2050_calculated.loc[:, metric, :, :]
        - paper_metrics_2050.loc[:, metric, :, :]
    )
    print(difference.describe())
