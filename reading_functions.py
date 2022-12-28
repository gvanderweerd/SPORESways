import string
import yaml
import os
import numpy as np
import pandas as pd
import csv

from friendly_data.converters import to_df
from frictionless.resource import Resource
from frictionless.package import Package


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


def generate_sim_data(data):

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
    for filename in data["2050"].keys():
        # Simulate a 2020 and 2030 dataset based on the 2050 dataset and a random factor
        if data["2050"][filename].dtype == np.float64:
            # Multiply selected spores data for 2050 with a randomising factor between 0 and 0.4
            sim_data["2030"][filename] = (
                data["2050"][filename][spores]
                * 0.4
                * np.random.random(len(data["2050"][filename][spores]))
            )
            # Multiply spore 0 data for 2050 with a randomising factor between 0 and 0.1
            sim_data["2020"][filename] = (
                data["2050"][filename][0]
                * 0.1
                * np.random.random(len(data["2050"][filename][0]))
            )
        else:
            sim_data["2030"][filename] = data["2050"][filename]
            sim_data["2020"][filename] = data["2050"][filename]

        if "spore" in data["2050"][filename].index.names:
            # Filter all data that contains the column "spore" on the selected spores
            sim_data["2050"][filename] = data["2050"][filename][spores]
            # Add a column "spore" to the simulated 2020 data and set "spore" to 0 (needed because functions rely on a column with the spore number)
            sim_data["2020"][filename] = pd.concat(
                {0: sim_data["2020"][filename]}, names=["spore"]
            )
    return sim_data


if __name__ == "__main__":
    REGION_MAPPING = {
        "ALB_1": "Albania",
        "AUT_1": "Austria",
        "AUT_2": "Austria",
        "AUT_3": "Austria",
        "BEL_1": "Belgium",
        "BGR_1": "Bulgaria",
        "BIH_1": "Bosnia",  # 'Bosniaand\nHerzegovina'
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
        "FIN_2": "France",
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
        "GBR_1": "Great Britain",
        "GBR_2": "Great Britain",
        "GBR_3": "Great Britain",
        "GBR_4": "Great Britain",
        "GBR_5": "Great Britain",
        "GBR_6": "Great Britain",
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
        "MKD_1": "Macedonia",  # 'North Macedonia'
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

    # Define paths to data
    paths = {
        "2050": os.path.join(os.getcwd(), "data", "euro-spores-results-v2022-05-13"),
        "ember_electricity_data": "data/ember_data/ember_electricitydata_yearly_full_release_long_format-1.csv",
    }
    # Define for which cost relaxation we want to read the data
    slack = "slack-10"
    # Define which files we want to read
    files = ["nameplate_capacity", "flow_out_sum"]

    data = {"2050": read_spores_data(paths["2050"], slack, files)}

    # Open the CSV file
    with open(paths["ember_electricity_data"], "r") as file:
        # Create a CSV reader object
        reader = csv.reader(file)

        # Read the header row and the data rows
        headers, *rows = reader

        # # Filter the rows based on a condition
        filtered_rows = [
            row
            for row in rows
            if (row[0] in COUNTRIES) and row[10] == "Capacity" and row[12] == "Solar"
        ]
        print(filtered_rows)

        # Print the headers
        print(headers)
        #
        # # Print the filtered rows
        # for row in filtered_rows:
        #     area = row[0]
        #     subcategory = row[2]
        #     variable = row[3]
        #     value = row[5]
        #     print(area, subcategory, variable, value)
