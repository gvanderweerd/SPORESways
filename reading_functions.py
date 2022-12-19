import string
import yaml
import os
import numpy as np

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
        resources = list(filter(lambda resource: resource["name"] in file_names, dpkg["resources"]))
    else:
        print("Reading all .csv files")

    data = {
        resource["name"]: to_df(resource).squeeze()
        for resource in resources
    }
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

    sim_data = {
        "2020": {},
        "2030": {},
        "2050": {}
    }
    for filename in data["2050"].keys():
        # Simulate a 2020 and 2030 dataset based on the 2050 dataset and a random factor
        if data["2050"][filename].dtype == np.float64:
            # Multiply selected spores data for 2050 with a randomising factor between 0 and 0.4
            sim_data["2030"][filename] = data["2050"][filename][spores] * 0.4 * np.random.random(len(data["2050"][filename][spores]))
            # Multiply spore 0 data for 2050 with a randomising factor between 0 and 0.1
            sim_data["2020"][filename] = data["2050"][filename][0] * 0.1 * np.random.random(len(data["2050"][filename][0]))
        else:
            sim_data["2030"][filename] = data["2050"][filename]
            sim_data["2020"][filename] = data["2050"][filename]

        if "spore" in data["2050"][filename].index.names:
            # Filter all data that contains the column "spore" on the selected spores
            sim_data["2050"][filename] = data["2050"][filename][spores]
            # Add a column "spore" to the simulated 2020 data and set "spore" to 0 (needed because functions rely on a column with the spore number)
            sim_data["2020"][filename] = pd.concat({0: sim_data["2020"][filename]}, names=["spore"])
    return sim_data

if __name__ == "__main__":

    paths = {
        "2050": os.path.join(os.getcwd(), "euro-spores-results-v2022-05-13")
    }
    slack = "slack-10"
    files = ["nameplate_capacity", "flow_out_sum"]

    data = {
        "2050": read_spores_data(paths["2050"], slack, files)
    }