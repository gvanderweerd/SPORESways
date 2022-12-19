import string
import yaml
import os

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



if __name__ == "__main__":

    paths = {
        "2050": os.path.join(os.getcwd(), "euro-spores-results-v2022-05-13")
    }
    slack = "slack-10"
    files = ["nameplate_capacity", "flow_out_sum"]

    data = {
        "2050": read_spores_data(paths["2050"], slack, files)
    }