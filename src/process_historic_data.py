import os
import pandas as pd

from src.utils.parameters import *

COLUMN_ORDER = ["year", "region", "technology", "capacity_gw"]


def read_irenastat_data(path, year):
    df = pd.read_csv(
        path,
        header=1,
    )
    # Transform data to GW
    df["Installed electricity capacity (MW)"] /= 1000
    # Set column names
    df.columns = ["region", "technology", "grid_connection", "year", "capacity_gw"]
    # Filter out older years and Off-grid connection
    df = df[
        (df["year"] == year)
        & (df["grid_connection"] == "On-grid")
        & df["technology"].isin(
            ["Offshore wind energy", "Onshore wind energy", "Solar photovoltaic"]
        )
    ]
    df = df.drop("grid_connection", axis=1).reset_index(drop=True)
    # Rename country
    df["region"] = df["region"].replace(COUNTRY_MAPPING_IRENASTAT)
    # Rename technologies
    df["technology"] = df["technology"].replace(TECH_MAPPING_IRENASTAT)
    return df[COLUMN_ORDER]


def read_ember_data(path, year):
    df = pd.read_csv(path, header=1, sep=";")
    df = df[
        (df["year"] == year)
        & (df["variable"].isin(["Bioenergy", "Coal", "Gas", "Hydro", "Nuclear"]))
    ].reset_index(drop=True)
    df.columns = ["region", "year", "technology", "capacity_gw"]
    # Rename technologies
    df["technology"] = df["technology"].replace(TECH_MAPPING_EMBER)

    return df[COLUMN_ORDER]


def process_historic_data(
    path_to_raw_data, path_to_processed_data, year=2022, save=False
):
    df = pd.DataFrame()
    for root, dirs, files in os.walk(path_to_raw_data):
        if ("irenastat_capacity_mw.csv" in files) & ("ember_capacity_gw.csv" in files):
            # Get Irenastat data
            df_irenastat = read_irenastat_data(
                os.path.join(root, "irenastat_capacity_mw.csv"), year=year
            )
            # Get Ember data
            df_ember = read_ember_data(
                os.path.join(root, "ember_capacity_gw.csv"), year=year
            )

            # Combine Irenastat and Ebmer data
            df = pd.concat([df, df_ember, df_irenastat]).reset_index(drop=True)
    df["spore"] = 0
    # Calculate continental capacity
    df_eu = (
        df.groupby(["year", "technology", "spore"])["capacity_gw"].sum().reset_index()
    )
    df_eu["region"] = "Europe"
    df_eu = df_eu[["year", "region", "technology", "capacity_gw", "spore"]]
    # Concatenate national and continental values in one Series
    power_capacity = pd.concat([df, df_eu]).fillna(0)

    # Save to .csv and return dataframe
    if save:
        save_processed_historic_data(
            processed_data=power_capacity.drop(columns=["year"]),
            path_to_result=os.path.join(path_to_processed_data, str(year)),
        )

    return df


def save_processed_historic_data(processed_data, path_to_result):
    # Make new folder for processed historic data if directory does not exist yet
    if not os.path.exists(path_to_result):
        os.makedirs(path_to_result)

    processed_data.to_csv(
        os.path.join(path_to_result, "power_capacity.csv"), index=False
    )


if __name__ == "__main__":
    df = process_historic_data(
        path_to_raw_data="../data/raw/historic-capacity",
        path_to_processed_data="../data/processed",
        year=2022,
        save=True,
    )
