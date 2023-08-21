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
        & df["technology"].isin(["Offshore wind energy", "Onshore wind energy"])
    ]
    df = df.drop("grid_connection", axis=1).reset_index(drop=True)
    # Rename country
    df["region"] = df["region"].replace(COUNTRY_MAPPING_IRENASTAT)
    # Rename technologies
    df["technology"] = df["technology"].replace(TECH_MAPPING_IRENASTAT)
    return df[COLUMN_ORDER]


def read_ember_data(path, year):
    # FIXME: should "Bio-energy" be added? If so; add to CCGT? Look up https://ember-climate.org/app/uploads/2022/03/GER22-Methodology.pdf
    df = pd.read_csv(path, header=1, sep=";")
    df = df[
        (df["year"] == year)
        & (df["variable"].isin(["Coal", "Gas", "Hydro", "Nuclear", "Solar"]))
    ].reset_index(drop=True)
    df.columns = ["region", "year", "technology", "capacity_gw"]
    # Rename technologies
    df["technology"] = df["technology"].replace(TECH_MAPPING_EMBER)
    # FIXME: Add CHP data and set to zero
    country = df["region"].iloc[0]
    chp_df = pd.DataFrame(
        {
            "year": [year],
            "region": [country],
            "technology": ["CHP"],
            "capacity_gw": [0.0],
        }
    )
    df = pd.concat([df, chp_df], ignore_index=True)

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

    # Save to .csv and return dataframe
    if save:
        save_processed_historic_data(
            processed_data=df,
            path_to_result=os.path.join(path_to_processed_data, str(year)),
        )

    return df


def save_processed_historic_data(processed_data, path_to_result):
    # Make new folder for processed historic data if directory does not exist yet
    if not os.path.exists(path_to_result):
        os.makedirs(path_to_result)

    processed_data.to_csv(os.path.join(path_to_result, "power_capacity.csv"))


if __name__ == "__main__":
    df = process_historic_data(
        path_to_raw_data="../data/raw/historic-capacity",
        path_to_processed_data="../data/processed",
        year=2022,
        save=True,
    )
