import os
import pandas as pd

from src.utils.data_io import *


def analyse_consumption_of_energy_carriers(final_consumption):
    pd.set_option("display.max_rows", None)
    print(
        final_consumption.groupby(level=["spore", "carriers"])
        .sum()
        .sort_values()
        # .describe()
    )


def find_realistic_spores_per_technology(power_capacity):
    for country in ["Europe", "France", "Germany", "Italy", "Spain", "United Kingdom"]:
        for technology in ["PV", "Onshore wind", "Offshore wind", "Coal"]:
            if technology == "Coal":
                mask = (
                    power_capacity.get("2030").loc[country, technology, :]
                    <= power_capacity.get("2022").loc[country, technology, :][0]
                )
            else:
                mask = (
                    power_capacity.get("2030").loc[country, technology, :]
                    >= power_capacity.get("2022").loc[country, technology, :][0]
                )
            realistic_spores = mask[mask].index.get_level_values(level="spore")
            print(
                f"{country} {technology}: Capacity 2022 = {power_capacity.get('2022').loc[country, technology, :][0]}, {len(realistic_spores)} Realistic SPORES in 2030"
            )


def find_realistic_spores_per_country(power_capacity):
    for country in ["Europe", "France", "Germany", "Italy", "Spain", "United Kingdom"]:
        mask = (
            (
                power_capacity.get("2030").loc[country, "Coal", :]
                <= power_capacity.get("2022").loc[country, "Coal", :][0]
            )
            & (
                power_capacity.get("2030").loc[country, "PV", :]
                >= power_capacity.get("2022").loc[country, "PV", :][0]
            )
            & (
                power_capacity.get("2030").loc[country, "Onshore wind", :]
                >= power_capacity.get("2022").loc[country, "Onshore wind", :][0]
            )
            & (
                power_capacity.get("2030").loc[country, "Offshore wind", :]
                >= power_capacity.get("2022").loc[country, "Offshore wind", :][0]
            )
        )
        realistic_spores = mask[mask].index.get_level_values(level="spore")
        print(
            f"{country}: {len(realistic_spores)} Realistic SPORES in 2030 (based on increase of PV, Offshore- and Onshore wind capacity and decrease of Coal capacity)"
        )


if __name__ == "__main__":
    """
    1. READ AND PREPARE DATA
    """
    historic_years = ["2022"]
    spores_years = ["2030", "2050"]
    path_to_processed_data = os.path.join(os.getcwd(), "..", "data", "processed")
    path_to_raw_data = os.path.join(os.getcwd(), "..", "data", "raw")

    power_capacity = load_processed_power_capacity(
        path_to_processed_data, years=(historic_years + spores_years)
    )
    final_consumption = load_raw_final_consumption(path_to_raw_data)
    tpes = load_raw_primary_energy_supply(path_to_raw_data)

    find_realistic_spores_per_technology(power_capacity)
    find_realistic_spores_per_country(power_capacity)

    print(power_capacity.get("2030").index.unique("technology"))
