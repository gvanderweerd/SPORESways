import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from main import *
from global_parameters import *
from processing_functions import *
from reading_functions import read_spores_data

# Define open circles for plotting 'All other SPORES'
pts = np.linspace(0, np.pi * 2, 24)
circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
vert = np.r_[circ, circ[::-1] * 0.7]
open_circle = mpl.path.Path(vert)

plot_order_years = ["2020", "2025", "2030", "2035", "2040", "2045", "2050"]
YEARS = range(2010, 2051)
POWER_TECH_ORDER = [
    "PV",
    "Onshore wind",
    "Offshore wind",
    "CHP",
    "CCGT",
    "Nuclear",
    "Hydro",
    "Bio to liquids",
]
COLORS = {
    "All other SPORES": "#b9b9b9",
    "Nuclear": "#cc0000",
    "nuclear": "#cc0000",
    "ccgt": "#8fce00",
    "chp": "#ce7e00",
    "PV": "#ffd966",
    "pv": "#ffd966",
    "Onshore wind": "#674ea7",
    "onshore wind": "#674ea7",
    "Offshore wind": "#e062db",
    "offshore wind": "#e062db",
    "Hydro": "#2986cc",
    "hydro": "#2986cc",
}

def plot_el_production_hubs(
    path_to_friendly_data,
    ax,
    spores=None,
    percentage_min=0.05,
    save_fig=False,
):
    spores_data = read_spores_data(path_to_friendly_data, slack="slack-10", file_names=["flow_out_sum"])

    # Calculate total electricity production in Europe in [TWh]
    df = (
        spores_data["flow_out_sum"]
        .xs("twh", level="unit")
        .xs("electricity", level="carriers")
        .unstack("spore")
    )
    prod_total = df.sum()

    # Get electricity production per producer for all countries
    prod_national = (
        df.groupby([REGION_MAPPING, ELECTRICITY_PRODUCERS_SPORES], level=["region", "technology"])
        .sum()
        .stack("spore")
        .reorder_levels(["spore", "region", "technology"])
        .sort_index()
    )

    # Filter technologies in the countries where they produce more than a given % of the total European electricity production
    prod_hubs_national = (
        prod_national.where(prod_national.div(prod_total) > percentage_min)
        .dropna()
    )

    # Filter specific spores when provided
    if spores is not None:
        prod_hubs_national = prod_hubs_national.loc[spores, :, :]

    sns.stripplot(
        data=prod_hubs_national,
        x="region",
        y=prod_hubs_national.array,
        hue="technology",
        dodge=True,
        palette=COLORS
    )

    # Set titles and labels
    ax.set_title(
        f"SPORES where a specific technology produces > {int(100*percentage_min)}% of total electricity production."
    )
    ax.set_xlabel("Countries")
    ax.set_xticks(range(len(list(prod_hubs_national.index.get_level_values("region").unique()))))
    ax.set_xticklabels(list(prod_hubs_national.index.get_level_values("region").unique()), fontsize=8, rotation=0)
    ax.set_ylabel("Electricity production [TWh]")

    # Save figure to output directory
    if save_fig:
        fig.savefig(
            f"figures/electricity_production_hubs.png",
            pad_inches=0,
        )

def plot_capacity_distribution_country(ax, data, country):
    capacity = data.loc[:, country, :, :]
    capacity_normalised = capacity.div(capacity.groupby("technology").max())
    capacity_ranges = capacity.groupby("technology").agg(["min", "max"])

    sns.stripplot(
        ax=ax,
        data=capacity_normalised,
        x="technology",
        y=capacity_normalised.array,
        order=POWER_TECH_ORDER,
        hue="year",
        hue_order=[2030, 2050],
        jitter=True,
        dodge=True,
        palette=["orange", "green"],
        marker=open_circle
    )
    ax.set_title(f"Capacity distribution of SPORES results for the power sector in {country}")
    ax.set_xlabel("")
    ax.set_ylabel("Normalised capacity")
    ax.set_xticks(range(len(POWER_TECH_ORDER)))
    ax.set_xticklabels(POWER_TECH_ORDER, fontsize=10)
    xticklabels = []
    for ticklabel in ax.get_xticklabels():
        technology = ticklabel.get_text()

        if technology in capacity.index.get_level_values("technology").unique():
            xticklabels.append(f"{technology}\n{capacity_ranges.loc[technology, 'min'].round(2)} - {capacity_ranges.loc[technology, 'max'].round(2)} [GW]")
        else:
            xticklabels.append(f"{technology}\n0.0 - 0.0 [GW]")
    ax.set_xticklabels(xticklabels, fontsize=10)

def plot_historic_and_spores_capacity(ax, capacity_2000_2021, capacity_spores):
    year_range = range(2000, 2051)

    sns.stripplot(
        data=capacity_2000_2021,
        x="year",
        y=capacity_2000_2021.array,
        ax=ax
    )
    sns.stripplot(
        data=capacity_spores,
        x="year",
        y=capacity_spores.array,
        order=year_range,
        ax=ax,
        jitter=False,
        marker=open_circle
    )

    ax.set_title("Title")
    ax.set_xticks(range(len(year_range)))
    ax.set_xticklabels(year_range)
    ax.set_xlabel("")
    ax.set_ylabel("Installed capacity [GW]")
    [label.set_visible(False) for index, label in enumerate(ax.get_xticklabels()) if index % 5 != 0]

if __name__ == '__main__':
    power_capacity = pd.read_csv("data/power_capacity.csv", index_col = ["year", "region", "technology", "spore"], squeeze=True)
    power_capacity_2000_2021 = pd.read_csv("data/power_capacity_irenastat.csv", index_col = ["region", "technology", "year"], squeeze=True)
    country = "Spain"
    technology = "PV"

    """
    Plotting top electricity producers
    """
    fig, ax = plt.subplots()

    plot_el_production_hubs(
        path_to_friendly_data="data/euro-spores-results-v2022-05-13",
        ax=ax
    )

    """
    Plotting normalised capacity distribution of power sector
    """
    fig, ax = plt.subplots()
    plot_capacity_distribution_country(ax, power_capacity, country)

    """
    Plotting IRENASTAT and SPORES capacities
    """
    fig, ax = plt.subplots()
    plot_historic_and_spores_capacity(
        ax=ax,
        capacity_2000_2021=power_capacity_2000_2021.loc[country, technology, :],
        capacity_spores=power_capacity.loc[:, country, technology, :]
    )

    plt.show()