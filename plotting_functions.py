import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import seaborn.objects as so

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
    spores_data = read_spores_data(
        path_to_friendly_data, slack="slack-10", file_names=["flow_out_sum"]
    )

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
        df.groupby(
            [REGION_MAPPING, ELECTRICITY_PRODUCERS_SPORES],
            level=["region", "technology"],
        )
        .sum()
        .stack("spore")
        .reorder_levels(["spore", "region", "technology"])
        .sort_index()
    )

    # Filter technologies in the countries where they produce more than a given % of the total European electricity production
    prod_hubs_national = prod_national.where(
        prod_national.div(prod_total) > percentage_min
    ).dropna()

    # Filter specific spores when provided
    if spores is not None:
        prod_hubs_national = prod_hubs_national.loc[spores, :, :]

    sns.stripplot(
        data=prod_hubs_national,
        x="region",
        y=prod_hubs_national.array,
        hue="technology",
        dodge=True,
        palette=COLORS,
    )

    # Set titles and labels
    ax.set_title(
        f"SPORES where a specific technology produces > {int(100*percentage_min)}% of total electricity production."
    )
    ax.set_xlabel("Countries")
    ax.set_xticks(
        range(len(list(prod_hubs_national.index.get_level_values("region").unique())))
    )
    ax.set_xticklabels(
        list(prod_hubs_national.index.get_level_values("region").unique()),
        fontsize=8,
        rotation=0,
    )
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
        marker=open_circle,
    )
    ax.set_title(
        f"Capacity distribution of SPORES results for the power sector in {country}"
    )
    ax.set_xlabel("")
    ax.set_ylabel("Normalised capacity")
    ax.set_xticks(range(len(POWER_TECH_ORDER)))
    ax.set_xticklabels(POWER_TECH_ORDER, fontsize=10)
    xticklabels = []
    for ticklabel in ax.get_xticklabels():
        technology = ticklabel.get_text()

        if technology in capacity.index.get_level_values("technology").unique():
            xticklabels.append(
                f"{technology}\n{capacity_ranges.loc[technology, 'min'].round(2)} - {capacity_ranges.loc[technology, 'max'].round(2)} [GW]"
            )
        else:
            xticklabels.append(f"{technology}\n0.0 - 0.0 [GW]")
    ax.set_xticklabels(xticklabels, fontsize=10)


def plot_capacity_pathway(
    capacity_2000_2021, capacity_2021_2030, capacity_spores, country, technology
):
    fig, axs = plt.subplots(nrows=2, ncols=1)

    # Prepare data for capacity pathway plot
    capacity_2015_2021 = capacity_2000_2021.loc[country, technology, 2015:]
    capacity_2021_2030 = capacity_2021_2030.loc[country, technology, :]
    spores_2030 = capacity_spores.loc[2030, country, technology, :]
    spores_2050 = capacity_spores.loc[2050, country, technology, :]

    # Count number of spores in each spores dataset
    n_spores_2030 = len(spores_2030.index.get_level_values("spore").unique())
    n_spores_2050 = len(spores_2050.index.get_level_values("spore").unique())

    # Define x-axis for past-, projected-, and spores-capacity data
    x = np.arange(2015, 2022)
    x_projection = np.arange(2021, 2031)
    x_spores_2030 = [2030] * n_spores_2030
    x_spores_2050 = [2050] * n_spores_2050

    # Add plots to figure
    sns.lineplot(ax=axs[0], data=capacity_2015_2021, x=x, y=capacity_2015_2021.array)
    sns.lineplot(
        ax=axs[0],
        data=capacity_2021_2030,
        x=x_projection,
        y=capacity_2021_2030.array,
        linestyle="--",
    )
    sns.scatterplot(
        ax=axs[0],
        data=spores_2030,
        x=x_spores_2030,
        y=spores_2030.array,
        marker=open_circle,
        color="grey",
    )
    sns.scatterplot(
        ax=axs[0],
        data=spores_2050,
        x=x_spores_2050,
        y=spores_2050.array,
        marker=open_circle,
        color="grey",
    )

    # Add band of uncertainty around projected plot
    x = x_projection
    y = capacity_2021_2030.array
    y_upper = y * (1 + 0.02 * (x - 2021))
    y_lower = y * (1 - 0.02 * (x - 2021))
    axs[0].fill_between(x=x, y1=y_upper, y2=y_lower, color="orange", alpha=0.5)

    # Color 2030 spores that are in the range of the projection
    mask = (spores_2030 >= y_lower[-1]) & (spores_2030 <= y_upper[-1])
    projected_spores = spores_2030[mask].index.get_level_values("spore").tolist()
    x_projected_spores = [2030] * len(projected_spores)
    sns.scatterplot(
        ax=axs[0], x=x_projected_spores, y=spores_2030[mask].array, color="orange"
    )

    # Plot linear continuation of the uncertainty band between the projected years 2021-2030
    x = np.arange(2030, 2051)
    y_upper = (y_upper[-1] - y_upper[-2]) * (x - 2030) + y_upper[-1]
    y_lower = (y_lower[-1] - y_lower[-2]) * (x - 2030) + y_lower[-1]
    axs[0].fill_between(x=x, y1=y_upper, y2=y_lower, color="lightgrey", alpha=0.5)

    # Plot exponential continuation of
    # cagr = 1.1
    y_exp_2030_2050 = [capacity_2021_2030.array[-1] * cagr ** (i - 2030) for i in x]
    sns.lineplot(ax=axs[0], x=x, y=y_exp_2030_2050, color="grey", linestyle="--")

    # Color 2050 spores that are need extra growth
    mask_extra_growth = spores_2050 > y_upper[-1]
    growth_spores = (
        spores_2050[mask_extra_growth].index.get_level_values("spore").tolist()
    )
    x_extra_growth_spores = [2050] * len(growth_spores)
    sns.scatterplot(
        ax=axs[0],
        x=x_extra_growth_spores,
        y=spores_2050[mask_extra_growth].array,
        color="red",
    )

    # Color 2050 spores that are within feasible range
    mask_feasible = (spores_2050 >= y_lower[-1]) & (spores_2050 <= y_upper[-1])
    feasible_spores = (
        spores_2050[mask_feasible].index.get_level_values("spore").tolist()
    )
    x_feasible_spores = [2050] * len(feasible_spores)
    sns.scatterplot(
        ax=axs[0],
        x=x_feasible_spores,
        y=spores_2050[mask_feasible].array,
        color="green",
    )

    # Set figure and axis titles
    axs[0].set_title(
        f"Capacity pathway of {technology} in {country} between 2015 and 2050"
    )
    axs[0].set_xlabel("Time [years]")
    axs[0].set_ylabel("Capacity [GW]")
    axs[0].set_yscale("log")
    # axs[0].legend(title="legend", labels=["Historic capacity (IRENASTAT)", "Projected capacity", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test"])

    # Prepare data for capacity distribution plot in 2050
    spores_2050 = capacity_spores.loc[2050, country, :, :]
    spores_2050_normalised = spores_2050.div(spores_2050.groupby("technology").max())
    spores_2050_ranges = spores_2050.groupby("technology").agg(["min", "max"])
    print(spores_2050_normalised)
    print(spores_2050_normalised.loc[:, :, :, feasible_spores])

    sns.stripplot(
        ax=axs[1],
        data=spores_2050_normalised,
        x="technology",
        y=spores_2050_normalised.array,
        order=POWER_TECH_ORDER,
        jitter=True,
        dodge=True,
        color="grey",
        marker=open_circle,
    )
    sns.stripplot(
        ax=axs[1],
        data=spores_2050_normalised.loc[:, :, :, feasible_spores],
        x="technology",
        y=spores_2050_normalised.loc[:, :, :, feasible_spores].array,
        order=POWER_TECH_ORDER,
        jitter=True,
        dodge=True,
        color="green",
        # marker=open_circle
    )
    sns.stripplot(
        ax=axs[1],
        data=spores_2050_normalised.loc[:, :, :, growth_spores],
        x="technology",
        y=spores_2050_normalised.loc[:, :, :, growth_spores].array,
        order=POWER_TECH_ORDER,
        jitter=True,
        dodge=True,
        color="red",
        # marker=open_circle
    )
    axs[1].set_title(
        f"Capacity distribution of 2050 SPORES results for the power sector in {country}"
    )
    axs[1].set_xlabel("")
    axs[1].set_ylabel("Capacity (normalised)")
    axs[1].set_xticks(range(len(POWER_TECH_ORDER)))
    axs[1].set_xticklabels(POWER_TECH_ORDER, fontsize=10)
    xticklabels = []
    for ticklabel in axs[1].get_xticklabels():
        technology = ticklabel.get_text()

        if (
            technology
            in spores_2050_normalised.index.get_level_values("technology").unique()
        ):
            xticklabels.append(
                f"{technology}\n{spores_2050_ranges.loc[technology, 'min'].round(2)} - {spores_2050_ranges.loc[technology, 'max'].round(2)} [GW]"
            )
        else:
            xticklabels.append(f"{technology}\n0.0 - 0.0 [GW]")
    axs[1].set_xticklabels(xticklabels, fontsize=10)


def plot_capacity_pathway_to_quartiles(
    ax, capacity_2000_2021, capacity_spores, country, technology
):
    # Prepare data for capacity pathway plot
    capacity_2015_2021 = capacity_2000_2021.loc[country, technology, 2015:]
    spores_2030 = capacity_spores.loc[2030, country, technology, :]
    spores_2050 = capacity_spores.loc[2050, country, technology, :]

    # Count number of spores in each spores dataset
    n_spores_2030 = len(spores_2030.index.get_level_values("spore").unique())
    n_spores_2050 = len(spores_2050.index.get_level_values("spore").unique())

    # Define x-axis for past-, projected-, and spores-capacity data
    x = np.arange(2015, 2022)
    x_2021_2030 = np.arange(2021, 2031)
    x_2021_2050 = np.arange(2021, 2051)
    x_spores_2030 = [2030] * n_spores_2030
    x_spores_2050 = [2050] * n_spores_2050

    calculate_quartile_capacities = (
        lambda spores: spores.groupby(["region", "technology"])
        .quantile(q=[0.25, 0.5, 0.75])
        .loc[country, technology, :]
    )
    # Calculate capacity values corresponding to first quartile, median, and 3rd quartile for 2030 spores
    (
        q1_capacity_2030,
        median_capacity_2030,
        q3_capacity_2030,
    ) = calculate_quartile_capacities(spores_2030)
    # Calculate capacity values of highest, and lowest capacity spores for 2030 spores
    min_capacity_2030 = spores_2030.min()
    max_capacity_2030 = spores_2030.max()

    calculate_growth_factor = lambda capacity, years: (
        capacity / capacity_2015_2021[-1]
    ) ** (1 / (years[-1] - years[0]))
    cagr_max = calculate_growth_factor(max_capacity_2030, x_2021_2030)
    cagr_q3 = calculate_growth_factor(q3_capacity_2030, x_2021_2030)
    cagr_median = calculate_growth_factor(median_capacity_2030, x_2021_2030)
    cagr_q1 = calculate_growth_factor(q1_capacity_2030, x_2021_2030)
    cagr_min = calculate_growth_factor(min_capacity_2030, x_2021_2030)

    calculate_exponential_growth = lambda growth_factor, years: [
        capacity_2015_2021[-1] * growth_factor ** (year - years[0]) for year in years
    ]
    y_max = calculate_exponential_growth(cagr_max, x_2021_2030)
    y_q1 = calculate_exponential_growth(cagr_q1, x_2021_2030)
    y_median = calculate_exponential_growth(cagr_median, x_2021_2030)
    y_q3 = calculate_exponential_growth(cagr_q3, x_2021_2030)
    y_min = calculate_exponential_growth(cagr_min, x_2021_2030)

    cagr_table = [
        [
            "Top 25% of SPORES",
            f"{100 * (cagr_q3 - 1):.1f}% - {100 * (cagr_max - 1):.1f}%",
            f"{y_q3[-1]:.1f} - {y_max[-1]:.1f} [GW]",
        ],
        [
            "Middle 50% of SPORES",
            f"{100 * (cagr_q1 - 1):.1f}% - {100 * (cagr_q3 - 1):.1f}%",
            f"{y_q1[-1]:.1f} - {y_q3[-1]:.1f} [GW]",
        ],
        [
            "Bottom 25% of SPORES",
            f"{100 * (cagr_min - 1):.1f}% - {100 * (cagr_q1 - 1):.1f}%",
            f"{y_min[-1]:.1f} - {y_q1[-1]:.1f} [GW]",
        ],
    ]

    # Plot historic capacity (2015-2021)
    sns.lineplot(
        ax=ax,
        data=capacity_2015_2021,
        x=x,
        y=capacity_2015_2021.array,
        label="Historic capacity",
    )
    # Plot capacity value in 2021
    ax.annotate(
        xy=(x[-1], capacity_2015_2021[-1]),
        xytext=(x[-1], capacity_2015_2021[-1] + 100),
        text=f"{capacity_2015_2021[-1]:.1f} [GW]",
        fontsize=12,
    )
    ax.scatter(x=x[-1], y=capacity_2015_2021[-1], marker="o", color="black")
    # Plot exponential growth to the median capacity value in 2030 (2021-2030)
    sns.lineplot(
        ax=ax,
        x=x_2021_2030,
        y=y_median,
        linestyle="--",
        color="orange",
        label="Exponential growth to median capacity in 2030",
    )
    # Plot exponential growth (2021-2030)
    ax.fill_between(
        x=x_2021_2030,
        y1=y_max,
        y2=y_q3,
        color="red",
        alpha=0.25,
        label="Exponential growth to top 25% of spores in 2030",
    )
    ax.fill_between(
        x=x_2021_2030,
        y1=y_q3,
        y2=y_q1,
        color="orange",
        alpha=0.25,
        label="Exponential growth to middle 50% of spores in 2030",
    )
    ax.fill_between(
        x=x_2021_2030,
        y1=y_q1,
        y2=y_min,
        color="green",
        alpha=0.25,
        label="Exponential growth to bottom 25% spores in 2030",
    )
    # Plot 2030 and 2050 spores
    sns.scatterplot(
        ax=ax,
        x=x_spores_2030,
        y=spores_2030.array,
        marker=open_circle,
        color="grey",
        label="SPORES 2030",
    )
    sns.scatterplot(
        ax=ax,
        x=x_spores_2050,
        y=spores_2050.array,
        marker=open_circle,
        color="grey",
        label="SPORES 2050",
    )

    # Set figure and axis titles
    ax.set_title(f"Capacity pathway of {technology} in {country} between 2015 and 2050")
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("Capacity [GW]")
    # ax.set_yscale("log")

    # Set figure legend
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")

    # # Add table with CAGR values
    # ax.table(
    #     cellText=cagr_table,
    #     colLabels=[
    #         "Scenario",
    #         "Required CAGR between 2021 and 2030",
    #         "Capacity range in 2030",
    #     ],
    #     # bbox_to_anchor=(1, 0.5), loc="upper left"
    #     bbox=(1, 0.5, 1, 0.4),
    # )


if __name__ == "__main__":
    power_capacity = pd.read_csv(
        "data/power_capacity.csv",
        index_col=["year", "region", "technology", "spore"],
        squeeze=True,
    )
    power_capacity_2000_2021 = pd.read_csv(
        "data/power_capacity_irenastat.csv",
        index_col=["region", "technology", "year"],
        squeeze=True,
    )
    country = "Spain"
    technology = "PV"

    # Get capacity data for specified technology in specified country between 2015 and 2021
    capacity_2015_2021 = power_capacity_2000_2021.loc[country, technology, 2015:]

    # Project exponential capacity growth for 2021-2030
    cagr = 1.20
    projected_years = range(2021, 2031)
    multiindex = pd.MultiIndex.from_product(
        [[country], [technology], projected_years],
        names=["region", "technology", "year"],
    )
    values_projection = [
        capacity_2015_2021[-1] * cagr ** (i - 2021) for i in projected_years
    ]
    power_capacity_2021_2030 = pd.Series(values_projection, index=multiindex)

    # Plotting top electricity producers
    fig, ax = plt.subplots()
    plot_el_production_hubs(
        path_to_friendly_data="data/euro-spores-results-v2022-05-13", ax=ax
    )

    # Plot normalised capacity distribution of power sector
    fig, ax = plt.subplots()
    plot_capacity_distribution_country(ax, power_capacity, country)

    # Plot figure
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_capacity_pathway_to_quartiles(
        ax=ax,
        capacity_2000_2021=power_capacity_2000_2021,
        capacity_spores=power_capacity,
        country=country,
        technology=technology,
    )
    fig.tight_layout()

    # Show plots
    plt.show()

    # """
    # Simulate historic capacity data
    # """
    # values = power_capacity_2000_2021.loc[country, technology, 2015:].array
    # multiindex = pd.MultiIndex.from_product(
    #     [["Spain"], ["PV"], range(2015, 2022)], names=["region", "technology", "year"]
    # )
    # cap_2021 = pd.Series(values, index=multiindex)
    # """
    # Simulate spores data for 2030
    # """
    # n_spores = 25
    # multiindex = pd.MultiIndex.from_product(
    #     [[2030], ["Spain"], ["PV"], range(n_spores)],
    #     names=["year", "region", "technology", "spore"],
    # )
    # values_spores = range(60, 260, 8)
    # cap_spores_2030 = pd.Series(values_spores, index=multiindex)
    # """
    # Simulate spores data for 2050
    # """
    # multiindex = pd.MultiIndex.from_product(
    #     [[2050], ["Spain"], ["PV"], range(n_spores)],
    #     names=["year", "region", "technology", "spore"],
    # )
    # values_spores = range(100, 300, 8)
    # cap_spores_2050 = pd.Series(values_spores, index=multiindex)
    # # Combine simulated spores data into 1 Series
    # cap_spores = pd.concat([cap_spores_2030, cap_spores_2050])
    # cap_spores_2030 = cap_spores.loc[2030, country, technology, :]
    # cap_spores_2050 = cap_spores.loc[2050, country, technology, :]
