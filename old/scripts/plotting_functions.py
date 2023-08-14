import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import seaborn.objects as so
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from main import *
from global_parameters import *
from processing_functions import *
from reading_functions import read_spores_data

# Define open circles for plotting 'All other SPORES'
pts = np.linspace(0, np.pi * 2, 24)
circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
vert = np.r_[circ, circ[::-1] * 0.7]
open_circle = mpl.path.Path(vert)

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

# def plot_capacity_distribution_2030_ontrack():

def plot_capacity_distribution_clusters_2050(ax, capacity, cluster_in_color=None):
    palette = {0:"grey", 1:"grey", 2:"grey", 3:"grey", 4:"grey", 5:"grey", 6:"grey", 7:"grey"}
    if cluster_in_color is not None:
        palette[cluster_in_color] = "blue"

    scaler = MinMaxScaler()
    all_techs = capacity.index.get_level_values("technology").unique()
    capacity_normalised = pd.DataFrame(
        scaler.fit_transform(capacity.unstack("technology")), columns=all_techs
    )
    capacity_ranges = capacity.groupby("technology").agg(["min", "max"])
    index = capacity.unstack("technology").index
    capacity_normalised.index = index
    capacity_normalised = capacity_normalised.stack("technology")

    print("TEST 3 may")
    print(capacity_normalised)
    print(palette)
    sns.stripplot(
        ax=ax,
        data=capacity_normalised,
        y=capacity_normalised.array,
        x="technology",
        hue="spore_cluster",
        marker=open_circle,
        #FIXME: This palette crashes the script somehow. I believe the idea was to plot one specific cluster in blue and the rest of the spores light grey
        # palette=palette
    )
    ax.set_title(
        f"Capacity distribution of 2050 SPORES for the power sector in Europe"
    )
    ax.set_xlabel("")
    ax.set_ylabel("Normalised capacity")
    ax.set_xticks(range(len(all_techs)))
    ax.set_xticklabels(all_techs, fontsize=10)
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

def plot_capacity_distribution(ax, data, year, country, spores_cluster, spores_rest):
    capacity = data
    scaler = MinMaxScaler()
    all_techs = capacity.index.get_level_values("technology").unique()
    capacity_normalised = pd.DataFrame(
        scaler.fit_transform(capacity.unstack("technology")), columns=all_techs
    )
    capacity_ranges = capacity.groupby("technology").agg(["min", "max"])

    sns.stripplot(
        ax=ax,
        data=capacity_normalised.loc[spores_rest],
        marker=open_circle,
        palette={"PV": "grey", "Onshore wind": "grey", "Offshore wind": "grey", "Hydro": "grey", "CCGT": "grey", "Nuclear": "grey", "CHP": "grey"},
    )
    sns.stripplot(
        ax=ax,
        data=capacity_normalised.loc[spores_cluster],
        marker="o",
        palette={"PV": "blue", "Onshore wind": "blue", "Offshore wind": "blue", "Hydro": "blue", "CCGT": "blue", "Nuclear": "blue", "CHP": "blue"},
        color="blue"
    )

    ax.set_title(
        f"Capacity distribution of {year} SPORES for the power sector in {country}"
    )
    ax.set_xlabel("")
    ax.set_ylabel("Normalised capacity")
    ax.set_xticks(range(len(all_techs)))
    ax.set_xticklabels(all_techs, fontsize=10)
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

    # for spore in [1, 300]:
    #     spore_capacity = capacity_normalised.loc[:, spore]
    #     sns.lineplot(
    #         ax=ax,
    #         data=spore_capacity,
    #         x="technology",
    #         y=spore_capacity.array,
    #         color="orange",
    #         label=f"SPORE number: {spore}"
    #     )


def plot_capacity_distribution_2030_2050(ax, data, country):
    capacity = data
    capacity_ranges = capacity.groupby("technology").agg(["min", "max"])
    capacity_normalised = capacity.div(capacity.groupby("technology").max())

    sns.stripplot(
        ax=ax,
        data=capacity_normalised,
        x="technology",
        y=capacity_normalised.array,
        hue="year",
        hue_order=[2030, 2050],
        jitter=True,
        dodge=True,
        palette=["grey", "grey"],
        marker=open_circle,
    )
    ax.set_title(
        "Capacity distribution of SPORES results for the power sector in Europe"
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


def plot_normalised_cluster_map(data, year, region):
    scaler = MinMaxScaler()
    all_techs = data.index.get_level_values("technology").unique()
    data_normalised = pd.DataFrame(
        scaler.fit_transform(data.unstack("technology")),
        columns=data.index.get_level_values("technology").unique(),
    )
    ranges = data.groupby("technology").agg(["min", "max"])
    print("TEST2")
    print(data)
    print(data_normalised)
    # FIXME: arange the order of the technologies such that they are visualised from top to bottom ('power', 'heat', 'grid', 'storage')
    # FIXME: find out how to set white spacing inbetween sectors such that all power technologies are together but there is a white space between the power technologies and the heat technologies
    #
    # Can we use this on row_columns? 'row_columns=df_colors' and chose a cmap that makes the row that is np.nan white
    # df_colors = pd.DataFrame({"color": np.repeat(1, df_normalised.shape[1])})
    # df_colors.loc[3, "color"] = np.nan

    cluster_map = sns.clustermap(
        data=data_normalised.T,
        method="ward",
        metric="euclidean",
        row_cluster=False,
        cmap="Spectral_r",
    )
    # Add figure title
    cluster_map.fig.suptitle(
        f"Normalised capacity clustermap of {year} SPORES results for the power sector in {region}"
    )
    cluster_map_axis = cluster_map.ax_heatmap

    # Set x and y axis labels
    cluster_map_axis.set_xlabel("X axis label")
    cluster_map_axis.set_ylabel("Y axis label")

    cluster_map_axis.set_yticks(range(len(all_techs)))
    cluster_map_axis.set_yticklabels(all_techs, fontsize=10)

    yticklabels = []
    # FIXME: add units corresponding to which sector
    for ticklabel in cluster_map_axis.get_yticklabels():
        technology = ticklabel.get_text()
        yticklabels.append(
            f"{technology} {ranges.loc[technology, 'min']:.1f} - {ranges.loc[technology, 'max']:.1f} [unit]"
        )
    yticklabels.sort(reverse=True)

    cluster_map_axis.set_yticklabels(yticklabels)


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

    # Color categorised_spores spores that are in the range of the projection
    mask = (spores_2030 >= y_lower[-1]) & (spores_2030 <= y_upper[-1])
    projected_spores = spores_2030[mask].index.get_level_values("spore").tolist()
    x_projected_spores = [2030] * len(projected_spores)
    sns.scatterplot(
        ax=axs[0], x=x_projected_spores, y=spores_2030[mask].array, color="orange"
    )

    # Plot linear continuation of the uncertainty band between the projected years 2021-categorised_spores
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


def add_historic_capacity_to_pathway(ax, capacity_historic):
    # Define the years for which to plot the historic capacity
    years_historic = list(capacity_historic.index.get_level_values("year").unique())
    # Plot historic capacity (2015-2021)
    sns.lineplot(
        ax=ax,
        data=capacity_historic,
        x=years_historic,
        y=capacity_historic.array,
        label="Historic capacity",
    )


def add_spores_capacity_to_pathway(ax, capacity_spores, year, label, color="grey", marker=open_circle):
    # Count number of SPORES in capacity spores data
    n_spores = len(capacity_spores.index.get_level_values("spore").unique())
    # Define x for capacity SPORES
    x_spores = [year] * n_spores
    # Plot capacity SPORES
    sns.scatterplot(
        ax=ax,
        x=x_spores,
        y=capacity_spores.array,
        marker=marker,
        color=color,
        label=label,
    )


def add_capacity_value_to_pathway(ax, year, value):
    # Plot black dot on the map of the value that is plotted
    ax.scatter(x=year, y=value, marker=open_circle, color="grey")
    # Plot capacity value text at x=year, y=value+10
    ax.annotate(
        xy=(year, value),
        xytext=(year, value + 10),
        text=f"{value:.1f} [GW]",
        fontsize=12,
    )


def add_exponential_growth_to_pathway(ax, projections, projection_years):
    for projection in projections:
        ax.fill_between(
            x=projection_years,
            y1=projection["y1"],
            y2=projection["y2"],
            color=projection["color"],
            label=projection["label"],
            alpha=0.25,
        )
def add_band_to_pathway(ax, x, y_upper, y_lower):
    ax.fill_between(x=x, y1=y_upper, y2=y_lower, color="grey", alpha=0.25)


def add_decline_to_lockin_capacity_2050(ax, projections):
    for projection in projections:
        sns.lineplot(x=years_2030_2050, y=projection, color="grey", linestyle="--")


def plot_technology_pathway(
    ax, technology, capacity_2000_2021, capacity_2030, capacity_2050, s_curve_params, capacity_2050_in_cluster=None
):
    L_max = capacity_2050_in_cluster.max()
    L_min = capacity_2050_in_cluster.min()
    y_min = s_curve(x=years_2000_2050, x0=s_curve_params.get("x0"), L=L_min, K=s_curve_params.get("K_min"))
    y_max = s_curve(x=years_2000_2050, x0=s_curve_params.get("x0"), L=L_max, K=s_curve_params.get("K_max"))

    # Split categorised_spores SPORES capacity series into 3 series; (1) a series with SPORES that are 'on track', (2) a series with SPORES that are 'over-invested', and (3) a series with SPORES that are 'uncer-invested' for the cluster of SPORES in 2050
    capacity_2030_on_track = capacity_2030.loc[(capacity_2030 > y_min[30]) & (capacity_2030 < y_max[30])].dropna()
    capacity_2030_overinvested = capacity_2030.loc[capacity_2030 >= y_max[30]].dropna()
    capacity_2030_underinvested = capacity_2030.loc[capacity_2030 <= y_min[30]].dropna()
    spores_2030_on_track = capacity_2030_on_track.index.get_level_values("spore").unique()
    spores_2030_overinvested = capacity_2030_overinvested.index.get_level_values("spore").unique()
    spores_2030_underinvested = capacity_2030_underinvested.index.get_level_values("spore").unique()

    # Add historic capacity
    add_historic_capacity_to_pathway(ax=ax, capacity_historic=capacity_2000_2021)

    # Add categorised_spores SPORES
    # _add_spores_capacity_to_pathway(ax=ax, capacity_spores=capacity_2030, year=categorised_spores)
    add_spores_capacity_to_pathway(ax=ax, capacity_spores=capacity_2030_overinvested, year=2030, label="SPORES that are over-invested for cluster 4 in 2050", color="green", marker="o")
    add_spores_capacity_to_pathway(ax=ax, capacity_spores=capacity_2030_on_track, year=2030, label="categorised_spores SPORES that are on track for cluster 4 in 2050", color="orange", marker="o")
    add_spores_capacity_to_pathway(ax=ax, capacity_spores=capacity_2030_underinvested, year=2030, label="categorised_spores SPORES that are under-invested for cluster 4 in 2050" ,color="red", marker="o")
    # Add 2050 SPORES
    add_spores_capacity_to_pathway(ax=ax, capacity_spores=capacity_2050, year=2050, label="2050 SPORES")
    if capacity_2050_in_cluster is not None:
        add_spores_capacity_to_pathway(ax=ax, capacity_spores=capacity_2050_in_cluster, year=2050, label="2050 SPORES in cluster 4", color="blue", marker="o")

    # Add s-curve projection to cluster in 2050
    add_band_to_pathway(ax=ax, x=years_2000_2050, y_upper=y_max, y_lower=y_min)


    # Add capacity value at x=2021
    add_capacity_value_to_pathway(ax=ax, year=2021, value=capacity_2000_2021.iloc[-1])
    # Add min and max capacity values at x=categorised_spores
    add_capacity_value_to_pathway(ax=ax, year=2030, value=capacity_2030.max())
    add_capacity_value_to_pathway(ax=ax, year=2030, value=capacity_2030.min())
    # Add min and max capacity values at x=2050
    #FIXME: figure out what values to plot? Min/Max of cluster, or total dataset?
    add_capacity_value_to_pathway(ax=ax, year=2050, value=capacity_2050_in_cluster.max())
    add_capacity_value_to_pathway(ax=ax, year=2050, value=capacity_2050_in_cluster.min())

    """
    These lines plot 
    
    # Calculate projection to categorised_spores
    projections_2021_2030, info_2030 = projection_to_spores_exponential(start_capacity=capacity_2000_2021.iloc[-1], spores_capacity=capacity_2030, years=years_2021_2030)
    projections_2021_2050, info_2050 = projection_to_spores_exponential(start_capacity=capacity_2000_2021.iloc[-1], spores_capacity=capacity_2050, years=years_2021_2050)
    # Calculate 'lock-in' projetion from categorised_spores to 2050
    projections_2030_2050 = projection_lock_in_2030_2050(capacity_2000_2021=capacity_2000_2021, capacity_2021_2030=projections_2021_2030, capacity_2030=capacity_2030, years=years_2030_2050, life_time=ELECTRICITY_PRODUCERS_LIFE.get("PV"))
    # Add exponential projection from 2021 until categorised_spores
    add_exponential_growth_to_pathway(ax=ax, projections=projections_2021_2030, projection_years=years_2021_2030)
    # # add_exponential_growth_to_pathway(ax=ax, projections=projections_2021_2050, projection_years=years_2021_2050)
    # Add lock-in effect from categorised_spores spores on 2050
    add_decline_to_lockin_capacity_2050(ax=ax, projections=projections_2030_2050)
    """

    # Set figure legend
    ax.legend(bbox_to_anchor=(0, 1), loc="upper left")
    # Set figure title
    ax.set_title(f"{technology}")
    # Set axis labels
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("Capacity [GW]")

    return spores_2030_underinvested, spores_2030_on_track, spores_2030_overinvested


def plot_boxplot_capacities_2030_2050(spores_2030, spores_2050, region, sector):
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(16, 9))
    sns.boxplot(ax=axs[0], data=spores_2030.unstack("technology"), orient="h")
    sns.boxplot(ax=axs[1], data=spores_2050.unstack("technology"), orient="h")
    axs[0].set_title("categorised_spores")
    axs[1].set_title("2050")
    axs[1].set_xlabel("Capacity [GW]")
    fig.suptitle(
        f"Distribution of categorised_spores and 2050 SPORES capacities for the {sector} sector in {region}",
        fontsize=14,
    )


def plot_boxplot_capacities_per_sector(power_spores, heat_spores, region, year):
    # FIXME: add grid and storage sectors
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    sns.boxplot(ax=axs[0], data=power_spores.unstack("technology"), orient="h")
    sns.boxplot(ax=axs[1], data=heat_spores.unstack("technology"), orient="h")
    axs[0].set_title("Power sector")
    axs[1].set_title("Heat sector")
    axs[1].set_xlabel("Capacity [GW]")
    fig.suptitle(
        f"Distribution of SPORES capacities for the power and heat sector in {region} ({year})",
        fontsize=14,
    )


if __name__ == "__main__":
    power_capacity = pd.read_csv(
        "../data/power_capacity.csv",
        index_col=["year", "region", "technology", "spore"],
        squeeze=True,
    )
    power_capacity_2000_2021 = pd.read_csv(
        "../data/power_capacity_irenastat.csv",
        index_col=["region", "technology", "year"],
        squeeze=True,
    )
    country = "Spain"
    technology = "PV"

    # Get capacity data for specified technology in specified country between 2015 and 2021
    capacity_2015_2021 = power_capacity_2000_2021.loc[country, technology, 2015:]

    # Project exponential capacity growth for 2021-categorised_spores
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
        path_to_friendly_data="../data/euro-spores-results-v2022-05-13", ax=ax
    )

    # Plot normalised capacity distribution of power sector for categorised_spores and 2050
    fig, ax = plt.subplots()
    plot_capacity_distribution_2030_2050(ax, power_capacity, country)

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
    # Simulate spores data for categorised_spores
    # """
    # n_spores = 25
    # multiindex = pd.MultiIndex.from_product(
    #     [[categorised_spores], ["Spain"], ["PV"], range(n_spores)],
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
    # cap_spores_2030 = cap_spores.loc[categorised_spores, country, technology, :]
    # cap_spores_2050 = cap_spores.loc[2050, country, technology, :]
