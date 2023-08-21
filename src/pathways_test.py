import itertools
import os
import pandas

from src.utils.parameters import *
from src.utils.visualisation import *
from src.utils.data_io import *


def add_capacity_ranges_to_x_axis(ax, ranges):
    xticklabels = []
    for ticklabel in ax.get_xticklabels():
        _label = ticklabel.get_text()
        label_range = ranges.loc[("Europe", _label), :].round(1)
        xticklabels.append(
            f"{_label}\n({label_range['min']} - {label_range['max']} GW)"
        )
    ax.set_xticklabels(xticklabels, fontsize=8)


def add_boxes_around_pathway_ranges(ax, ranges, color):
    _x = 0
    for ticklabel in ax.get_xticklabels():
        _label = ticklabel.get_text()

        width = 0.2
        # Plot 2030 box
        y_min_2030 = ranges.loc[2030, "Europe", _label]["min"]
        y_max_2030 = ranges.loc[2030, "Europe", _label]["max"]
        height = y_max_2030 - y_min_2030
        height += 0.018
        ax.add_patch(
            mpl.patches.Rectangle(
                xy=(_x - 0.3, y_min_2030 - 0.009),
                height=height,
                width=width,
                fc="None",
                ec=color,
                linestyle="--",
                lw=0.75,
                zorder=10,
            ),
        )

        # Plot 2050 box
        if _label == "Coal":
            y_max_2050 = 0
            y_min_2050 = 0
        else:
            y_max_2050 = ranges.loc[2050, "Europe", _label]["max"]
            y_min_2050 = ranges.loc[2050, "Europe", _label]["min"]

        height = y_max_2050 - y_min_2050
        height += 0.018
        ax.add_patch(
            mpl.patches.Rectangle(
                xy=(_x + 0.1, y_min_2050 - 0.009),
                height=height,
                width=width,
                fc="None",
                ec=color,
                linestyle="--",
                lw=0.75,
                zorder=10,
            ),
        )

        # Color paths between 2030 and 2050
        x_path = [_x - 0.1, _x + 0.1]
        y_path_max = [y_max_2030 + 0.009, y_max_2050 + 0.009]
        y_path_min = [y_min_2030 - 0.009, y_min_2050 - 0.009]
        ax.fill_between(x=x_path, y1=y_path_max, y2=y_path_min, color=color, alpha=0.25)

        _x += 1


def calc_pathway_growth(power_capacity_2030, power_capacity_2050, pathway):
    # Prepare data
    power_capacity_2030 = power_capacity_2030.reset_index()
    power_capacity_2050 = power_capacity_2050.reset_index()
    power_capacity_2030["year"] = 2030
    power_capacity_2050["year"] = 2050
    power_capacity = pd.concat([power_capacity_2030, power_capacity_2050])

    # Calculate capacity ranges for chosen pathway
    power_capacity_in_path = power_capacity.query(
        f"(cluster == {pathway[0]} and year == 2030) or (cluster == {pathway[1]} and year == 2050)"
    )

    # Pathway ranges
    pathway_ranges = power_capacity_in_path.groupby(["year", "region", "technology"])[
        "capacity_gw"
    ].agg(["min", "max"])
    print(pathway_ranges)

    # Calculate linear growth rates
    n_years = 20
    growth_rates = (pathway_ranges.loc[2050] - pathway_ranges.loc[2030]) / n_years
    print(growth_rates)

    # Calculate compound annual growth rate
    cagr = ((pathway_ranges.loc[2050] / pathway_ranges.loc[2030]) ** (1 / n_years)) - 1
    print(cagr * 100)


def plot_capacity_pathway(power_capacity_2030, power_capacity_2050, pathway):
    # Prepare data
    power_capacity_2030 = power_capacity_2030.reset_index()
    power_capacity_2050 = power_capacity_2050.reset_index()
    power_capacity_2030["year"] = 2030
    power_capacity_2050["year"] = 2050
    power_capacity = pd.concat([power_capacity_2030, power_capacity_2050])
    technologies = power_capacity["technology"].unique()
    features = power_capacity.groupby(["region", "technology"]).size()

    # Calculate capacity ranges
    capacity_ranges = power_capacity.groupby(["region", "technology"])[
        "capacity_gw"
    ].agg(["min", "max"])

    # Normalise data
    power_capacity["capacity_normalised"] = power_capacity.apply(
        lambda row: row["capacity_gw"]
        / capacity_ranges.loc[(row["region"], row["technology"]), "max"],
        axis=1,
    )

    # Split data in SPORES that belong to pathway and other SPORES
    power_capacity_colored = power_capacity.query(
        f"(cluster == {pathway[0]} and year == 2030) or (cluster == {pathway[1]} and year == 2050)"
    )
    power_capacity_grey = power_capacity.query(
        f"(cluster != {pathway[0]} and year == 2030) or (cluster != {pathway[1]} and year == 2050)"
    )

    # Pathway ranges
    pathway_ranges = power_capacity_colored.groupby(["year", "region", "technology"])[
        "capacity_normalised"
    ].agg(["min", "max"])

    # Get color palette

    # Plot figure
    fig, ax = plt.subplots(1, 1, figsize=(FIGWIDTH, 10 * FIGWIDTH / 20))

    # Plot grey SPORES (that do not belong to the pathway)
    sns.stripplot(
        ax=ax,
        data=power_capacity_grey,
        x="technology",
        y="capacity_normalised",
        hue="year",
        hue_order=[2030, 2050],
        dodge=True,
        marker="o",
        palette={
            2030: sns.color_palette("bright")[-3],
            2050: sns.color_palette("bright")[-3],
        },
        alpha=0.5,
        s=3,
    )
    # Plot colored SPORES (that belong to the pathway)
    color = sns.color_palette("bright")[1]
    sns.stripplot(
        ax=ax,
        data=power_capacity_colored,
        x="technology",
        y="capacity_normalised",
        hue="year",
        hue_order=[2030, 2050],
        dodge=True,
        marker="o",
        palette={
            2030: color,
            2050: color,
        },
        alpha=0.5,
        s=3,
    )

    ax.set_xticks(range(len(capacity_ranges.index)))

    # Add boxes around range of pathway
    add_boxes_around_pathway_ranges(ax=ax, ranges=pathway_ranges, color=color)

    # Add ranges to x-axis
    add_capacity_ranges_to_x_axis(ax=ax, ranges=capacity_ranges)

    # Add title
    ax.set_title(
        f"Pathway from scenario {pathway[0]} in 2030 to scenario {pathway[1]} in 2050"
    )

    # Format y-axis
    ax.set_ylabel("Capacity (Normalised to SPORE with highest capacity)")


if __name__ == "__main__":
    """
    0. SET PARAMETERS
    """
    # Set spatial granularity for which to run the analysis ("national", or "continental")
    spatial_resolution = "continental"

    """
    1. READ AND PREPARE DATA
    """
    # Read data
    path_to_processed_data = os.path.join(os.getcwd(), "..", "data", "processed")
    years = find_years(path_to_processed_data=path_to_processed_data)
    power_capacity, paper_metrics = get_processed_data(
        path_to_processed_data=path_to_processed_data, resolution=spatial_resolution
    )

    spore_to_scenario_maps = get_spore_to_scenario_maps(
        path_to_processed_data=path_to_processed_data, resolution=spatial_resolution
    )

    # Add cluster index to data
    for year in years:
        power_capacity[year] = add_cluster_index_to_series(
            data=power_capacity.get(year),
            cluster_mapper=spore_to_scenario_maps.get(year),
        )
        paper_metrics[year] = add_cluster_index_to_series(
            data=paper_metrics.get(year),
            cluster_mapper=spore_to_scenario_maps.get(year),
        )
        # Calculate amount of spore in each cluster
        count_spores_per_cluster(power_capacity.get(year))

    """
    Plot pathway
    """
    # Find all possible pathways (= all combinations of clusters in 2030 with clusters in 2050)
    pathways = list(
        itertools.product(
            power_capacity.get("2030").index.unique(level="cluster"),
            power_capacity.get("2050").index.unique(level="cluster"),
        )
    )
    # Choose pathway: pathway = (cluster_2030, cluster_2050)
    pathway = (3, 6)
    print(pathway)
    print(pathways[:10])

    for pathway in pathways:
        plot_capacity_pathway(
            power_capacity_2030=power_capacity.get("2030"),
            power_capacity_2050=power_capacity.get("2050"),
            pathway=pathway,
        )
    plt.show()

    #
    calc_pathway_growth(
        power_capacity_2030=power_capacity.get("2030"),
        power_capacity_2050=power_capacity.get("2050"),
        pathway=pathway,
    )
