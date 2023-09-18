import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from string import ascii_letters
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon, Rectangle

from src.utils.data_io import *
from src.utils.visualisation import *


def filter_data_on_countries_of_interest(data, countries_of_interest):
    for year in data.keys():
        data[year] = data.get(year)[
            data.get(year).index.get_level_values("region").isin(countries_of_interest)
        ]
    return data


def plot_primary_energy_distribution(
    primary_energy_data, region="Europe", distribution="per_source"
):
    def prepare_primary_energy_data_for_distribution_plot_per_source(
        primary_energy_data, region="Europe"
    ):
        primary_energy_data_2030 = primary_energy_data.get("2030").copy()
        primary_energy_data_2050 = primary_energy_data.get("2050").copy()

        # Concatenate 2030 and 2050
        primary_energy_data_2030.index = pd.MultiIndex.from_tuples(
            [
                ("2030", idx[0], idx[1], idx[2])
                for idx in primary_energy_data_2030.index
            ],
            names=["year", "region", "carriers", "spore"],
        )
        primary_energy_data_2050.index = pd.MultiIndex.from_tuples(
            [
                ("2050", idx[0], idx[1], idx[2])
                for idx in primary_energy_data_2050.index
            ],
            names=["year", "region", "carriers", "spore"],
        )
        primary_energy_data = pd.concat(
            [primary_energy_data_2030, primary_energy_data_2050]
        )

        # Filter on region
        primary_energy_data = primary_energy_data.xs(
            region, level="region", drop_level=False
        )
        if region == "Europe":
            primary_energy_data = primary_energy_data.drop(
                [
                    "Natural gas, carbon-neutral net imports",
                    "Oil, carbon-neutral net imports",
                    "Net electricity import",
                ],
                level="carriers",
            )
        else:
            primary_energy_data = primary_energy_data.rename(
                index={
                    "Natural gas, carbon-neutral net imports": "Net natural gas import",
                    "Oil, carbon-neutral net imports": "Net oil import",
                }
            )
            # TODO: add all import/export carriers to one import export group?

        # Calc nuclear heat input to account for energy loss (as electricity is output)
        mask = (
            primary_energy_data.index.get_level_values("carriers")
            == "Nuclear electricity"
        )
        primary_energy_data[mask] *= NUCLEAR_HEAT_MULTIPLIER
        primary_energy_data = primary_energy_data.rename(
            index={"Nuclear electricity": "Nuclear heat"}
        )

        # Calculate ranges
        primary_energy_ranges = (
            primary_energy_data.groupby(level=["region", "carriers"])
            .agg(["min", "max"])
            .droplevel("region")
        )

        # primary_energy_normalised capacity
        primary_energy_normalised = (
            primary_energy_data.groupby(level=["region", "carriers"])
            .transform(normalise_to_max)
            .reset_index()
        )

        return primary_energy_normalised, primary_energy_ranges

    def prepare_primary_energy_data_for_distribution_plot_per_country(
        primary_energy_data,
    ):
        primary_energy_data_2030 = (
            primary_energy_data.get("2030").copy().groupby(["region", "spore"]).sum()
        )
        primary_energy_data_2050 = (
            primary_energy_data.get("2050").copy().groupby(["region", "spore"]).sum()
        )

        # Concatenate 2030 and 2050
        primary_energy_data_2030.index = pd.MultiIndex.from_tuples(
            [("2030", idx[0], idx[1]) for idx in primary_energy_data_2030.index],
            names=["year", "region", "spore"],
        )
        primary_energy_data_2050.index = pd.MultiIndex.from_tuples(
            [("2050", idx[0], idx[1]) for idx in primary_energy_data_2050.index],
            names=["year", "region", "spore"],
        )
        primary_energy_data = pd.concat(
            [primary_energy_data_2030, primary_energy_data_2050]
        )

        # Calculate ranges
        primary_energy_ranges = primary_energy_data.groupby(level="region").agg(
            ["min", "max"]
        )

        # primary_energy_normalised capacity
        primary_energy_normalised = (
            primary_energy_data.groupby(level="region")
            .transform(normalise_to_max)
            .reset_index()
        )

        return primary_energy_normalised, primary_energy_ranges

    def plot_distribution(data, distribution, ax):
        if distribution == "per_source":
            x = "carriers"
        elif distribution == "per_country":
            x = "region"
        sns.stripplot(
            ax=ax,
            data=data,
            x=x,
            y="primary_energy_supply_twh",
            hue="year",
            hue_order=["2030", "2050"],
            jitter=True,
            dodge=True,
            # palette=[sns.color_palette("bright")[-3], sns.color_palette("bright")[-3]],
            marker="o",
            s=3,
            alpha=0.5,
        )

    def add_ranges_to_xticks(ranges, ax):
        xticklabels = []

        for ticklabel in ax.get_xticklabels():
            _label = ticklabel.get_text()
            # Format x-axis labels
            if _label in ["Net natural gas import", "Net oil import"]:
                energy_range = ranges.loc[_label].round(1).astype(float)
                if energy_range["max"] == -0.0:
                    energy_range["max"] = 0.0
            else:
                energy_range = ranges.loc[_label].round(0).astype(int)

            xticklabels.append(
                f"{primary_energy_plot_names[_label]}\n({energy_range['min']} - {energy_range['max']}) twh"
            )
        ax.set_xticklabels(xticklabels, fontsize=9)

    # Prepare data
    if distribution == "per_source":
        (
            primary_energy_normalised,
            primary_energy_ranges,
        ) = prepare_primary_energy_data_for_distribution_plot_per_source(
            primary_energy_data, region
        )
    elif distribution == "per_country":
        (
            primary_energy_normalised,
            primary_energy_ranges,
        ) = prepare_primary_energy_data_for_distribution_plot_per_country(
            primary_energy_data
        )

    # Plot figure
    fig, ax = plt.subplots(figsize=(1.5 * FIGWIDTH, 0.75 * FIGWIDTH * 9 / 16))
    plot_distribution(primary_energy_normalised, distribution, ax)

    # Remove top and right spines
    remove_top_and_right_spines(ax)
    # Format y-axis
    ax.set_ylabel(
        "Normalised primary energy supply",
        weight="bold",
    )
    # Format x-axis
    if distribution == "per_source":
        _labels = list(primary_energy_normalised["carriers"].unique())
    elif distribution == "per_country":
        _labels = list(primary_energy_normalised["region"].unique())
    ax.set_xlabel("")
    ax.set_xticks(range(len(_labels)))
    ax.set_xticklabels(_labels, fontsize=10)
    add_ranges_to_xticks(primary_energy_ranges, ax)
    # Set title
    ax.set_title(f"{region}", weight="bold")

    if primary_energy_normalised["primary_energy_supply_twh"].min() < 0:
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    plt.tight_layout()


def plot_power_capacity_distribution(power_data, region="Europe"):
    def prepare_power_data_for_distribution_plot(power_data, region="Europe"):
        capacity_data_2030 = power_data.get("2030").copy()
        capacity_data_2050 = power_data.get("2050").copy()

        # Concatenate 2030 and 2050
        capacity_data_2030.index = pd.MultiIndex.from_tuples(
            [("2030", idx[0], idx[1], idx[2]) for idx in capacity_data_2030.index],
            names=["year", "region", "technology", "spore"],
        )
        capacity_data_2050.index = pd.MultiIndex.from_tuples(
            [("2050", idx[0], idx[1], idx[2]) for idx in capacity_data_2050.index],
            names=["year", "region", "technology", "spore"],
        )
        capacity_data = pd.concat([capacity_data_2030, capacity_data_2050])

        # Filter on region
        capacity_data = capacity_data.xs(region, level="region", drop_level=False)

        # Calculate ranges
        capacity_ranges = (
            capacity_data.groupby(level=["region", "technology"])
            .agg(["min", "max"])
            .droplevel("region")
        )

        # Normalise capacity
        capacity_normalised = (
            capacity_data.groupby(level=["region", "technology"])
            .transform(normalise_to_max)
            .reset_index()
        ).rename(columns={"0": "capacity"})

        return capacity_normalised, capacity_ranges

    def plot_distribution(data, ax):
        sns.stripplot(
            ax=ax,
            data=data,
            x="technology",
            y="capacity_gw",
            hue="year",
            hue_order=["2030", "2050"],
            jitter=True,
            dodge=True,
            # palette=[sns.color_palette("bright")[-3], sns.color_palette("bright")[-3]],
            marker="o",
            s=3,
            alpha=0.5,
        )

    def add_ranges_to_xticks(ranges, ax):
        xticklabels = []
        print(ranges)
        for ticklabel in ax.get_xticklabels():
            _tech = ticklabel.get_text()

            # Format x-axis labels
            capacity_range = ranges.loc[_tech].round(0).astype(int)
            if _tech == "Battery":
                _unit = "gwh"
            else:
                _unit = "gw"
            xticklabels.append(
                f"{tech_plot_names[_tech]}\n({capacity_range['min']} - {capacity_range['max']}) {_unit}"
            )
        ax.set_xticklabels(xticklabels, fontsize=9)

    # Prepare data
    capacity_normalised, capacity_ranges = prepare_power_data_for_distribution_plot(
        power_data, region
    )
    # Plot figure
    fig, ax = plt.subplots(figsize=(1.5 * FIGWIDTH, 0.75 * FIGWIDTH * 9 / 16))
    plot_distribution(capacity_normalised, ax)
    # Remove top and right spines
    remove_top_and_right_spines(ax)
    tech_labels = list(capacity_normalised["technology"].unique())

    # Format y-axis
    ax.set_ylabel(
        "Normalised installed capacity",
        weight="bold",
    )
    # Format x-axis
    ax.set_xlabel("")
    ax.set_xticks(range(len(tech_labels)))
    ax.set_xticklabels(tech_labels, fontsize=10)
    add_ranges_to_xticks(capacity_ranges, ax)
    # Set title
    ax.set_title(f"{region}", weight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()


def plot_metrics_distribution(metrics_data, region="Europe"):
    def prepare_metrics_data_for_distribution_plot(metrics_data):
        metrics_data_2030 = metrics_data.get("2030").copy()
        metrics_data_2050 = metrics_data.get("2050").copy()
        print("metric ranges 2030 & 2050:")
        print(metrics_data_2030.groupby("metric").agg(["min", "max"]))
        print(metrics_data_2050.groupby("metric").agg(["min", "max"]))
        # Concatenate 2030 and 2050
        metrics_data_2030.index = pd.MultiIndex.from_tuples(
            [("2030", idx[1], idx[2], idx[0]) for idx in metrics_data_2030.index],
            names=["year", "metric", "unit", "spore"],
        )
        metrics_data_2050.index = pd.MultiIndex.from_tuples(
            [("2050", idx[1], idx[2], idx[0]) for idx in metrics_data_2050.index],
            names=["year", "metric", "unit", "spore"],
        )
        metrics_data = pd.concat([metrics_data_2030, metrics_data_2050])

        # Calculate ranges
        metrics_ranges = metrics_data.groupby("metric").agg(["min", "max"])

        # Normalise capacity
        metrics_normalised = (
            metrics_data.groupby("metric").transform(normalise_to_max).reset_index()
        ).rename(columns={"paper_metrics": "metric_value"})

        return metrics_normalised, metrics_ranges

    def plot_distribution(data, ax):
        sns.stripplot(
            ax=ax,
            data=data,
            x="metric",
            y="metric_value",
            hue="year",
            hue_order=["2030", "2050"],
            jitter=True,
            dodge=True,
            # palette=[sns.color_palette("bright")[-3], sns.color_palette("bright")[-3]],
            marker="o",
            s=3,
            alpha=0.5,
        )

    def add_ranges_to_xticks(ranges, units, ax):
        xticklabels = []
        for ticklabel in ax.get_xticklabels():
            _metric = ticklabel.get_text()
            # Format x-axis labels
            metric_range = ranges.apply(metric_range_formatting[_metric]).loc[_metric]
            _unit = units.get(_metric)
            if _unit == "percentage":
                _unit = " %"
            elif _unit == "fraction":
                _unit = ""
            else:
                _unit = " " + _unit
            xticklabels.append(
                f"{metric_plot_names[_metric]}\n({metric_range['min']} - {metric_range['max']}){_unit}"
            )
        ax.set_xticklabels(xticklabels, fontsize=9)

    # Prepare data
    metrics_normalised, metric_ranges = prepare_metrics_data_for_distribution_plot(
        metrics_data
    )
    metric_units = (
        metrics_data.get("2030")
        .index.to_frame(index=False)
        .drop_duplicates(subset="metric")
        .set_index("metric")["unit"]
        .to_dict()
    )
    # Plot figure
    fig, ax = plt.subplots(figsize=(1.5 * FIGWIDTH, 0.75 * FIGWIDTH * 9 / 16))
    plot_distribution(metrics_normalised, ax)
    # Remove top and right spines
    remove_top_and_right_spines(ax)
    metric_labels = list(metrics_normalised["metric"].unique())
    # Format y-axis
    ax.set_ylabel(
        "Normalised metric",
        weight="bold",
    )
    # Format x-axis
    ax.set_xlabel("")
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=10)
    add_ranges_to_xticks(metric_ranges, metric_units, ax)
    # Set title
    ax.set_title(f"{region}", weight="bold")
    plt.tight_layout()


def plot_trade_offs_as_correlation_heatmap(power_data, metrics_data, year):
    def prepaare_data_for_correlation_plot(power_data, metrics_data):
        power_data = power_data.get(year).copy()
        metrics_data = metrics_data.get(year).copy()
        # Drop hydro & nuclear (not useful because they do not vary across spores)
        power_data = power_data.loc[
            ~power_data.index.get_level_values("technology").isin(["Hydro", "Nuclear"])
        ].rename(index={"United Kingdom": "UK"})

        # Transform data
        df_power = power_data.unstack(["region", "technology"])

        df_metrics = metrics_data
        df_metrics.index = pd.MultiIndex.from_tuples(
            [("Europe", metric, spore) for (spore, metric, unit) in metrics_data.index]
        ).set_names(["region", "metric", "spore"])
        df_metrics = df_metrics.rename(index=RENAME_METRICS).unstack(
            ["region", "metric"]
        )

        df = pd.concat([df_power, df_metrics], axis=1).rename_axis(
            columns=["region", "technology"]
        )

        # Copmute correlation matrics
        correlation_matrix = df.corr()
        correlation_matrix.fillna(0, inplace=True)

        return correlation_matrix

    def plot_custom_heatmap(corr, cmap, ax):
        n = corr.shape[0]
        size_factor = np.abs(corr) / np.max(np.abs(corr))

        for i in range(n):
            for j in range(n):
                # Skip diagonal entries
                if i == j:
                    continue

                color = cmap(corr.iloc[i, j])
                x_start = j + 0.5 - size_factor.iloc[i, j] / 2
                y_start = i + 0.5 - size_factor.iloc[i, j] / 2

                rect = plt.Rectangle(
                    (x_start, y_start),
                    width=size_factor.iloc[i, j],
                    height=size_factor.iloc[i, j],
                    facecolor=color,
                    edgecolor="none",
                )
                ax.add_patch(rect)

        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_xticklabels(
            corr.columns.get_level_values("technology"), rotation=90, fontsize=8
        )
        ax.set_yticklabels(corr.index.get_level_values("technology"), fontsize=8)
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.grid(False)
        ax.set_aspect("equal")

        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
        ax.set_title(f"{year}", weight="bold")

    def plot_tapered_colorbar(cmap, ax):
        n_colors = cmap.N

        # Create array of widths
        y = np.linspace(-1, 1, n_colors)
        widths = np.interp(y, [-1, 0, 1], [1, 0, 1])
        max_width = max(widths)

        # Generate tapered colorbar
        img = np.zeros((n_colors, int(n_colors * max_width), 4))
        for i, width in enumerate(widths):
            start_col = int((max_width - width) * n_colors / 2)
            end_col = start_col + int(width * n_colors)
            color = cmap(i)
            img[i, start_col:end_col] = color

        ax.imshow(
            img,
            aspect="auto",
            origin="lower",
            extent=[0, max_width, -1, 1],
        )
        ax.set_yticks([-1, 0, 1])
        ax.yaxis.tick_right()
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def draw_lines_between_countries(corr, ax):
        n = corr.shape[0]
        country_changes_y = (
            [0]
            + [i for i in range(1, n) if corr.index[i][0] != corr.index[i - 1][0]]
            + [n]
        )
        country_changes_x = (
            [0]
            + [i for i in range(1, n) if corr.columns[i][0] != corr.columns[i - 1][0]]
            + [n]
        )
        # Add horizontal and vertical lines between countries
        for y in country_changes_y[1:-1]:
            ax.axhline(y=y, color="black", linewidth=0.5)
        for x in country_changes_x[1:-1]:
            ax.axvline(x=x, color="black", linewidth=0.5)

        # Add country annotations
        for i in range(len(country_changes_y) - 1):
            y_pos = (country_changes_y[i] + country_changes_y[i + 1]) / 2
            ax.text(
                -16,
                y_pos,
                corr.index[country_changes_y[i]][0],
                rotation=90,
                verticalalignment="center",
                color="black",
                weight="bold",
                fontsize=10,
            )
        for i in range(len(country_changes_x) - 1):
            x_pos = (country_changes_x[i] + country_changes_x[i + 1]) / 2
            ax.text(
                x_pos,
                -16,
                corr.columns[country_changes_y[i]][0],
                horizontalalignment="center",
                color="black",
                weight="bold",
                fontsize=10,
            )
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
        plt.tight_layout()

    # Generate figure
    ax = {}
    fig = plt.subplots(figsize=(1.2 * FIGWIDTH, 1.2 * FIGWIDTH))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 8, 1], width_ratios=[39, 1])
    ax["heatmap"] = plt.subplot(gs[0:, 0], frameon=True)
    ax["colorbar"] = plt.subplot(gs[1, 1], frameon=False)

    # Prepare data
    corr = prepaare_data_for_correlation_plot(power_data, metrics_data)

    # Generate a custom diverging colormap
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    # Plot correlation heatmap
    plot_custom_heatmap(corr, cmap, ax["heatmap"])

    # Plot colorbar legend
    plot_tapered_colorbar(cmap, ax["colorbar"])

    # Draw lines to seperate countries
    draw_lines_between_countries(corr, ax["heatmap"])


def add_battery_and_grid_capacity_to_power_capacity(
    power_data, storagge_data, grid_data
):
    for year in power_data.keys():
        # Compute battery capacity per country
        battery_capacity = storagge_data.get(year).xs(
            "Battery", level="technology", drop_level=False
        )

        # Compute international grid capacity per country
        total_grid_capacity = (
            grid_data.get(year).groupby(["importing_region", "spore"]).sum()
        )
        total_grid_capacity.index = pd.MultiIndex.from_tuples(
            [(index[0], "Grid", index[1]) for index in total_grid_capacity.index]
        ).set_names(["region", "technology", "spore"])

        # Calculate continental capacity
        total_grid_capacity_eu = total_grid_capacity.groupby("spore").sum()
        total_grid_capacity_eu.index = pd.MultiIndex.from_tuples(
            [("Europe", "Grid", spore) for spore in total_grid_capacity_eu.index]
        ).set_names(["region", "technology", "spore"])

        # Add grid and battery capacity to power capacity data
        power_data[year] = pd.concat(
            [
                power_data.get(year),
                battery_capacity,
                total_grid_capacity,
                total_grid_capacity_eu,
            ]
        ).sort_index(level=["region", "technology", "spore"])

    return power_data


if __name__ == "__main__":
    years = ["2030", "2050"]
    path_to_processed_data = os.path.join(os.getcwd(), "..", "data", "processed")
    region_of_interest = "Germany"
    regions_to_analyse = [
        "France",
        "Germany",
        "Italy",
        "Spain",
        "United Kingdom",
        "Europe",
    ]

    """
    1. READ AND PREPARE DATA
    """
    power_capacity = load_processed_power_capacity(path_to_processed_data, years)
    paper_metrics = load_processed_paper_metrics(path_to_processed_data, years)
    grid_capacity = load_processed_grid_transfer_capacity(path_to_processed_data, years)
    storage_capacity = load_processed_storage_capacity(path_to_processed_data, years)
    final_consumption = load_processed_final_consumption(path_to_processed_data, years)
    primary_energy_supply = load_processed_primary_energy_supply(
        path_to_processed_data, years
    )

    # Biggest country in terms of TPES
    # print(
    #     primary_energy_supply.get("2030")
    #     .groupby(["region", "spore"])
    #     .sum()
    #     .groupby("region")
    #     .median()
    #     .sort_values()
    # )
    # print(
    #     primary_energy_supply.get("2050")
    #     .groupby(["region", "spore"])
    #     .sum()
    #     .groupby("region")
    #     .median()
    #     .sort_values()
    # )

    # Filter data for regions to analyse
    power_capacity = filter_data_on_countries_of_interest(
        power_capacity, regions_to_analyse
    )
    primary_energy_supply = filter_data_on_countries_of_interest(
        primary_energy_supply, regions_to_analyse
    )

    """
    2. CHARACTERISTICS
    """
    # TPES per source
    plot_primary_energy_distribution(primary_energy_supply, region=region_of_interest)
    # # TPES per country
    # plot_primary_energy_distribution(
    #     primary_energy_supply, region=region_of_interest, distribution="per_country"
    # )

    # POWER
    plot_power_capacity_distribution(power_capacity, region=region_of_interest)
    # METRICS
    plot_metrics_distribution(paper_metrics)

    """
    3. TRADE-OFFS: CORRELATION BETWEEN POWER CAPACITY & SYSTEM METRICS
    """
    plot_trade_offs_as_correlation_heatmap(power_capacity, paper_metrics, "2030")
    plot_trade_offs_as_correlation_heatmap(power_capacity, paper_metrics, "2050")
    plt.show()
