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


def analyse_energy_consumption_or_supply_per_country(data):
    df_2030 = pd.DataFrame(
        data.get("2030").groupby(["region", "spore"]).sum()
    ).reset_index()
    df_2050 = pd.DataFrame(
        data.get("2050").groupby(["region", "spore"]).sum()
    ).reset_index()
    df_2030["year"] = 2030
    df_2050["year"] = 2050
    df = pd.concat([df_2030, df_2050], ignore_index=True)
    df = df[df["region"] != "Europe"]
    # print(df)
    median_2030 = (
        df[df["year"] == 2030]
        .groupby("region")[data.get("2030").name]
        .median()
        .sort_values()
    )
    country_order = median_2030.index.tolist()
    top_10_regions = median_2030.tail(5).index.tolist()
    df_top_10 = df[df["region"].isin(top_10_regions)]
    # pd.set_option("display.max_rows", None)
    # print(data.get("2030").loc[top_10_regions, :, [1, 2, 3, 4, 5, 6]])
    plt.figure(figsize=(2.5 * FIGWIDTH, 1.25 * FIGWIDTH * 9 / 16))
    sns.boxplot(
        data=df_top_10,
        x="region",
        y=data.get("2030").name,
        hue="year",
        hue_order=[2030, 2050],
        order=top_10_regions,
        showfliers=False,
    )
    plt.xticks(rotation=45)


def plot_metrics(metrics):
    fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGWIDTH))
    # Calculate metric ranges and get units
    metric_ranges = metrics.groupby(level="metric").agg(["min", "max"])
    metric_units = (
        metrics.index.to_frame(index=False)
        .drop_duplicates(subset="metric")
        .set_index("metric")["unit"]
        .to_dict()
    )

    # Normalise metrics
    metrics_normalised = (
        metrics.groupby(["metric"]).transform(normalise_to_max).reset_index()
    )
    #
    sns.stripplot(
        ax=ax,
        data=metrics_normalised,
        x="metric",
        y="paper_metrics",
        marker="o",
        color=sns.color_palette("bright")[-3],
        alpha=0.5,
        s=3,
    )


def plot_characteristics_as_distribution_for_2030_and_2050():
    pass


def plot_trade_offs_as_correlation_heatmap(power_data, metrics_data, year, ax):
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
            [("", metric, spore) for (spore, metric, unit) in metrics_data.index]
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
        size_factor = np.abs(corr) / np.max(np.abs(corr)) * 0.9

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
            corr.columns.get_level_values("technology"), rotation=90, fontsize=9
        )
        ax.set_yticklabels(corr.index.get_level_values("technology"), fontsize=9)
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
                -12,
                y_pos,
                corr.index[country_changes_y[i]][0],
                rotation=90,
                verticalalignment="center",
                color="black",
                weight="bold",
                fontsize=11,
            )
        for i in range(len(country_changes_x) - 1):
            x_pos = (country_changes_x[i] + country_changes_x[i + 1]) / 2
            ax.text(
                x_pos,
                -12,
                corr.columns[country_changes_y[i]][0],
                horizontalalignment="center",
                color="black",
                weight="bold",
                fontsize=11,
            )
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
        plt.tight_layout()

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


def filter_data_on_countries_of_interest(data, countries_of_interest):
    for year in data.keys():
        data[year] = data.get(year)[
            data.get(year).index.get_level_values("region").isin(countries_of_interest)
        ]
    return data


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

    # Add battery and grid capacity to power capacity data
    power_capacity = add_battery_and_grid_capacity_to_power_capacity(
        power_capacity, storage_capacity, grid_capacity
    )

    # Filter data for regions to analyse
    power_capacity = filter_data_on_countries_of_interest(
        power_capacity, regions_to_analyse
    )
    primary_energy_supply = filter_data_on_countries_of_interest(
        primary_energy_supply, regions_to_analyse
    )

    """
    TPES
    """

    # analyse_energy_consumption_or_supply_per_country(primary_energy_supply)
    # plt.ylabel("Total Primary Energy Supply [TWh]")

    """
    TRADE-OFFS: CORRELATION BETWEEN POWER CAPACITY & SYSTEM METRICS
    """
    ax = {}
    fig = plt.subplots(figsize=(1.2 * FIGWIDTH, 1.2 * FIGWIDTH))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 8, 1], width_ratios=[39, 1])
    ax["heatmap"] = plt.subplot(gs[0:, 0], frameon=True)
    ax["colorbar"] = plt.subplot(gs[1, 1], frameon=False)

    plot_trade_offs_as_correlation_heatmap(power_capacity, paper_metrics, "2030", ax)

    """
    METRICS
    """
    plot_metrics(metrics=paper_metrics.get("2030"))
    plot_metrics(metrics=paper_metrics.get("2050"))

    # plt.show()
