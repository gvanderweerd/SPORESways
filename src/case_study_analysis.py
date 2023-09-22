import json
import os
import pandas as pd
import matplotlib.patches as mpatches

# Functions from own source code
from src.utils.parameters import *
from src.utils.visualisation import *
from src.utils.data_io import *


def plot_technology_target_capacity(
    power_data,
    historic_years,
    spores_years,
    target_country,
    target_capacity,
    target_technology,
):
    def _prepare_data_for_plot(power_data, target_technology):
        # Filter data for country & technology
        power_data_filtered = {}
        for year in power_data.keys():
            power_data_filtered[year] = power_data.get(year).loc[
                target_country, target_technology, :
            ]
        return power_data_filtered

    def _get_color_maps(power_data_filtered, target_capacity):
        y_2050 = _get_projected_2050_value(power_data_filtered)

        colors_2030 = [
            "green" if capacity >= target_capacity else "red"
            for capacity in power_data_filtered.get("2030")
        ]
        colors_2050 = [
            "grey"
            if capacity < target_capacity
            else ("blue" if 215 <= capacity < y_2050 else "orange")
            for capacity in power_data_filtered.get("2050")
        ]
        colors = {"2030": colors_2030, "2050": colors_2050}
        palettes = {
            "2030": ["green", "red"],
            "2050": ["mediumorchid", "cornflowerblue", "orange"],
        }
        return colors, palettes

    def _get_projected_2050_value(power_data):
        # Define points
        x1, y1 = 2022, power_data.get("2022")[0]
        x2, y2 = 2030, target_capacity

        # Linear function parameters
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        return m * 2050 + b

    def _add_capacity_value(ax, year, value, x_offset, y_offset):
        # Plot black dot on the map of the value that is plotted
        ax.scatter(x=year, y=value, marker=open_circle, color="black")
        # Plot capacity value text at x=year
        ax.annotate(
            xy=(year, value),
            xytext=(year + x_offset, value + y_offset),
            text=f"{value:.1f}",
            fontsize=12,
        )

    def _add_historic_capacity(ax, power_data, historic_years):
        historic_capacity = []
        for year in historic_years:
            historic_capacity.append(power_data.get(year)[0])

        sns.lineplot(
            ax=ax,
            x=[int(year) for year in historic_years],
            y=historic_capacity,
            color="black",
            marker="o",
        )

    def _add_spores_capacity(ax, power_data, spores_years):
        power_cap = power_data.copy()
        colors, palettes = _get_color_maps(power_cap, target_capacity)
        for year in spores_years:
            # Count number of SPORES in capacity spores data
            n_spores = len(power_cap.get(year).index.unique(level="spore"))
            # Define small jitter to increase visibility of SPORES
            jitter = 0.3
            # Define x for capacity SPORES
            x_spores = [
                int(year) + np.random.uniform(-jitter, jitter) for _ in range(n_spores)
            ]
            # Plot capacity SPORES
            sns.scatterplot(
                ax=ax,
                x=x_spores,
                y=power_cap.get(year),
                marker="o",
                hue=colors.get(year),
                palette=palettes.get(year),
                # edgecolor=None,
                legend=True,
                s=10,
                # alpha=0.5,
            )

    def _add_projection_lines(ax, power_data, target_capacity):
        # Define points
        x1, y1 = 2022, power_data.get("2022")[0]
        x2, y2 = 2030, target_capacity

        # Linear function parameters
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # Generate lines
        x_linear_growth = np.linspace(2022, 2050, 100)
        y_linear_growth = m * x_linear_growth + b
        x_maintain_capacity = np.linspace(2030, 2050, 100)
        y_maintain_capacity = [target_capacity] * 100

        # Plot lines
        ax.plot(x_linear_growth, y_linear_growth, "k--", linewidth=1)
        ax.plot(x_maintain_capacity, y_maintain_capacity, "k--", linewidth=1)

        # Calculate y value for x = 2040
        y_2040 = m * 2040 + b

        # Plot average growth rate text at x=2040
        ax.annotate(
            text=f"Avg. growth rate: {m:.1f} GW/year",
            xy=(2040, y_2040),
            xytext=(2035, y_2040 + 120),
            fontsize=12,
        )

    def _format_axis(ax, target_technology, target_country):
        ax.set_ylabel(f"Installed Capacity [GW]", weight="bold")
        ax.set_title(f"{target_country}, {target_technology}", weight="bold")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    def _format_legend(ax):
        legend = ax.get_legend()
        legend_labels = [
            "Target 2030 met",
            "Target 2030 missed",
            "Reduction to 2050",
            "Steady growth to 2050",
            "Growth acceleration to 2050",
        ]
        for label, text in zip(legend.get_texts(), legend_labels):
            label.set_text(text)

    # Prepare data
    power_data_filtered = _prepare_data_for_plot(power_data, target_technology)

    # Define figure
    fig, ax = plt.subplots(figsize=(1.5 * FIGWIDTH, 0.75 * FIGWIDTH * 9 / 16))

    # Add plot elements
    _add_historic_capacity(ax, power_data_filtered, historic_years)
    _add_spores_capacity(ax, power_data_filtered, spores_years)
    _add_projection_lines(ax, power_data_filtered, target_capacity)
    _add_capacity_value(
        ax, 2022, power_data_filtered.get("2022")[0], x_offset=0, y_offset=-70
    )
    _add_capacity_value(ax, 2030, target_capacity, x_offset=0.4, y_offset=-60)
    _add_capacity_value(
        ax,
        2050,
        _get_projected_2050_value(power_data_filtered),
        x_offset=0.4,
        y_offset=-60,
    )
    _add_capacity_value(ax, 2050, target_capacity, x_offset=0.6, y_offset=-60)
    _format_axis(ax, target_technology, target_country)
    _format_legend(ax)
    plt.show()


if __name__ == "__main__":
    """
    0. SET PARAMETERS
    """
    # Set spatial granularity for which to run the analysis ("national", or "continental")
    spatial_resolution = "Germany"

    # Define policy target for 2030 to analyse
    target_technology = "PV"
    target_capacity = 215  # GW

    """
    1. READ AND PREPARE DATA
    """
    historic_years = ["2020", "2021", "2022"]
    spores_years = ["2030", "2050"]
    path_to_processed_data = os.path.join(os.getcwd(), "..", "data", "processed")
    power_capacity = load_processed_power_capacity(
        path_to_processed_data, (historic_years + spores_years)
    )
    paper_metrics = load_processed_paper_metrics(path_to_processed_data, (spores_years))
    spore_to_scenario_maps = get_spore_to_scenario_maps(
        path_to_processed_data=path_to_processed_data,
        resolution=spatial_resolution,
        years=spores_years,
    )

    """
    2. Visualise target
    """
    plot_technology_target_capacity(
        power_data=power_capacity,
        historic_years=historic_years,
        spores_years=spores_years,
        target_country=spatial_resolution,
        target_capacity=target_capacity,
        target_technology=target_technology,
    )
