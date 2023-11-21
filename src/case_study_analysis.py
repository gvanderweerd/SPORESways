import json
import os
import pandas as pd
import matplotlib.patches as mpatches

# Functions from own source code
from src.utils.parameters import *
from src.utils.visualisation import *
from src.utils.data_io import *


def filter_power_capacity(power_data, region, tech):
    # Filter data for country & technology
    power_data_filtered = {}
    for year in power_data.keys():
        power_data_filtered[year] = power_data.get(year).loc[region, tech, :]
    return power_data_filtered


def filter_data_on_countries_of_interest(data, countries_of_interest):
    for year in data.keys():
        data[year] = data.get(year)[
            data.get(year).index.get_level_values("region").isin(countries_of_interest)
        ]
    return data


def get_colors(power_data_filtered, target_capacity_2030, acceleration_capacity_2050):
    colors_2030 = [
        "green" if capacity >= target_capacity_2030 else "red"
        for capacity in power_data_filtered.get("2030")
    ]
    colors_2050 = [
        "mediumorchid"
        if capacity < target_capacity
        else (
            "blue"
            if target_capacity_2030 <= capacity < acceleration_capacity_2050
            else "orange"
        )
        for capacity in power_data_filtered.get("2050")
    ]
    colors = {"2030": colors_2030, "2050": colors_2050}

    return colors


def get_growth_acceleration_threshold_2050(power_data_filtered, target_capacity_2030):
    # Define points
    x1, y1 = 2022, power_data_filtered.get("2022")[0]
    x2, y2 = 2030, target_capacity_2030

    # Linear function parameters
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    return m * 2050 + b

def plot_technology_target_capacity(
    power_data,
    historic_years,
    spores_years,
    target_country,
    target_capacity,
    target_technology,
    acceleration_threshold_2050,
    colors_per_spore,
):
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

    def _add_spores_capacity(
        ax, power_data, spores_years, target_capacity, acceleration_threshold
    ):
        power_cap = power_data.copy()
        # colors = get_colors(power_data, target_capacity, acceleration_threshold)
        for year in spores_years:
            # Count number of SPORES in capacity spores data
            n_spores = len(power_cap.get(year).index.unique(level="spore"))
            # Define small jitter to increase visibility of SPORES
            jitter = 0.3
            # Define x for capacity SPORES
            x_spores = [
                int(year) + np.random.uniform(-jitter, jitter) for _ in range(n_spores)
            ]
            # Define data including corresponding color
            df = pd.DataFrame(
                {
                    "x": x_spores,
                    "y": power_cap.get(year),
                    "color": colors_per_spore.get(year),
                }
            )

            # Plot capacity SPORES
            sns.scatterplot(
                ax=ax,
                data=df,
                x="x",
                y="y",
                marker="o",
                hue="color",
                palette=palette_dict,
                legend=True,
                s=10,
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
            xytext=(2035, y_2040 + 30),
            fontsize=12,
        )

    def _format_axis(ax, target_technology, target_country):
        ax.set_ylabel(f"Installed Capacity [GW]", weight="bold")
        ax.set_title(f"{target_country}, {target_technology}", weight="bold")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    def _format_legend(ax, target_capacity, acceleration_threshold_2050):
        legend = ax.get_legend()
        legend_labels = [
            f"Capacity 2030 > {target_capacity:.1f} GW",
            f"Capacity 2030 < {target_capacity:.1f} GW",
            f"Capacity 2050 < {target_capacity:.1f} GW",
            f"{target_capacity:.1f} GW < Capacity 2050 < {acceleration_threshold_2050:.1f} GW",
            f"Capacity 2050 > {acceleration_threshold_2050:.1f} GW"
        ]
        for label, text in zip(legend.get_texts(), legend_labels):
            label.set_text(text)

    # Prepare data
    # power_data_filtered = _prepare_data_for_plot(power_data, target_technology)
    power_data_filtered = filter_power_capacity(
        power_data, target_country, target_technology
    )

    # Define figure
    fig, ax = plt.subplots(figsize=(1.5 * FIGWIDTH, 0.75 * FIGWIDTH * 9 / 16))

    # Add plot elements
    _add_historic_capacity(ax, power_data_filtered, historic_years)
    _add_spores_capacity(
        ax,
        power_data_filtered,
        spores_years,
        target_capacity,
        acceleration_threshold_2050,
    )
    _add_projection_lines(ax, power_data_filtered, target_capacity)
    _add_capacity_value(
        ax, 2022, power_data_filtered.get("2022")[0], x_offset=0, y_offset=15
    )
    _add_capacity_value(ax, 2030, target_capacity, x_offset=0.4, y_offset=15)
    _add_capacity_value(
        ax,
        2050,
        acceleration_threshold_2050,
        x_offset=0.4,
        y_offset=10,
    )
    _add_capacity_value(ax, 2050, target_capacity, x_offset=0.4, y_offset=10)
    _format_axis(ax, target_technology, target_country)
    _format_legend(ax, target_capacity, acceleration_threshold_2050)


def plot_power_capacity_distribution(power_data, colored_2030_spores, colored_2050_spores, color_2030, color_2050, region="Europe", colored_2050_spores_2=None, color_2050_2=None):
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
        # Find indices that correspond to the data that needs to be colored in a certain way
        colored_2030_data_idx = data[
            (data["year"] == "2030") & (data["spore"].isin(colored_2030_spores))
        ].index
        colored_2050_data_idx = data[
            (data["year"] == "2050") & (data["spore"].isin(colored_2050_spores))
        ].index
        if colored_2050_spores_2 is not None:
            colored_2050_data_idx2 = data[
                (data["year"] == "2050") & (data["spore"].isin(colored_2050_spores_2))
            ].index
            grey_data_idx = data.index.difference(colored_2030_data_idx).difference(colored_2050_data_idx).difference(colored_2050_data_idx2)
        else:
            grey_data_idx = data.index.difference(colored_2030_data_idx).difference(colored_2050_data_idx)


        # Set title
        ax.set_title(
            fig_title,
            weight="bold",
        )

        # Plot grey SPORES in 2030 and 2050
        sns.stripplot(
            ax=ax,
            data=data[
                data.index.isin(grey_data_idx)
            ],
            x="technology",
            y="capacity_gw",
            hue="year",
            hue_order=["2030", "2050"],
            jitter=True,
            dodge=True,
            palette=[sns.color_palette("bright")[-3], sns.color_palette("bright")[-3]],
            marker=open_circle,
            s=3,
            # alpha=0.5,
            legend=False,
        )

        # Plot colored 2030 SPORES
        sns.stripplot(
            ax=ax,
            data=data[data.index.isin(colored_2030_data_idx)],
            x="technology",
            y="capacity_gw",
            hue="year",
            hue_order=["2030", "2050"],
            jitter=True,
            dodge=True,
            palette=[color_2030, color_2030],
            marker=open_circle,
            s=3,
            # alpha=0.5,
            legend=False,
        )

        # Plot colored 2050 SPORES
        sns.stripplot(
            ax=ax,
            data=data[data.index.isin(colored_2050_data_idx)],
            x="technology",
            y="capacity_gw",
            hue="year",
            hue_order=["2030", "2050"],
            jitter=True,
            dodge=True,
            palette=[color_2050, color_2050],
            marker=open_circle,
            s=3,
            # alpha=0.5,
            legend=False,
        )

        # Plot second set of colored 2050 SPORES if they are defined
        if colored_2050_spores_2 is not None:
            sns.stripplot(
                ax=ax,
                data=data[data.index.isin(colored_2050_data_idx2)],
                x="technology",
                y="capacity_gw",
                hue="year",
                hue_order=["2030", "2050"],
                jitter=True,
                dodge=True,
                palette=[color_2050_2, color_2050_2],
                marker=open_circle,
                s=3,
                # alpha=0.5,
                legend=False,
            )

    def add_ranges_to_xticks(ranges, ax):
        xticklabels = []

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

    # ax.legend(loc="upper left")
    plt.tight_layout()


def plot_primary_energy_distribution(
    primary_energy_data, colored_2030_spores, colored_2050_spores, color_2030, color_2050, region="Europe", distribution="per_source", colored_2050_spores_2=None, color_2050_2=None
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

    def plot_distribution(data, distribution, ax):
        # Find indices that correspond to the data that needs to be colored in a certain way
        colored_2030_data_idx = data[
            (data["year"] == "2030") & (data["spore"].isin(colored_2030_spores))
        ].index
        colored_2050_data_idx = data[
            (data["year"] == "2050") & (data["spore"].isin(colored_2050_spores))
        ].index
        if colored_2050_spores_2 is not None:
            colored_2050_data_idx2 = data[
                (data["year"] == "2050") & (data["spore"].isin(colored_2050_spores_2))
            ].index
            grey_data_idx = data.index.difference(colored_2030_data_idx).difference(colored_2050_data_idx).difference(colored_2050_data_idx2)
        else:
            grey_data_idx = data.index.difference(colored_2030_data_idx).difference(colored_2050_data_idx)

        # Set title
        ax.set_title(
            fig_title,
            weight="bold",
        )


        # Plot grey SPORES in 2030 and 2050
        sns.stripplot(
            ax=ax,
            data=data[
                data.index.isin(grey_data_idx)
            ],
            x="carriers",
            y="primary_energy_supply_twh",
            hue="year",
            hue_order=["2030", "2050"],
            jitter=True,
            dodge=True,
            palette=[sns.color_palette("bright")[-3], sns.color_palette("bright")[-3]],
            marker=open_circle,
            s=3,
            # alpha=0.5,
            legend=False,
        )

        # Plot colored 2030 SPORES
        sns.stripplot(
            ax=ax,
            data=data[data.index.isin(colored_2030_data_idx)],
            x="carriers",
            y="primary_energy_supply_twh",
            hue="year",
            hue_order=["2030", "2050"],
            jitter=True,
            dodge=True,
            palette=[color_2030, color_2030],
            marker=open_circle,
            s=3,
            # alpha=0.5,
            legend=False,
        )

        # Plot colored 2050 SPORES
        sns.stripplot(
            ax=ax,
            data=data[data.index.isin(colored_2050_data_idx)],
            x="carriers",
            y="primary_energy_supply_twh",
            hue="year",
            hue_order=["2030", "2050"],
            jitter=True,
            dodge=True,
            palette=[color_2050, color_2050],
            marker=open_circle,
            s=3,
            # alpha=0.5,
            legend=False,
        )

        # Plot second set of colored 2050 SPORES if they are defined
        if colored_2050_spores_2 is not None:
            sns.stripplot(
                ax=ax,
                data=data[data.index.isin(colored_2050_data_idx2)],
                x="carriers",
                y="primary_energy_supply_twh",
                hue="year",
                hue_order=["2030", "2050"],
                jitter=True,
                dodge=True,
                palette=[color_2050_2, color_2050_2],
                marker=open_circle,
                s=3,
                # alpha=0.5,
                legend=False,
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

    if primary_energy_normalised["primary_energy_supply_twh"].min() < 0:
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    plt.tight_layout()


def get_spores_for_target(power_data, target_2030, acceleration_threshold_2050):
    # Define masks to filter spores based on 2030 and 2050 thresholds
    target_2030_met_mask = power_data.get("2030") >= target_2030
    target_2030_failed_mask = power_data.get("2030") < target_2030
    acceleration_2050_mask = power_data.get("2050") > acceleration_threshold_2050
    reduction_2050_mask = power_data.get("2050") < target_2030
    steady_growth_2050_mask = (power_data.get("2050") >= target_2030) & (
        power_data.get("2050") <= acceleration_threshold_2050
    )

    # Find SPORES corresponding to filters based on 2030 and 2050 thresholds
    target_2030_met_spores = target_2030_met_mask[target_2030_met_mask].index
    target_2030_failed_spores = target_2030_failed_mask[target_2030_failed_mask].index
    acceleration_2050_spores = acceleration_2050_mask[acceleration_2050_mask].index
    reduction_2050_spores = reduction_2050_mask[reduction_2050_mask].index
    steady_growth_2050_spores = steady_growth_2050_mask[steady_growth_2050_mask].index
    print(
        f"Number of SPORES per category for {spatial_resolution} {target_technology} {target_capacity} GW case study"
    )
    print("2030 target met")
    print(len(target_2030_met_spores))
    print("2030 target failed")
    print(len(target_2030_failed_spores))
    print("2050 acceleration")
    print(len(acceleration_2050_spores))
    print("2050 steady")
    print(len(steady_growth_2050_spores))
    print("2050 reduction")
    print(len(reduction_2050_spores))

    return (
        target_2030_met_spores,
        target_2030_failed_spores,
        acceleration_2050_spores,
        reduction_2050_spores,
        steady_growth_2050_spores,
    )


def print_target_impact_on_power_capacity_distribution():
    print("Distribution of all 2030 SPORES")
    print(
        power_capacity.get("2030").groupby(["region", "technology"]).agg(["min", "max"]).loc[spatial_resolution]
    )
    print("Distribution of 2030 SPORES when target is achieved")
    print(
        power_capacity.get("2030")
        .loc[:, :, target_2030_met_spores]
        .groupby(["region", "technology"])
        .agg(["min", "max"])
    )
    print("Distribution of 2030 SPORES when target is failed")
    print(
        power_capacity.get("2030")
        .loc[:, :, target_2030_failed_spores]
        .groupby(["region", "technology"])
        .agg(["min", "max"])
    )

    print("Distribution of all 2050 SPORES")
    print(
        power_capacity.get("2050").groupby(["region", "technology"]).agg(["min", "max"]).loc[spatial_resolution]
    )

    print(f"Distribution of 2030 SPORES when {target_technology} capacity between {target_capacity}-{acceleration_threshold_2050} GW")
    print(
        power_capacity.get("2050")
        .loc[:, :, steady_growth_2050_spores]
        .groupby(["region", "technology"])
        .agg(["min", "max"])
    )
    print(f"Distribution of 2050 SPORES when {target_technology} capacity is greater than {acceleration_threshold_2050} GW")
    print(
        power_capacity.get("2050")
        .loc[:, :, reduction_2050_spores]
        .groupby(["region", "technology"])
        .agg(["min", "max"])
    )
    print("Distribution of 2050 SPORES when target in 2030 is also failed in 2050")
    print(
        power_capacity.get("2050")
        .loc[:, :, reduction_2050_spores]
        .groupby(["region", "technology"])
        .agg(["min", "max"])
    )


def print_target_impact_on_tpes_distribution():
    print("Distribution of all 2030 SPORES")
    print(
        primary_energy_supply.get("2030").groupby(["region", "carriers"]).agg(["min", "max"]).loc[spatial_resolution]
    )
    print("Distribution of 2030 SPORES when target is achieved")
    print(
        primary_energy_supply.get("2030")
        .loc[:, :, target_2030_met_spores]
        .groupby(["region", "carriers"])
        .agg(["min", "max"])
    )
    print("Distribution of 2030 SPORES when target is failed")
    print(
        primary_energy_supply.get("2030")
        .loc[:, :, target_2030_failed_spores]
        .groupby(["region", "carriers"])
        .agg(["min", "max"])
    )

    print("Distribution of all 2050 SPORES")
    print(
        primary_energy_supply.get("2050").groupby(["region", "carriers"]).agg(["min", "max"]).loc[spatial_resolution]
    )

    print(f"Distribution of 2030 SPORES when {target_technology} capacity between {target_capacity}-{acceleration_threshold_2050} GW")
    print(
        primary_energy_supply.get("2050")
        .loc[:, :, steady_growth_2050_spores]
        .groupby(["region", "carriers"])
        .agg(["min", "max"])
    )
    print(f"Distribution of 2050 SPORES when {target_technology} capacity is greater than {acceleration_threshold_2050} GW")
    print(
        primary_energy_supply.get("2050")
        .loc[:, :, reduction_2050_spores]
        .groupby(["region", "carriers"])
        .agg(["min", "max"])
    )
    print("Distribution of 2050 SPORES when target in 2030 is also failed in 2050")
    print(
        primary_energy_supply.get("2050")
        .loc[:, :, reduction_2050_spores]
        .groupby(["region", "carriers"])
        .agg(["min", "max"])
    )


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
    primary_energy_supply = load_processed_primary_energy_supply(
        path_to_processed_data, spores_years
    )

    power_data_filtered = filter_power_capacity(
        power_capacity, spatial_resolution, target_technology
    )
    acceleration_threshold_2050 = get_growth_acceleration_threshold_2050(
        power_data_filtered, target_capacity
    )
    colors_per_spore = get_colors(
        power_data_filtered, target_capacity, acceleration_threshold_2050
    )
    palette_dict = {
        "green": "green",
        "red": "red",
        "blue": "blue",
        "orange": "orange",
        "mediumorchid": "mediumorchid",
    }
    (
        target_2030_met_spores,
        target_2030_failed_spores,
        acceleration_2050_spores,
        reduction_2050_spores,
        steady_growth_2050_spores,
    ) = get_spores_for_target(
        power_data_filtered, target_capacity, acceleration_threshold_2050
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
        acceleration_threshold_2050=acceleration_threshold_2050,
        colors_per_spore=colors_per_spore,
    )

    """
    3. Visualise impact on 2030 and 2050 distribution
    """

    # Filter data for regions to analyse
    power_capacity = filter_data_on_countries_of_interest(
        power_capacity, [spatial_resolution]
    )

    primary_energy_supply = filter_data_on_countries_of_interest(
        primary_energy_supply, [spatial_resolution]
    )
    # print_target_impact_on_power_capacity_distribution()
    # print_target_impact_on_tpes_distribution()


    # Plot power capacity distribution in case the target is met
    fig_title = f"{spatial_resolution}, {target_technology} (2030 & 2050) > {target_capacity} GW"
    plot_power_capacity_distribution(
        power_data=power_capacity, region=spatial_resolution, colored_2030_spores=target_2030_met_spores, colored_2050_spores=steady_growth_2050_spores, colored_2050_spores_2=acceleration_2050_spores, color_2030="green", color_2050="blue", color_2050_2="orange"
    )
    # Plot power capacity distribution in case the target is failed
    fig_title = f"{spatial_resolution}, {target_technology} (2030 & 2050) < {target_capacity} GW"
    plot_power_capacity_distribution(
        power_data=power_capacity, region=spatial_resolution, colored_2030_spores=target_2030_failed_spores, colored_2050_spores=reduction_2050_spores, color_2030="red", color_2050="mediumorchid"
    )

    # Plot primary energy distribution in case the target is met
    fig_title = f"{spatial_resolution}, {target_technology} (2030 & 2050)  > {target_capacity} GW"
    plot_primary_energy_distribution(primary_energy_supply, region=spatial_resolution, colored_2030_spores=target_2030_met_spores, colored_2050_spores=steady_growth_2050_spores, colored_2050_spores_2=acceleration_2050_spores, color_2030="green", color_2050="blue", color_2050_2="orange")
    # Plot primary energy distribution in case the target is failed
    fig_title = f"{spatial_resolution}, {target_technology} (2030 & 2050) < {target_capacity} GW"
    plot_primary_energy_distribution(primary_energy_supply, region=spatial_resolution, colored_2030_spores=target_2030_failed_spores, colored_2050_spores=reduction_2050_spores, color_2030="red", color_2050="mediumorchid")


    plt.show()
