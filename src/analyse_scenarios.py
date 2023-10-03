import json
import os
import pandas as pd
import plotly.express as px

# Functions from own source code
from src.utils.parameters import *
from src.utils.visualisation import *
from src.utils.data_io import *

plt.rcParams.update(
    {"svg.fonttype": "none", "font.family": "sans-serif", "font.sans-serif": "Arial"}
)


def load_data_for_scenario_analysis(path_to_processed_data, years, resolution):
    power_capacity = load_processed_power_capacity(path_to_processed_data, years)
    paper_metrics = load_processed_paper_metrics(path_to_processed_data, years)
    grid_capacity = load_processed_grid_transfer_capacity(path_to_processed_data, years)

    spore_to_scenario_maps = get_spore_to_scenario_maps(
        path_to_processed_data=path_to_processed_data,
        years=years,
        resolution=resolution,
    )

    n_spores_per_cluster = {}
    for year in years:
        # Filter data based on spatial resolution
        power_capacity[year] = filter_power_capacity(power_capacity[year], resolution)

        # Add cluster index to data
        power_capacity[year] = add_cluster_index_to_series(
            data=power_capacity.get(year),
            cluster_mapper=spore_to_scenario_maps.get(year),
        )
        paper_metrics[year] = add_cluster_index_to_series(
            data=paper_metrics.get(year),
            cluster_mapper=spore_to_scenario_maps.get(year),
        )
        grid_capacity[year] = add_cluster_index_to_series(
            data=grid_capacity.get(year),
            cluster_mapper=spore_to_scenario_maps.get(year),
        )

        # Calculate amount of spore in each cluster
        n_spores_per_cluster[year] = count_spores_per_cluster(power_capacity.get(year))

    return power_capacity, grid_capacity, paper_metrics, n_spores_per_cluster


def check_scenario_impact_on_europe_scenarios(
    focus_scenario,
    year,
    years,
    resolution,
):
    spore_to_scenario_maps = get_spore_to_scenario_maps(
        path_to_processed_data=path_to_processed_data,
        years=years,
        resolution=resolution,
    )
    spore_to_scenario_maps_Europe = get_spore_to_scenario_maps(
        path_to_processed_data=path_to_processed_data,
        years=years,
        resolution="Europe",
    )

    spores_in_national_scenario = [
        spore
        for spore, scenario in spore_to_scenario_maps.get(year).items()
        if scenario == focus_scenario
    ]
    feasible_european_scenarios = list(
        set(
            {
                spore: spore_to_scenario_maps_Europe.get(year)[spore]
                for spore in spores_in_national_scenario
            }.values()
        )
    )
    all_european_scenarios = list(
        set([scenario for scenario in spore_to_scenario_maps_Europe.get(year).values()])
    )
    infeasible_european_scenarios = [
        scenario
        for scenario in all_european_scenarios
        if scenario not in feasible_european_scenarios
    ]

    return feasible_european_scenarios, infeasible_european_scenarios


def test_gridspec():
    with sns.plotting_context("paper", font_scale=1.5):
        nspores = 2
        nrows = nspores * 2 + 2
        ncols = 5
        ax = {}
        fig = plt.figure(figsize=(20, 2 * 7 + 4))
        gs = mpl.gridspec.GridSpec(
            nrows=nrows,
            ncols=ncols,
            figure=fig,
            hspace=0.1,
            wspace=0.1,
            width_ratios=[1, 1, 23, 25, 25],
            height_ratios=[2, 18] * 2 + [2, 2],
        )
        for row in range(nrows):
            for col in range(ncols):
                ax = fig.add_subplot(gs[row, col])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(
                    0.5, 0.5, f"R{row}C{col}", ha="center", va="center", fontsize=10
                )

        fig.tight_layout()


def test_gridspec2(
    power_capacity,
    grid_capacity,
    paper_metrics,
    spatial_resolution,
    scenario_description,
    scenario_description_eu,
    infeasible_european_scenarios,
    n_spores_per_cluster_eu,
    scenario_number,
    year,
):
    n_spores_per_cluster = count_spores_per_cluster(paper_metrics.get(year))
    n_spores = n_spores_per_cluster.get(scenario_number)
    n_spores_total = len(power_capacity.get(year).index.unique(level="spore"))
    # FIXME: this data transformation is now done on multiple places --> do this in a more efficient way on a logical place
    # See: plot_scenario_analysis_barchart
    # See: plot_capacity_bar
    scenario_values_eu = (
        scenario_description_eu.get(year)
        .loc[:, ["cluster", "technology", "mean"]]
        .pivot_table(index="cluster", columns="technology")["mean"]
    )

    with sns.plotting_context("paper", font_scale=1.5):
        nrows = 3
        ncols = 5
        ax = {}
        fig = plt.figure(figsize=(FIGWIDTH, FIGWIDTH * 9 / 16))
        gs = mpl.gridspec.GridSpec(
            nrows=nrows,
            ncols=ncols,
            figure=fig,
            hspace=0.3,
            wspace=0.1,
            width_ratios=[2, 2, 23, 25, 25],
            height_ratios=[2, 18, 18],
        )
        frame = True
        # Plot figure title
        ax["title"] = plt.subplot(gs[0, :], frameon=frame)
        plot_title(
            ax["title"],
            title=f"{spatial_resolution} scenario {scenario_number}, {year} ({n_spores} / {n_spores_total} SPORES)",
        )

        # Plot capacity bar for focus scenario
        ax["A"] = plt.subplot(gs[1, 0], frameon=frame)
        plot_subfigure_letter(ax["A"], "A")
        ax["capacity_bar"] = plt.subplot(gs[1, 1], frameon=frame)
        plot_capacity_bar(
            ax=ax["capacity_bar"],
            scenario_description=scenario_description,
            year=year,
            focus_scenario=scenario_number,
            value_to_plot="mean",
        )

        # Plot capacity distribution for focus scenario
        # ax["B"] = plt.subplot(gs[1, 2], frameon=frame)
        # plot_subfigure_letter(ax["B"], "B")
        # Plot capacity distribution for focus scenario
        ax["capacity_distribution"] = plt.subplot(gs[1, 2], frameon=True)
        plot_capacity_distribution(
            ax=ax["capacity_distribution"],
            capacity=power_capacity.get(year),
            year=year,
            resolution=spatial_resolution,
            focus_cluster=scenario_number,
        )

        ax["C"] = plt.subplot(gs[1, 4], frameon=frame)
        plot_subfigure_letter(ax["C"], "C")

        ax["D"] = plt.subplot(gs[2, 0], frameon=frame)
        plot_subfigure_letter(ax["D"], "D")

        ax["E"] = plt.subplot(gs[2, 4], frameon=frame)
        plot_subfigure_letter(ax["E"], "E")

        fig.tight_layout()


def plot_scenario_heatmap(power_capacity, spatial_resolution):
    # Normalise capacity
    capacity_normalised = power_capacity.groupby(
        level=["region", "technology"]
    ).transform(normalise_to_max)
    # Calculate median capacity for each technology in each scenario
    median_capacity = (
        power_capacity.groupby(level=["cluster", "technology"]).median().unstack()
    )
    median_capacity_normalised = (
        capacity_normalised.groupby(level=["cluster", "technology"]).median().unstack()
    )

    plt.figure(figsize=(FIGWIDTH, FIGWIDTH))
    sns.heatmap(
        data=median_capacity_normalised,
        annot=median_capacity,
        cmap="RdBu_r",
        fmt=".2f",
        vmin=0,
        vmax=1,
        linewidth=0.5,
        cbar_kws={
            "label": "Median technology capacity (scaled per technology to maximum accross all SPORES)"
        },
    )
    plt.ylabel(f"Scenarios in {spatial_resolution}")
    plt.xlabel("Technologies")


def plot_scenario_analysis_v3(
    power_capacity,
    spatial_resolution,
    scenario_description,
    scenario_number,
    year,
    value_to_plot="median",
):
    n_rows = 2
    n_cols = 3
    n_spores_per_cluster = count_spores_per_cluster(paper_metrics.get(year))
    n_spores = n_spores_per_cluster.get(scenario_number)
    n_spores_total = len(power_capacity.get(year).index.unique(level="spore"))

    with sns.plotting_context("paper", font_scale=1.5):
        ax = {}
        fig = plt.figure(figsize=(2 * FIGWIDTH, FIGWIDTH * 9 / 16))
        gs = gridspec.GridSpec(
            n_rows,
            n_cols,
            height_ratios=[2, 18],
            width_ratios=[1, 24, 4],
            hspace=0.2,
            wspace=0.3,
        )
        _row = 0
        alpha_idx = 0

        # Plot figure title
        ax["title"] = plt.subplot(gs[0, :], frameon=False)
        plot_title(
            ax["title"],
            title=f"{spatial_resolution} scenario {scenario_number}, {year} ({n_spores} / {n_spores_total} SPORES)",
        )

        # Plot capacity bar for focus scenario
        ax["capacity_bar"] = plt.subplot(gs[1, 0], frameon=False)
        plot_capacity_bar(
            ax=ax["capacity_bar"],
            scenario_description=scenario_description,
            year=year,
            focus_scenario=scenario_number,
            value_to_plot=value_to_plot,
        )
        ax["capacity_bar"].annotate(
            "A",
            fontweight="bold",
            xy=(0, 1.1),
            xycoords="axes fraction",
            horizontalalignment="left",
            fontsize="small",
        )

        # Plot capacity distribution for focus scenario
        ax["capacity_distribution"] = plt.subplot(gs[1, 1], frameon=True)
        plot_capacity_distribution(
            ax=ax["capacity_distribution"],
            capacity=power_capacity.get(year),
            year=year,
            resolution=spatial_resolution,
            focus_cluster=scenario_number,
        )
        ax["capacity_distribution"].annotate(
            "B",
            fontweight="bold",
            xy=(0, 1.1),
            xycoords="axes fraction",
            horizontalalignment="left",
            fontsize="small",
        )

        # Add legend
        handles, labels = ax["capacity_bar"].get_legend_handles_labels()

        ax["european_scenarios_legend"] = plt.subplot(gs[1, 2], frameon=False)
        plot_scenario_barchart_legend(
            ax=ax["european_scenarios_legend"],
            handles=handles,
            labels=labels,
            year=year,
            n_spores=n_spores_total,
        )

        plt.tight_layout(pad=1)
        plt.savefig(
            f"../figures/appendices/scenario_analysis/v3/{spatial_resolution}_{year}_sc{scenario_number}.png",
            bbox_inches="tight",
            dpi=120,
        )


def generate_scenario_names(power_data, summary_stats):
    # Define thresholds for 'high' and 'low' technology deployments
    high_thresholds = summary_stats["75%"]
    low_thresholds = summary_stats["25%"]

    # Evaluate deployment of each technoology in the cluster
    high_deployment = power_data[power_data > high_thresholds].index.to_list()
    low_deployment = power_data[power_data < low_thresholds].index.to_list()

    # Generate a name for the cluster based on the evaluations
    name_parts = []
    if high_deployment:
        name_parts.append("High " + " \& ".join(high_deployment))
    if low_deployment:
        name_parts.append("Low " + " \& ".join(low_deployment))

    if not name_parts:
        return "No extreme deployments"

    return ", ".join(name_parts)


def plot_treemap(power_data, scenario_names):
    tree_data = power_data.groupby(["cluster", "technology"]).mean().reset_index()
    tree_data["cluster_name"] = tree_data["cluster"].apply(
        lambda x: scenario_names[scenario_names["cluster"] == x]["name"].values[0]
    )

    # Plot figure
    fig = px.treemap(
        tree_data,
        path=["cluster_name", "technology"],
        values="capacity_gw",
        title="Treemap of Technology Deployments Accross Scenarios",
        color="capacity_gw",
        color_continuous_scale="RdBu",
        labels={"capacity_gw": "Capacity [GW]"},
    )
    fig.show()


if __name__ == "__main__":
    """
    0. SET PARAMETERS
    """
    # Choose scenario_number to analyse
    focus_scenario = 1
    focus_year = "2050"
    generate_figures_for_all_scenarios = False

    # Set spatial granularity for which to run the analysis ("national", or "continental")
    spatial_resolution = "Germany"

    """
    1. READ AND PREPARE DATA
    """
    # Read data
    path_to_processed_data = os.path.join(os.getcwd(), "..", "data", "processed")
    years = ["2030", "2050"]
    (
        power_capacity,
        grid_capacity,
        paper_metrics,
        n_spores_per_cluster,
    ) = load_data_for_scenario_analysis(
        path_to_processed_data, years, spatial_resolution
    )
    (
        power_capacity_eu,
        grid_capacity_eu,
        paper_metrics_eu,
        n_spores_per_cluster_eu,
    ) = load_data_for_scenario_analysis(path_to_processed_data, years, "Europe")

    # Calculate how scenarios are distributed (min, max, mean, median)
    scenario_description = {}
    scenario_description_eu = {}
    for year in years:
        scenario_description[year] = (
            power_capacity.get(year)
            .groupby(["cluster", "technology"])
            .agg(["min", "mean", "median", "max"])
            .reset_index()
        )
        scenario_description_eu[year] = (
            power_capacity_eu.get(year)
            .groupby(["cluster", "technology"])
            .agg(["min", "mean", "median", "max"])
            .reset_index()
        )

    all_scenarios = power_capacity.get(focus_year).index.unique(level="cluster")

    max_link_capacity = max(
        grid_capacity.get("2030").max(), grid_capacity.get("2050").max()
    )

    """
    2. VISUALISE SCENARIOS
    """
    # Visualise all scenarios for chosen country
    plot_scenario_analysis_barchart(
        scenario_description=scenario_description,
        n_spores_per_cluster=n_spores_per_cluster,
        resolution=spatial_resolution,
        value_to_plot="median",
    )
    # # Visualise all scenarios for Europe
    # plot_scenario_analysis_barchart(
    #     scenario_description=scenario_description_eu,
    #     n_spores_per_cluster=n_spores_per_cluster_eu,
    #     resolution="Europe",
    #     value_to_plot="median",
    # )

    print(power_capacity.get("2030"))

    power_capacity.get("2030").to_csv("ger_scenarios_2030.csv")

    # Visualise plot to analyse 1 chosen scenario in 1 chosen year
    if generate_figures_for_all_scenarios:
        for scenario in all_scenarios:
            (
                feasible_european_scenarios,
                infeasible_european_scenarios,
            ) = check_scenario_impact_on_europe_scenarios(
                focus_scenario=scenario,
                year=focus_year,
                years=years,
                resolution=spatial_resolution,
            )

            plot_scenario_analysis_v3(
                power_capacity=power_capacity,
                spatial_resolution=spatial_resolution,
                scenario_description=scenario_description,
                scenario_number=scenario,
                year=focus_year,
            )
    else:
        (
            feasible_european_scenarios,
            infeasible_european_scenarios,
        ) = check_scenario_impact_on_europe_scenarios(
            focus_scenario=focus_scenario,
            year=focus_year,
            years=years,
            resolution=spatial_resolution,
        )

        plot_scenario_analysis_v3(
            power_capacity=power_capacity,
            spatial_resolution=spatial_resolution,
            scenario_description=scenario_description,
            scenario_number=focus_scenario,
            year=focus_year,
        )

    # plot_scenario_heatmap(
    #     power_capacity=power_capacity_eu.get("2030"),
    #     spatial_resolution=spatial_resolution,
    # )
    # plot_scenario_heatmap(
    #     power_capacity=power_capacity_eu.get("2050"),
    #     spatial_resolution=spatial_resolution,
    # )
    # plt.show()

    # fig, ax = plt.subplots()
    # plot_grid_capacity_map(ax, grid_capacity.get(focus_year), focus_scenario)

    """
    3. Name scenarios based on low/high deployment of technologies
    """
    print(f"Scenario names for scenarios in {spatial_resolution}")
    for year in ["2030", "2050"]:
        print(f"Year: {year}")
        scenario_median_values = (
            power_capacity.get(year)
            .groupby(["cluster", "technology"])
            .median()
            .unstack()
            .reset_index()
        )
        summary_stats = power_capacity.get(year).groupby("technology").describe()
        scenario_median_values["name"] = scenario_median_values.drop(
            "cluster", axis=1
        ).apply(generate_scenario_names, axis=1, args=(summary_stats,))
        scenario_names = scenario_median_values[["cluster", "name"]]
        pd.options.display.max_colwidth = 200
        print(scenario_names)
