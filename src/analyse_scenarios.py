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


if __name__ == "__main__":
    # This script requires a 'spore_to_scenario{region}.json'. This can be generated using cluster_spores_to_scenarios.py (see the README file).

    """
    0. SET PARAMETERS
    """
    # Choose scenario_number to analyse
    focus_scenario = 1
    focus_year = "2050"
    generate_figures_for_all_scenarios = True

    # Set spatial granularity for which to run the analysis. Choose a country or "Europe" to analyse scenarios for the European energy sector as a whole.
    spatial_resolution = "Netherlands"

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
    print(all_scenarios)
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

            plot_scenario_analysis(
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

        plot_scenario_analysis(
            power_capacity=power_capacity,
            spatial_resolution=spatial_resolution,
            scenario_description=scenario_description,
            scenario_number=focus_scenario,
            year=focus_year,
        )


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
    plt.show()
