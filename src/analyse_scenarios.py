import json
import os
import pandas as pd

# Functions from own source code
from src.utils.parameters import *
from src.utils.visualisation import *
from src.utils.data_io import *

plt.rcParams.update(
    {"svg.fonttype": "none", "font.family": "sans-serif", "font.sans-serif": "Arial"}
)


def load_data_for_scenario_analysis(path_to_processed_data, years, resolution):
    power_capacity, paper_metrics = get_processed_data(
        path_to_processed_data=path_to_processed_data, years=years
    )
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

        # Calculate amount of spore in each cluster
        n_spores_per_cluster[year] = count_spores_per_cluster(power_capacity.get(year))

    return power_capacity, paper_metrics, n_spores_per_cluster


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


if __name__ == "__main__":
    """
    0. SET PARAMETERS
    """
    # Choose scenario_number to analyse
    focus_scenario = 4
    focus_year = "2030"

    # Set spatial granularity for which to run the analysis ("national", or "continental")
    spatial_resolution = "Netherlands"

    """
    1. READ AND PREPARE DATA
    """
    # Read data
    path_to_processed_data = os.path.join(os.getcwd(), "..", "data", "processed")
    years = ["2030", "2050"]
    (
        power_capacity,
        paper_metrics,
        n_spores_per_cluster,
    ) = load_data_for_scenario_analysis(
        path_to_processed_data, years, spatial_resolution
    )
    (
        power_capacity_eu,
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

    (
        feasible_european_scenarios,
        infeasible_european_scenarios,
    ) = check_scenario_impact_on_europe_scenarios(
        focus_scenario=focus_scenario,
        year=focus_year,
        years=years,
        resolution=spatial_resolution,
    )

    """
    2. VISUALISE SCENARIOS
    """
    # Visualise all scenarios for chosen country
    plot_scenario_analysis_barchart(
        scenario_description=scenario_description,
        n_spores_per_cluster=n_spores_per_cluster,
        resolution=spatial_resolution,
        value_to_plot="mean",
    )
    # Visualise all scenarios for Europe
    plot_scenario_analysis_barchart(
        scenario_description=scenario_description_eu,
        n_spores_per_cluster=n_spores_per_cluster_eu,
        resolution="Europe",
        value_to_plot="mean",
    )

    # Visualise plot to analyse 1 chosen scenario in 1 chosen year
    plot_scenario_analysis_new(
        power_capacity=power_capacity,
        paper_metrics=paper_metrics,
        spatial_resolution=spatial_resolution,
        scenario_description=scenario_description,
        scenario_description_eu=scenario_description_eu,
        infeasible_european_scenarios=infeasible_european_scenarios,
        n_spores_per_cluster_eu=n_spores_per_cluster_eu,
        scenario_number=focus_scenario,
        year=focus_year,
    )

    plt.show()

    # FIXME: Get power capacity with european values and european clusters (=scenarios)
    # FIXME: Make scenario description
    # FIXME: calculate mean values for european scenarios
    # scenario_values = (
    #     scenario_description.get("2050")
    #     .loc[:, ["cluster", "technology", "mean"]]
    #     .pivot_table(index="cluster", columns="technology")["mean"]
    # )

    # FIXME: plot barchart for european scenarios with the ones that are not feasble greyed out
    # plot_scenario_capacity_stacked_barchart(
    #     scenario_values,
    #     n_spores_per_cluster,
    #     year="2050",
    #     ax,
    #     spores_amount_y_offset=20,
    #     greyed_out_scenarios=infeasible_european_scenarios,
    # )

    # FIXME: plot_scenario_analysis()
    # - think of ways to include geographical plots?
