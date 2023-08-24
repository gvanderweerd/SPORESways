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
        resolution=spatial_resolution,
    )

    # Filter data based on spatial resolution
    for year in years:
        power_capacity[year] = filter_power_capacity(power_capacity[year], resolution)

    n_spores_per_cluster = {}
    for year in years:
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


if __name__ == "__main__":
    """
    0. SET PARAMETERS
    """
    # Choose scenario_number to analyse
    focus_scenario_2030 = 1
    focus_scenario_2050 = 4

    # Set spatial granularity for which to run the analysis ("national", or "continental")
    spatial_resolution = "Europe"

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

    scenario_description = {}
    for year in years:
        # Describe scenarios
        scenario_description[year] = (
            power_capacity.get(year)
            .groupby(["cluster", "technology"])
            .agg(["min", "mean", "median", "max"])
            .reset_index()
            #
        )

    """
    2. VISUALISE SCENARIOS
    """
    plot_scenario_analysis_barchart(
        scenario_description=scenario_description,
        n_spores_per_cluster=n_spores_per_cluster,
        resolution=spatial_resolution,
        value_to_plot="mean",
    )

    plot_scenario_analysis(
        power_capacity=power_capacity,
        paper_metrics=paper_metrics,
        spatial_resolution=spatial_resolution,
        scenario_description=scenario_description,
        scenario_number=focus_scenario_2050,
        year="2050",
    )

    plt.show()

    # FIXME: plot_scenario_analysis()
    # - think of ways to include geographical plots?
