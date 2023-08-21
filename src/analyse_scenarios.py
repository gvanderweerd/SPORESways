import json
import os
import pandas as pd

# Packages for clustering
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial import distance_matrix
from scipy.cluster.hierarchy import dendrogram

# Functions from own source code
from src.utils.parameters import *
from src.utils.visualisation import *
from src.utils.data_io import *

# Packages for plotting
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update(
    {"svg.fonttype": "none", "font.family": "sans-serif", "font.sans-serif": "Arial"}
)

# Setting printing options
# pd.set_option("display.max_rows", 500)
# pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 200)
# np.set_printoptions(linewidth=150)
# np.set_printoptions(threshold=np.inf, edgeitems=10)


if __name__ == "__main__":
    """
    0. SET PARAMETERS
    """
    # Choose scenario_number to analyse
    focus_scenario = 1
    # Set spatial granularity for which to run the analysis ("national", or "continental")
    spatial_resolution = "Europe"

    """
    1. READ AND PREPARE DATA
    """
    # Read data
    path_to_processed_data = os.path.join(os.getcwd(), "..", "data", "processed")
    years = ["2030", "2050"]
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
        power_capacity[year] = filter_power_capacity(
            power_capacity[year], spatial_resolution
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
    2. VISUALISE SCENARIOS
    """

    for year in years:
        # Plot mean installed capacity for each scenario
        print(power_capacity.get(year))
        plot_scenarios_mean_capacities(
            power_capacity_clustered=power_capacity.get(year), year=year
        )

        # Plot metrics distribution for focus scenario
        plot_metrics_distribution(
            metrics=paper_metrics.get(year), year=year, focus_cluster=focus_scenario
        )

        # plot_capacity_distribution()

    plt.show()

    # - check how many clusters it finds if we cluster on national level
    # FIXME: plot_scenarios_mean_capacities()
    # FIXME: plot_scenario_analysis()
    # - plot_metrics_distribution()
    # - plot_capacity_distribution()
    # - think of ways to include geographical plots?
