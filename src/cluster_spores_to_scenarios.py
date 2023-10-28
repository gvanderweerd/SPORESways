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


def get_spore_to_cluster_map(clustered_data):
    return (
        clustered_data.index.to_frame(index=False)
        .drop_duplicates(subset="spore")
        .set_index("spore")["cluster"]
        .to_dict()
    )


def cluster_spores(data_series, n_clusters):
    # Prepare data for clustering
    data_series_prepared = prepare_data_for_clustering(data_series)
    # Set K-Means algorithm parameters
    kmeans = KMeans(init="random", n_clusters=n_clusters, n_init=10, random_state=42)
    # Cluster scaled data
    data_series_prepared["cluster"] = kmeans.fit_predict(data_series_prepared)
    data_series_clustered = (
        data_series_prepared.reset_index()
        .set_index(["spore", "cluster"])
        .stack(["region", "technology"])
    )
    # Name dataseries
    data_series_clustered.name = data_series.name

    return data_series_clustered


def find_n_clusters(
    data_series, min_clusters=2, max_clusters=10, plot=True
):
    # Prepare data for clustering
    data_series_prepared = prepare_data_for_clustering(data_series)

    # Within-Cluster Sum of Squared distances (Elbow Method)
    wcss = []
    # Sihouette Scores (Silhouette Method)
    silhouette_scores = []
    for n_clusters in range(min_clusters, max_clusters + 1):
        # Set K-Means algorithm parameters
        kmeans = KMeans(init="random", n_clusters=n_clusters, n_init=10, random_state=1)
        cluster_labels = kmeans.fit_predict(data_series_prepared)
        wcss.append(kmeans.inertia_)

        # Silhouette score method requires at least 2 clusters
        if n_clusters > 1:
            silhouette_avg = silhouette_score(data_series_prepared, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(None)

    # Find optimal number of clusters for both methods
    silhouette_optimal_clusters = np.argmax(silhouette_scores) + min_clusters

    print(
        f"Optimal number of clusters found by Silhouette method: {silhouette_optimal_clusters} (Best average silhouette score = {max(silhouette_scores)})"
    )

    # Plot figures for selecting optimal number of clusters
    if plot:
        # Elbow method
        plot_elbow_figure(wcss, min_clusters, max_clusters, spatial_resolution, year)
        # Silhouette method
        plot_silhouette_score(
            silhouette_scores, min_clusters, max_clusters, spatial_resolution, year
        )

    return silhouette_optimal_clusters


def prepare_data_for_clustering(data_series):
    # Reformat data such that the rows are the amount of SPORES and the columns represent the features to cluster on (feature is a unique combination of technology capacity and location)
    data_series_reformatted = data_series.unstack(["region", "technology"])
    # Standardise data to ensure that clustering algorithm considers every feature equally important
    scaler = StandardScaler()
    data_series_scaled = scaler.fit_transform(data_series_reformatted)

    return pd.DataFrame(
        data_series_scaled,
        index=data_series_reformatted.index,
        columns=data_series_reformatted.columns,
    )


def save_cluster_map(cluster_map, path_to_directory, spatial_granularity):
    file_path = os.path.join(
        path_to_directory, f"spore_to_scenario_{spatial_granularity}.json"
    )
    with open(file_path, "w") as file:
        json.dump(cluster_map, file)


if __name__ == "__main__":
    """
    0. SET PARAMETERS
    """
    # Choose scenario_number to analyse
    focus_scenario = 0
    # Set spatial granularity for which to run the analysis ("national", or "continental")
    spatial_resolution = "Europe"
    # Set to True if you want to manually force the clustering algorithm to find a number of clusters
    manually_set_n_clusters = False
    if manually_set_n_clusters:
        n_clusters_manual = {"2030": 6, "2050": 5}

    """
    1. READ AND PREPARE DATA
    """
    # Read data
    path_to_processed_data = os.path.join(os.getcwd(), "..", "data", "processed")
    years = ["2030", "2050"]
    power_capacity = load_processed_power_capacity(path_to_processed_data, years)
    paper_metrics = load_processed_paper_metrics(path_to_processed_data, years)

    # Filter data based on spatial resolution
    for year in years:
        power_capacity[year] = filter_power_capacity(
            power_capacity[year], spatial_resolution
        )

    """
    2. CLUSTER SPORES TO SCENARIOS
    """
    for year in years:
        print(
            f"\n For clustering {year} SPORES results based on installed capacity of power generation technologies in {spatial_resolution} we find: \n"
        )
        # Find optimal number of clusters based on Silhouette score method (also plot WCSS to see results for elbow method --> elbow method requires manual picking of clusters based on the WCSS plot. Number of clusters can be set manually in lines 135-137)
        n_clusters = find_n_clusters(
            data_series=power_capacity.get(year),
            min_clusters=5, # set to 5 because lower than five clusters gives clusters with very high WCSS, meaning you get clusters with SPORES very far from the centroid cluster (WCSS usually drops of very quickly)
            max_clusters=15,
            plot=True,
        )
        plt.show()

        if manually_set_n_clusters:
            # Set number of clusters to a manually chosen integer value
            n_clusters = n_clusters_manual.get(year)
        # else:
        #     # Prompt user to input the number of clusters based on the graph
        #     n_clusters = int(
        #         input(f"\n Enter the optimal number of clusters for the year {year}:  ")
        #     )

        # Cluster SPORES using K-Means clustering
        power_capacity[year] = cluster_spores(power_capacity.get(year), n_clusters)

        # Add cluster as an index to paper_metrics data
        spore_to_cluster_map = get_spore_to_cluster_map(power_capacity.get(year))
        paper_metrics[year] = add_cluster_index_to_series(
            data=paper_metrics.get(year), cluster_mapper=spore_to_cluster_map
        )

        # Save spore_to_cluster_map to processed to processed data
        save_cluster_map(
            spore_to_cluster_map,
            os.path.join(path_to_processed_data, year),
            spatial_resolution,
        )
