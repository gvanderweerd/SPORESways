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


def add_cluster_index_to_series(data, cluster_mapper):
    # Get name of the data
    data_name = data.name

    # Add cluster using cluster mapper
    data = data.reset_index()
    data["cluster"] = data["spore"].map(cluster_mapper)

    # Get index names
    index_names = list(data.columns)
    index_names.remove(data_name)

    # Return clustered data as a multi index series
    return data.set_index(index_names)[data_name]


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
    data_series, min_clusters=2, max_clusters=15, method="silhouette", plot=True
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
    # FIXME: the elbow method does not find the elbow point correctly --> default to silhouette method
    second_derivative_wcss = np.diff(np.diff(wcss))
    elbow_optimal_clusters = np.argmax(second_derivative_wcss) + min_clusters
    silhouette_optimal_clusters = np.argmax(silhouette_scores) + min_clusters

    print(
        f"Optimal number of clusters found by Silhouette method: {silhouette_optimal_clusters}"
    )
    print(f"Optimal number of clusters found by Elbow method: {elbow_optimal_clusters}")

    # Plot figures for selecting optimal number of clusters
    if plot:
        # Using Elbow method
        plt.figure()
        plt.plot(range(min_clusters, max_clusters + 1), wcss, marker="o")
        plt.xlabel("Number of Clusters")
        plt.ylabel("WCSS")
        plt.title(f"Elbow Method for {year}")

        # Using Silhouette method
        plt.figure()
        plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker="o")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.title(f"Silhouette Score Method for {year}")

    if method == "silhouette":
        return silhouette_optimal_clusters
    elif method == "elbow":
        return elbow_optimal_clusters


def get_processed_data(path_to_processed_data):
    years = find_years(path_to_processed_data)

    power_capacity = {}
    paper_metrics = {}

    for year in years:
        power_capacity[year] = pd.read_csv(
            os.path.join(path_to_processed_data, year, "power_capacity.csv"),
            index_col=["region", "technology", "spore"],
            # squeeze=True,
        ).squeeze()
        # power_capacity[year].name = os.path.basename(file_path)

        paper_metrics[year] = pd.read_csv(
            os.path.join(path_to_processed_data, year, "paper_metrics.csv"),
            index_col=["spore", "metric", "unit"],
            # squeeze=True,
        ).squeeze()

    return power_capacity, paper_metrics


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


def save_cluster_map(cluster_map, path_to_directory):
    file_path = os.path.join(path_to_directory, "spore_to_scenario.json")
    with open(file_path, "w") as file:
        json.dump(cluster_map, file)


if __name__ == "__main__":
    """
    0. SET PARAMETERS
    """
    # Choose scenario_number to analyse
    focus_scenario = 0
    # Set spatial granularity for which to run the analysis ("national", or "continental")
    spatial_granularity = "national"
    # Set to True if you want to manually force the clustering algorithm to find a number of clusters
    manually_set_n_clusters = True
    n_clusters_manual = {"2030": 14, "2050": 14}

    """
    1. READ AND PREPARE DATA
    """
    # Read data
    path_to_processed_data = os.path.join(os.getcwd(), "..", "data", "processed")
    years = ["2030", "2050"]
    # years = find_years(path_to_processed_data=path_to_processed_data)
    power_capacity, paper_metrics = get_processed_data(
        path_to_processed_data=path_to_processed_data
    )

    # Split power capacity into continental granularity and national granularity
    power_capacity_continental = {}
    power_capacity_national = {}
    for year in years:
        power_capacity_continental[year] = power_capacity.get(year).xs(
            "Europe", level="region", drop_level=False
        )
        power_capacity_national[year] = power_capacity.get(year).drop(
            index="Europe", level="region"
        )

    # Choose for which granularity to run the analysis
    if spatial_granularity == "national":
        _power_capacity = power_capacity_national
    elif spatial_granularity == "continental":
        _power_capacity = power_capacity_continental
    else:
        # Write a line of code that crashes the program and raises and error
        raise ValueError(
            "Invalid spatial granularity. Please choose 'national' or 'continental'."
        )

    """
    2. CLUSTER SPORES TO SCENARIOS
    """
    for year in years:
        if manually_set_n_clusters:
            # Set number of clusters to a manually chosen integer value
            n_clusters = n_clusters_manual.get(year)
        else:
            # Find optimal number of clusters based on Elbow and Silhouette score methods
            n_clusters = find_n_clusters(
                data_series=_power_capacity.get(year), min_clusters=2, max_clusters=15
            )

        # Cluster SPORES using K-Means clustering
        _power_capacity[year] = cluster_spores(_power_capacity.get(year), n_clusters)

        # Add cluster as an index to paper_metrics data
        spore_to_cluster_map = get_spore_to_cluster_map(_power_capacity.get(year))
        paper_metrics[year] = add_cluster_index_to_series(
            data=paper_metrics.get(year), cluster_mapper=spore_to_cluster_map
        )

        # Save spore_to_cluster_map to processed to processed data
        save_cluster_map(
            spore_to_cluster_map, os.path.join(path_to_processed_data, year)
        )
