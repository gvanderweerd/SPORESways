import os
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial import distance_matrix
from scipy.cluster.hierarchy import dendrogram

from src.utils.parameters import *
from src.utils.visualisation import *
from src.utils.data_io import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
# np.set_printoptions(linewidth=150)
# np.set_printoptions(threshold=np.inf, edgeitems=10)

plt.rcParams.update({
    "svg.fonttype": 'none',
    'font.family':'sans-serif',
    'font.sans-serif':'Arial'
})

def plot_3D_fig(df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x = df["CHP"]
    y = df["PV"]
    z = df["Wind"]
    clusters = df["cluster"]
    num_clusters = len(clusters.unique())
    color_map = cm.get_cmap("viridis", num_clusters)

    for cluster, color in zip(clusters.unique(), color_map.colors):
        idx = clusters == cluster
        ax.scatter(x[idx], y[idx], z[idx], c=[color], marker="o", label=f"Cluster {cluster}")

    ax.set_xlabel("CHP capacity [GW]")
    ax.set_ylabel("PV capacity [GW]")
    ax.set_zlabel("Wind capacity [GW]")
    ax.legend()
    ax.set_title("3D Scatter Plot for Clustering SPORES")


def plot_elbow_fig(num_clusters_range, average_distances):
    plt.figure()
    sns.lineplot(x=num_clusters_range, y=average_distances, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Average Distance")
    plt.title("Elbow Method to Determine Optimal Number of Clusters")


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


def cluster_spores_data(spores_data, num_clusters):
    # FIXME: make this compatible with national spores data (I think columns = ["technology", "region"]
    data_df = (
        spores_data.reset_index()
        .pivot(index=["spore"], columns="technology", values="capacity_gw")
        .reset_index()
        .drop(["spore"], axis=1)
    )
    kmeans = KMeans(n_clusters=num_clusters)
    data_df["cluster"] = kmeans.fit_predict(data_df)

    # Reformat data_df
    data_df.index.name = "spore"
    data_df = (
        data_df.reset_index().set_index(["spore", "cluster"]).stack("technology")
    )

    # Get dictionary to map the spore number to a corresponding cluster number
    cluster_mapper = data_df.index.to_frame(index=False).drop_duplicates(subset="spore").set_index("spore")["cluster"].to_dict()

    # Calculate cluster centroids
    centroids = kmeans.cluster_centers_

    return data_df, cluster_mapper, centroids


def find_optimal_clusters(spores_data):

    num_clusters_list = list(range(2, 20))
    average_distances = []
    for num_clusters in num_clusters_list:
        print(f"Number of clusters: {num_clusters}")

        # Cluster SPORES data for a given number of clusters
        spores_data_clustered, cluster_mapper, centroids = cluster_spores_data(spores_data, num_clusters)

        # Calculate Euclidean distances of each SPORE to all centroids (distances has n rows for each spore 1:n and m columns for each cluster 1:m)
        # distances = pairwise_distances(spores_data_clustered.drop("cluster", axis=1), centroids, metric="euclidean")
        distance_to_all_clusters = pairwise_distances(spores_data.unstack(level="technology").values, centroids, metric="euclidean")
        # Calculate the square of the distance to the closest cluster
        distance_to_cluster_squared = distance_to_all_clusters.min(axis=1)**2
        # Put distance to the cluster centroid for each spore in a series with corresponding cluster as index
        spore_cluster_index = pd.MultiIndex.from_tuples(cluster_mapper.items(), names=["spore", "cluster"])
        distance_series = pd.Series(distance_to_cluster_squared, index=spore_cluster_index)
        # Calculate the average distance for this number of clusters (total average distance is calculated by taking the mean of the mean distance for each cluster)
        average_distance = distance_series.groupby("cluster").mean().mean()

        # Add average distance for given number of clusters to the list that represents the y-axis in the elbow plot
        average_distances.append(average_distance)

    # Plot Elbow curve to determine the number of clusters
    plot_elbow_fig(num_clusters_list, average_distances)


def plot_scenarios_as_stacked_barcharts(clustered_spores_data, value, year):
    scenarios = clustered_spores_data.groupby(["cluster", "technology"]).agg(["min", "max", "mean", "median"]).reset_index()
    scenarios_avg = scenarios.pivot_table(index="cluster", columns="technology", values="mean")
    scenarios_median = scenarios.pivot_table(index="cluster", columns="technology", values="median")
    scenarios_min = scenarios.pivot_table(index="cluster", columns="technology", values="min")
    scenarios_max = scenarios.pivot_table(index="cluster", columns="technology", values="max")

    design_space = clustered_spores_data.groupby(["technology"]).agg(["min", "max", "mean"]).T

    # plt.figure(figsize=(8, 4.5))
    scenarios_avg.plot(kind="bar", stacked=True, color=POWER_TECH_COLORS)

    plt.xlabel("Scenario")
    plt.ylabel("Installed capacity [GW]")
    plt.title(f"Stacked barchart of installed power capacity for each scenario in {year}")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout(pad=0)


def normalise_to_max(x):
    max_value = x.max()
    return x / max_value


def plot_scenarios_capacity_distribution(clustered_spores_data, year, region):
    capacity = clustered_spores_data
    scaler = MinMaxScaler()
    all_techs = capacity.index.get_level_values("technology").unique()
    capacity_normalised = pd.DataFrame(
        scaler.fit_transform(capacity.unstack("technology")), columns=all_techs
    )
    capacity_ranges = capacity.groupby("technology").agg(["min", "max"])
    capacity_normalised2 = capacity.groupby(["technology"]).transform(normalise_to_max)
    capacity_normalised2 = capacity_normalised2.unstack("technology")

    # fig, ax = plt.subplots()
    # sns.stripplot(
    #     ax=ax,
    #     data=capacity_normalised2.unstack("technology").reset_index(),
    #     x="technology",
    #     y="value",
    #     marker=open_circle,
    #     hue="cluster"
    # )

    fig, ax = plt.subplots()
    sns.stripplot(
        ax=ax,
        data=capacity_normalised2,
        marker=open_circle,
        palette=POWER_TECH_COLORS
    )

    capacity_normalised.index.name = "spore"
    capacity_normalised = capacity_normalised.stack("technology")

    ax.set_title(
        f"Capacity distribution of {year} SPORES for the power sector in {region}"
    )
    ax.set_xlabel("")
    ax.set_ylabel("Normalised capacity")
    ax.set_xticks(range(len(all_techs)))
    ax.set_xticklabels(all_techs, fontsize=10)
    xticklabels = []
    for ticklabel in ax.get_xticklabels():
        technology = ticklabel.get_text()
        if technology in capacity.index.get_level_values("technology").unique():
            xticklabels.append(
                f"{technology}\n{capacity_ranges.loc[technology, 'min'].round(2)} - {capacity_ranges.loc[technology, 'max'].round(2)} [GW]"
            )
        else:
            xticklabels.append(f"{technology}\n0.0 - 0.0 [GW]")
    ax.set_xticklabels(xticklabels, fontsize=10)


def reorganise_metrics(df):
    return df.iloc[df['metric'].map(metric_plot_order).argsort()]


def plot_metrics_distribution(metrics, year, focus_cluster=None):
    # Calculate metric ranges and get units
    metric_ranges = metrics.groupby(level="metric").agg(["min", "max"])
    metric_units = metrics.index.to_frame(index=False).drop_duplicates(subset="metric").set_index("metric")["unit"].to_dict()

    # Normalise metrics
    metrics_normalised = metrics.groupby(["metric"]).transform(normalise_to_max).reset_index()

    # Calculate min, max for focus cluster (for plotting boxes)
    max_scenario = metrics_normalised.loc[
        (metrics_normalised.cluster == focus_cluster), ["paper_metrics", "metric"]
    ].groupby("metric").max()["paper_metrics"]
    min_scenario = metrics_normalised.loc[
        (metrics_normalised.cluster == focus_cluster), ["paper_metrics", "metric"]
    ].groupby("metric").min()["paper_metrics"]


    # Get colors dictionary for metric
    metric_labels = list(metrics.index.unique(level="metric"))
    cluster_labels = list(metrics.index.unique(level="cluster"))
    metric_colors = get_color_dict2(metric_labels)
    cluster_colors = get_color_dict2(cluster_labels)

    # Get SPORES to plot in color (SPORES that belong to a specific cluster)
    # spores_to_plot_in_color = metrics.xs(focus_cluster, level="cluster").index.unique(level="spore")
    spores_to_plot_in_color = metrics_normalised[
        (metrics_normalised.cluster == focus_cluster)
        ].spore.values

    #
    fig, ax = plt.subplots(1, 1, figsize=(FIGWIDTH, 10 * FIGWIDTH / 20))
    sns.stripplot(
        ax=ax,
        data=metrics_normalised[~metrics_normalised.spore.isin(spores_to_plot_in_color)],
        x="metric",
        y="paper_metrics",
        marker=open_circle,
        color=cluster_colors["rest_color"],
        alpha=.5,
        s=3
    )

    # Format y-axis
    ax.set_ylabel("Metric score (scaled per metric to maximum across all SPORES)")

    # Format x-axis
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=10)
    xticklabels = []
    _x = 0
    for ticklabel in ax.get_xticklabels():
        _metric = ticklabel.get_text()

        # Plot boxes
        xmin = _x - 0.2
        xmax = _x + 0.2
        height = max_scenario.loc[_metric] - min_scenario.loc[_metric]
        height += 0.018
        ax.add_patch(
            mpl.patches.Rectangle(
                xy=(xmin, min_scenario.loc[_metric] - 0.009),
                height=height,
                width=(xmax - xmin),
                fc="None",
                ec=cluster_colors[focus_cluster],
                linestyle="--",
                lw=0.75,

                zorder=10
            ),
        )
        _x += 1

        # Format x-axis labels
        metric_range = metric_ranges.apply(metric_range_formatting[_metric]).loc[_metric]
        _unit = metric_units.get(_metric)
        if _unit == "percentage":
            _unit = " %"
        elif _unit == "fraction":
            _unit = ""
        else:
            _unit = " " + _unit
        xticklabels.append(
            f"{metric_plot_names[_metric]}\n({metric_range['min']} - {metric_range['max']}){_unit}"
        )
    ax.set_xticklabels(xticklabels, fontsize=6)

    # Color focus scenario
    if focus_cluster is not None:
        sns.stripplot(
            data=metrics_normalised[metrics_normalised.spore.isin(spores_to_plot_in_color)],
            x="metric", y="paper_metrics", alpha=.5, ax=ax, marker="o", color=cluster_colors[focus_cluster], s=3
        )

    #FIXME: get unique spores only
    # print(spores_to_plot_in_color)

    # Set figure title
    ax.set_title(f"Scenario {focus_cluster}: {spores_per_cluster[focus_cluster]} SPORES")



    # Plot boxes around scenario range




    # for ticklabel in ax.get_xticklabels():
    #     _metric = ticklabel.get_text()
    #     print(_metric)
    #
    # _x = 0
    # for idx in max_scenario.index:
    #     print(idx)
    #     xmin = _x - 0.2
    #     xmax = _x + 0.2
    #     height = max_scenario.loc[idx] - min_scenario.loc[idx]
    #     height += 0.018
    #     ax.add_patch(
    #         mpl.patches.Rectangle(
    #             xy=(xmin, min_scenario.loc[idx] - 0.009),
    #             height=height,
    #             width=(xmax - xmin),
    #             fc="None",
    #             ec=cluster_colors[focus_cluster],
    #             linestyle="--",
    #             lw=0.75,
    #
    #             zorder=10
    #         ),
    #     )
    #     _x += 1



    # if focus_metric is not None:
    #     sns.stripplot(
    #         data=metric_df[metric_df.spore.isin(spores_to_plot_in_color)],
    #         x="metric", y=y, alpha=focus_metric_alpha, ax=ax, marker="o", color=colours[focus_metric],
    #         s=3
    #     )
    #
    # ax.set_xticklabels(xticklabels, fontsize=6)
    # handles = {}
    # handles["other"] = mpl.lines.Line2D(
    #     [0], [0],
    #     marker=open_circle,
    #     color="w",
    #     markerfacecolor=colours["All other SPORES"],
    #     markeredgecolor=colours["All other SPORES"],
    #     label='All other SPORES',
    #     markersize=4
    # )
    # if incl_15pp_boxes:
    #     handles["best"] = mpl.patches.Rectangle(
    #         xy=(xmin, min_best.loc[idx] - 0.005),
    #         height=height,
    #         width=(xmax - xmin),
    #         fc="None",
    #         ec="black",
    #         linestyle="--",
    #         lw=0.75,
    #         label='SPORE +15pp range',
    #
    #     )  #
    # if focus_metric is not None:
    #     handles["linked_spores"] = mpl.lines.Line2D(
    #         [0], [0],
    #         marker='o',
    #         color="w",
    #         markerfacecolor=colours[focus_metric],
    #         label=f"SPORES linked to {plot_names[focus_metric].lower()} +15pp range",
    #         markersize=5
    #     )
    #
    # ax.legend(handles=handles.values(), frameon=False, loc="lower right", bbox_to_anchor=(0.8, 0))
    # sns.despine(ax=ax)


def get_processed_data(path_to_processed_data):
    years = find_years(path_to_processed_data)

    power_capacity = {}
    paper_metrics = {}

    for year in years:
        power_capacity[year] = pd.read_csv(
            os.path.join(path_to_processed_data, year, "power_capacity.csv"),
            index_col=["region", "technology", "spore"],
            squeeze=True,
        )
        paper_metrics[year] = pd.read_csv(
            os.path.join(path_to_processed_data, year, "paper_metrics.csv"),
            index_col=["spore", "metric", "unit"],
            squeeze=True,
        )
    return power_capacity, paper_metrics


def main():
    for root, dirs, files in os.walk(os.path.join(os.getcwd(), "..", "data", "processed")):
        print(root, dirs, files)


if __name__ == '__main__':
    # MRQ:
    # "How do investment decisions affect the manoeuvring space of a climate neutral European energy system design in 2050?"


    """
    0. SET PARAMETERS
    """
    # Choose region for which we want to run the analysis
    region = "Europe"
    elbow_figures = True
    year = "2050"

    """
    1. PREPARING AND READING DATA
    """
    path_to_processed_data = os.path.join(os.getcwd(), "..", "data", "processed")
    years = find_years(path_to_processed_data=path_to_processed_data)
    power_capacity, paper_metrics = get_processed_data(path_to_processed_data=path_to_processed_data)

    # Transform capacity to continental and national scale
    power_capacity_continental_2050 = power_capacity.get(year).loc["Europe", :, :]
    power_capacity_national_2050 = power_capacity.get(year).drop(index="Europe", level="region")

    print(power_capacity_continental_2050)
    """
    2. CLUSTER SPORES TO SCENARIOS
    """
    # SQ1:
    # "What are the key characteristics and trade-offs of future configurations of the European energy system?"

    # Find optimal number of clusters of SPORES (=scenario's) by using KMeans clustering and looking for the "elbow" in the Elbow curve
    if elbow_figures:
        find_optimal_clusters(spores_data=power_capacity_continental_2050)

    # Pick a number of clusters based on the elbow plot
    n_clusters_2050 = 11

    # Cluster SPORES data for the chosen number of clusters
    power_capacity_2050_clustered, cluster_mapper_2050, centroids = cluster_spores_data(power_capacity_continental_2050, n_clusters_2050)

    print(power_capacity_2050_clustered)
    print(cluster_mapper_2050)

    # FIXME: obtain a dataframe with 441 rows and 204 (=techs*countries) columns such that we can cluster a dataset with more variables
    # print("test")
    # print(power_capacity_national_2050)
    # # pd.set_option("display.max_rows", None)
    # print(power_capacity_national_2050.unstack(["technology", "region"]))

    # Add cluster column to paper metrics
    # FIXME: cluster mapping goes wrong because the spores in 2030 are not numbered but named
    paper_metrics[year] = add_cluster_index_to_series(data=paper_metrics.get(year), cluster_mapper=cluster_mapper_2050)

    print(paper_metrics[year])

    """
    3. ANALYSE AND VISUALISE "MANOEUVRING SPACE" OF 2030 AND 2050
    """

    # Visualise spores clusters in a technology distribution plot with colored boxes for each cluster
    # plot_scenarios_capacity_distribution(clustered_spores_data=power_capacity_2030_clustered, year=2030, region=region)
    # plot_scenarios_capacity_distribution(clustered_spores_data=power_capacity_2050_clustered, year=2050, region=region)

    # Calculate average capacities for each scenario
    scenarios_2050 = power_capacity_2050_clustered.groupby(["cluster", "technology"]).agg(["min", "max", "mean", "median"])

    # Visualise average capacities for each scenario in geographical plot (like in the paper)
    plot_scenarios_as_stacked_barcharts(clustered_spores_data=power_capacity_2050_clustered, value="mean", year=year)

    #FIXME: check if the clustering algorithm finds the same clusters each time!
    cluster_df = pd.DataFrame.from_dict(cluster_mapper_2050, orient="index", columns=["cluster"])
    spores_per_cluster = cluster_df.groupby("cluster").size()
    print(spores_per_cluster)
    print(spores_per_cluster.sum())
    print(spores_per_cluster[5])

    # Plot 2050 clusters
    for cluster in power_capacity_2050_clustered.index.unique(level="cluster"):
        # Visualise trade-offs between technologies using stacked bar charts for each clustered scenario
        plot_metrics_distribution(metrics=paper_metrics.get(year), focus_cluster=cluster, year=year)

    plt.show()

    #FIXME: make some form of a decision tree to describe scenario's?