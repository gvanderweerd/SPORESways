import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import seaborn.objects as so
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial import distance_matrix
from scipy.cluster.hierarchy import dendrogram

from main import *
from global_parameters import *
from plotting_functions import plot_capacity_distribution
from reading_functions import read_spores_data, get_power_capacity


def plot_normalised_cluster_map(data, year, region):
    df = data.loc[year, region, :, :]
    scaler = MinMaxScaler()
    df_normalised = pd.DataFrame(scaler.fit_transform(df.unstack("technology")), columns=df.index.get_level_values("technology").unique())

    print("here")
    print(df_normalised)
    cluster_map = sns.clustermap(
        data=df_normalised.T,
        method="ward",
        metric="euclidean",
        row_cluster=False,
    )
    # Add figure title
    cluster_map.fig.suptitle(f"Normalised capacity clustermap of SPORES results for the power sector in {region}")

    #FIXME: add horizontal boxplots of all technologies in df (not normalised) right next to the clustermap figure to show the installed capacity distribution
    cluster_map.gs.update(left=0.05, right=0.45)
    box_plot = mpl.gridspec.GridSpec(1, 1, left=0.6)
    ax2 = cluster_map.fig.add_subplot(box_plot[0])
    sns.boxplot(ax=ax2, data=df.unstack("technology"), orient="h")

def plot_cluster_capacity_distribution(data, year, region):
    df = data.loc[year, region, :, :]
    # Calculate capacity range for all technologies
    capacity_ranges = df.groupby("technology").agg(["min", "max"])

    # Normalising data; minimum value = 0, maximum value = 1
    scaler = MinMaxScaler()
    df_normalised = pd.DataFrame(scaler.fit_transform(df.unstack("technology")), columns=df.index.get_level_values("technology").unique())

    # Make 6 clusters off spores based on the normalised capacity capacity 'profile' for each spore
    n_clusters = 5
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(df_normalised)

    # Add cluster number
    new_index = pd.MultiIndex.from_tuples([(spore, clustering.labels_[spore], tech) for tech, spore in df.index], names=["spore", "spore_cluster", "technology"])
    df = pd.Series(df.array, index=new_index)
    new_index2 = pd.MultiIndex.from_tuples([(spore, clustering.labels_[spore], tech) for spore, tech in df_normalised.stack("technology").index], names=["spore", "spore_cluster", "technology"])
    df_normalised2 = pd.Series(df_normalised.stack("technology").array, index=new_index2)

    # Calculate capacity range for all technology in each cluster
    cluster_ranges = (
        df
        .groupby(["spore_cluster", "technology"])
        .agg(["min", "max"])
    )
    cluster_ranges_normalised = (
        df_normalised2
        .groupby(["spore_cluster", "technology"])
        .agg(["min", "max"])
    )


    fig, axs = plt.subplots(nrows=3, ncols=1)

    sns.stripplot(
        ax=axs[0],
        data=df_normalised.stack("technology"),
        x="technology",
        y=df_normalised.stack("technology").array,
        order=POWER_TECH_ORDER,
        marker=open_circle,
        color="grey",
    )

    axs[0].set_title(
        f"Capacity distribution of SPORES results for the power sector in {region}"
    )
    axs[0].set_xlabel("")
    axs[0].set_ylabel("Normalised capacity")
    axs[0].set_xticks(range(len(POWER_TECH_ORDER)))
    axs[0].set_xticklabels(POWER_TECH_ORDER, fontsize=10)
    xticklabels = []
    for ticklabel in axs[0].get_xticklabels():
        technology = ticklabel.get_text()

        if technology in df.index.get_level_values("technology").unique():
            xticklabels.append(
                f"{technology}\n{capacity_ranges.loc[technology, 'min'].round(2)} - {capacity_ranges.loc[technology, 'max'].round(2)} [GW]"
            )
        else:
            xticklabels.append(f"{technology}\n0.0 - 0.0 [GW]")
    axs[0].set_xticklabels(xticklabels, fontsize=10)


    print(df)
    # sns.boxplot(ax=axs[1], data=df, x="technology", y=df.array)


    print(cluster_ranges_normalised)
    x = cluster_ranges_normalised.index.get_level_values("technology").unique()
    for cluster in range(n_clusters):
        axs[2].fill_between(x=x, y1=cluster_ranges_normalised.loc[cluster, :]["min"], y2=cluster_ranges_normalised.loc[cluster, :]["max"], alpha=0.5, label=f"SPORE cluster: {cluster}")
    #FIXME: re order the x-ticks to match to order of the stripplot

    # Set figure legend
    axs[2].legend(bbox_to_anchor=(1, 1), loc="upper left")


if __name__ == "__main__":
    # Define paths to data
    paths = {"2050": os.path.join(os.getcwd(), "../data", "euro-spores-results-v2022-05-13"), }
    # Define for which cost relaxation we want to read the data
    slack = "slack-10"
    # Define which files we want to read
    files = ["nameplate_capacity"]
    # Read the data
    data = {"2050": read_spores_data(paths["2050"], slack, files)}
    # Get capacity data for power sector
    power_capacity = get_power_capacity(spores_data=data, save_to_csv=False)

    plot_normalised_cluster_map(data=power_capacity, year="2050", region="Europe")
    plot_normalised_cluster_map(data=power_capacity, year="2050", region="Netherlands")

    """
    Choose country
    """
    region = "Netherlands"


    plot_cluster_capacity_distribution(data=power_capacity, year="2050", region=region)

    # 1. Prepare DataFrame
    df = power_capacity.loc["2050", region, :, :]
    # 2a. Standardise df: -mean, /std (optional)
    standard_scaler = StandardScaler()
    df_standardised = pd.DataFrame(standard_scaler.fit_transform(df.unstack("technology")), columns=df.index.get_level_values("technology").unique())
    # 2b. Normalise df: -min(), /max() (optional)
    normal_scaler = MinMaxScaler()
    df_normalised = pd.DataFrame(normal_scaler.fit_transform(df.unstack("technology")), columns=df.index.get_level_values("technology").unique())

    best_n_clusters = None
    best_score = -1             #Because silhouette score ranges between -1 and 1
    for n_clusters in range(2, 100):
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = agg_clustering.fit_predict(df_normalised)
        score = silhouette_score(df_normalised, cluster_labels)
        print(f"For {n_clusters}, the silhouette score = {score}")
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters

    print(f"Optimal number of clusters = {best_n_clusters}, highest silhouette score = {best_score}")

    # spores = [353, 364, 343, 361, 346, 358, 351, 352, 348, 350, 345, 342, 363, 354, 349, 355, 365, 359, 360, 347, 357, 344, 362]
    # print(power_capacity_Europe.loc[:, spores].unstack("spore"))
    # print(power_capacity_Europe.loc[:, spores].unstack("spore").min(axis=1))
    # print(power_capacity_Europe.loc[:, spores].unstack("spore").max(axis=1))


    clustering = AgglomerativeClustering(n_clusters=6).fit(df_normalised)

    # Make new DataFrame that contains the corresponding cluster number for each spore
    new_index = pd.MultiIndex.from_tuples([(spore, clustering.labels_[spore], tech) for tech, spore in df.index], names=["spore", "spore_cluster", "technology"])
    df_with_cluster = pd.Series(df.array, index=new_index)

    # Calculate capacity range for all technology in each cluster
    cluster_ranges = (
        df_with_cluster
        .groupby(["spore_cluster", "technology"])
        .agg(["min", "max"])
    )

    plt.show()
