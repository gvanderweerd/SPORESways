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

from src.utils.parameters import *
from src.utils.data_io import *

# Define open circles for plotting 'All other SPORES'
pts = np.linspace(0, np.pi * 2, 24)
circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
vert = np.r_[circ, circ[::-1] * 0.7]
open_circle = mpl.path.Path(vert)


POWER_TECH_COLORS = {
    "All other SPORES": "#b9b9b9",
    "Nuclear": "#cc0000",
    "nuclear": "#cc0000",
    "CCGT": "#8fce00",
    "ccgt": "#8fce00",
    "CHP": "#ce7e00",
    "chp": "#ce7e00",
    "PV": "#ffd966",
    "pv": "#ffd966",
    "Onshore wind": "#674ea7",
    "onshore wind": "#674ea7",
    "Offshore wind": "#e062db",
    "offshore wind": "#e062db",
    "Wind": "#674ea7",
    "Hydro": "#2986cc",
    "hydro": "#2986cc",
    "Coal": "#000000",  # Black = #000000, Grey = #808080
    "coal": "#000000",
}
metric_plot_order = {
    "curtailment": 2,
    "electricity_production_gini": 5,
    "average_national_import": 4,
    "fuel_autarky_gini": 6,
    "storage_discharge_capacity": 1,
    "transport_electrification": 9,
    "heat_electrification": 8,
    "biofuel_utilisation": 3,
    "ev_flexibility": 7,
}
metric_plot_names = {
    "curtailment": "Curtailment",
    "electricity_production_gini": "Electricity production\nGini coefficient",
    "average_national_import": "Average\nnational import",
    "fuel_autarky_gini": "Fuel autarky\nGini coefficient",
    "storage_discharge_capacity": "Storage discharge\ncapacity",
    "transport_electrification": "Transport\nelectrification",
    "heat_electrification": "Heat\nelectrification",
    "biofuel_utilisation": "Biofuel\nutilisation",
    "ev_flexibility": "EV as flexibility",
}
metric_range_formatting = {
    "curtailment": lambda x: x.round(0).astype(int),
    "electricity_production_gini": lambda x: x.round(2),
    "average_national_import": lambda x: x.round(0).astype(int),
    "fuel_autarky_gini": lambda x: x.round(2),
    "storage_discharge_capacity": lambda x: x.round(0).astype(int)
    if x.name == "max"
    else x.round(2),
    "transport_electrification": lambda x: x.round(0).astype(int),
    "heat_electrification": lambda x: x.round(0).astype(int),
    "biofuel_utilisation": lambda x: x.round(0).astype(int),
    "ev_flexibility": lambda x: x.round(2),
}
FIGWIDTH = 6.77165


def normalise_to_max(x):
    max_value = x.max()
    return x / max_value


def get_color_dict2(label_list):
    color_palette = sns.color_palette("bright")
    colors = {
        label_list[i]: color_palette[i % len(color_palette)]
        for i in range(len(label_list))
    }
    colors["rest_color"] = color_palette[-3]
    return colors


def get_color_dict(label_list):
    # Define colours
    colors = {
        label_list[i]: (
            sns.color_palette("bright")[:-3] + sns.color_palette("bright")[-2:]
        )[i]
        for i in range(len(label_list))
    }
    colors["rest_color"] = sns.color_palette("bright")[-3]
    return colors


def plot_scenarios_mean_capacities(power_capacity_clustered, year):
    # FIXME: this function is only intended for continental resolution. Is this the right figure if we want to visualise clusters that where clustered on national resolution?
    fig, ax = plt.subplots(1, 1, figsize=(FIGWIDTH, 10 * FIGWIDTH / 20))

    # Prepare data
    scenarios_avg = (
        power_capacity_clustered.groupby(["cluster", "technology"])
        .agg("mean")
        .reset_index()
        .pivot_table(index="cluster", columns="technology", values="capacity_gw")
    )
    n_spores_per_cluster = count_spores_per_cluster(power_capacity_clustered)
    print()

    # Plot figure
    scenarios_avg.plot(ax=ax, kind="bar", stacked=True, color=POWER_TECH_COLORS)
    # FIXME: rotate x-labels 90 degrees
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Installed capacity [GW]")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.title(f"{year}")
    plt.tight_layout(pad=1)

    # Annotate number of SPORE in each cluster on top of each bar
    for cluster_number, spores_count in enumerate(n_spores_per_cluster):
        ax.text(
            cluster_number,
            scenarios_avg.sum(axis=1)[cluster_number] * 1.05,
            str(spores_count),
            va="center",
            ha="center",
        )


def plot_scenario_analysis():
    fig, ax = plt.subplots(1, 1, figsize=(FIGWIDTH, 10 * FIGWIDTH / 20))


def plot_metrics_distribution(metrics, year, focus_cluster=None):
    print(metrics)
    # Calculate metric ranges and get units
    metric_ranges = metrics.groupby(level="metric").agg(["min", "max"])
    metric_units = (
        metrics.index.to_frame(index=False)
        .drop_duplicates(subset="metric")
        .set_index("metric")["unit"]
        .to_dict()
    )

    # Normalise metrics
    metrics_normalised = (
        metrics.groupby(["metric"]).transform(normalise_to_max).reset_index()
    )

    # Calculate min, max for focus cluster (for plotting boxes)
    max_scenario = (
        metrics_normalised.loc[
            (metrics_normalised.cluster == focus_cluster), ["paper_metrics", "metric"]
        ]
        .groupby("metric")
        .max()["paper_metrics"]
    )
    min_scenario = (
        metrics_normalised.loc[
            (metrics_normalised.cluster == focus_cluster), ["paper_metrics", "metric"]
        ]
        .groupby("metric")
        .min()["paper_metrics"]
    )

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
        data=metrics_normalised[
            ~metrics_normalised.spore.isin(spores_to_plot_in_color)
        ],
        x="metric",
        y="paper_metrics",
        marker=open_circle,
        color=cluster_colors["rest_color"],
        alpha=0.5,
        s=3,
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
                zorder=10,
            ),
        )
        _x += 1

        # Format x-axis labels
        metric_range = metric_ranges.apply(metric_range_formatting[_metric]).loc[
            _metric
        ]
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
            data=metrics_normalised[
                metrics_normalised.spore.isin(spores_to_plot_in_color)
            ],
            x="metric",
            y="paper_metrics",
            alpha=0.5,
            ax=ax,
            marker="o",
            color=cluster_colors[focus_cluster],
            s=3,
        )

    # FIXME: get unique spores only
    # print(spores_to_plot_in_color)

    # Set figure title
    n_spores_per_cluster = count_spores_per_cluster(metrics)
    ax.set_title(
        f"Scenario {focus_cluster} ({year}): {n_spores_per_cluster[focus_cluster]} SPORES"
    )

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


def plot_capacity_distribution(capacity, year, focus_cluster=None):
    pass


def plot_capacity_pathway(power_capacity):
    pass


def plot_elbow_figure(wcss, min_clusters, max_clusters, spatial_resolution, year):
    # Using Elbow method
    plt.figure()
    plt.plot(range(min_clusters, max_clusters + 1), wcss, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.savefig(
        f"../figures/appendices/finding_optimal_n_scenarios/elbow_method_{spatial_resolution}_{year}.png",
        bbox_inches="tight",
        dpi=120,
    )
    plt.title(f"Elbow Method for {year}")


def plot_silhouette_score(
    silhouette_scores, min_clusters, max_clusters, spatial_resolution, year
):
    plt.figure()
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.savefig(
        f"../figures/appendices/finding_optimal_n_scenarios/silhouette_method_{spatial_resolution}_{year}.png",
        bbox_inches="tight",
        dpi=120,
    )
    plt.title(f"Silhouette Score Method for {year}")


if __name__ == "__main__":
    """
    0. SET PARAMETERS
    """
    # Choose scenario_number to analyse
    focus_scenario = 0

    """
    1. READ AND PREPARE DATA
    """
