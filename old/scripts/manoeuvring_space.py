import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler

from global_parameters import *
from plotting_functions import *
from processing_functions import *

def normalise_df_columns(df):
    df_normalised = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
    return df_normalised.fillna(0)

def plot_clustermap(normalised_spores, min_max_values, color_map, show_ranges=False):
    # Convert color map into a Series that we can input into sns.clustermap
    row_colors = pd.Series(list(normalised_spores.T.index), index=normalised_spores.T.index)
    row_colors = row_colors.map(color_map)

    all_techs = list(capacity_spores_normalised.columns)

    clustermap = sns.clustermap(
        data=normalised_spores.T,
        method="ward",
        metric="euclidean",
        row_cluster=False,
        cmap="Spectral_r",
        row_colors=row_colors
    )
    clustermap_axis = clustermap.ax_heatmap

    #FIXME: change the labels of the y axis (can I use the method set_text()? on a matplotlib.text
    if show_ranges:
        # y_labels = clustermap_axis.get_yticklabels()
        # print(y_labels)
        # print(y_labels[0])
        # print(type(y_labels[0]))
        # print(dir(y_labels[0]))
        # y_labels[0].set_text()
        # print(y_labels)
        # y_labels_with_ranges = [label.set_text(f"{label.get_text()} {min_max_values.loc[label.get_text(), 'min']:.1f} - {min_max_values.loc[label.get_text(), 'max']:.1f} [{TECH_UNITS.get(label.get_text())}]") for label in y_labels]
        # print(y_labels_with_ranges)

        # # Set y labels for all technologies with corresponding capacity range and unit
        # clustermap_axis.set_yticks(range(len(all_techs)))
        # clustermap_axis.set_yticklabels(all_techs, fontsize=10)
        # yticklabels = []
        # for ticklabel in clustermap_axis.get_yticklabels():
        #     technology = ticklabel.get_text()
        #     yticklabels.append(
        #         f"{technology} {min_max_values.loc[technology, 'min']:.1f} - {min_max_values.loc[technology, 'max']:.1f} [{TECH_UNITS.get(technology)}]"
        #     )
        # clustermap_axis.set_yticklabels(yticklabels)

if __name__ == "__main__":
    """
    Read spores capacity data
    """
    # Get SPORES heat capacity in Europe for categorised_spores and 2050
    heat_spores_gw = pd.read_csv(
        "../data/heat_capacity.csv",
        index_col=["year", "region", "technology", "spore"],
        squeeze=True,
    ).loc[:, "Europe", :, :]

    # Get SPORES power capacity in Europe for categorised_spores and 2050
    power_spores_gw = pd.read_csv(
        "../data/power_capacity.csv",
        index_col=["year", "region", "technology", "spore"],
        squeeze=True,
    ).loc[:, "Europe", :, :]

    color_map = {
        "Boiler": "r",
        "Electric heater": "r",
        "Heat pump": "r",
        "PV": "b",
        "Onshore wind": "b",
        "Offshore wind": "b",
        "Nuclear": "b",
        "Hydro": "b",
        "CCGT": "b",
        "CHP": "b",
    }

    # Concatenate heat and power spores capacity into one series
    capacity_spores_gw = pd.concat(
        [power_spores_gw, heat_spores_gw]
    )

    """
    Plot clustermap for power and heat
    """
    # obtain power and heat spores capacity for 2050
    capacity_spores_gw_2050 = capacity_spores_gw.loc[2050, :, :]
    # normalise spores
    capacity_spores_normalised = normalise_df_columns(capacity_spores_gw_2050.unstack("technology"))
    # calculate min and max capacity of each technology
    capacity_ranges_2050 = capacity_spores_gw_2050.groupby("technology").agg(["min", "max"])
    # Plot cluster map
    plot_clustermap(normalised_spores=capacity_spores_normalised, min_max_values=capacity_ranges_2050, color_map=color_map)
    plot_clustermap(normalised_spores=capacity_spores_normalised, min_max_values=capacity_ranges_2050, color_map=color_map, show_ranges=True)

    plt.show()