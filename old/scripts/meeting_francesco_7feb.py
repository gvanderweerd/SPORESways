import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from global_parameters import *
from plotting_functions import *
from processing_functions import *

if __name__ == "__main__":
    # Get SPORES power capacity in Europe for categorised_spores and 2050
    power_capacity = pd.read_csv(
        "../data/power_capacity.csv",
        index_col=["year", "region", "technology", "spore"],
        squeeze=True,
    )
    # Get SPORES heat capacity in Europe for categorised_spores and 2050
    heat_capacity = pd.read_csv(
        "../data/heat_capacity.csv",
        index_col=["year", "region", "technology", "spore"],
        squeeze=True,
    ).loc[:, "Europe", :, :]
    # Get SPORES storage capacity in Europe for categorised_spores and 2050
    storage_capacity = pd.read_csv(
        "../data/storage_capacity.csv",
        index_col=["year", "region", "technology", "spore"],
        squeeze=True,
    ).loc[:, "Europe", :, :]
    # Get SPORES grid capacity in Europe for categorised_spores and 2050
    grid_capacity = (
        pd.read_csv(
            "../data/grid_transfer_capacity.csv",
            index_col=[
                "year",
                "importing_region",
                "exporting_region",
                "technology",
                "spore",
            ],
            squeeze=True,
        )
        .groupby(["year", "technology", "spore"])
        .sum()
    )
    # FIXME: add fuel generation? (like hydrogen, methane production etc.)

    # Get historical power capacity data (2000-2021) obtained from IRENASTAT
    # FIXME: 1. Find out why theses countries are missing in irenastat data. Misses: "Iceland", "Latvia", "Luxembourg", "Slovenia"
    # FIXME: 2. Get complete historical dataset that also includes CCGT, CHP, Coal, Gas etc.
    power_capacity_2000_2021 = pd.read_csv(
        "../data/power_capacity_irenastat.csv",
        index_col=["region", "technology", "year"],
        squeeze=True,
    )
    # FIXME: find and add historical data for heat, storage, and grid capacity (maybe we need to switch to only 2020/2021 (or 2022 if possible) values to make this easier

    power_capacity = power_capacity.loc[:, "Europe", :, :]
    power_capacity_2000_2021 = power_capacity_2000_2021.groupby(
        ["technology", "year"]
    ).sum()

    # print(grid_capacity)
    # # Add sector names as an index with name "sector"
    # power_capacity = pd.concat({"Power": power_capacity}, names=["sector"])
    # heat_capacity = pd.concat({"Heat": heat_capacity}, names=["sector"])
    # storage_capacity = pd.concat({"Storage": storage_capacity}, names=["sector"])
    # grid_capacity = pd.concat({"Grid": grid_capacity}, names=["sector"])
    # print(grid_capacity)

    # Combine capacities of all sectors to plot the clustermap: (groupby is needed again to sum CHP for electricity and heat)
    capacity = pd.concat(
        [power_capacity, heat_capacity, storage_capacity, grid_capacity]
    )
    capacity_2030 = capacity.loc[2030, :, :].groupby(["technology", "spore"]).sum()
    capacity_2050 = capacity.loc[2050, :, :].groupby(["technology", "spore"]).sum()

    """
    FIGURE 1: cluster map
    """
    print("TEST")
    print(capacity_2050)
    plot_normalised_cluster_map(data=capacity_2050, year=2050, region="Europe")

    # Find optimal amount of clusters for Agglomorative Hierarchical clustering using silhouette similarity score
    capacities_2050_clustered = find_highest_similarity_clusters(data=capacity_2050)

    # Pick cluster number to focus on
    focus_cluster = 4
    capacity_2050_in_cluster = capacities_2050_clustered[
        capacities_2050_clustered.index.get_level_values("spore_cluster")
        == focus_cluster
    ].reset_index(level="spore_cluster", drop=True)
    capacity_2050_rest = capacities_2050_clustered[
        capacities_2050_clustered.index.get_level_values("spore_cluster")
        != focus_cluster
    ].reset_index(level="spore_cluster", drop=True)
    # Find the spores spore numbers that belong to the chosen cluster and the spore numbers that do not belong to the chosen cluster
    spores_cluster = capacity_2050_in_cluster.index.get_level_values("spore").unique()
    spores_rest = capacity_2050_rest.index.get_level_values("spore").unique()

    """
    FIGURE 2: 2050 SPORES capacity distribution for clusters. Color all 2050 spores that belong to the chosen cluster blue.
    """
    n_clusters = capacities_2050_clustered.index.get_level_values(
        "spore_cluster"
    ).unique()
    fig, axs = plt.subplots(nrows=1, ncols=1)
    plot_capacity_distribution_clusters_2050(
        ax=axs,
        capacity=capacities_2050_clustered.loc[
            ["PV", "Onshore wind", "Offshore wind", "Hydro", "CCGT", "CHP", "Nuclear"],
            :,
            :,
        ],
        cluster_in_color=4,  #This should select a cluster number in the function in plottig_functions.py
    )

    """
    FIGURE 3: pathway plot
    """

    technologies_to_plot = [
        "PV",
        "Onshore wind",
        # "Offshore wind",
        # "Hydro",
        # "Nuclear",
        # "CCGT"
    ]  # Other technologies are: "Hydro", "Nuclear", "CCGT", "CHP" (We do not have historic data for CHP)
    fig, axs = plt.subplots(nrows=len(technologies_to_plot), ncols=1, sharex=True)

    spores_on_track_overlap = range(441)
    i = 0
    spores_2030_categories = {}
    for tech in technologies_to_plot:
        """
        Make technology pathway plot for "PV" in Europe if we go for the cluster with the most amount of spores
        """
        # Prepare spore data for technology pathway plot
        pv_2000_2021 = power_capacity_2000_2021.loc[tech, :].reset_index(
            level="technology", drop=True
        )
        # Drop the "technology" index (this index is not needed since we have filtered out all other technologies)
        pv_2030 = capacity_2030.loc[tech, :].reset_index(level="technology", drop=True)
        pv_2050 = capacity_2050.loc[tech, :].reset_index(level="technology", drop=True)
        # Split the technology capacity for 2050 into two series; 1. the 2050 SPORES that belong to the cluster and 2. the rest of the 2050 SPORES
        pv_2050_in_cluster = capacity_2050_in_cluster.loc[tech, :].reset_index(
            level="technology", drop=True
        )
        pv_2050_rest = capacity_2050_rest.loc[tech, :].reset_index(
            level="technology", drop=True
        )

        # Plot pathway for technology and output the categorised_spores spores that are 'on track' for the pathway for this technology
        spores_2030_categories[tech] = [
            spores_underinvested,
            spores_on_track,
            spores_overinvested,
        ] = plot_technology_pathway(
            ax=axs[i],
            technology=tech,
            capacity_2000_2021=pv_2000_2021,
            capacity_2030=pv_2030,
            capacity_2050=pv_2050_rest,
            capacity_2050_in_cluster=pv_2050_in_cluster,
            s_curve_params=s_curve_params_power_sector.get(tech),
        )
        i += 1



        # Find categorised_spores SOPRES that are on track for every technology
        # print(
        #     f"These categorised_spores SPORES are on track for the capacity of {tech} for 2050 SPORES in cluster 4: \n {list(spores_on_track)}"
        # )
        spores_on_track_overlap = sorted(
            list(set(spores_on_track_overlap) & set(spores_on_track))
        )

    """
    Add the spores that are on track for all technologies as black dots to the pathway plot
    """
    i = 0
    for tech in technologies_to_plot:
        print(tech, i)
        capacity_2030_on_track_for_all_techs = capacity_2030.loc[tech, list(spores_on_track_overlap)].reset_index(level="technology", drop=True)
        add_spores_capacity_to_pathway(ax=axs[i], capacity_spores=capacity_2030_on_track_for_all_techs, year=2030, label="SPORES that are on track in all technologies for cluster 4 in 2050", color="black", marker="o")
        i += 1


    """
    FIGURE 4: categorised_spores SPORES capacity distribution for power sector. Color categorised_spores SPORES "on-track", "under-invested", "over-inveseted".
    """
    fig, axs = plt.subplots(nrows=2, ncols=1)

    # Find underinvested, on-track, and overinvested categorised_spores SPORES for each technology
    #FIXME: 1. save the categorised_spores SPORES that are output by the pathway plot for each technology in a dictionary for later access.
    plot_capacity_distribution_2030_2050(ax=axs[0], data=power_capacity, country="Europe")

    print(spores_2030_categories.get("PV")[1])
    print(spores_2030_categories.get("Onshore wind")[1])
    print(power_capacity)

    # Prepare capacity data (test with power capacity)
    #FIXME: 2. get all power capacity data (from above)

    #FIXME: 3. normalise al power capacity data between min and max

    # Figure out how to add the underinveseted, ontrack, overinvested colors for categorised_spores (category in series?)
    # Figure out how to add the 2050 cluster spores in blue

    # Plot figure
    #FIXME: 4. plot noramlised power capacity data with categorised_spores and 2050 as a hue (all spores in grey)


    plt.show()