import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from global_parameters import *
from plotting_functions import *

if __name__ == "__main__":
    # Get SPORES power capacity for categorised_spores and 2050
    power_capacity = pd.read_csv(
        "../data/power_capacity.csv",
        index_col=["year", "region", "technology", "spore"],
        squeeze=True,
    )
    # Get SPORES heat capacity for categorised_spores and 2050
    heat_capacity = pd.read_csv(
        "../data/heat_capacity.csv",
        index_col=["year", "region", "technology", "spore"],
        squeeze=True,
    )
    # Get SPORES storage capacity for categorised_spores and 2050
    storage_capacity = pd.read_csv(
        "../data/storage_capacity.csv",
        index_col=["year", "region", "technology", "spore"],
        squeeze=True,
    )
    # Get SPORES grid capacity for categorised_spores and 2050
    grid_capacity = pd.read_csv("../data/grid_transfer_capacity.csv", index_col = ["importing_region", "exporting_region", "year", "spore"], squeeze=True)

    # Get historical power capacity data (2000-2021) obtained from IRENASTAT
    power_capacity_2000_2021 = pd.read_csv(
        "../data/power_capacity_irenastat.csv",
        index_col=["region", "technology", "year"],
        squeeze=True,
    )

    """
    INPUT: Choose region and technology to focus on:
    """
    region_of_interest = "United Kingdom"
    technology_of_interest = "Onshore wind"

    """
    1. Prepare the data for all the functions
    """
    # Get power capacity data for categorised_spores and 2050 SPORES, and historic capacity data for 2000-2021
    #FIXME: rename to p_capacity_2030 etc. (p = power)
    capacity_2030 = power_capacity.loc[2030, region_of_interest, :, :]
    capacity_2050 = power_capacity.loc[2050, region_of_interest, :, :]
    capacity_2000_2021 = power_capacity_2000_2021.loc[region_of_interest, technology_of_interest, :]
    h_capacity_2030 = heat_capacity.loc[2030, region_of_interest, :, :]
    h_capacity_2050 = heat_capacity.loc[2050, region_of_interest, :, :]


    # fig, ax = plt.subplots(figsize=(20,9))
    # plot_technology_pathway(ax=ax, capacity_2000_2021=power_capacity_2000_2021, capacity_2030=capacity_2030, capacity_2050=capacity_2050, country=region_of_interest, technology=technology_of_interest)


    # Filter out categorised_spores spores that are infeasible to due to capacity that is currently locked in
    infeasible_spores = set()
    lock_in_2030 = []
    for tech in [technology_of_interest]: #capacity_2030.index.get_level_values("technology").unique():
        # Get life time for each technology
        life_time = ELECTRICITY_PRODUCERS_LIFE.get(tech)
        # Get installed capacity in 2021
        capacity_present = power_capacity_2000_2021.get((region_of_interest, tech, 2021), 0)    # returns 0 if technology is not found in IRENASTAT data
        # Get capacity that will be decomissioned before categorised_spores
        capacity_decommissioned = power_capacity_2000_2021.get((region_of_interest, tech, (2030 - life_time)), 0) # returns 0 if technology or year (if lifetime > 30 years) is not found in IRENASTAT data
        # Calculate capacity that is locked in beyond categorised_spores
        capacity_minimum = capacity_present - capacity_decommissioned
        # Find which SPORES are feasible (above locked in capacity) and which are infeasible (below lock in capacity)
        tech_capacity_2030 = capacity_2030.loc[2030, region_of_interest, tech]
        mask = tech_capacity_2030 > capacity_minimum
        feasible_spores = tech_capacity_2030[mask].index.get_level_values("spore")
        infeasible_spores.update(tech_capacity_2030[~mask].index.get_level_values("spore"))
        # Filter out infeasible categorised_spores SPORES due to lock-in
        capacity_2030 = capacity_2030.loc[:, :, :, feasible_spores]
        # Add capacity values to table and infeasible spores to list
        table_row = [tech, f"{life_time} [years]",  f"{capacity_present:.2f} [GW]", f"{capacity_decommissioned:.2f} [GW]", f"{capacity_minimum:.2f} [GW]"]
        lock_in_2030.append(table_row)
        print(f"{tech}, feasible: {len(feasible_spores)}, in-feasible: {len(infeasible_spores)}")
    # Define table with capacity values for each technology
    lock_in_2030 = pd.DataFrame(np.array(lock_in_2030), columns=["Technology", "Life time", "Capacity 2021", "Decomisioned capacity before categorised_spores", "Locked-in capacity in categorised_spores"])
    infeasible_spores = list(infeasible_spores)
    pd.set_option("display.max_columns", None)
    print(lock_in_2030)
    print(f"The following categorised_spores SPORES were deamed infeasible due to capacity that is already locked in beyond categorised_spores: \n {infeasible_spores}")


    """
    2. Plot boxplot of capacities for power sector categorised_spores and 2050 compararison
    """
    # # Plot boxplot of capacities for categorised_spores and 2050 comparison
    plot_boxplot_capacities_2030_2050(capacity_2030, capacity_2050, region_of_interest, "power")
    plot_boxplot_capacities_2030_2050(h_capacity_2030, h_capacity_2050, region_of_interest, "heat")
    # # Plot boxplot of capacities for power and heat sector comparison in 2050
    # plot_boxplot_capacities_per_sector(capacity_2050, h_capacity_2050, region_of_interest, 2050)

    """
    3. Plot capacity clustermap
    """
    plot_normalised_cluster_map(data=power_capacity.loc[2030, region_of_interest, :, :], year=2030, region=region_of_interest)
    plot_normalised_cluster_map(data=power_capacity.loc[2050, region_of_interest, :, :], year=2050, region=region_of_interest)

    """
    4. Plot capacity distribution:
    """
    # fig, ax = plt.subplots(figsize=(20,9))
    # plot_capacity_distribution_2030_2050(ax=ax, data=power_capacity, country=region_of_interest)

    """
    5. Plot pathway for PV
    """
    fig, ax = plt.subplots(figsize=(20,9))
    # Get spores that below to each of the quartiles
    quartile_spores_2030 = get_national_quartile_spores(capacity_spores=capacity_2030, region=region_of_interest, technology=technology_of_interest, year=2030)
    # Get exponential projections to each of the quartiles
    projections_2021_2030, info_2030 = projection_to_spores_exponential(start_capacity=capacity_2000_2021[-1], spores_capacity=capacity_2030.loc[2030, region_of_interest, technology_of_interest, :], years=years_2021_2030)
    print(info_2030)
    # Get "lock-in" trajectory if no new capacity is installed after categorised_spores
    lock_in_info_2050 = lock_in_projection(capacity_2000_2021=capacity_2000_2021, projections=projections_2021_2030, capacity_2030=capacity_2030, region=region_of_interest, technology=technology_of_interest, technology_life=ELECTRICITY_PRODUCERS_LIFE.get(technology_of_interest))
    # print(pd.merge(info_2030, lock_in_info_2050, on="SPORES categorised_spores"))


    # Prepare data for technology pathway
    pv_2000_2021 = power_capacity_2000_2021.loc[region_of_interest, technology_of_interest, 2000:]
    pv_2030 = capacity_2030.loc[2030, region_of_interest, technology_of_interest, :]
    pv_2050 = capacity_2050.loc[2050, region_of_interest, technology_of_interest, :]

    # Plot technology pathway
    plot_technology_pathway(ax=ax, capacity_2000_2021=pv_2000_2021, capacity_2030=pv_2030, capacity_2050=pv_2050, country=region_of_interest, technology=technology_of_interest)



    """
    7. Plot impact of Top 25% PV decision
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 9))
    #FIXME: plot only the spores in capacity_2030 that are not "quartile_spores_2030.get("Q4_spores") to avoid Q1-Q3 being plot twice
    plot_capacity_distribution(ax=ax, data=capacity_2050, year=2050, country=region_of_interest)
    # capacity_normalised = capacity_2030.div(capacity_2030.groupby("technology").max())
    # capacity_top_spores = capacity_normalised.loc[:, :, :, quartile_spores_2030.get("Q4_spores")]
    # sns.stripplot(
    #     ax=axs[0],
    #     data=capacity_top_spores,
    #     x="technology",
    #     y=capacity_top_spores.array,
    #     order=POWER_TECH_ORDER,
    #     color="red",
    # )




    # """
    # TEST: measure/visualise "lock-in" effect in pathway plot
    # """
    # fig, ax = plt.subplots(figsize=(20,9))
    # plot_capacity_pathway_without_projection(ax=ax, capacity_2000_2021=power_capacity_2000_2021, capacity_spores=power_capacity, country=region_of_interest, technology=technology)
    #
    # # Calculate growth rate required to reach maximum spores capacity in categorised_spores
    # max_growth = calculate_growth_factor(capacity_2000_2021[-1], max_capacity_2030, 2021, categorised_spores)
    #
    # # Calculate projected capacity between 2021 and categorised_spores with exponential growth
    # cap_20_percent_growth = [capacity_2000_2021[-1] * 1.2 ** (year - years_2021_2030[0]) for year in years_2021_2030]
    # cap_max_growth = [capacity_2000_2021[-1] * max_growth ** (year - years_2021_2030[0]) for year in years_2021_2030]
    #
    # # Calculate projected capacity decline if nothing new is built between categorised_spores and 2050
    # cap_lock_in_from_max = [max_capacity_2030]
    # print(cap_lock_in_from_max[-1])
    # print(capacity_2000_2021)
    # print(capacity_2000_2021.loc[:, :, (2031 - pv_life_years)])
    # #FIXME: calculate capacity decline from categorised_spores to 2050 using a life time of 25 years assuming nothing no new capacity gets installed
    #
    #
    #
    # plt.plot(years_2021_2030, cap_20_percent_growth, linestyle="--", color="orange", label="Exponential growth with a CAGR of 20%")
    # plt.plot(years_2021_2030, cap_max_growth, linestyle="--", color="red", label=f"Exponential growth with a CAGR of {100 * (max_growth - 1):.1f} %")
    #
    # # Set figure legend
    # ax.legend(bbox_to_anchor=(0, 1), loc="upper left")

    plt.show()