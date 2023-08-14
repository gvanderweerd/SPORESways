import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Import other scripts in this repository
from main import *
from global_parameters import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

YEARS = range(2010, 2051)

calculate_quartile_capacities = (
    lambda spores, region, technology: spores.groupby(["region", "technology"])
    .quantile(q=[0.25, 0.5, 0.75])
    .loc[region, technology, :]
)
calculate_exponential_growth = lambda start_capacity, growth_factor, years: [
    start_capacity * growth_factor ** (year - years[0]) for year in years
]

def s_curve(x, K, x0, L):
    return L / (1 + np.exp(-K * (x - x0)))

def capacity_growth_s_curve(capacity_2000_2021, capacity_2050):
    # x = [2020, 2021, 2050]
    x = list(np.arange(2000, 2022))
    x.append(2050)
    print(x)
    y = list(capacity_2000_2021.array)
    y.append(capacity_2050)
    print(y)

    popt, pcov = curve_fit(s_curve, x, y)
    K, x0, L = popt
    print(K, x0, L)

    years_2000_2050 = np.arange(2000, 2051)
    y = s_curve(x=years_2000_2050, K=K, x0=x0, L=L)

    return y

def get_storage_capacity_year(
    spores_data, year, technologies, carrier=None, spores=None, normalise=False
):
    cap_data = spores_data[year]["storage_capacity"]
    if spores is not None:
        cap_data = cap_data[spores]

    capacity = (
        cap_data.xs("twh", level="unit")
        .unstack("spore")
        .groupby(
            [REGION_MAPPING, technologies, "carriers"],
            level=["region", "technology", "carriers"],
        )
        .sum()
        .stack("spore")
    )

    if normalise:
        capacity = capacity.div(
            capacity.groupby(["region", "technology", "carriers"]).max()
        )

    if carrier is not None:
        capacity = capacity.xs(carrier, level="carriers")

    # Add the year as an index
    capacity = pd.concat({year: capacity}, names=["year"])

    return capacity


def get_energy_capacity_year(
    spores_data, year, technologies, carrier=None, spores=None, normalise=False
):
    """
    This functions calculates capacities of given technologies. Capacities are calculated for all countries and spores, in a given year.

    :param spores_data:     Contains the spores data for all years
    :param year:            The year for which you want to calculate the capacities
    :param technologies:    The set of technologies for which you want to calculate capacities
    :param carrier:         The carrier of the technologies (to avoid double counting in case a technology outputs multiple carriers, e.g. chp)
    :param spores:          A list of spores filters data on specific spores when provided
    :param normalise:       A bolean that normalises the data to the maximum value when set to True
    :return:                A MultiIndex Series that contains a capacity value [TW] for each combination of: "year", "region", "technology", "spore"
    """

    cap_data = spores_data[year]["nameplate_capacity"]
    if spores is not None:
        cap_data = cap_data[spores]

    capacity = (
        cap_data.xs("tw", level="unit")
        .unstack("spore")
        .groupby(
            [REGION_MAPPING, technologies, "carriers"],
            level=["region", "technology", "carriers"],
        )
        .sum()
        .stack("spore")
    )

    if normalise:
        capacity = capacity.div(
            capacity.groupby(["region", "technology", "carriers"]).max()
        )

    if carrier is not None:
        capacity = capacity.xs(carrier, level="carriers")

    # Add the year as an index
    capacity = pd.concat({year: capacity}, names=["year"])

    return capacity


def get_energy_capacity(spores_data, technologies, carrier, resolution="continental"):
    """
    This functions calculates capacities of given technologies. Capacities are calculated for all years, countries and spores in the data.

    :param spores_data:     Contains the spores data for all years
    :param technologies:    The set of technologies for which you want to calculate capacities
    :param carrier:         The carrier of the technologies (to avoid double counting in case a technology outputs multiple carriers, e.g. chp)
    :param resolution:      "national" or "continental"
    :return:                A MultiIndex Series that contains a capacity value [TW] for each combination of: "year", "region", "technology", "spore"
    """

    output = pd.Series(dtype="float64")
    for year in spores_data.keys():
        if resolution == "national":
            capacity = get_energy_capacity_year(
                spores_data, year, technologies, carrier
            )
        else:
            capacity = (
                get_energy_capacity_year(spores_data, year, technologies, carrier)
                .groupby(["year", "technology", "spore"])
                .sum()
            )
            # Add "Europe" as an index with name "region"
            capacity = pd.concat({"Europe": capacity}, names=["region"])

        # Make sure the index names
        index_names = ["year", "region", "technology", "spore"]
        capacity = capacity.reorder_levels(index_names)

        output = output.append(capacity)
        index = pd.MultiIndex.from_tuples(output.index, names=index_names)

    return pd.Series(output.array, index=index)


def get_energy_output_year(
    spores_data, year, technologies, carrier=None, spores=None, normalise=False
):
    """
    This functions calculates energy output of given technologies. Energy outputs are calculated for all countries and spores in a given year.

    :param spores_data:     Contains the spores data for all years
    :param year:            The year for which you want to calculate the energy output
    :param technologies:    The set of technologies for which you want to calculate the energy output
    :param carrier:         The carrier of the technologies (to avoid double counting in case a technology outputs multiple carriers, e.g. chp)
    :param spores:          A list of spores filters data on specific spores when provided
    :param normalise:       A bolean that normalises the data to the maximum value when set to True
    :return:                A MultiIndex Series that contains energy output value [TWh] for each combination of: "year", "region", "technology", "spore"
    """

    prod_data = spores_data[year]["flow_out_sum"]
    if spores is not None:
        prod_data = prod_data[spores]

    production = (
        prod_data.xs("twh", level="unit")
        .unstack("spore")
        .groupby(
            [REGION_MAPPING, technologies, "carriers"],
            level=["region", "technology", "carriers"],
        )
        .sum()
        .stack("spore")
    )

    if normalise:
        production = production.div(
            production.groupby(["region", "technology", "carriers"]).max()
        )

    if technologies == PRIMARY_ENERGY_SOURCES:
        # Nuclear electricity output is multiplied by 1/0.4 (efficiency = 40%) to obtain its primary energy supply as is convention
        production.loc[:, "Nuclear heat", :] = production.mul(NUCLEAR_HEAT_MULTIPLIER)
    else:
        production = production.xs(carrier, level="carriers")

    production = pd.concat({year: production}, names=["year"])

    return production


def get_energy_output(
    spores_data, technologies, carrier=None, resolution="continental"
):
    """
    This functions calculates energy output of given technologies. Energy outputs are calculated for all years, countries and spores in the data.


    :param spores_data:     Contains the spores data for all years
    :param technologies:    The set of technologies for which you want to calculate the energy output
    :param carrier:         The carrier of the technologies (to avoid double counting in case a technology outputs multiple carriers, e.g. chp)
    :param resolution:      "national" or "continental"
    :return:                A MultiIndex Series that contains energy output value [TWh] for each combination of: "year", "region", "technology", "spore"
    """

    output = pd.Series(dtype="float64")
    for year in spores_data.keys():
        if resolution == "national":
            production = get_energy_output_year(
                spores_data, year, technologies, carrier
            )
            index_names = ["year", "region", "technology", "spore"]
        else:
            production = (
                get_energy_output_year(spores_data, year, technologies, carrier)
                .groupby(["year", "technology", "spore"])
                .sum()
            )
            index_names = ["year", "technology", "spore"]

        output = output.append(production)
        index = pd.MultiIndex.from_tuples(output.index, names=index_names)

    return pd.Series(output.array, index=index)


def energy_capacity_projection_linear(data, technology, national_growth_linear):
    """
    This function projects linear capacity growth of a specified technology based on constant annual growth specified for each country

    :param data:                        Capacity data as a Series
    :param technology:                  The technology for which the growth projection is calculated
    :param national_growth_linear:      The annually installed additional capacity for each country
    :return:                            A series with the projected capacity values for all years between 2020 and categorised_spores
    """

    capacity_data = get_energy_capacity(
        data, ELECTRICITY_PRODUCERS_SPORES, "electricity", "national"
    )
    capacity_2020 = capacity_data.loc["2020", :, technology, 0].droplevel("spore")
    capacity_2020 = pd.concat({"linear": capacity_2020}, names=["method"])

    capacity_projected = capacity_2020

    # Calculate projected capacity for all years after 2020 with a 5 year time interval
    for year in YEARS[10:20]:
        next_interval = capacity_2020.add(
            (year - 2020) * national_growth_linear.div(1000)
        ).dropna()
        next_interval = next_interval.rename({"2020": str(year)})
        capacity_projected = pd.concat([capacity_projected, next_interval])

    return capacity_projected


def energy_capacity_projection_exponential(
    data, technology, national_growth_exponential
):
    """
    This function projects the exponential capacity growth of a specified technology based on growth factor specified for each country

    :param data:                            Capacity data as a Series
    :param technology:                      The technology for which the growth projection is calculated
    :param national_growth_exponential:     The annual growth factors for each country
    :return:                                A series with the projected capacity values for all years between 2020 and categorised_spores
    """

    capacity_data = get_energy_capacity(
        data, ELECTRICITY_PRODUCERS_SPORES, "electricity", "national"
    )
    capacity_2020 = capacity_data.loc["2020", :, technology, 0].droplevel("spore")
    capacity_2020 = pd.concat({"exponential": capacity_2020}, names=["method"])
    capacity_projected = capacity_2020

    # Calculate projected capacity for all years between 2020 and categorised_spores (YEARS = range(2010, 2051))
    for year in YEARS[10:20]:
        next_interval = capacity_2020.multiply(
            national_growth_exponential ** (year - 2020)
        ).dropna()
        next_interval = next_interval.rename({"2020": str(year)})
        capacity_projected = pd.concat([capacity_projected, next_interval])

    return capacity_projected


def get_national_quartile_spores(capacity_spores, technology, region, year):
    # FIXME: get rid of "region" and "technology" input. national_series should be calculated after the decision is made in the main file.

    """
    This function finds and returns the bottom 25 percentile of spores, the middle 50 percentile of spores, and the top 25 percentile of spores for the years categorised_spores and 2050. It does so for a given technology in a given region.

    :param national_series:     Input data as a multi-index series that contains values for each combination of indices: "year", "region", "technology"
    :param technology:          The technology for which to find the spores in each quartile
    :param region:              The region for which to find the spores in each quartile
    :return:                    A dictionary containing "Q1_spores", "Q2_spores", "Q3_spores", and "Q4_spores", (Q1 = lowest 25%, Q4 = highest 25% of spores)
    """

    quartiles = (
        capacity_spores.loc[year]
        .groupby(["region", "technology"])
        .quantile(q=[0.25, 0.5, 0.75])
    )
    q1, median, q3 = quartiles.loc[region, technology, :]

    s = capacity_spores.loc[year, region, technology, :]
    mask_1 = s <= q1
    mask_2 = (s > q1) & (s <= median)
    mask_3 = (s > median) & (s <= q3)
    mask_4 = s > q3

    bottom_spores = s.index[mask_1].get_level_values("spore").tolist()
    quartile_2_spores = s.index[mask_2].get_level_values("spore").tolist()
    quartile_3_spores = s.index[mask_3].get_level_values("spore").tolist()
    top_spores = s.index[mask_4].get_level_values("spore").tolist()

    output = {
        "Q1_spores": bottom_spores,
        "Q2_spores": quartile_2_spores,
        "Q3_spores": quartile_3_spores,
        "Q4_spores": top_spores,
    }
    return output


def get_quartile_range_values(capacity_spores):
    ## Calculate capacity values corresponding to first quartile, median, and 3rd quartile for spores capacities
    (
        q1,
        median,
        q3,
    ) = capacity_spores.quantile(q=[0.25, 0.5, 0.75])
    # Calculate capacity values of highest, and lowest capacity spores for categorised_spores spores
    min = capacity_spores.min()
    max = capacity_spores.max()
    return [min, q1, median, q3, max]


def projection_to_spores_exponential(start_capacity, spores_capacity, years):
    # Get capacity values for the the range of each capacity quartile in categorised_spores (quartile_range_value = [min_capacity, q1_capacity, median_capacity, q3_capacity, max_capacity])
    quartile_range_value = get_quartile_range_values(spores_capacity)
    # Calculate cagr for each quartile range in quartile_range_value
    cagrs = [
        calculate_growth_factor(start_capacity, end_capacity, years[0], years[-1])
        for end_capacity in quartile_range_value
    ]
    # Calculate projected exponential growth
    y_projections = [
        calculate_exponential_growth(start_capacity, cagr, years) for cagr in cagrs
    ]

    projections = [
        {
            "y1": y_projections[3],
            "y2": y_projections[4],
            "color": "red",
            "label": "Exponential growth to top 25% spores in categorised_spores",
        },
        {
            "y1": y_projections[2],
            "y2": y_projections[3],
            "color": "orange",
            "label": "Exponential growth to second 25% spores in categorised_spores",
        },
        {
            "y1": y_projections[1],
            "y2": y_projections[2],
            "color": "yellow",
            "label": "Exponential growth to third 25% spores in categorised_spores",
        },
        {
            "y1": y_projections[0],
            "y2": y_projections[1],
            "color": "green",
            "label": "Exponential growth to bottom 25% capacity in categorised_spores",
        },
    ]

    # FIXME: (1) define table_row in a loop and append to an empty list (info_table = []), make into a pd.DataFrame after loop
    # FIXME: (2) define a list with average growth rates (like cagrs and quartile_range_value), figure out how to deal with negative values (set to 0?)"
    info_table = pd.DataFrame(
        [
            [
                "Top 25% of SPORES",
                f"{quartile_range_value[3]:.1f} - {quartile_range_value[4]:.1f} [GW]",
                f"{100 * (cagrs[3] - 1):.1f}% - {100 * (cagrs[4] - 1):.1f}%",
                f"{(quartile_range_value[3]-start_capacity)/(years[-1]-years[0]):.1f} - {(quartile_range_value[4]-start_capacity)/(years[-1]-years[0]):.1f} [GW/year]",
            ],
            [
                "Second 25% of SPORES",
                f"{quartile_range_value[2]:.1f} - {quartile_range_value[3]:.1f} [GW]",
                f"{100 * (cagrs[2] - 1):.1f}% - {100 * (cagrs[3] - 1):.1f}%",
                f"{(quartile_range_value[2] - start_capacity) / (years[-1] - years[0]):.1f} - {(quartile_range_value[3] - start_capacity) / (years[-1] - years[0]):.1f} [GW/year]",
            ],
            [
                "Third 25% of SPORES",
                f"{quartile_range_value[1]:.1f} - {quartile_range_value[2]:.1f} [GW]",
                f"{100 * (cagrs[1] - 1):.1f}% - {100 * (cagrs[2] - 1):.1f}%",
                f"{(quartile_range_value[1]-start_capacity)/(years[-1]-years[0]):.1f} - {(quartile_range_value[2]-start_capacity)/(years[-1]-years[0]):.1f} [GW/year]",
            ],
            [
                "Bottom 25% of SPORES",
                f"{quartile_range_value[0]:.1f} - {quartile_range_value[1]:.1f} [GW]",
                f"{100 * (cagrs[0] - 1):.1f}% - {100 * (cagrs[1] - 1):.1f}%",
                f"{(quartile_range_value[0]-start_capacity)/(years[-1]-years[0]):.1f} - {(quartile_range_value[1]-start_capacity)/(years[-1]-years[0]):.1f} [GW/year]",
            ],
        ],
        columns=[
            "SPORES categorised_spores",
            "Capacity categorised_spores",
            "Required CAGR",
            "Required average growth",
        ],
    )

    return projections, info_table


def projection_lock_in_2030_2050(
    capacity_2000_2021, capacity_2021_2030, capacity_2030, years, life_time
):
    # Get capacity values for the the range of each capacity quartile in categorised_spores (quartile_range_value = [min_capacity, q1_capacity, median_capacity, q3_capacity, max_capacity])
    quartile_range_values = get_quartile_range_values(capacity_2030)
    # change order of start capacities from so that it matches order in which capacity_2021_2030 is projected (Top spores first [0], Bottom spores last [4])
    start_capacities = [
        quartile_range_values[4],
        quartile_range_values[3],
        quartile_range_values[2],
        quartile_range_values[1],
        quartile_range_values[0],
    ]

    projections = []

    # FIXME:
    # this is the ugly solution that gets uses the upper band of the top SPORES in capacity_2021_2030
    # after this loop we do the same for all 4 lower limist of each projection in (top-, second-, third-, bottom SPORES)
    # for now, it works but this function needs to be made cleaner
    capacity_2000_2030 = list(capacity_2000_2021.array)
    capacity_2000_2030.extend(capacity_2021_2030[0].get("y2")[1:])
    y = [start_capacities[0]]
    for year in years_2030_2050[1:]:
        # Calculate the index that corresponds to the capacity value in capacity_2000_2030 (i=0 means year 2000, i=10 means 2010)
        i = year - life_time - 2000
        # Calculate capacity that will be decomisioned
        capacity_decomissioned = capacity_2000_2030[i] - capacity_2000_2030[i - 1]
        # Caculate capacity value of next year (last years capacity value y[-1] minus capacity that will be decomissined in that year), capacity cannot be negative
        capacity_next_year = max(0, y[-1] - capacity_decomissioned)
        y.append(capacity_next_year)
    projections.append(y)

    for n in range(4):
        capacity_2000_2030 = list(capacity_2000_2021.array)
        capacity_2000_2030.extend(capacity_2021_2030[n].get("y1")[1:])
        # FIXME: this must be n+1 because we define y[0] earlier
        y = [start_capacities[n + 1]]
        for year in years_2030_2050[1:]:
            # Calculate the index that corresponds to the capacity value in capacity_2000_2030 (i=0 means year 2000, i=10 means 2010)
            i = year - life_time - 2000
            # Calculate capacity that will be decomisioned
            capacity_decomissioned = max(
                0, capacity_2000_2030[i] - capacity_2000_2030[i - 1]
            )
            # Caculate capacity value of next year (last years capacity value y[-1] minus capacity that will be decomissined in that year), capacity cannot be negative
            capacity_next_year = max(0, y[-1] - capacity_decomissioned)
            y.append(capacity_next_year)
        projections.append(y)
    return projections


def lock_in_projection(
    capacity_2000_2021, projections, capacity_2030, region, technology, technology_life
):
    # Get capacity values for the the range of each capacity quartile in categorised_spores
    [
        min_capacity_2030,
        q1_capacity_2030,
        median_capacity_2030,
        q3_capacity_2030,
        max_capacity_2030,
    ] = get_quartile_range_values(capacity_2030)

    # To assess the capacity that is 'locked-in' in 2050 we need to calculate the capacity that reaches end of its lifetime between categorised_spores and 2050.
    # For a lifetime of 25 years, all capcity that is built between before 2025 will expire

    # All installed capacity before "year_of_retired_capacity" is at the end of its lifetime
    year_of_retired_capacity = 2050 - technology_life

    locked_in_capacities = []
    for i in range(4):
        capacity_value_2030 = projections[i].get("y2")[-1]
        if year_of_retired_capacity > 2021:
            expired_capacity = projections[i].get("y2")[year_of_retired_capacity - 2021]
        else:
            expired_capacity = capacity_2000_2021.loc[
                region, technology, year_of_retired_capacity
            ]
        locked_in_capacities.append(max(0, capacity_value_2030 - expired_capacity))
    labels = [
        "Top 25% of SPORES",
        "Second 25% of SPORES",
        "Third 25% of SPORES",
        "Bottom 25% of SPORES",
    ]
    info_table = pd.DataFrame(
        {
            "SPORES categorised_spores": labels,
            "Locked-in capacity in 2050": [
                f"{value:.1f} [GW]" for value in locked_in_capacities
            ],
        }
    )
    # FIXME: visualise declining projections from categorised_spores to 2050 to the 'locked-in' capacity value in 2050 that corresponds to the choice of spores in categorised_spores

    return info_table

    # y = [max_capacity_2030]
    # y_2000_2021 = capacity_2000_2021
    # y_2021_2030 = projections[0].get("y1")
    # print(y_2000_2021)
    # print(y_2021_2030)
    # for i in range(len(years_2030_2050)-1):
    #     year_of_retired_capacity = i+categorised_spores-technology_life
    #     if year_of_retired_capacity <= 2021:
    #         y.append(y[-1] - (y_2000_2021[year_of_retired_capacity - 2000] - y_2000_2021[year_of_retired_capacity - 1 - 2000]))
    #         print(year_of_retired_capacity)
    #         print((y_2000_2021[year_of_retired_capacity - 2000] - y_2000_2021[year_of_retired_capacity - 1 - 2000]))
    #     else:
    #         y.append(y[-1] - (y_2021_2030[year_of_retired_capacity - 2021] - y_2021_2030[year_of_retired_capacity - 1 - 2021]))
    #         print(year_of_retired_capacity)
    #         print((y_2021_2030[year_of_retired_capacity - 2021] - y_2021_2030[year_of_retired_capacity - 1 - 2021]))
    #
    # print(y)

    # return projections, info_table


def find_highest_similarity_clusters(data):
    scaler = MinMaxScaler()
    all_techs = data.index.get_level_values("technology").unique()
    df_normalised = pd.DataFrame(
        scaler.fit_transform(data.unstack("technology")), columns=all_techs
    )

    # Find optimal number of clusters
    best_n_clusters = None
    best_score = -1  # Because silhouette score ranges between -1 and 1
    for n_clusters in range(2, 100):
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = agg_clustering.fit_predict(df_normalised)
        score = silhouette_score(df_normalised, cluster_labels)
        # print(f"For {n_clusters}, the silhouette score = {score}")
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters

    print(
        f"Optimal number of clusters = {best_n_clusters}, highest silhouette score = {best_score}"
    )

    # Cluster spores with optimal number of clusters
    clustering = AgglomerativeClustering(n_clusters=best_n_clusters).fit(df_normalised)

    # Make new DataFrame that contains the corresponding cluster number for each spore
    new_index = pd.MultiIndex.from_tuples(
        [(tech, spore, clustering.labels_[spore]) for tech, spore in data.index],
        names=["technology", "spore", "spore_cluster"],
    )
    data_with_cluster = pd.Series(data.array, index=new_index)

    return data_with_cluster


if __name__ == "__main__":
    print("test")
