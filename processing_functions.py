import pandas as pd

# Import other scripts in this repository
from main import *

YEARS = range(2010, 2051)


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
            index_names = ["year", "region", "technology", "spore"]
        else:
            capacity = (
                get_energy_capacity_year(spores_data, year, technologies, carrier)
                .groupby(["year", "technology", "spore"])
                .sum()
            )
            index_names = ["year", "technology", "spore"]

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
    :return:                            A series with the projected capacity values for all years between 2020 and 2030
    """

    capacity_data = get_energy_capacity(
        data, ELECTRICITY_PRODUCERS, "electricity", "national"
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
    :return:                                A series with the projected capacity values for all years between 2020 and 2030
    """

    capacity_data = get_energy_capacity(
        data, ELECTRICITY_PRODUCERS, "electricity", "national"
    )
    capacity_2020 = capacity_data.loc["2020", :, technology, 0].droplevel("spore")
    capacity_2020 = pd.concat({"exponential": capacity_2020}, names=["method"])

    capacity_projected = capacity_2020

    # Calculate projected capacity for all years between 2020 and 2030 (YEARS = range(2010, 2051))
    for year in YEARS[10:20]:
        next_interval = capacity_2020.multiply(
            national_growth_exponential ** (year - 2020)
        ).dropna()
        next_interval = next_interval.rename({"2020": str(year)})
        capacity_projected = pd.concat([capacity_projected, next_interval])

    return capacity_projected


def get_national_quartile_spores(national_series, technology, region):
    """
    This function finds and returns the bottom 25 percentile of spores, the middle 50 percentile of spores, and the top 25 percentile of spores for the years 2030 and 2050. It does so for a given technology in a given region.

    :param national_series:     Input data as a multi-index series that contains values for each combination of indices: "year", "region", "technology"
    :param technology:          The technology for which to find the bottom, middle, and top spores
    :param region:              The region for which to find the bottom, middle, and top spores
    :return:                    Two dictionaries (one for 2030 and one for 2050) each containing "bottom_spores", "middle_spores", and "top_spores"
    """

    output = {}
    for year in ["2030", "2050"]:
        quartiles = (
            national_series.loc[year]
            .groupby(["region", "technology"])
            .quantile(q=[0.25, 0.5, 0.75])
        )
        q1, median, q3 = quartiles.loc[region, technology, :]

        s = national_series.loc[year, region, technology, :]
        mask_1 = s <= q1
        mask_2 = (s > q1) & (s <= q3)
        mask_3 = s > q3

        bottom_spores = s.index[mask_1].get_level_values("spore").tolist()
        interquartile_spores = s.index[mask_2].get_level_values("spore").tolist()
        top_spores = s.index[mask_3].get_level_values("spore").tolist()

        output[year] = {
            "bottom_spores": bottom_spores,
            "middle_spores": interquartile_spores,
            "top_spores": top_spores,
        }
    return output["2030"], output["2050"]


if __name__ == "__main__":
    print("test")
