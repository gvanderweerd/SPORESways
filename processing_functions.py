# Import other scripts in this repository
from reading_functions import *
from main import *


def get_energy_capacity_year(
    spores_data, year, technologies, carrier=None, spores=None, normalise=False
):
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
        capacity = capacity.groupby(
            [REGION_MAPPING, technologies, "carriers"],
            level=["region", "technology", "carriers"],
        )

    if carrier is not None:
        capacity = capacity.xs(carrier, level="carriers")

    capacity = pd.concat({year: capacity}, names=["year"])

    return capacity


def get_energy_capacity(data, technologies, carrier, resolution="continental"):
    output = pd.Series(dtype="float64")
    for year in data.keys():
        if resolution == "national":
            capacity = get_energy_capacity_year(data, year, technologies, carrier)
            index_names = ["year", "region", "technology", "spore"]
        else:
            capacity = (
                get_energy_capacity_year(data, year, technologies, carrier)
                .groupby(["year", "technology", "spore"])
                .sum()
            )
            index_names = ["year", "technology", "spore"]

        output = output.append(capacity)
        index = pd.MultiIndex.from_tuples(output.index, names=index_names)

    return pd.Series(output.array, index=index)


if __name__ == "__main__":
    print("test")
