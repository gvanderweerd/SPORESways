import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from reading_functions import *
from processing_functions import *
from plotting_functions import *
from global_parameters import *


COLORS = {
    "Oil": "#5d5d5d",
    "Natural gas": "#b9b9b9",
    "Other fossils": "#181818",
    "Nuclear heat": "#cc0000",
    "Biofuels": "#8fce00",
    "PV": "#ffd966",
    "pv": "#ffd966",
    "Onshore wind": "#674ea7",
    "onshore wind": "#674ea7",
    "Offshore wind": "#e062db",
    "offshore wind": "#e062db",
    "Hydro": "#2986cc",
    "Waste": "#ce7e00",
    "Direct heat": "#f6b26b",
    "Electricity": "#2986cc",
}

def plot_transmission_expansion(
    ax_map,
    link_caps, link_cap_max,
    spore_num,
    units_m, colors, centroids,
):

    def _add_line(ax, point_from, point_to, color, lw, dashes=(None, None), linestyle='-'):
        ax.add_line(mpl.lines.Line2D(
            xdata=(point_from[0], point_to[0]), ydata=(point_from[1], point_to[1]),
            color=color, lw=lw,
            linestyle=linestyle, dashes=dashes
        ))

    def _add_links(ax, link_caps, cap_max):
        links_completed = []
        for link in link_caps.index:
            if sorted(link) in links_completed:  # every link comes up twice
                continue
            cap = link_caps.loc[link]
            point_from = (centroids.loc[link[0]].x, centroids.loc[link[0]].y)
            point_to = (centroids.loc[link[1]].x, centroids.loc[link[1]].y)
            _add_line(
                ax, point_from, point_to,
                color='#a9a9a999',
                lw=0.5
            )
            _add_line(
                ax, point_from, point_to,
                color='blue',
                lw= 5 * cap / cap_max
            )

            links_completed.append(sorted(link))

    # ax_map.set_xlim((2.6e6, 6.55e6))
    units_m.plot(ax=ax_map, facecolor=colors, ec="white", lw=0.5, legend=True)
    _add_links(ax_map, link_caps[spore_num], link_cap_max)

    ax_map.annotate(
        f"Transmission capacity expansion\n(Total: + {link_caps[spore_num].sum() / 10:.1f} TW)",
        fontweight="bold",
        xy=(0.5, 1.1),
        xycoords='axes fraction',
        horizontalalignment="center",
        fontsize="small"
    )
    ax_map.axis("off")


if __name__ == '__main__':

    power_capacity = pd.read_csv("../data/power_capacity.csv", index_col = ["year", "region", "technology", "carriers", "spore"], squeeze=True)
    grid_expansion = pd.read_csv("../data/grid_capacity_expansion.csv", index_col = ["importing_region", "exporting_region", "year", "spore"], squeeze=True)
    power_capacity_2000_2021 = pd.read_csv("../data/power_capacity_irenastat.csv", index_col = ["region", "technology", "year"], squeeze=True)



    # """
    # Linear and exponential projection
    # """
    # #FIXME: store this data in csv file and make a function in reading_functions.py that reads data necessary for growth projection
    # pv_annual_growth_linear_tw = {
    #     "Switzerland": 1,
    #     "Austria": 1,
    #     "Albania": 1,
    #     "Belgium": 1,
    #     "Bosnia": 1,
    #     "Bulgaria": 1,
    #     "Croatia": 1,
    #     "Cyprus": 1,
    #     "Czechia": 1,
    #     "Denmark": 1.5,     #Expected growth for 2022, obtained from EU Market Outlook for Solar Power (2022-2026)
    #     "Estonia": 1,
    #     "Finland": 1,
    #     "France": 2.7,      #Expected growth for 2022, obtained from EU Market Outlook for Solar Power (2022-2026)
    #     "Germany": 7.9,     #Expected growth for 2022, obtained from EU Market Outlook for Solar Power (2022-2026)
    #     "Great Britain": 1,
    #     "Greece": 1.4,      #Expected growth for 2022, obtained from EU Market Outlook for Solar Power (2022-2026)
    #     "Hungary": 1,
    #     "Iceland": 1,
    #     "Ireland": 1,
    #     "Italy": 2.6,       #Expected growth for 2022, obtained from EU Market Outlook for Solar Power (2022-2026)
    #     "Latvia": 1,
    #     "Lithuania": 1,
    #     "Luxembourg": 1,
    #     "Macedonia": 1,
    #     "Montenegro": 1,
    #     "Netherlands": 4,   #Expected growth for 2022, obtained from EU Market Outlook for Solar Power (2022-2026)
    #     "Norway": 1,
    #     "Poland": 4.9,      #Expected growth for 2022, obtained from EU Market Outlook for Solar Power (2022-2026)
    #     "Portugal": 2.5,    #Expected growth for 2022, obtained from EU Market Outlook for Solar Power (2022-2026)
    #     "Romania": 1,
    #     "Serbia": 1,
    #     "Slovakia": 1,
    #     "Slovenia": 1,
    #     "Spain": 7.5,       #Expected growth for 2022, obtained from EU Market Outlook for Solar Power (2022-2026)
    #     "Sweden": 1.1       #Expected growth for 2022, obtained from EU Market Outlook for Solar Power (2022-2026)
    # }
    # pv_growth_rates_exponential = {
    #     "Switzerland": 1.1,
    #     "Austria": 1.1,
    #     "Albania": 1.1,
    #     "Belgium": 1.13,        #GAGR estimated by EU Market Outlook for Solar Power (2022-2026) converted to growth factor
    #     "Bosnia": 1.1,
    #     "Bulgaria": 1.1,
    #     "Croatia": 1.1,
    #     "Cyprus": 1.1,
    #     "Czechia": 1.1,
    #     "Denmark": 1.1,
    #     "Estonia": 1.1,
    #     "Finland": 1.1,
    #     "France": 1.21,         #GAGR estimated by EU Market Outlook for Solar Power (2022-2026) converted to growth factor
    #     "Germany": 1.18,        #GAGR estimated by EU Market Outlook for Solar Power (2022-2026) converted to growth factor
    #     "Great Britain": 1.1,
    #     "Greece": 1.3,          #GAGR estimated by EU Market Outlook for Solar Power (2022-2026) converted to growth factor
    #     "Hungary": 1.23,        #GAGR estimated by EU Market Outlook for Solar Power (2022-2026) converted to growth factor
    #     "Iceland": 1.1,
    #     "Ireland": 1.9,         #GAGR estimated by EU Market Outlook for Solar Power (2022-2026) converted to growth factor
    #     "Italy": 1.17,          #GAGR estimated by EU Market Outlook for Solar Power (2022-2026) converted to growth factor
    #     "Latvia": 1.1,
    #     "Lithuania": 1.1,
    #     "Luxembourg": 1.1,
    #     "Macedonia": 1.1,
    #     "Montenegro": 1.1,
    #     "Netherlands": 1.2,     #GAGR estimated by EU Market Outlook for Solar Power (2022-2026) converted to growth factor
    #     "Norway": 1.1,
    #     "Poland": 1.29,         #GAGR estimated by EU Market Outlook for Solar Power (2022-2026) converted to growth factor
    #     "Portugal": 1.36,       #GAGR estimated by EU Market Outlook for Solar Power (2022-2026) converted to growth factor
    #     "Romania": 1.44,        #GAGR estimated by EU Market Outlook for Solar Power (2022-2026) converted to growth factor
    #     "Serbia": 1.1,
    #     "Slovakia": 1.1,
    #     "Slovenia": 1.1,
    #     "Spain": 1.31,          #GAGR estimated by EU Market Outlook for Solar Power (2022-2026) converted to growth factor
    #     "Sweden": 1.1
    # }
    # pv_annual_growth_linear_tw = pd.Series(pv_annual_growth_linear_tw)
    # pv_annual_growth_linear_tw = pv_annual_growth_linear_tw.rename_axis("region")
    # pv_growth_rates_exponential = pd.Series(pv_growth_rates_exponential)
    # pv_growth_rates_exponential = pv_growth_rates_exponential.rename_axis("region")
    #
    # # 1. Get projections
    # pv_projection_linear = energy_capacity_projection_linear(data, "PV", pv_annual_growth_linear_tw)
    # pv_projection_exp = energy_capacity_projection_exponential(data, "PV", pv_growth_rates_exponential)
    #
    # pv_cap_NL = power_capacity.xs(("PV", "Netherlands"), level=["technology", "region"])
    # pv_projection_NL = pv_projection_exp.xs("Netherlands", level="region")
    #
    # """
    # Plotting capacity distributions
    # """
    # fig, axs = plt.subplots(nrows=3, ncols=1)
    # plot_spores_capacity(axs[0], pv_cap_NL)
    # plot_capacity_distribution_projection(axs[1], pv_cap_NL, pv_projection_NL)
    # plot_capacity_distribution(axs[2], power_capacity.xs("Netherlands", level="region"))


    """
    Selecting top, bottom or middle spores
    """
    country_of_interest = "Netherlands"
    pv_NL_spores_2030 = get_national_quartile_spores(power_capacity, "PV", country_of_interest, 2030)

    bottom_pv_NL = pv_NL_spores_2030.get("bottom_spores")
    q2_pv_NL = pv_NL_spores_2030.get("quartile_2_spores")
    q3_pv_NL = pv_NL_spores_2030.get("quartile_3_spores")
    top_pv_NL = pv_NL_spores_2030.get("top_spores")


    """
    Get transmission capacity data
    """
    # grid_expansion = data.get("2050").get("grid_capacity_expansion")
    link_caps = (
        grid_expansion
        .unstack("spore")
        .groupby(level=["importing_region", "exporting_region"])
        .sum()
        .where(lambda x: x > 1e-5)
        .fillna(0)
    )
    link_cap_max = link_caps.max().max()
    country_regions = [region for region, country in REGION_MAPPING.items() if country == country_of_interest]

    grid_expansion_NL_gw = grid_expansion[grid_expansion.index.get_level_values("exporting_region").isin(country_regions)] * 1000
    neighbour_regions = list(grid_expansion_NL_gw.index.get_level_values("importing_region").unique())
    regions = list(set((country_regions + neighbour_regions)))


    """
    Plotting grid link map
    """
    units = gpd.read_file("../data/eurospores_units.geojson").set_index("id")
    # Transform geoemtries to new coordinate reference system
    units_m = units.to_crs("EPSG:3035")
    units_m = units_m[units_m.index.get_level_values("id").isin(regions)]

    #Color all regions of the country with a different colour than the other regions
    mask = units_m.index.get_level_values("id").isin(country_regions)
    colors = ["#add8e6" if mask else "#e8f4f8" for mask in mask]


    link_caps = link_caps[link_caps.index.get_level_values("importing_region").isin(country_regions)]
    fig, ax = plt.subplots(nrows=2, ncols=2)

    #Link caps for bottom and top PV in NL
    link_caps_bottom = link_caps[bottom_pv_NL]
    link_caps_top = link_caps[top_pv_NL]
    link_caps_min_max_bottom = link_caps_bottom.agg(["min", "max"], axis=1)
    link_caps_min_max_top = link_caps_top.agg(["min", "max"], axis=1)
    plot_transmission_expansion(ax_map=ax[0, 0], link_caps=link_caps_min_max_bottom, link_cap_max=link_cap_max, spore_num="min", units_m=units_m, colors=colors, centroids=units_m.centroid)
    plot_transmission_expansion(ax_map=ax[0, 1], link_caps=link_caps_min_max_bottom, link_cap_max=link_cap_max, spore_num="max", units_m=units_m, colors=colors, centroids=units_m.centroid)

    plt.show()