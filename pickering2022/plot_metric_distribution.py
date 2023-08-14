import os

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from friendly_data.converters import to_df
from frictionless.package import Package


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
    "Electricity": "#2986cc"
}
ENERGY_PRODUCERS = {
    "waste_supply": "Waste",
    "biofuel_supply": "Biofuels",
    "hydro_reservoir": "Hydro",
    "hydro_run_of_river": "Hydro",
    "nuclear": "Nuclear heat",
    "open_field_pv": "PV",
    "roof_mounted_pv": "PV",
    "wind_offshore": "Offshore wind",
    "wind_onshore": "Onshore wind",
}
NUCLEAR_HEAT_MUTIPLIER = 1 / 0.4  # our model uses an efficiency of 40% for nuclear

# QUESTION1: How to categorise 'battery' / 'hydrogen_storage' / 'pumped_hydro'
VARIABLE_RENEWABLE_TECHS = [
    'open_field_pv',
    'roof_mounted_pv',
    'wind_offshore',
    'wind_onshore'
]
FIRM_RENEWABLE_TECHS = [
    'ccgt',
    'chp_biofuel_extraction',
    'chp_methane_extraction',
    'chp_wte_back_pressure',
    'hydro_reservoir',
    'hydro_run_of_river',
    'nuclear'
]
HEAT_TECHS = [
    'biofuel_boiler',
    'chp_biofuel_extraction',
    'chp_methane_extraction',
    'chp_wte_back_pressure',
    'electric_heater',
    'hp',
    'methane_boiler'
]
SUPPLY_MAPPING = {
    'open_field_pv': 'pv',
    'roof_mounted_pv': 'pv',
    'wind_offshore': 'offshore wind',
    'wind_onshore': 'onshore wind',
    'ccgt': 'ccgt',
    'chp_biofuel_extraction': 'chp',
    'chp_methane_extraction': 'chp',
    'chp_wte_back_pressure': 'chp',
    'hydro_reservoir': 'hydro',
    'hydro_run_of_river': 'hydro',
    'nuclear': 'nuclear'
}
SUPPLY_TYPE_MAPPING = {
    'open_field_pv': 'Variable',
    'roof_mounted_pv': 'Variable',
    'wind_offshore': 'Variable',
    'wind_onshore': 'Variable',
    'ccgt': 'Firm',
    'chp_biofuel_extraction': 'Firm',
    'chp_methane_extraction': 'Firm',
    'chp_wte_back_pressure': 'Firm',
    'hydro_reservoir': 'Firm',
    'hydro_run_of_river': 'Firm',
    'nuclear': 'Firm'
}
RENEWABLES_MAPPING = {
    'open_field_pv': 'pv',
    'roof_mounted_pv': 'pv',
    'wind_offshore': 'offshore wind',
    'wind_onshore': 'onshore wind',
    'wind_onshore_monopoly': 'onshore wind',
    'biofuel_supply': 'biofuel_supply'
}
FIRM_SUPPLY_MAPPING = {
    "ccgt": "ccgt",
    "chp_biofuel_extraction": "chp",
    "chp_methane_extraction": "chp",
    "chp_wte_back_pressure": "chp",
    "nuclear": "nuclear",
    "hydro_reservoir": "hydro",
    "hydro_run_of_river": "hydro",
}
REGION_MAPPING = {
    'ALB_1': 'Albania',
    'AUT_1': 'Austria',
    'AUT_2': 'Austria',
    'AUT_3': 'Austria',
    'BEL_1': 'Belgium',
    'BGR_1': 'Bulgaria',
    'BIH_1': 'Bosnia and Herzegovina',
    'CHE_1': 'Switzerland',
    'CHE_2': 'Switzerland',
    'CYP_1': 'Cyprus',
    'CZE_1': 'Czechia',
    'CZE_2': 'Czechia',
    'DEU_1': 'Germany',
    'DEU_2': 'Germany',
    'DEU_3': 'Germany',
    'DEU_4': 'Germany',
    'DEU_5': 'Germany',
    'DEU_6': 'Germany',
    'DEU_7': 'Germany',
    'DNK_1': 'Denmark',
    'DNK_2': 'Denmark',
    'ESP_1': 'Spain',
    'ESP_10': 'Spain',
    'ESP_11': 'Spain',
    'ESP_2': 'Spain',
    'ESP_3': 'Spain',
    'ESP_4': 'Spain',
    'ESP_5': 'Spain',
    'ESP_6': 'Spain',
    'ESP_7': 'Spain',
    'ESP_8': 'Spain',
    'ESP_9': 'Spain',
    'EST_1': 'Estonia',
    'FIN_1': 'Finland',
    'FIN_2': 'France',
    'FRA_1': 'France',
    'FRA_10': 'France',
    'FRA_11': 'France',
    'FRA_12': 'France',
    'FRA_13': 'France',
    'FRA_14': 'France',
    'FRA_15': 'France',
    'FRA_2': 'France',
    'FRA_3': 'France',
    'FRA_4': 'France',
    'FRA_5': 'France',
    'FRA_6': 'France',
    'FRA_7': 'France',
    'FRA_8': 'France',
    'FRA_9': 'France',
    'GBR_1': 'Great Britain',
    'GBR_2': 'Great Britain',
    'GBR_3': 'Great Britain',
    'GBR_4': 'Great Britain',
    'GBR_5': 'Great Britain',
    'GBR_6': 'Great Britain',
    'GRC_1': 'Greece',
    'GRC_2': 'Greece',
    'HRV_1': 'Croatia',
    'HUN_1': 'Hungary',
    'IRL_1': 'Ireland',
    'ISL_1': 'Iceland',
    'ITA_1': 'Italy',
    'ITA_2': 'Italy',
    'ITA_3': 'Italy',
    'ITA_4': 'Italy',
    'ITA_5': 'Italy',
    'ITA_6': 'Italy',
    'LTU_1': 'Lithuania',
    'LUX_1': 'Luxembourg',
    'LVA_1': 'Latvia',
    'MKD_1': 'North Macedonia',
    'MNE_1': 'Montenegro',
    'NLD_1': 'Netherlands',
    'NOR_1': 'Norway',
    'NOR_2': 'Norway',
    'NOR_3': 'Norway',
    'NOR_4': 'Norway',
    'NOR_5': 'Norway',
    'NOR_6': 'Norway',
    'NOR_7': 'Norway',
    'POL_1': 'Poland',
    'POL_2': 'Poland',
    'POL_3': 'Poland',
    'POL_4': 'Poland',
    'POL_5': 'Poland',
    'PRT_1': 'Portugal',
    'PRT_2': 'Portugal',
    'ROU_1': 'Romania',
    'ROU_2': 'Romania',
    'ROU_3': 'Romania',
    'SRB_1': 'Serbia',
    'SVK_1': 'Slovakia',
    'SVN_1': 'Slovenia',
    'SWE_1': 'Sweden',
    'SWE_2': 'Sweden',
    'SWE_3': 'Sweden',
    'SWE_4': 'Sweden'
}

plt.rcParams.update({
    "svg.fonttype": 'none',
    'font.family':'sans-serif',
    'font.sans-serif':'Arial'
})
FIGWIDTH = 6.77165

result_dirs = [
    'slack-5',
    'slack-15',
    'slack-10',
    'cost-opt',
    'slack-10-demand-update',
    "cost-opt-demand-update"
]
plot_order = {
    "curtailment": 2,
    "electricity_production_gini": 5,
    "average_national_import": 4,
    "fuel_autarky_gini": 6,
    "storage_discharge_capacity": 1,
    "transport_electrification": 9,
    "heat_electrification": 8,
    "biofuel_utilisation": 3,
    "ev_flexibility": 7
}

plot_names = {
    "curtailment": 'Curtailment',
    "electricity_production_gini":  'Electricity production\nGini coefficient',
    "average_national_import": 'Average\nnational import',
    "fuel_autarky_gini": 'Fuel autarky\nGini coefficient',
    "storage_discharge_capacity": 'Storage discharge\ncapacity',
    "transport_electrification": 'Transport\nelectrification',
    "heat_electrification": 'Heat\nelectrification',
    "biofuel_utilisation": 'Biofuel\nutilisation',
    "ev_flexibility": 'EV as flexibility'
}
# FIXME what does this do?
min_max_range_formatting = {
    "curtailment": lambda x: x.round(0).astype(int),
    "electricity_production_gini": lambda x: x.round(2),
    "average_national_import": lambda x: x.round(0).astype(int),
    "fuel_autarky_gini": lambda x: x.round(2),
    "storage_discharge_capacity": lambda x: x.round(0).astype(int) if x.name == "max" else x.round(2),
    "transport_electrification": lambda x: x.round(0).astype(int),
    "heat_electrification": lambda x: x.round(0).astype(int),
    "biofuel_utilisation": lambda x: x.round(0).astype(int),
    "ev_flexibility": lambda x: x.round(2)
}

def read_data(path_to_friendly_data, filenames):
    dpkg_filepath = os.path.join(path_to_friendly_data, "datapackage.json")
    spores_datapackage = Package(dpkg_filepath)

    spores_data = {}

    for resource in spores_datapackage["resources"]:
        if resource['name'] in filenames:
            spores_data[resource['name']] = pd.read_csv(
                os.path.join(path_to_friendly_data, resource['path']),
                index_col=resource['schema']['primaryKey']).squeeze(1)

    return spores_data

def get_cap_df(path_to_friendly_data):
    spores_data = read_data(path_to_friendly_data, ['nameplate_capacity'])

    df = (
        spores_data['nameplate_capacity']
        .xs('tw', level='unit')
        .xs('electricity', level='carriers')
        .unstack('spore')
    )
    continental_cap = (
        df
        .groupby(SUPPLY_MAPPING, level='techs')
        .sum()
        .stack('spore')
        .rename('capacity').to_frame().reset_index()
    )
    national_cap = (
        df
        .groupby([REGION_MAPPING, SUPPLY_MAPPING], level=['locs', 'techs'])
        .sum()
        .stack('spore')
        .reorder_levels(['spore', 'locs', 'techs']).sort_index()
        .rename('capacity').to_frame().reset_index()
    )
    regional_cap = (
        df
        .groupby(['locs', SUPPLY_MAPPING], level=['locs', 'techs'])
        .sum()
        .stack('spore')
        .reorder_levels(['spore', 'locs', 'techs']).sort_index()
        .rename('capacity').to_frame().reset_index()
    )
    return continental_cap, national_cap

def get_best_performing_spores(metric_series, metric, percent_deviation=10, normalised=False):
    """
    Return SPORES that sit within X% of the lowest value across all SPORES.
    If normalised is True, take X% to be the percentage point difference, not the percentage difference.
    """
    metric_data = metric_series[metric_series.index.get_level_values("metric") == metric]
    min_val = metric_data.min()
    if normalised:
        max_val = min_val + percent_deviation / 100
    else:
        max_val = min_val * (1 + percent_deviation / 100)
    return metric_data[metric_data <= max_val]

def reorganise_metrics(df):
    return df.iloc[df['metric'].map(plot_order).argsort()]

def get_metric_plot_df(path_to_friendly_data, normalise=True, percentage_best=10):
    data_dict = read_data(path_to_friendly_data, ['paper_metrics'])

    all_metric_array = []
    for metric in data_dict["paper_metrics"].index.get_level_values("metric").unique():
        if normalise:
            normalised_series = data_dict["paper_metrics"].div(data_dict["paper_metrics"].max(level="metric"))
            best_spores = get_best_performing_spores(
                normalised_series, metric, percentage_best, normalised=True
            ).index
            df = normalised_series[normalised_series.index.get_level_values("metric") == metric].to_frame("Normalised metric score")
        else:
            best_spores = get_best_performing_spores(
                data_dict["paper_metrics"],
                metric, percentage_best
            ).index
            df = data_dict["paper_metrics"][data_dict["paper_metrics"].index.get_level_values("metric") == metric].to_frame("Metric score")
        df = df.assign(best="All other SPORES")
        df.loc[best_spores, "best"] = "'Best' Performing SPORES"

        all_metric_array.append(df.reset_index())
    return reorganise_metrics(pd.concat(all_metric_array))

def plot_metric_spores(metric_df, focus_metric, colours, ax, focus_metric_alpha=0.5, incl_15pp_boxes=True):
    data_dict = read_data(path_to_friendly_data, ['paper_metrics'])
    baseline_metric_ranges = data_dict['paper_metrics'].groupby(level='metric').agg(["min", "max"])

    if "Normalised metric score" in metric_df.columns:
        y = "Normalised metric score"
    else:
        y = "Metric score"
    metrics = metric_df.metric.unique()
    metric_units = metric_df[["metric", "unit"]].drop_duplicates().set_index("metric")
    print("METRIC")
    print(metric_df)
    spores_to_plot_in_color = metric_df[
        (metric_df.metric == focus_metric) & (metric_df.best != "All other SPORES")
        ].spore.values
    print('spores to plot in color:')
    print(spores_to_plot_in_color)

    pts = np.linspace(0, np.pi * 2, 24)
    circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
    vert = np.r_[circ, circ[::-1] * .7]
    open_circle = mpl.path.Path(vert)

    sns.stripplot(
        data=metric_df[~metric_df.spore.isin(spores_to_plot_in_color)], x="metric", y=y,
        color=colours["All other SPORES"], marker=open_circle, alpha=0.5, ax=ax, s=3
    )

    max_best = metric_df.loc[
        (metric_df.best != "All other SPORES"), [y, "metric"]
    ].groupby("metric").max().squeeze().reindex(metrics)
    min_best = metric_df.loc[
        (metric_df.best != "All other SPORES"), [y, "metric"]
    ].groupby("metric").min().squeeze().reindex(metrics)

    print(max_best)
    print(min_best)

    if incl_15pp_boxes:
        _x = 0
        for idx in max_best.index:
            xmin = _x - 0.2
            xmax = _x + 0.2
            height = max_best.loc[idx] - min_best.loc[idx]
            height += 0.018
            ax.add_patch(
                mpl.patches.Rectangle(
                    xy=(xmin, min_best.loc[idx] - 0.009),
                    height=height,
                    width=(xmax - xmin),
                    fc="None",
                    ec=colours[idx],
                    linestyle="--",
                    lw=0.75,

                    zorder=10
                ),
            )
            _x += 1

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=10)
    xticklabels = []

    for ticklabel in ax.get_xticklabels():
        _metric = ticklabel.get_text()
        metric_range = baseline_metric_ranges.apply(min_max_range_formatting[_metric]).loc[_metric]
        _unit = metric_units.loc[_metric].item()
        if _unit == "percentage":
            _unit = " %"
        elif _unit == "fraction":
            _unit = ""
        else:
            _unit = " " + _unit
        xticklabels.append(
            f"{plot_names[_metric]}\n({metric_range['min']} - {metric_range['max']}){_unit}"
        )

    if focus_metric is not None:
        sns.stripplot(
            data=metric_df[metric_df.spore.isin(spores_to_plot_in_color)],
            x="metric", y=y, alpha=focus_metric_alpha, ax=ax, marker="o", color=colours[focus_metric],
            s=3
        )

    ax.set_xticklabels(xticklabels, fontsize=6)
    handles = {}
    handles["other"] = mpl.lines.Line2D(
        [0], [0],
        marker=open_circle,
        color="w",
        markerfacecolor=colours["All other SPORES"],
        markeredgecolor=colours["All other SPORES"],
        label='All other SPORES',
        markersize=4
    )
    if incl_15pp_boxes:
        handles["best"] = mpl.patches.Rectangle(
            xy=(xmin, min_best.loc[idx] - 0.005),
            height=height,
            width=(xmax - xmin),
            fc="None",
            ec="black",
            linestyle="--",
            lw=0.75,
            label='SPORE +15pp range',

        )  #
    if focus_metric is not None:
        handles["linked_spores"] = mpl.lines.Line2D(
            [0], [0],
            marker='o',
            color="w",
            markerfacecolor=colours[focus_metric],
            label=f"SPORES linked to {plot_names[focus_metric].lower()} +15pp range",
            markersize=5
        )

    ax.legend(handles=handles.values(), frameon=False, loc="lower right", bbox_to_anchor=(0.8, 0))
    sns.despine(ax=ax)


def get_lowest_cap_spores(cap_series, tech, percent_deviation=10, normalised=False):
    """
    Return SPORES that sit within 10% of the lowest value across all SPORES.
    """
    metric_data = cap_series[cap_series.index.get_level_values('techs') == tech]
    lower_bound = metric_data.min()
    if normalised:
        upper_bound = lower_bound + percent_deviation / 100
    else:
        upper_bound = lower_bound * (1 + percent_deviation / 100)
    return metric_data[metric_data <= upper_bound]

def get_highest_cap_spores(cap_series, tech, percent_deviation=10, normalised=False):
    """
    Return SPORES that sit within 10% of the highest value across all SPORES.
    """
    metric_data = cap_series[cap_series.index.get_level_values('techs') == tech]
    upper_bound = metric_data.max()
    if normalised:
        lower_bound = upper_bound - percent_deviation / 100
    else:
        lower_bound = upper_bound * (1 - percent_deviation / 100)
    return metric_data[metric_data >= lower_bound]

def get_low_high_cap_spores(cap_series, tech, percent_deviation=10, normalised=False):
    """
    Return SPORES that sit within 10% of the lowest value, and within 10% of the highest value across all SPORES.
    """
    cap_data = cap_series[cap_series.index.get_level_values('techs') == tech]
    lowest_cap = cap_data.min()
    highest_cap = cap_data.max()

    if normalised:
        upper_bound = lowest_cap + percent_deviation / 100
        lower_bound = highest_cap - percent_deviation / 100
    else:
        upper_bound = lowest_cap * (1 + percent_deviation / 100)
        lower_bound = highest_cap * (1 - percent_deviation / 100)

    low_cap_spores = cap_data[cap_data <= upper_bound]
    high_cap_spores = cap_data[cap_data >= lower_bound]
    return low_cap_spores, high_cap_spores

def get_euro_capacity_plot_df(path_to_friendly_data, normalise=True, percentage_extremes=30):
    data_dict = read_data(path_to_friendly_data, ['nameplate_capacity'])

    s = (
        data_dict['nameplate_capacity']
        .xs('tw', level='unit')
        .xs('electricity', level='carriers')
        .unstack('spore')
        .groupby([SUPPLY_MAPPING], level='techs').sum()
        .stack('spore')
    )

    all_metric_array = []
    for tech in s.index.get_level_values('techs').unique():
        if normalise:
            s_normalised = s.div(s.groupby('techs').max())
            low_cap, high_cap = get_low_high_cap_spores(s_normalised, tech, percentage_extremes, normalised=True)
            df = s_normalised[s_normalised.index.get_level_values('techs') == tech].to_frame('Normalised capacity')
        else:
            low_cap, high_cap = get_low_high_cap_spores(s_normalised, tech, percentage_extremes)
            df = s[s.index.get_level_values('techs') == tech].to_frame('Capacity')
        df = df.assign(type="All other SPORES")
        df.loc[low_cap.index, 'type'] = 'Lowest Capacity SPORES'
        df.loc[high_cap.index, 'type'] = 'Highest Capacity SPORES'

        all_metric_array.append(df.reset_index())

    all_metric_array = pd.concat(all_metric_array)
    all_metric_array.loc[:, 'unit'] = 'tw'

    return all_metric_array

def get_tech_capacity_plot_df(path_to_friendly_data, normalise=True, percentage_extremes=10):
    data_dict = read_data(path_to_friendly_data, ['nameplate_capacity'])

    data = (
        data_dict['nameplate_capacity']
        .xs('tw', level='unit')
        .xs('electricity', level='carriers')
        .unstack('spore')
        .groupby([REGION_MAPPING, SUPPLY_MAPPING], level=['locs', 'techs']).sum()
        .stack('spore')
    )

    all_metric_array = []
    country_tech_combinations = []
    for country in data.index.get_level_values('locs').unique():
        s = data.xs(country, level='locs')
        for tech in s.index.get_level_values('techs').unique():
            country_tech_combinations.append(country + '_' + tech)
            if normalise:
                s_normalised = s.div(s.groupby('techs').max())
                low_cap, high_cap = get_low_high_cap_spores(s_normalised, tech, percentage_extremes, normalised=True)
                df = s_normalised[s_normalised.index.get_level_values('techs') == tech].to_frame('Normalised capacity')
            else:
                low_cap, high_cap = get_low_high_cap_spores(s_normalised, tech, percentage_extremes)
                df = s[s.index.get_level_values('techs') == tech].to_frame('Capacity')
            df = df.assign(type="All other SPORES")
            df.loc[low_cap.index, 'type'] = 'Lowest Capacity SPORES'
            df.loc[high_cap.index, 'type'] = 'Highest Capacity SPORES'
            df.loc[:, 'locs'] = country
            all_metric_array.append(df.reset_index())

    all_metric_array = pd.concat(all_metric_array)
    all_metric_array.loc[:, 'unit'] = 'tw'
    return all_metric_array

    # All metric array is 89523 rows long because there exist 203 out of 245 technology country combinations. Check the missing combinations with the code below:
    # missing_combinations = []
    # for country in data.index.get_level_values('locs').unique():
    #     for tech in data.index.get_level_values('techs').unique():
    #         if country + '_' + tech not in country_tech_combinations:
    #             missing_combinations.append(country + '_' + tech)
    # print(missing_combinations)
    # print(len(missing_combinations))


    return all_metric_array




def plot_euro_cap_distribution(cap_df_eu, cap_df_countries, focus_tech, colours, ax, alpha=0.5, incl_15pp_boxes=True, upper_15pp_boxes=True):
    data_dict = read_data(path_to_friendly_data, ['nameplate_capacity'])

    # Determine the range of capacities for each electricity producing technology
    cap_ranges = (
        data_dict['nameplate_capacity']
        .xs('tw', level='unit')
        .xs('electricity', level='carriers')
        .groupby([SUPPLY_MAPPING], level='techs').agg(["min", "max"])
    )
    if 'Normalised capacity' in cap_df_eu.columns:
        y = 'Normalised capacity'
    else:
        y = 'Capacity'

    # Define technologies and corresponding units
    techs = cap_df_eu.techs.unique()
    tech_units = cap_df_eu[['techs', 'unit']].drop_duplicates().set_index('techs')

    pts = np.linspace(0, np.pi * 2, 24)
    circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
    vert = np.r_[circ, circ[::-1] * .7]
    open_circle = mpl.path.Path(vert)

    if upper_15pp_boxes:
        spore_type = 'Highest Capacity SPORES'
    else:
        spore_type = 'Lowest Capacity SPORES'

    max_best = cap_df_eu.loc[
        (cap_df_eu.type == spore_type), [y, 'techs']
    ].groupby('techs').max().squeeze().reindex(techs)
    min_best = cap_df_eu.loc[
        (cap_df_eu.type == spore_type), [y, 'techs']
    ].groupby('techs').min().squeeze().reindex(techs)

    if incl_15pp_boxes:
        _x = 0
        for idx in max_best.index:
            xmin = _x - 0.2
            xmax = _x + 0.2
            height = max_best.loc[idx] - min_best.loc[idx]
            height += 0.018
            ax.add_patch(
                mpl.patches.Rectangle(
                    xy=(xmin, min_best.loc[idx] - 0.009),
                    height=height,
                    width=(xmax - xmin),
                    fc="None",
                    ec=colours[idx],
                    linestyle="--",
                    lw=0.75,

                    zorder=10
                ),
            )
            _x += 1

    #FIXME: insert the part of the function that makes the labels for the x-axis



    # Determine SPORES that need to be plotted in colour
    spores_to_plot_in_color = cap_df_eu[
        (cap_df_eu.techs == focus_tech) & (cap_df_eu.type == spore_type)
    ].spore.values

    # Plot capacities in grey for technologies in Europe
    sns.stripplot(
        data=cap_df_eu[~cap_df_eu.spore.isin(spores_to_plot_in_color)], x='techs', y=y,
        color=colours["All other SPORES"], marker=open_circle, alpha=0.5, ax=ax, s=3
    )
    # Plot capacities in colour for technologies in Europe
    if focus_tech is not None:
        sns.stripplot(data=cap_df_eu[cap_df_eu.spore.isin(spores_to_plot_in_color)], x='techs', y=y, color=colours[focus_tech], s=3)



    # # Plot capacities in grey for focus technology in all Countries
    # df = cap_df_countries[cap_df_countries['techs'] == focus_tech]
    #
    #
    # sns.stripplot(
    #     data=df[~df.spore.isin(spores_to_plot_in_color)], x='locs', y=y,
    #     color=colours["All other SPORES"], marker=open_circle, alpha=0.5, ax=ax, s=3
    # )
    # # Plot capacities in color for focus technology in all Countries
    # if focus_tech is not None:
    #     sns.stripplot(data=df[df.spore.isin(spores_to_plot_in_color)], x='locs', y=y, color=colours[focus_tech], s=3)
    #
    # # plt.show()
    #
    #
    # # plt.show(block=False)



def make_plot(
    path_to_friendly_data
):
    # Get normalised capacity DataFrame for all technologies on continental scale
    cap_df_eu = get_euro_capacity_plot_df(path_to_friendly_data, normalise=True, percentage_extremes=10)
    # Get normalised capacity DataFrame for all technologies on national scale
    cap_df_countries = get_tech_capacity_plot_df(path_to_friendly_data, normalise=True, percentage_extremes=10)


    nrows = 2

    with sns.plotting_context("paper", font_scale=1.5):
        ax = {}
        fig = plt.figure(figsize=(20, 2 * 7 + 4))
        gs = mpl.gridspec.GridSpec(
            nrows=nrows,
            ncols=1,
            figure=fig,
            hspace=0.1,
            wspace=0.1,
        )

if __name__ == "__main__":
    path_to_friendly_data = '../data/raw/euro-spores-results-2050/aggregated-slack-10/'
    path_to_units = '../data/raw/eurospores_units.geojson'
    path_to_unit_groups = '../data/raw/plotting_unit_groups.csv'
    path_to_output = '../figures/'

    # eu_caps, nat_caps = get_cap_df(path_to_friendly_data)



    # Get capacity DataFrame
    cap_df = get_euro_capacity_plot_df(path_to_friendly_data)
    tech_cap_df = get_tech_capacity_plot_df(path_to_friendly_data)

    techs = cap_df.techs.unique()
    # Define colours
    tech_colours = {
        techs[i]: (sns.color_palette("bright")[:-3] + sns.color_palette("bright")[-2:])[i]
        for i in range(len(techs))
    }
    tech_colours["All other SPORES"] = sns.color_palette("bright")[-3]
    print(tech_colours)

    # Plot function: capacity distribution in Europe
    # fig, ax = plt.subplots(1, 1, figsize=(FIGWIDTH, 10 * FIGWIDTH / 20))
    # plot_euro_cap_distribution(cap_df, 'offshore wind', tech_colours, ax=ax, upper_15pp_boxes=True)

    # Plot function: capacity distribution in Europe AND offshore wind capacities in different countries
    fig, ax = plt.subplots(1, 1, figsize=(FIGWIDTH, 10 * FIGWIDTH / 20))
    plot_euro_cap_distribution(cap_df, tech_cap_df, 'chp', tech_colours, ax=ax, upper_15pp_boxes=True)


    # calculate capacity ranges for variability score of technology deployment
    norm_ranges1 = cap_df.groupby('techs').agg({'Normalised capacity': ['min', 'max']})
    norm_ranges = cap_df.groupby('techs')['Normalised capacity'].agg(['min', 'max'])
    norm_ranges.loc[:, 'range'] = norm_ranges['max'] - norm_ranges['min']
    score = norm_ranges['range'].sum()

    # Inspect dataframe
    # print(cap_df)
    # pd.set_option('display.max_rows', None)
    # print(cap_df[cap_df['techs'] == 'offshore wind'])


    # Find 10 countries with biggest offshore wind deployment




    # prod_sum = prod_sum.where(prod_sum.div(prod_sum.sum()) > 0.01).dropna(how="all")








    # Metrics example

    # Get metrics DataFrame
    metric_df = get_metric_plot_df(path_to_friendly_data)
    # Define metric colours
    metrics = metric_df.metric.unique()
    metric_colours = {
        metrics[i]: (sns.color_palette("bright")[:-3] + sns.color_palette("bright")[-2:])[i]
        for i in range(len(metrics))
    }
    metric_colours["All other SPORES"] = sns.color_palette("bright")[-3]
    # Choose metric to be plotted

    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0.06, right=1, top=1)
    np.random.seed(123)

    for metric in metric_df.metric.unique():
        print(metric)
        fig, ax = plt.subplots(1, 1, figsize=(FIGWIDTH, 10 * FIGWIDTH / 20))
        plot_metric_spores(metric_df, metric, metric_colours, ax)
        ax.set_xlabel("")
        ax.set_ylabel("Metric score (scaled per metric to maximum across all SPORES)", weight="bold")
        # fig.savefig(f"/output/figures/spores_metric_scores_{metric}_focus.pdf", bbox_inches="tight", dpi=300)
        # filename = str(metric)
        # path = 'output/figures/' + filename + '.png'
        # print(path)
        # fig.savefig(f"output/spores_metric_scores_{metric}_focus.pdf", bbox_inches='tight', dpi=300)
    plt.show()