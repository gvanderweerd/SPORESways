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
    "Coal": "#000000", # Black = #000000, Grey = #808080
    "coal": "#000000"
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
    "ev_flexibility": 7
}
metric_plot_names = {
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
metric_range_formatting = {
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
FIGWIDTH = 6.77165


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
        label_list[i]: (sns.color_palette("bright")[:-3] + sns.color_palette("bright")[-2:])[i]
        for i in range(len(label_list))
    }
    colors["rest_color"] = sns.color_palette("bright")[-3]
    return colors


if __name__ == '__main__':
    print("")