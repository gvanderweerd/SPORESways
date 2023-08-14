import pandas as pd
import geopandas as gpd
import numpy as np
import os
import re
import shutil
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import seaborn.objects as so
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from main import *
from global_parameters import *
from processing_functions import *
from reading_functions import *

# Define open circles for plotting 'All other SPORES'
pts = np.linspace(0, np.pi * 2, 24)
circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
vert = np.r_[circ, circ[::-1] * 0.7]
open_circle = mpl.path.Path(vert)

YEARS = range(2010, 2051)
POWER_TECH_ORDER = [
    "PV",
    "Onshore wind",
    "Offshore wind",
    "CHP",
    "CCGT",
    "Nuclear",
    "Hydro",
    "Bio to liquids",
]
COLORS = {
    "All other SPORES": "#b9b9b9",
    "Nuclear": "#cc0000",
    "nuclear": "#cc0000",
    "ccgt": "#8fce00",
    "chp": "#ce7e00",
    "PV": "#ffd966",
    "pv": "#ffd966",
    "Onshore wind": "#674ea7",
    "onshore wind": "#674ea7",
    "Offshore wind": "#e062db",
    "offshore wind": "#e062db",
    "Hydro": "#2986cc",
    "hydro": "#2986cc",
}

if __name__ == "__main__":
    # MRQ:
    # "How do investment decisions affect the manoeuvring space of a climate neutral European energy system design in 2050?"

    """
    0. PREPARING AND READING DATA
    """
    path_to_processed_data = os.path.join(os.getcwd(), "..", "data", "processed")

    # Get SPORES power capacity in Europe for 2030 and 2050
    power_capacity = pd.read_csv(
        os.path.join(path_to_processed_data, "power_capacity.csv"),
        index_col=["year", "region", "technology", "spore"],
        squeeze=True,
    )
    print(power_capacity)
    #FIXME 0.6: obtain historical capacity data for 2020, 2021, 2022 for all technologies

    """
    1. ANALYSE AND VISUALISE "MANOEUVRING SPACE" OF categorised_spores AND 2050 (CLUSTERPLOTS + DEFINE CLUSTERS)
    """
    # SQ1:
    # "What are the key characteristics and trade-offs of future configurations of the European energy system?"

    #FIXME 1.1: plot clustermaps of categorised_spores and 2050

    #FIXME 1.2: decide on number of clusters (clustering algorithm)

    #FIXME 1.3: describe each cluster as a scenario

    #FIXME 1.4: plot scenarios in a capacity distribution plot (techs on x=axis, normalised capacitieson y-axis, SPORES to each scenario in color with a colored box around them)

    #FIXME 1.5: make some form of a decision tree to describe scenario's?

    """
    2. CONSTRUCT PATHWAYS BETWEEN PRESENT, categorised_spores, AND 2050
    """
    # SQ2:
    # "How can pathways of technology deployment decisions transform the European energy system of today to desirable future configurations?"
    #FIXME SQ2: consider reformulating SQ2. The answer to the question should be a table with pathways

    #FIXME 2.1: plot scenario's in time

    #FIXME 2.2: find a way to choose a number of pathways
    # - all possible combinations of categorised_spores scenario's and 2050 scenario's
    # - forward looking: pathways exactly through categorised_spores scenario's and extrapolated to 2050 scenario's
    # - backward looking: pathways exactly through 2050 scenario's and passing through multiple categorised_spores scenario's

    #FIXME 2.3: describe pathways in a table (pathways on the rows, technologies on the columns, deployment rates (CAGR & Nominal deployment rates)


    """
    3. SCORE PATHWAYS ON FLEXIBILITY IN MANOEUVERING SPACE --> THINK OF HOW THIS WILL FINALLY LEAD TO ANSWER MRQ 
    """
    # SQ3:
    # FIXME: ONCE WE GET HERE, DISCUSS HOW TO PROCEED IN A MEETING WITH FRANCESCO
