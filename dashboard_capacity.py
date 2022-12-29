import os
import dash
import plotly.express as px
import pandas as pd
import numpy as np

from dash import dcc
from dash import html
from reading_functions import read_spores_data
from processing_functions import get_energy_capacity_year

# Defines the country to which each region belongs
REGION_MAPPING = {
    "ALB_1": "Albania",
    "AUT_1": "Austria",
    "AUT_2": "Austria",
    "AUT_3": "Austria",
    "BEL_1": "Belgium",
    "BGR_1": "Bulgaria",
    "BIH_1": "Bosnia",  #'Bosniaand\nHerzegovina'
    "CHE_1": "Switzerland",
    "CHE_2": "Switzerland",
    "CYP_1": "Cyprus",
    "CZE_1": "Czechia",
    "CZE_2": "Czechia",
    "DEU_1": "Germany",
    "DEU_2": "Germany",
    "DEU_3": "Germany",
    "DEU_4": "Germany",
    "DEU_5": "Germany",
    "DEU_6": "Germany",
    "DEU_7": "Germany",
    "DNK_1": "Denmark",
    "DNK_2": "Denmark",
    "ESP_1": "Spain",
    "ESP_10": "Spain",
    "ESP_11": "Spain",
    "ESP_2": "Spain",
    "ESP_3": "Spain",
    "ESP_4": "Spain",
    "ESP_5": "Spain",
    "ESP_6": "Spain",
    "ESP_7": "Spain",
    "ESP_8": "Spain",
    "ESP_9": "Spain",
    "EST_1": "Estonia",
    "FIN_1": "Finland",
    "FIN_2": "France",
    "FRA_1": "France",
    "FRA_10": "France",
    "FRA_11": "France",
    "FRA_12": "France",
    "FRA_13": "France",
    "FRA_14": "France",
    "FRA_15": "France",
    "FRA_2": "France",
    "FRA_3": "France",
    "FRA_4": "France",
    "FRA_5": "France",
    "FRA_6": "France",
    "FRA_7": "France",
    "FRA_8": "France",
    "FRA_9": "France",
    "GBR_1": "Great Britain",
    "GBR_2": "Great Britain",
    "GBR_3": "Great Britain",
    "GBR_4": "Great Britain",
    "GBR_5": "Great Britain",
    "GBR_6": "Great Britain",
    "GRC_1": "Greece",
    "GRC_2": "Greece",
    "HRV_1": "Croatia",
    "HUN_1": "Hungary",
    "IRL_1": "Ireland",
    "ISL_1": "Iceland",
    "ITA_1": "Italy",
    "ITA_2": "Italy",
    "ITA_3": "Italy",
    "ITA_4": "Italy",
    "ITA_5": "Italy",
    "ITA_6": "Italy",
    "LTU_1": "Lithuania",
    "LUX_1": "Luxembourg",
    "LVA_1": "Latvia",
    "MKD_1": "Macedonia",  #'North Macedonia'
    "MNE_1": "Montenegro",
    "NLD_1": "Netherlands",
    "NOR_1": "Norway",
    "NOR_2": "Norway",
    "NOR_3": "Norway",
    "NOR_4": "Norway",
    "NOR_5": "Norway",
    "NOR_6": "Norway",
    "NOR_7": "Norway",
    "POL_1": "Poland",
    "POL_2": "Poland",
    "POL_3": "Poland",
    "POL_4": "Poland",
    "POL_5": "Poland",
    "PRT_1": "Portugal",
    "PRT_2": "Portugal",
    "ROU_1": "Romania",
    "ROU_2": "Romania",
    "ROU_3": "Romania",
    "SRB_1": "Serbia",
    "SVK_1": "Slovakia",
    "SVN_1": "Slovenia",
    "SWE_1": "Sweden",
    "SWE_2": "Sweden",
    "SWE_3": "Sweden",
    "SWE_4": "Sweden",
}
COUNTRIES = np.unique(list(REGION_MAPPING.values()))
YEARS = ["2030", "2050"]

# Defines the technology family to which each technology belongs for technologies in the power sector
ELECTRICITY_PRODUCERS = {
    "open_field_pv": "PV",
    "roof_mounted_pv": "PV",
    "wind_offshore": "Offshore wind",
    "wind_onshore": "Onshore wind",
    "ccgt": "CCGT",
    "chp_biofuel_extraction": "CHP",
    "chp_methane_extraction": "CHP",
    "chp_wte_back_pressure": "CHP",
    "hydro_reservoir": "Hydro",
    "hydro_run_of_river": "Hydro",
    "nuclear": "Nuclear",
    "biofuel_to_liquids": "Bio to liquids",
}
# Defines the technology family to which each technology belongs for technologies in the heat sector
HEAT_PRODUCERS = {
    "biofuel_boiler": "Boiler",
    "chp_biofuel_extraction": "CHP",
    "chp_methane_extraction": "CHP",
    "chp_wte_back_pressure": "CHP",
    "electric_heater": "Electric heater",
    "hp": "Heat pump",
    "methane_boiler": "Boiler",
}
SECTORS = ["Power", "Heat"]

"""
Obtaining data
"""
# Define to spores data
paths = {
    "2050": os.path.join(os.getcwd(), "data", "euro-spores-results-v2022-05-13"),
}
# Define for which cost relaxation we want to read the data
slack = "slack-10"
# Define which files we want to read
files = ["nameplate_capacity"]
# Reading data
data = {
    "2050": read_spores_data(paths["2050"], slack, files),
}

year = "2050"
s_heat = get_energy_capacity_year(
    spores_data=data,
    year=year,
    technologies=HEAT_PRODUCERS,
    carrier="heat",
    normalise=True,
)
s_heat.name = "capacity"
df_heat = s_heat.to_frame().reset_index()
s_power = get_energy_capacity_year(
    spores_data=data,
    year=year,
    technologies=ELECTRICITY_PRODUCERS,
    carrier="electricity",
    normalise=True,
)
s_power.name = "capacity"
df_power = s_power.to_frame().reset_index()

# Initialize the app
app = dash.Dash()

# Define the layout of the app
app.layout = html.Div(
    [
        # Add a dropdown menu for selecting the region
        html.Div(
            [
                dcc.Dropdown(
                    id="region-dropdown",
                    options=[
                        {"label": region, "value": region} for region in COUNTRIES
                    ],
                    value=COUNTRIES[0],
                )
            ]
        ),
        # Add a dropdown menu for selecting the sector
        html.Div(
            [
                dcc.Dropdown(
                    id="sector-dropdown",
                    options=[{"label": sector, "value": sector} for sector in SECTORS],
                    value=SECTORS[0],
                )
            ]
        ),
        # Add a dropdown menu for selecting the year
        html.Div(
            [
                dcc.Dropdown(
                    id="year-dropdown",
                    options=[{"label": year, "value": year} for year in YEARS],
                    value=YEARS[1],
                )
            ]
        ),
        html.Div(
            [
                # Add a Graph component for displaying the stripplot
                dcc.Graph(id="stripplot", style={"width": "50%"}),
                # Add a container for the sliders
                html.Div(id="sliders")
            ]
        )
    ]
)

# Define the callback function for updating the stripplot
@app.callback(
    dash.dependencies.Output("stripplot", "figure"),
    [
        dash.dependencies.Input("region-dropdown", "value"),
        dash.dependencies.Input("sector-dropdown", "value"),
        dash.dependencies.Input("year-dropdown", "value"),
    ],
)
def update_stripplot(region, sector, year):
    if sector == "Power":
        df = df_power
    else:
        df = df_heat

    # Filter the dataframe by year and region
    df_filtered = df[(df["year"] == year) & (df["region"] == region)].copy()
    # Create the stripplot
    fig = px.strip(df_filtered, x="technology", y="capacity")
    fig.update_layout(yaxis_title="Capacity [GW]")

    # Add a slider for each technology
    for technology in df_filtered["technology"].unique():
        print(technology)
        app.layout.children.append(html.Div([
            dcc.RangeSlider(
                id=f"slider-{technology}",
                min=0,
                max=1,
                step=0.01,
                value=[0, 1],
                vertical=False
            )
        ]))

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
