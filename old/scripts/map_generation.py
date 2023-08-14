# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:55:55 2022

@author: sanvi
"""
#%% import libraries
import pandas as pd
import geopandas as gpd
import json
import plotly.express as px

#%% import parameters

scenario_list = ([ '2050_agg_unc_2h_nolog', '2050_agg_unc_2h_log', '2050_agg_v1g_2h_costhigh', '2050_agg_v1g_2h_costlow',
                  '2050_agg_v2g_2h_costhigh', '2050_agg_v2g_2h_costlow', '2050_agg_v2g_2h_costlow_conn2',
                  '2050_agg_v2g_2h_costlow_conn3', '2050_agg_v2g_2h_costpar' ])

nord_list = (['NORD','R1','R2','R3','R4','R5','R6','R7','R8'])
cnor_list = (['CNORD','R9','R10','R11'])
csud_list = (['CSUD','R12','R13','R14'])
sud_list = (['SUD','R15','R16','R17','R18'])
border_countries = (['AT','CH','FR','GR','SI'])
single_reg = (['AT','CH','FR','GR','SI', 'SICI', 'SARD'])

dict = {'NORD': ['NORD','R1','R2','R3','R4','R5','R6','R7','R8'],
        'CNOR': ['CNOR','R9','R10','R11'],
        'CSUD': ['CSUD','R12','R13','R14'],
        'SUD': ['SUD','R15','R16','R17','R18'],
        'SICI': 'SICI',
        'SARD': 'SARD',
        'AT': 'AT',
        'CH': 'CH',
        'FR': 'FR',
        'GR': 'GR',
        'SI': 'SI'        
        }

sub_region_list = (['AT','CH','FR','GR','SI', 'NORD', 'CNOR', 'CSUD', 'SUD', 'SICI', 'SARD' ])

#%% dataframe parsing

name = '2050_agg_v1g_2h_costhigh'
folder = f'Res_{name}'
file = f'capacities_{name}.csv'

for name in scenario_list:
    dff = pd.DataFrame(columns=df_cap.columns)
    folder = f'Res_{name}'
    file = f'capacities_{name}.csv'
    
    df_cap = pd.read_csv('{}\{}'.format(folder,file), index_col = 0).fillna(0)
    tot_cap = df_cap.sum()
    tot_cap.name = 'tot cap'
    df_cap.loc['Total'] = tot_cap.transpose()
    
    for sub_r in sub_region_list:
        if sub_r in single_reg:
            dff.loc[sub_r]=df_cap.loc[dict[sub_r]].transpose()
        else:
            dff.loc[sub_r]=df_cap.loc[dict[sub_r]].sum().transpose()
    
    dff = dff/10**6

    nord_cap = df_cap.loc[nord_list].sum()
    df[]

df = pd.read_csv('df.csv', index_col = 0).fillna(0)/10**6

df_cap = pd.read_csv('Res_2050_agg_unc_2h_nolog\capacities_2050_agg_unc_2h_nolog.csv', index_col = 0).fillna(0)



#%% import geography

italy = gpd.read_file('limits_IT_regions.geojson')
region_list = (['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','R11','R12','R13','R14','R15','R16','R17','R18','SICI','SARD'])
italy.reg_name = region_list

df = pd.read_csv('general_info.csv', index_col = None).fillna(0)
df['mobility_demand']= df.mobility_demand/10**9
df['cars']= df.cars/10**6
df['battery_capacity']= df.battery_capacity/10**6

#%% Plotting 20-region map

col = 'battery_capacity'

fig = px.choropleth(df, geojson=italy, color=col,
                    color_continuous_scale="viridis_r",
                    locations="region", featureidkey="properties.reg_name",
                   )
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(coloraxis_colorbar=dict(
    title="Vehicle battery capacity",
    thicknessmode="pixels", thickness=25,
    lenmode="pixels", len=350,
    # yanchor="top", y=0.7,
    orientation="v",
    ticksuffix=' GWh',
    ticks='outside'
    # ticks="outside", ticksuffix=" bills",
    # dtick=5
))
fig.update_layout(height=400, width=1000,margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
fig.write_image(f"{col}_map.png") 

#%% 6-node geography creation

from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import copy

italy_bz = italy.copy(deep=True)
italy_bz["reg"] = ['NORD', 'CNOR', 'CSUD', 'SUD', 'SICI', 'SARD']

nord_geo = cascaded_union( [italy.geometry[0],italy.geometry[1], italy.geometry[2], italy.geometry[3], italy.geometry[4], italy.geometry[5],
        italy.geometry[6], italy.geometry[7]])
cnor_geo = cascaded_union( [italy.geometry[8],italy.geometry[9], italy.geometry[10]])
csud_geo = cascaded_union( [italy.geometry[11],italy.geometry[12], italy.geometry[13]])
sud_geo = cascaded_union( [italy.geometry[14],italy.geometry[15], italy.geometry[16], italy.geometry[17]])
sici_geo = italy.geometry[18]
sard_geo = italy.geometry[19]

db = [nord_geo, cnor_geo, csud_geo, sud_geo, sici_geo, sard_geo]

italy_bz = gpd.GeoDataFrame()
italy_bz["regions"]=['NORD', 'CNOR', 'CSUD', 'SUD', 'SICI', 'SARD']
italy_bz["geometry"]=db

#%% Plotting 6-node map










