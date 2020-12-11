import numpy as np
import pandas as pd

from math import *
from numpy.core.numeric import NaN

from qwikidata.entity import WikidataItem, WikidataLexeme, WikidataProperty
from qwikidata.linked_data_interface import get_entity_dict_from_api

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
import pandas as pd

# get current location
import geocoder
location = geocoder.ip('me')
loc = location.latlng
# lat = 49.282761666666666
# lon = -123.12364166666666

def Map(loc, title):
    mean_lat=loc['lat'].mean()
    mean_lon=loc['lon'].mean()

    token ='pk.eyJ1IjoiY2FyaW5hemhhbyIsImEiOiJja2ljYm1scWYwaDJuMndwM3lxaWZrd2VjIn0.ZwPLmscfNLPAF4VI6Uy4PA'
    fig = go.Figure(go.Scattermapbox(
        mode = "markers+text",
        lat = loc['lat'],
        lon = loc['lon'],
        hovertext=loc['name'],
        text=loc['name'],
        textfont=dict(size=13, color='red'),
        marker = {'size': 12,
                  'symbol': 'restaurant'},
        textposition = "bottom right"))

    fig.update_layout(
        title=title,
        # height=2300,
        # width=2030,
        height=700,
        width=700,
        mapbox = {
            'accesstoken': token,
            'center':{'lon':mean_lon,'lat':mean_lat},
            'style': 'outdoors',
            'zoom': 10})
    fig.show()

def ScatterMap(df, title, name, zoom, opacity, size_max):
    fig = px.scatter_mapbox(df, lat='lat', lon='lon', hover_name=name,
                            zoom=zoom, size=[1] * len(df), opacity=opacity,
                            color_discrete_sequence=px.colors.qualitative.G10,
                            size_max=size_max
                            )
    fig.update_layout(mapbox_style='open-street-map', title=title)
    fig.show()


def ScatterColorMap(df, color, title, name, zoom, opacity, size_max):
    fig = px.scatter_mapbox(df, lat='lat', lon='lon', hover_name=name,
                            zoom=zoom, size=[1] * len(df), color = color, opacity=opacity,
                            color_discrete_sequence=px.colors.qualitative.G10,
                            size_max=size_max
                            )
    fig.update_layout(mapbox_style='open-street-map', title=title)
    fig.show()



#reference: https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
def haversine(point):
    p = np.pi/180
    d = 0.5 - np.cos((point['lat'] - loc[0]) * p)/2 + np.cos(loc[0] * p) * np.cos(point['lat'] * p) * (1 - np.cos((point['lon'] - loc[1]) * p))/2
    return 12742000 * np.arcsin(np.sqrt(d))
def haversine2(lat1, lon1, lat2, lon2):
    # haversine function reference:
    # https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
    a = (np.sin(np.radians(lat2 - lat1) / 2)**2
         + np.cos(np.radians(lat1))
         * np.cos(np.radians(lat2))
         * np.sin(np.radians(lon2 - lon1) / 2)**2)
    return 12742000 * np.arcsin(np.sqrt(a))

def nearMe(df, loc):
    dis = haversine2(df['lat'], df['lon'], loc[0], loc[1])
    return dis

file = 'Data/osm/amenities-vancouver.json.gz'
df = pd.read_json(file, lines=True)
# tags = df['tags'].apply(pd.Series)

# df['name'] = df.apply(fill, axis=1)
# df.info()

restaurant = df[df['amenity'] == 'restaurant']
restaurant = restaurant.reset_index(drop=True)
restaurant['dist'] = restaurant.apply(nearMe, axis=1)

nearest = restaurant[restaurant['dist'] < 300]
# Map(restaurant, 'all restaurant')
Map(nearest, 'nearest')
tags = restaurant['tags'].apply(pd.Series)
nums = tags.groupby('cuisine').size().reset_index(name='counts')
name = nums.loc[nums['counts'] == nums.counts.max()].values[0][0]
themax = tags[tags['cuisine'] == name]
themax = pd.merge(restaurant, themax, left_index=True, right_index=True)
# Map(themax, 'the most restaurant with cuisine: '+name)
ScatterMap(themax, 'the most restaurant with cuisine: '+name, themax['name'], 11, 0.8,8)

# different cuisine restaurant
cuisine = tags[tags['cuisine'].notna()]
cuisine = pd.merge(restaurant, cuisine, left_index=True, right_index=True)
ScatterColorMap(cuisine, cuisine['cuisine'], 'different cuisine restaurant in vancouver', cuisine['name'], 11, 0.8, 8)

# with chain restaurant
chains = tags[tags['brand:wikidata'].notna()]
chains = pd.merge(restaurant, chains, left_index=True, right_index=True)
ScatterColorMap(chains, chains['name'], 'the chain restaurant in vancouver', chains['name'], 11, 0.8, 8)

# non chain restaurant
nonchains = restaurant[~restaurant.isin(chains)]
ScatterMap(nonchains, 'non chains restaurant in vancouver', nonchains['name'], 11, 0.8,8)

# exif = getGPSData('1.jpg')
# latitude = exif['Latitude']
# longitude = exif['Longitude']
# print(latitude)
# print(longitude)
