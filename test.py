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

def fill(X):
    name = X['name']
    if name is NaN:
        if 'brand:wikidata' in X['tags']:
            wikidata = X['tags']['brand:wikidata']
            q_dict = get_entity_dict_from_api(wikidata)
            name = WikidataItem(q_dict).get_label()
            # print(X.loc[i, ['name', 'tags']])
        elif 'brand:wikipedia' in X['tags']:
            wikipedia = X['tags']['brand:wikipedia']
            name = wikipedia[3:]
        # else:
        #   print(X[['amenity', 'tags']])
    return name


#reference: https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
def haversine(point):
    p = np.pi/180
    d = 0.5 - np.cos((point['lat'] - loc[0]) * p)/2 + np.cos(loc[0] * p) * np.cos(point['lat'] * p) * (1 - np.cos((point['lon'] - loc[1]) * p))/2
    return 12742000 * np.arcsin(np.sqrt(d))

def scatterMap(df, title, name=None):
    if name is None:
        fig = px.scatter_mapbox(lat=df["lat"], lon=df["lon"])
    else:
        fig = px.scatter_mapbox(lat=df["lat"], lon=df["lon"], hover_name=name)
        
    fig.update_layout(mapbox_style="open-street-map", title=title)
    fig.show()

file = 'Data/osm/amenities-vancouver.json.gz'
df = pd.read_json(file, lines=True)
# tags = df['tags'].apply(pd.Series)

# df['name'] = df.apply(fill, axis=1)
# df.info()

restaurant = df[df['amenity'] == 'restaurant']
restaurant = restaurant.reset_index(drop=True)
restaurant['dist'] = restaurant.apply(haversine,axis=1)
# nearest = restaurant[restaurant['dist'] == restaurant.agg('min')['dist']]
nearest = restaurant[restaurant['dist'] < 300]
# Map(restaurant, 'all restaurant')
Map(nearest, 'nearest')
tags = restaurant['tags'].apply(pd.Series)
nums = tags.groupby('cuisine').size().reset_index(name='counts')
name = nums.loc[nums['counts'] == nums.counts.max()].values[0][0]
themax = tags[tags['cuisine'] == name]
themax = pd.merge(restaurant, themax, left_index=True, right_index=True)
# Map(themax, 'the most restaurant with cuisine: '+name)
scatterMap(themax, 'the most restaurant with cuisine: '+name)

# with chain restaurant
chains = tags[tags['brand:wikidata'].notna()]
chains = pd.merge(restaurant, chains, left_index=True, right_index=True)
scatterMap(chains, 'the chain restaurant in vancouver')

# non chain restaurant
nonchains = restaurant[~restaurant.isin(chains)]
scatterMap(nonchains, 'non chains restaurant in vancouver')

# exif = getGPSData('1.jpg')
# latitude = exif['Latitude']
# longitude = exif['Longitude']
# print(latitude)
# print(longitude)
