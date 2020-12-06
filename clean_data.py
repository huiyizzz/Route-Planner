import numpy as np
from numpy.core.numeric import NaN
from numpy.lib.type_check import _imag_dispatcher
import pandas as pd
from pandas.core.base import DataError
from qwikidata.entity import WikidataItem
from qwikidata.linked_data_interface import get_entity_dict_from_api
from GPSPhoto.gpsphoto import getGPSData
import exifread
import os
from datetime import datetime
#from get_map import *
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
import pandas as pd


def fill(X):
    name = X['name']
    if name is NaN:
        if 'official_name' in X['tags']:
            name = X['tags']['official_name']
        elif 'operator' in X['tags']:
            name = X['tags']['operator']
        elif 'brand:wikidata' in X['tags']:
            wikidata = X['tags']['brand:wikidata']
            q_dict = get_entity_dict_from_api(wikidata)
            name = WikidataItem(q_dict).get_label()
            # print(X.loc[i, ['name', 'tags']])
        elif 'brand:wikipedia' in X['tags']:
            wikipedia = X['tags']['brand:wikipedia']
            name = wikipedia[3:]
        # else:
        #    print(X['tags'])
            
            
    return name

def clean_data(df):
    df['name'] = df.apply(fill, axis=1)
    df.dropna(inplace=True)
    df.drop(['timestamp', 'tags'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_img_exif(path):
    # inital image dataframe
    df = pd.DataFrame(columns=['img', 'lat', 'lon', 'datetime'])
    file_list = os.listdir(path)
    for file in file_list:
        img = {}
        img_file = os.path.join(path, file)
        f = open(img_file, 'rb')
        tags = exifread.process_file(f)
        date_str = tags['EXIF DateTimeOriginal'].__str__()
        date = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
        exif = getGPSData(img_file)
        img['img'] = file
        img['lat'] = exif['Latitude']
        img['lon'] = exif['Longitude']
        img['datetime'] = date #datetime.timestamp(date)
        df = df.append(img, ignore_index=True)
    return df


def haversine(lat1, lon1, lat2, lon2):
    # haversine function reference:
    # https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
    a = (np.sin(np.radians(lat2 - lat1) / 2)**2
         + np.cos(np.radians(lat1))
         * np.cos(np.radians(lat2))
         * np.sin(np.radians(lon2 - lon1) / 2)**2)
    return 12742000 * np.arcsin(np.sqrt(a))

def find_amenity(img, osm):
    dis = haversine(osm['lat'], osm['lon'], img['lat'], img['lon'])
    return np.argmin(dis)

def sMap(df, title):
    #fig = px.scatter_mapbox(lat=img_df["lat_x"], lon=img_df["lon_x"])
    fig = go.Figure(px.scatter_mapbox(lat=img_df["lat_x"], lon=img_df["lon_x"]))
    # fig.add_trace(go.Scattermapbox(mode = 'markers',                                
    #                                    lat=img_df["lat_y"],
    #                                    lon=img_df["lon_y"],
    #                                    marker = {'size':6},
    #                                    showlegend = False))
    # fig.update_layout(title=title)
    fig.show()

osm_file = './Data/osm/amenities-vancouver.json.gz'
osm_df = pd.read_json(osm_file, lines=True)
#osm_df.info()
osm_df = clean_data(osm_df)
#osm_df.info()
img_path = './Data/image'
img_df = load_img_exif(img_path)
#print(img_df)


#scatterMap(osm_df, 'OSM Location in Vancouver', osm_df['name'])
#scatterMap(img_df, 'Images Location')

img_df['index'] = img_df.apply(find_amenity, osm=osm_df, axis=1)
img_df = pd.merge(img_df,osm_df,left_on='index',right_index=True)
sMap(img_df,'title')
