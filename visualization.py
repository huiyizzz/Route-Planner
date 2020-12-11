import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud


def Map(loc, title):
    mean_lat = loc['lat'].mean()
    mean_lon = loc['lon'].mean()

    token = 'pk.eyJ1IjoiY2FyaW5hemhhbyIsImEiOiJja2ljYm1scWYwaDJuMndwM3lxaWZrd2VjIn0.ZwPLmscfNLPAF4VI6Uy4PA'
    fig = go.Figure(go.Scattermapbox(
        mode='markers+lines+text',
        lat=loc['lat'],
        lon=loc['lon'],
        hovertext=loc['name'],
        text=loc['name'],
        textfont=dict(size=13, color='red'),
        marker={'size': 12,
                'symbol': 'bus'},
        textposition='bottom right'))

    fig.update_layout(
        title=title,
        height=700,
        width=700,
        mapbox={
            'accesstoken': token,
            'center': {'lon': mean_lon, 'lat': mean_lat},
            'style': 'outdoors',
            'zoom': 9})
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
                            zoom=zoom, size=[1] * len(df), color=color, opacity=opacity,
                            color_discrete_sequence=px.colors.qualitative.G10,
                            size_max=size_max
                            )
    fig.update_layout(mapbox_style='open-street-map', title=title)
    fig.show()


def RouteMap(df1, labels, lat, lon, title, zoom, opacity, size_max):
    fig = px.scatter_mapbox(df1, lat='lat_y', lon='lon_y', labels=labels,
                            zoom=zoom, color=labels,
                            hover_name='name', size=[1] * len(df1), opacity=opacity,
                            color_discrete_sequence=px.colors.qualitative.G10,
                            size_max=size_max
                            )
    fig.add_trace(go.Scattermapbox(mode='lines+markers',
                                   lat=lat,
                                   lon=lon,
                                   name='Actual',
                                   marker={
                                       'size': 12, 'color': 'LightSlateGray', 'opacity': 0.7},
                                   line={'color': 'rgba(100, 100, 100, 0.4)'}
                                   ))

    fig.update_layout(
        title=title,
        autosize=True,
        mapbox_style='open-street-map')
    fig.show()


# prepare for calculate word frequency
def formate(word):
    res = ''
    for item in word:
        if res == '':
            res = list(item.values())
            res = res[0]
        else:
            t = list(item.values())
            res.extend(t[0])

    return res


# reference from CMPT459 Assignmet1
# create wordcloud
def wordcloud(nltk_count, title):
    mask = np.array(Image.open('./Image/pic.jpg'))
    wordcloud = WordCloud(background_color='white',
                          contour_width=3, mask=mask,
                          contour_color='blue'
                          ).generate_from_frequencies(nltk_count)

    plt.figure(figsize=(20, 17))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=30)


# prepare for wordcloud
def pre_wordcloud(nlp_list):
    word_list = list(nlp_list.explode())
    nltk_count = nltk.FreqDist(word_list)
    return nltk_count


def showRoute():
    # read the recommended path
    loc = pd.read_json('./Data/hot_locs.json', orient='record')
    loc = loc.set_index('name')
    file = open('./Data/route.txt')
    order = file.read().replace('\n', '->')
    name = re.split(r'->', order)

    # sorted by the order of path (start from origin)
    order = pd.DataFrame(
        {'name': name[:-2], 'order': range(0, len(name[:-2]))})
    order = order.set_index('name')
    loc['order'] = order['order']
    loc = loc.sort_values(by=['order'])
    loc = loc.reset_index()

    # generate map
    Map(loc, 'Interesting Route in Vancouver')

    # generate wordcloud
    # top-5
    loc['nlp_list'] = loc['word_list'].apply(formate)
    nltk_count = pre_wordcloud(loc['nlp_list'])
    title = 'Top-5 Rank Spots'
    wordcloud(nltk_count, title)

    # last-5
    low = pd.read_json('./Data/low.json', orient='record')
    low['nlp_list'] = low['word_list'].apply(formate)
    nltk_count_low = pre_wordcloud(low['nlp_list'])
    title = 'Last-5 Rank Spots'
    wordcloud(nltk_count_low, title)


def showImgRoute(osm_df, img_df, merged_df, near_df):
    ScatterMap(osm_df, 'OSM Location in Vancouver', osm_df['name'], 8, 0.6, 5)
    ScatterMap(img_df, 'Imgae Location', img_df['img'], 11, 0.8, 8)
    RouteMap(merged_df, 'img', merged_df['lat_x'],
             merged_df['lon_x'], 'User\'s Route with Their Nearest Locations in the OSM', 12, 0.9, 6)
    RouteMap(near_df, 'img', img_df['lat'], img_df['lon'],
             'User\'s Route Generated from Images with Near Amenities', 12, 0.9, 4)


def showRestaurant(nearest, restaurant, name, themax, cuisine, chains, nonchains):
    Map(nearest, 'The nearest restaurant')
    ScatterColorMap(
        cuisine, cuisine['style'], 'The different cuisine restaurants in vancouver', cuisine['name'], 8, 0.8, 6)
    ScatterMap(themax, 'The most restaurant with cuisine: ' +
               name, themax['name'], 8, 0.8, 6)
    ScatterColorMap(
        chains, chains['name'], 'The chain restaurant in vancouver', chains['name'], 8, 0.8, 7)
    ScatterMap(nonchains, 'The non-chain restaurant in vancouver',
               nonchains['name'], 8, 0.8, 6)
