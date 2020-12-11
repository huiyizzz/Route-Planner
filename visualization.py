import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
import nltk
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, get_single_color_func


def Map(loc, title):
    mean_lat = loc['lat'].mean()
    mean_lon = loc['lon'].mean()

    token = 'pk.eyJ1IjoiY2FyaW5hemhhbyIsImEiOiJja2ljYm1scWYwaDJuMndwM3lxaWZrd2VjIn0.ZwPLmscfNLPAF4VI6Uy4PA'
    fig = go.Figure(go.Scattermapbox(
        mode="markers+lines+text",
        lat=loc['lat'],
        lon=loc['lon'],
        hovertext=loc['name'],
        text=loc['name'],
        textfont=dict(size=13, color='red'),
        marker={'size': 12,
                'symbol': 'bus'},
        textposition="bottom right"))

    fig.update_layout(
        title=title,
        # height=2300,
        # width=2030,
        height=700,
        width=700,
        mapbox={
            'accesstoken': token,
            'center': {'lon': mean_lon, 'lat': mean_lat},
            'style': 'outdoors',
            'zoom': 10})
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
def wordcloud(nltk_count):
    mask = np.array(Image.open('./Image/pic.jpg'))
    wordcloud = WordCloud(background_color="white",
                          contour_width=3, mask=mask,
                          contour_color='blue'
                          ).generate_from_frequencies(nltk_count)

    plt.figure(figsize=(20, 17))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')


# prepare for wordcloud
def pre_wordcloud(nlp_list):
    # all_words_unique_list = (nlp_list.explode()).unique()
    word_list = list(nlp_list.explode())
    nltk_count = nltk.FreqDist(word_list)
    return nltk_count


def showRoute():
    # read the recommended path
    loc = pd.read_json('./Data/hot_locs.json', orient='record')
    loc = loc.set_index('name')
    file = open("./Data/route.txt")
    order = file.read().replace("\n", "->")
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
    wordcloud(nltk_count)
    # last-5
    low = pd.read_json('./Data/low.json', orient='record')
    low['nlp_list'] = low['word_list'].apply(formate)
    nltk_count_low = pre_wordcloud(low['nlp_list'])
    wordcloud(nltk_count_low)
