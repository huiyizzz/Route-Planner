from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from os import path
from numpy.core.numeric import NaN
import pandas as pd
from qwikidata.entity import WikidataItem
from qwikidata.linked_data_interface import get_entity_dict_from_api

import googlemaps
import warnings
import time
import re

from nltk.corpus import stopwords


warnings.filterwarnings("ignore")
global unique_name
unique_name = set()
API_KEY = 'AIzaSyC3O-BdZRrBuOmC_nCvWcbnCdmxEWTztLg'
gmaps = googlemaps.Client(key=API_KEY)
global stop_words
stop_words = set(stopwords.words('english'))


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
        elif 'brand:wikipedia' in X['tags']:
            wikipedia = X['tags']['brand:wikipedia']
            name = wikipedia[3:]
    return name


def clean_data(df):
    df['name'] = df.apply(fill, axis=1)
    df.dropna(inplace=True)
    df.drop(['timestamp', 'tags'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# find attraction reviews
def find_reivew(name):
    if name not in unique_name:
        unique_name.add(name)
        places_result = gmaps.find_place(
            input=name, input_type='textquery', fields=['place_id'])
        if len(places_result['candidates']) != 0:
            place_id = places_result['candidates'][0]['place_id']
            place_details = gmaps.place(
                place_id=place_id, fields=['name', 'user_ratings_total', 'rating', 'review'])
            return place_details['result']


def find_review_id(place):
    if place['name'] not in unique_name:
        unique_name.add(place['name'])
        place_details = gmaps.place(
            place_id=place['place_id'], fields=['name', 'user_ratings_total', 'rating', 'review'])
        return place_details['result']

# cleaning name


def extract(name):
    name = str(name)
    if '-' in name:
        index = name.index('-')
        name = name[:index]
    return name

# create df that contains reviews


def create_df(result):

    lat = [place["geometry"]['location']['lat'] for place in result['results']]
    lon = [place["geometry"]['location']['lng'] for place in result['results']]
    name = [place['name'] for place in result['results']]
    id = [place['place_id'] for place in result['results']]
    data = pd.DataFrame(
        data={'name': name, 'place_id': id, 'lat': lat, 'lon': lon})

    data = data.loc[(data['lat'] > 49) & (
        data['lat'] < 49.5)]
    data = data.loc[(data['lon'] > -123.5) & (
        data['lon'] < -122)]
    review = data.apply(find_review_id, axis=1)
    return review

# add additional attraction


def addition():

    result = gmaps.places_nearby(location='49.252336254279, -123.1142930446478',
                                 radius=50000, open_now=False, type='tourist_attraction')
    review = create_df(result)
    review.to_json('./Data/addition.json')
    i = 1
    while 'next_page_token' in result:
        next_token = result['next_page_token']
        time.sleep(3)
        result = (gmaps.places_nearby(page_token=next_token))
        review = create_df(result)
        review.to_json('./Data/addition'+str(i)+'.json')
        i = i+1


def remove_url_punctuation(X):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    replace_url = url_pattern.sub(r'', str(X))
    punct_pattern = re.compile(r'[^\w\s]')
    no_punct = punct_pattern.sub(r'', replace_url).lower()
    return no_punct


def split_words(X):
    split_words_list = X.split(' ')
    return split_words_list


def remove_stopwords(X):
    global stop_words
    words = []
    for word in X:
        if word not in stop_words and len(word) > 2 and word != 'nan':
            words.append(word)
    return words
# --------------------------------------------


def processing_text(commend):
    res = []
    for text in commend:
        text = text['text']
        text = remove_url_punctuation(text)
        text = split_words(text)
        text = remove_stopwords(text)
        lemmatizer = WordNetLemmatizer()
        text = [lemmatizer.lemmatize(t) for t in text]
        text = [lemmatizer.lemmatize(t, 'v') for t in text]
        res.append({'word': text})
    return res


# check spelling
def spelling(reviews):
    res = []
    for commend in reviews:
        if commend['language'] == 'en':
            text = commend['text']
            temp = TextBlob(text)
            text = temp.correct()
            res.append({'text': str(text)})
    return res

# find attraction location


def find_location(row):
    loc = gmaps.find_place(input=row['name'], input_type='textquery',
                           fields=['geometry/location'])
    loc = loc['candidates'][0]['geometry']
    row['lat'] = loc['location']['lat']
    row['lon'] = loc['location']['lng']
    return row


def generateReview(out_directory):

    # read files and fill missing information
    file = './Data/osm/amenities-vancouver.json.gz'
    osm_df = pd.read_json(file, lines=True)
    osm_df['name'] = osm_df.apply(fill, axis=1)

    # select useful amenity
    attraction = osm_df.loc[(osm_df['amenity'] == 'clock')
                            | (osm_df['amenity'] == 'bicycle_rental') | (osm_df['amenity'] == 'arts_centre') | (osm_df['amenity'] == 'park') | (osm_df['amenity'] == 'nightclub')]

    attraction['name'] = attraction['name'].apply(extract)

    # find visiters' commends
    if not (path.exists('./Data/reviews.json')):
        reviews = attraction['name'].apply(find_reivew)
        reviews.to_json('./Data/reviews.json')

    # add more attraction in Vancouver
    if not(path.exists('./Data/addition.json')):
        addition()

    # combine comments df
    reviews = (pd.read_json('./Data/reviews.json', orient='records')).T
    add1 = (pd.read_json('./Data/addition.json', orient='records')).T
    add2 = (pd.read_json('./Data/addition1.json', orient='records')).T
    add3 = (pd.read_json('./Data/addition2.json', orient='records')).T
    reviews = pd.concat([reviews, add1, add2, add3])

    # clean data
    reviews = (reviews.reset_index()).drop(columns=['index'])
    reviews = reviews.dropna()
    reviews = reviews.loc[reviews['user_ratings_total'] > 1000]

    # cleaning commends
    reviews['clean_text'] = reviews['reviews'].apply(spelling)
    reviews = reviews.loc[reviews['clean_text'] != '']
    reviews['word_list'] = reviews['clean_text'].apply(processing_text)
    reviews = reviews.loc[reviews['word_list'] != '']
    reviews = reviews.drop_duplicates(subset=['name'])

    # select attraction with rank >4.5
    reviews = reviews.loc[reviews['rating'] > 4.5]
    reviews = reviews.apply(find_location, axis=1)
    # save and use for analyzing.py
    reviews.to_json(out_directory)
