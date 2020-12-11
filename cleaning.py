from io import open_code
from os import path
import os
from math import *
from numpy.core.numeric import NaN
from numpy.lib.type_check import _imag_dispatcher
import pandas as pd
import numpy as np
from datetime import datetime
from GPSPhoto.gpsphoto import getGPSData
from pandas.core.base import DataError
from qwikidata.entity import WikidataItem
from qwikidata.linked_data_interface import get_entity_dict_from_api
import exifread
import googlemaps
import warnings
import time
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import geocoder

warnings.filterwarnings('ignore')
global unique_name
unique_name = set()
API_KEY = 'AIzaSyC3O-BdZRrBuOmC_nCvWcbnCdmxEWTztLg'
gmaps = googlemaps.Client(key=API_KEY)
global stop_words
stop_words = set(stopwords.words('english'))
file = 'Data/osm/amenities-vancouver.json.gz'
#location = geocoder.ip('me')
#loc = location.latlng
loc = [49.282761666666666, -123.12364166666666]

cuisine_style = ['acadian', 'afghan', 'american', 'arab', 'brazilian',
                 'buddhist', 'burmese', 'cambodian', 'caribbean', 'chinese',
                 'cuban', 'czech', 'dutch', 'ethiopian', 'filipino', 'french',
                 'german', 'greek', 'hong kong', 'indian', 'indonesian', 'irish',
                 'italian', 'jamaican', 'japanese', 'korean', 'lebanese', 'malaysian',
                 'malaysian', 'mediterranean', 'mexican', 'mexican', 'mongolian',
                 'moroccan', 'persian', 'peruvian', 'portuguese', 'singaporean',
                 'taiwanese', 'thai', 'turkish', 'ukranian', 'vietnamese', 'west_coast']


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
    print('The osm data with filling names:')
    df.info()
    print()
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

    lat = [place['geometry']['location']['lat'] for place in result['results']]
    lon = [place['geometry']['location']['lng'] for place in result['results']]
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
        review.to_json('./Data/addition' + str(i) + '.json')
        i = i + 1


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
        img['datetime'] = date  # datetime.timestamp(date)
        df = df.append(img, ignore_index=True)
    df.sort_values(by='datetime', inplace=True)
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
    dis = haversine(img['lat'], img['lon'], osm['lat'], osm['lon'])
    return np.argmin(dis)


def find_amenities(img, osm):
    a = haversine(img['lat'], img['lon'], img['next_lat'], img['next_lon'])
    b = haversine(img['lat'], img['lon'], osm['lat'], osm['lon'])
    c = haversine(img['next_lat'], img['next_lon'], osm['lat'], osm['lon'])
    semi_p = (a + b + c) / 2
    area = np.sqrt(semi_p * (semi_p - a) * (semi_p - b) * (semi_p - c))
    dis = (2 * area) / a
    triangle = abs(b**2 - c**2) - a**2
    return list(dis[(dis < 100) & (triangle <= 0) | (b < 100) | (c < 100)].index)


def nearMe(df):
    dis = haversine(df['lat'], df['lon'], loc[0], loc[1])
    return dis


def generateReview(out_directory):
    # read files and fill missing information
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
    if not (path.exists('./Data/addition.json')):
        addition()

    # combine comments df
    reviews = (pd.read_json('./Data/reviews.json', orient='records')).T
    add1 = (pd.read_json('./Data/addition.json', orient='records')).T
    add2 = (pd.read_json('./Data/addition1.json', orient='records')).T
    add3 = (pd.read_json('./Data/addition2.json', orient='records')).T
    reviews = pd.concat([reviews, add1, add2, add3])
    print('The reviews data:')
    print(reviews.head())
    print()
    reviews.info()

    # clean data
    reviews = (reviews.reset_index()).drop(columns=['index'])
    reviews = reviews.dropna()
    reviews = reviews.loc[reviews['user_ratings_total'] > 1000]

    # cleaning commends
    # select attraction with rank >4.5
    reviews = reviews.loc[reviews['rating'] > 4.5]
    reviews['clean_text'] = reviews['reviews'].apply(spelling)
    reviews = reviews.loc[reviews['clean_text'] != '']
    reviews['word_list'] = reviews['clean_text'].apply(processing_text)
    reviews = reviews.loc[reviews['word_list'] != '']
    reviews = reviews.drop_duplicates(subset=['name'])

    reviews = reviews.apply(find_location, axis=1)
    print('The reviews data after cleaning:')
    print(reviews.head())
    print()
    reviews.info()
    print()
    # save and use for analyzing.py
    # reviews.to_json(out_directory)


def generateImg():
    osm_df = pd.read_json(file, lines=True)
    print('The osm data:')
    osm_df.info()
    print()
    osm_df = clean_data(osm_df)
    print('The osm data after cleaning:')
    osm_df.info()
    print()
    img_path = './Image/photo'
    img_df = load_img_exif(img_path)
    print('The image data:')
    print(img_df)
    print()
    index = img_df.apply(find_amenity, osm=osm_df, axis=1)
    merged_df = pd.merge(img_df, osm_df, left_on=index, right_index=True)
    print('The image data with nearest locatons in osm data:')
    print(merged_df)
    print()
    img_df['next_lat'] = img_df['lat'].shift(-1)
    img_df['next_lon'] = img_df['lon'].shift(-1)

    osm_index = img_df.apply(find_amenities, osm=osm_df, axis=1)
    near_df = pd.DataFrame(columns=['img_index', 'osm_index'])

    for i in range(len(osm_index)):
        temp_df = pd.DataFrame(
            {'img_index': [i] * len(osm_index[i]), 'osm_index': osm_index[i]})
        near_df = near_df.append(temp_df, ignore_index=True)

    near_df = pd.merge(img_df, near_df, left_index=True,
                       right_on='img_index', how='inner')
    near_df = near_df.drop_duplicates('osm_index').reset_index(drop=True)
    near_df = pd.merge(near_df, osm_df, left_on='osm_index',
                       right_index=True, how='left')
    print('The near amenities merged with user\'s route:')
    print(near_df)
    return osm_df, img_df, merged_df, near_df


def getStyle(string):
    string = str(string).lower()
    for style in cuisine_style:
        if style in string:
            return style
    return 'other'


def generateRestaurant():
    df = pd.read_json(file, lines=True)
    restaurant = df[df['amenity'] == 'restaurant']
    restaurant = restaurant.reset_index(drop=True)
    restaurant['dist'] = restaurant.apply(nearMe, axis=1)
    nearest = restaurant[restaurant['dist'] < 300]

    tags = restaurant['tags'].apply(pd.Series)
    tags['style'] = tags['cuisine'].apply(getStyle)
    nums = tags.groupby('style').size().reset_index(name='counts')
    nums = nums.drop(nums[nums['style'] == 'other'].index)
    name = nums.loc[nums['counts'] == nums.counts.max()].values[0][0]
    themax = tags[tags['style'] == name]
    themax = pd.merge(restaurant, themax, left_index=True, right_index=True)

    # different cuisine restaurant
    cuisine = tags[tags['style'].notna()]
    cuisine = pd.merge(restaurant, cuisine, left_index=True, right_index=True)
    cuisine = cuisine.drop(cuisine[cuisine['style'] == 'other'].index)

    # with chain restaurant
    chains = tags[tags['brand:wikidata'].notna()]
    chains = pd.merge(restaurant, chains, left_index=True, right_index=True)

    # non chain restaurant
    nonchains = restaurant[~restaurant.isin(chains)].dropna()

    return nearest, restaurant, name, themax, cuisine, chains, nonchains
