from opencage.geocoder import OpenCageGeocode
from typing import Text
from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np
from textblob import TextBlob
import googlemaps
from os import path
import re


API_KEY = 'AIzaSyC3O-BdZRrBuOmC_nCvWcbnCdmxEWTztLg'
gmaps = googlemaps.Client(key=API_KEY)
# reference from https://verimake.com/topics/90
# find the shortest path after visiting all selected attraction


class TSP:
    def __init__(self, matrix, names):
        matrix.append([0] * len(matrix))
        matrix.insert(0, [0] * len(matrix))
        for i in range(len(matrix)):
            matrix[i].insert(0, 0)
            matrix[i].append(0)
        self.matrix = matrix
        self.names = names
        self.num = len(names)  # 城市个数
        self.temp_path = [i for i in range(0, self.num + 1)]  # 当前路径
        self.temp_len = 0  # 当前路径长度
        self.best_path = [0] * (self.num + 1)  # 最短路径
        self.best_len = 99999999  # 最短路径的长度

    def __backtrack(self, node):
        if node > self.num:
            v = self.matrix[self.temp_path[self.num]][1]
            if (v != 0) and (v + self.temp_len < self.best_len):
                self.best_len = v + self.temp_len
                for i in range(1, self.num + 1):
                    self.best_path[i] = self.temp_path[i]
        else:
            for i in range(node, self.num + 1):
                v = self.matrix[self.temp_path[node - 1]][self.temp_path[i]]
                if (v != 0) and (v + self.temp_len < self.best_len):
                    self.temp_path[node], self.temp_path[i] = self.temp_path[i], self.   temp_path[node]
                    self.temp_len += self.matrix[self.temp_path[node - 1]
                                                 ][self.temp_path[node]]
                    self.__backtrack(node + 1)
                    self.temp_len -= self.matrix[self.temp_path[node - 1]
                                                 ][self.temp_path[node]]
                    self.temp_path[node], self.temp_path[i] = self.temp_path[i], self.   temp_path[node]
        return

    def solve(self):

        self.__backtrack(2)

        path_str = ""
        for i in range(1, self.num + 1):
            path_str += self.names[self.best_path[i]] + '->'
        path_str += self.names[1]

        return path_str, self.best_len

# calculate review scores


def create_df(data):
    out = []
    for index, row in data.iterrows():
        for item in row['clean_text']:
            out.append(
                {'name': row['name'], 'comment': item['text'], 'rating': row['rating'], 'lat': row['lat'], 'lon': row['lon']})
    return pd.DataFrame(out)


def score(row):
    text = row['comment']
    row['polarity'] = round(TextBlob(text).sentiment.polarity, 3)
    row['subjectivity'] = round(TextBlob(text).sentiment.subjectivity, 3)
    return row


def label(score):
    if score > 0:
        label = 'pos'
    elif score < 0:
        label = 'neg'
    else:
        label = 'neu'
    return label

# format transportation time


def format(row):
    for idx in range(len(name)):
        if row[name[idx]] != 0:
            temp = re.split(r'[a-z]+', row[name[idx]])
            temp.remove('')
            mins = 0
            if len(temp) == 2:
                mins += int(temp[0]) * 60 + int(temp[1])
            else:
                mins += int(temp[0])
            row[name[idx]] = mins
    return row

# return the distance matrix between each attraction


def distance_matrix(transport, time):
    for i in range(len(transport)):
        row = transport[i]
        origin = str(row[1]) + ',' + str(row[2])
        for j in range(i + 1, len(transport)):
            row2 = transport[j]
            dest = str(row2[1]) + ',' + str(row2[2])
            res = gmaps.distance_matrix(
                origins=origin, destinations=dest, region='.ca')
            res = res['rows'][0]['elements'][0]
            dur = res['duration']['text']
            time[row[0]][row2[0]] = dur
    return time

# prepare for calculating shortest path


def pre_TSP(data, time):
    member = list(data['name'])
    city_name = {}
    i = 1
    city_name[i] = 'Origin'
    for name in member:
        i += 1
        city_name[i] = name
    s = len(city_name)
    rows, cols = (s, s)
    arr = [[0 for i in range(cols)] for j in range(rows)]
    for i in range(s):
        for j in range(i + 1, s):
            spot1 = city_name[i + 1]
            spot2 = city_name[j + 1]
            t = time[spot1][spot2]
            if t == 0:
                t = time[spot2][spot1]
            arr[i][j] = t
            arr[j][i] = t

    return city_name, arr

# get attraction time


def calculate_time(data, sign):
    filename = './Data/time_matrix' + str(sign) + '.json'
    if not (path.exists(filename)):
        lat = 49.281579
        lon = -122.996366
        transport = data[['name', 'lat', 'lon', 'word_list']]
        transport = transport.append(
            {'name': 'Origin', 'lat': lat, 'lon': lon, 'word_list': ''}, ignore_index=True)
        transport.to_json('./Data/hot_locs.json')
        transport = transport.drop(['word_list'], axis=1)
        time = pd.DataFrame(columns=transport['name'], index=transport['name'])
        transport = transport.values.tolist()
        time = distance_matrix(transport, time)
        time.to_json(filename)

    time = pd.read_json(filename, orient='record')
    global name
    name = time.columns
    time = time.fillna(0)
    time = time.apply(format, axis=1)

    return time

# return route plan and trip time


def router_plan(data, sign):
    router = list()
    if sign == 0:
        # rate first
        data = data.sort_values(by=['rate'], ascending=False)
    else:
        # heat first
        data = data.sort_values(by=['heat'], ascending=False)

    low = data.iloc[-5:]
    data = data.iloc[0:5]
    low.to_json('./Data/low.json')
    time = calculate_time(data, sign)
    spot, arr = pre_TSP(data, time)
    tsp = TSP(arr, spot)
    shortest_path, shortest_time = tsp.solve()

    return shortest_path, shortest_time

def generateRoute(output):

    data = pd.read_json('./Data/data.json')

    # calculate review scores
    comments = create_df(data)
    comments = comments.apply(score, axis=1)
    comments['label'] = comments['polarity'].apply(label)

    data = data.set_index('name')
    stat = comments.groupby('name').agg(
        {'polarity': 'mean', 'subjectivity': 'mean'})
    data['polarity'] = stat['polarity']
    data['subjectivity'] = stat['subjectivity']

    # calculate heat and rate for spots
    data['heat'] = data['rating'] * 5 + data['user_ratings_total'] * 3
    data['rate'] = data['rating'] * 5 + \
        data['polarity'] * 2 + data['subjectivity'] * 2
    data = data.drop(['reviews', 'clean_text', 'polarity',
                      'subjectivity', 'rating', 'user_ratings_total'], axis=1)
    data = data.reset_index()

    # find shortest path
    sign = 0
    shortest_path, shortest_time = router_plan(data, sign)

    # write to file and use for visualization
    output_name = './Data/' + output
    file = open(output_name, "w")
    file.write(shortest_path + '\n')
    file.write(str(shortest_time))
    file.close()