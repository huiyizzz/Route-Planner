import numpy as np
import pandas as pd
from pandas.core.base import DataError
from qwikidata.datavalue import MonolingualText
from qwikidata.entity import WikidataItem
from qwikidata.linked_data_interface import get_entity_dict_from_api
from numpy.core.numeric import NaN
from GPSPhoto.gpsphoto import getGPSData


# create an item representing "Douglas Adams"


file = './Data/osm/amenities-vancouver.json.gz'
osm_df = pd.read_json(file, lines=True)
# osm_df.info()
# if 'brand:wikidata' in osm_df['tags'][0]:
# print('yes')

#print(osm_df[['name','tags']])



a = getGPSData('./Data/1.jpg')
print(a)

#print(osm_df[osm_df['brand:wikidata' in osm_df['tags']]])
# osm_df.dropna(inplace=True)
# osm_df.reset_index(inplace=True)
# print(osm_df)
# amenity = set(osm_df['amenity'])
#Q37158
Q_DOUGLAS_ADAMS = "Q37158"
qdict = get_entity_dict_from_api(Q_DOUGLAS_ADAMS)
q = WikidataItem(qdict)
#print(q.get_label())