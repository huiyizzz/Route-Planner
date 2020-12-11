from cleaning import *
from analyzing import *
from visualization import *

def imageInfo():
    print('preprocessing data and analyzing data...')
    print()
    osm_df, img_df, merged_df, near_df = generateImg()
    print('done')
    print()
    showImgRoute(osm_df, img_df, merged_df, near_df)

def restaurant():
    print('preprocessing data and analyzing data...')
    print()
    nearest, restaurant, name, themax, cuisine, chains, nonchains = generateRestaurant()
    print('done')
    print()
    #showRestaurant(nearest, restaurant, name, themax, cuisine, chains, nonchains)

def interestingRoute():
    print('preprocessing data...')
    print()
    generateReview('./Data/data.json')
    print('analyzing data...')
    print()
    generateRoute('route.txt')
    print('done!')
    print()
    showRoute()

if __name__ == '__main__':
    imageInfo()
    restaurant()
    interestingRoute()