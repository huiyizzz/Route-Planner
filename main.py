from cleaning import *
from analyzing import *
from visualization import *
import sys


def imageInfo():
    print('preprocessing data...')
    print()
    osm_df, img_df = generateImg()
    print('done')
    print()
    print('analyzing data...')
    print()
    merged_df, near_df = combineImgData(osm_df, img_df)
    print()
    showImgRoute(osm_df, img_df, merged_df, near_df)


def restaurant():
    print('preprocessing data and analyzing data...')
    print()
    nearest, restaurant, name, themax, cuisine, chains, nonchains = generateRestaurant()
    print('done')
    print()
    showRestaurant(nearest, restaurant, name, themax,
                   cuisine, chains, nonchains)


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
    if len(sys.argv) != 2:
        print('Please input the correct command.')
    else:
        if sys.argv[1] == 'photo':
            imageInfo()
        elif sys.argv[1] == 'restaurant':
            restaurant()
        elif sys.argv[1] == 'route':
            interestingRoute()
        elif sys.argv[1] == 'all':
            interestingRoute()
            restaurant()
            imageInfo()
        else:
            print('Wrong input! Please input the correct command.')
