from cleaning import *
from analyzing import *
from visualization import *

def imageInfo():
    osm_df, img_df, merged_df, near_df = generateImg()
    showImgRoute(osm_df, img_df, merged_df, near_df)

def restaurant():
    nearest, restaurant, name, themax, cuisine, chains, nonchains = generateRestaurant()
    showRestaurant(nearest, restaurant, name, themax, cuisine, chains, nonchains)

def interestingRoute():
    #generateReview('./Data/data.json')
    #generateRoute('route.txt')
    showRoute()

if __name__ == '__main__':
    imageInfo()
    restaurant()
    interestingRoute()
    
