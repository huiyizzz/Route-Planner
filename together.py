from cleaning import *
from analyzing import *
from visualization import *


def interestingRoute():
    # generateReview('./Data/data.json')
    # generateRoute('route.txt')
    showRoute()


def imageInfo():
    osm_df, img_df, merged_df, near_df = generateImg()
    generateImgRoute(osm_df, img_df, merged_df, near_df)


if __name__ == '__main__':
    interestingRoute()
    imageInfo()
