from cleaning import *
from analyzing import *
from visualization import *
out_directory = './Data/data.json'
# carina cleaning
generateReview(out_directory)
generateRoute('route.txt')
showRoute()
