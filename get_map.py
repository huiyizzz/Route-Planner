import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
import pandas as pd
import math
import numpy as np

def scatterMap(df, title, name=None):
    if name is None:
        fig = px.scatter_mapbox(lat=df["lat"], lon=df["lon"])
    else:
        fig = px.scatter_mapbox(lat=df["lat"], lon=df["lon"], hover_name=name)
        
    fig.update_layout(mapbox_style='open-street-map', title=title)
    fig.show()
a = np.cos(math.pi/3)
print(a)