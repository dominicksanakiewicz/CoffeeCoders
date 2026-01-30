import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt
from shapely import wkt
import contextily as cx

path = "C:/Users/amand/student30538-w26/CoffeeCoders/data"
os.chdir(path)


transport = pd.read_csv("CTA_-_Bus_Routes_20260129.csv")

transport['geometry'] = transport['the_geom'].apply(wkt.loads)

gdf = gpd.GeoDataFrame(
    transport,
    geometry="geometry",
    crs="EPSG:4326"
)

gdf_plot = gdf.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(10, 10))

gdf_plot.plot(
    ax=ax,
    linewidth=2
)

cx.add_basemap(
    ax,
    source=cx.providers.CartoDB.Positron
)

ax.set_axis_off()
plt.show()