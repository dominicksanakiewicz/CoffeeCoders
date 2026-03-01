import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt
from shapely import wkt
import contextily as cx
import csv
import numpy as np
from geopy.geocoders import ArcGIS
import time
from shapely.geometry import Point
import requests

path = "C:/Users/amand/student30538-w26/CoffeeCoders/data"
os.chdir(path)

demo = pd.read_csv("cook_county_high_school_demographics.csv")
transport = pd.read_csv("CTA_-_Bus_Routes_20260129.csv")
grocery = gpd.read_file('grocery_store_shapefile_v0/grocery_stores.shp')
income = pd.read_csv('median_income_by_zip.csv')

academic = pd.read_csv("panel_yx_highschools.csv")
cook = gpd.read_file('cook_county/PVS_25_v2_tracts2020_17031.shp')
income = pd.read_csv('cook_tract_income_v7_2016_2024.csv')
desert = pd.read_csv('cook_food_access_2019.csv')

school_loc = pd.read_csv('school_name_location.csv')
#########################
# Clean census level data, keeping only columns with less than 50% NAs
cook_3857 = cook.to_crs(epsg=3857)
threshold = 0.5
cook_clean = cook_3857.loc[:, cook_3857.apply(lambda col: ((col.isna()) | (col.isnull())).mean() <= threshold)]
cook_clean = cook_clean[['TRACTID', 'TRACTLABEL', 'POP20', 'geometry']]
cook_clean['TRACTID'] = cook_clean['TRACTID'].astype(int)
cook_clean.head()
len(cook_clean['TRACTID'].unique())
len(cook_clean)

#############################
# clean the income data, adding np.nan to negative income values, and merging with cook_income
income['year'].unique()
income['median_hh_income'] = income['median_hh_income'].apply(
    lambda x: np.nan if x is not None and x < 0 else x
)
cook_income = pd.merge(cook_clean, income, left_on='TRACTID', right_on='GEOID', how='right')
len(cook_income)
cook_income.tail()

#######################
#Dominick's code for matching school names & geometry

# schools_url = "https://data-nces.opendata.arcgis.com/datasets/nces::public-school-locations-2021-22.geojson"
# schools = gpd.read_file(schools_url)

# counties_url = "https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip"

# counties = gpd.read_file(counties_url)

# cook = counties[(counties["STATEFP"] == "17") & (counties["COUNTYFP"] == "031")].copy()

# target_crs = "EPSG:3435"  # NAD83 / Illinois East (ftUS)
# schools = schools.to_crs(target_crs)
# cook = cook.to_crs(target_crs)

# # Keep only points within Cook County
# schools_cook = gpd.sjoin(
#     schools,
#     cook[["geometry"]],
#     how="inner",
#     predicate="within"
# ).drop(columns=["index_right"]).copy()


# if {"GSLO", "GSHI"}.issubset(schools_cook.columns):
#     schools_cook["GSLO_num"] = pd.to_numeric(schools_cook["GSLO"], errors="coerce")
#     schools_cook["GSHI_num"] = pd.to_numeric(schools_cook["GSHI"], errors="coerce")
#     # High school if it serves grade 9+ AND includes grade 12
#     schools_cook = schools_cook[(schools_cook["GSLO_num"] <= 9) & (schools_cook["GSHI_num"] >= 12)].copy()

# elif "LEVEL" in schools_cook.columns:
#     schools_cook["LEVEL_num"] = pd.to_numeric(schools_cook["LEVEL"], errors="coerce")
#     schools_cook = schools_cook[schools_cook["LEVEL_num"] == 3].copy()

# else:
#     name_col = "NAME" if "NAME" in schools_cook.columns else None
#     if name_col is None:
#         raise ValueError("Can't identify high schools: no (GSLO,GSHI), no LEVEL, and no NAME column.")
#     schools_cook = schools_cook[schools_cook[name_col].astype(str).str.contains(r"\bHIGH\b", case=False, na=False)].copy()

############################################################
demo = demo.replace("*", np.nan)
demographic = demo.loc[:, demo.isna().mean() <= 0.5]
demographic = demographic[demographic['County']=='Cook']

demo_aca = pd.merge(demographic, academic, left_on = 'School Name', right_on ='school_name', how='outer')
demo_aca.head()
len(demo_aca)

school_loc_unique = school_loc.drop_duplicates(subset='NAME')

merge_schools = pd.merge(demo_aca, school_loc_unique, left_on='School Name', right_on='NAME', how='right')
merge_schools.head()
len(merge_schools)

# missing_schools = [
#     {'School Name': 'Acero Chtr Sch Network -  Major Hector P Garcia MD H S', 'LAT': 41.8085443, 'LON': -87.7333591},
#     {'School Name': 'Acero Chtr Sch Network- Sor Juana Ines de la Cruz K-12', 'LAT': 42.0160989, 'LON': -87.6871158},
#     {'School Name': 'Collins Academy STEAM High School', 'LAT': 41.8640799, 'LON': -87.7036933}
# ]

# merge_schools.columns

# for school in missing_schools:
#     mask = merge_schools["School Name"] == school["School Name"]
#     merge_schools.loc[mask, "LAT"] = school["LAT"]
#     merge_schools.loc[mask, "LON"] = school["LON"]

# merge_schools.to_csv('demo_aca_test.csv')

# merge_schools = pd.read_csv('demo_aca_geo.csv')

# Make sure geometry is correct and reprojected
merge_schools_geo = gpd.GeoDataFrame(
    merge_schools,
    geometry=[Point(xy) for xy in zip(merge_schools['LON'], merge_schools['LAT'])],
    crs="EPSG:4326"  # WGS84 lat/lon
)

merge_schools_geo = merge_schools_geo.to_crs(epsg=3857)

merge_schools_geo.crs
cook_clean.crs

merge_schools_geo = merge_schools_geo.to_crs(cook_clean.crs)
merge_schools_buffered = merge_schools_geo.copy()
merge_schools_buffered['geometry'] = merge_schools_buffered.geometry.buffer(1)  # 1 meter buffer

demo_aca_merge = gpd.sjoin(
    cook_clean,
    merge_schools_buffered,
    how='right',
    predicate='intersects'
)

len(demo_aca_merge)
demo_aca_merge.columns

##################################################################################

transport['geometry'] = transport['the_geom'].apply(wkt.loads)

gdf_transport = gpd.GeoDataFrame(
    transport,
    geometry="geometry",
    crs="EPSG:4326"
)

gdf_transport_3857 = gdf_transport.to_crs(epsg=3857)
transport_length = gdf_transport_3857.copy()
transport_length['length'] = gdf_transport_3857.geometry.length

transport_length = transport_length[['geometry', 'length']]
transport_length = transport_length.rename(columns={'length': 'transport_length'})
transport_length
len(transport_length)

transport_merge = gpd.sjoin(cook_clean, transport_length, how='left', predicate='intersects')
transport_merge

transport_sum = (
    transport_merge.groupby(transport_merge.index)['transport_length']
    .sum()
)
transport_sum

transport_merge2 = cook_clean.copy()
transport_merge2['transport_length'] = transport_sum

######################################################################
desert.head()
desert['CensusTract']= desert['CensusTract'].astype(int)
desert_merge = pd.merge(cook_clean, desert, left_on='TRACTID', right_on='CensusTract')
desert_merge.head()

######################################################################
#big merge
def drop_census_cols(gdf):
    cols_to_drop = ["geometry", "TRACTLABEL", "POP20"]
    # Only drop columns that actually exist
    existing_cols = [col for col in cols_to_drop if col in gdf.columns]
    return gdf.drop(columns=existing_cols)

desert_drop = drop_census_cols(desert_merge)
income_drop = drop_census_cols(cook_income)
transport_drop= drop_census_cols(transport_merge2)
demo_aca_drop = drop_census_cols(demo_aca_merge)

merged = cook_clean.merge(demo_aca_drop, on="TRACTID", how="right")
merged = merged.merge(income_drop, on=["TRACTID", 'year'], how="left")
merged = merged.merge(desert_drop, on="TRACTID", how="left")
merged = merged.merge(transport_drop, on="TRACTID", how="left")

merged.to_csv('big_merged.csv')


