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
from datetime import datetime

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Final_Data")
demo = pd.read_csv(os.path.join(DATA, "cook_county_high_school_demographics.csv"))
transport = pd.read_csv(os.path.join(DATA,'CTA_-_Bus_Routes_20260129.csv'))
academic = pd.read_csv(os.path.join(DATA,"panel_yx_highschools_base_new.csv"))
cook = gpd.read_file(os.path.join(DATA,'cook_county/PVS_25_v2_tracts2020_17031.shp'))
income = pd.read_csv(os.path.join(DATA,'cook_tract_income_v7_2016_2024.csv'))
desert = pd.read_csv(os.path.join(DATA,'cook_food_access_2019.csv'))
seg = pd.read_csv(os.path.join(DATA,'school_segregation_all_final.csv'))

school_loc = gpd.read_file(os.path.join(DATA,'Public_School_Locations_2021-22.geojson'))

###########################################################################
# Clean census level data, keeping only columns with less than 50% NAs
cook_3857 = cook.to_crs(epsg=3857)
cook_3857.head()
threshold = 0.5
cook_clean = cook_3857.loc[:, cook_3857.apply(lambda col: ((col.isna()) | (col.isnull())).mean() <= threshold)]
cook_clean = cook_clean[['TRACTID', 'TRACTLABEL', 'POP20', 'geometry']]
cook_clean['TRACTID'] = cook_clean['TRACTID'].astype(int)


###########################################################################
# clean the income data, adding np.nan to negative income values, and merging with cook_income
income['year'].unique()
income['median_hh_income'] = income['median_hh_income'].apply(
    lambda x: np.nan if x is not None and x < 0 else x
)
income['GEOID'] = income['GEOID'].astype(int)
cook_income = pd.merge(cook_clean, income, left_on='TRACTID', right_on='GEOID', how='right')
cook_income = cook_income[cook_income['TRACTID'].notna()]

# income.columns
# na_counts = cook_income.isna().sum()
# na_counts
# len(income)
# len(cook_clean)
# rows_with_na = cook_income[cook_income.isna().any(axis=1)]
# rows_with_na


###########################################################################
demo = demo.replace("*", np.nan)
demographic = demo.loc[:, demo.isna().mean() <= 0.5]
demographic = demographic[demographic['County']=='Cook']
demographic = demographic[['RCDTS', 'School Name']]

academic.columns
academic_seg = pd.merge(academic, seg, left_on=['school_name', 'year'], right_on=['school', 'year'], how='left')

demo_aca = pd.merge(demographic, academic_seg, left_on = 'RCDTS', right_on ='school_id', how='outer')
school_loc_unique = school_loc.drop_duplicates(subset='NAME')

merge_schools = pd.merge(demo_aca, school_loc_unique, left_on='School Name', right_on='NAME', how='left')

###########################################################################

missing_schools = [
    {'School Name': 'Acero Chtr Sch Network -  Major Hector P Garcia MD H S', 'LAT': 41.8085443, 'LON': -87.7333591},
    {'School Name': 'Acero Chtr Sch Network- Sor Juana Ines de la Cruz K-12', 'LAT': 42.0160989, 'LON': -87.6871158},
    {'School Name': 'Collins Academy STEAM High School', 'LAT': 41.8640799, 'LON': -87.7036933}
]

for school in missing_schools:
    mask = merge_schools["School Name"] == school["School Name"]
    merge_schools.loc[mask, "LAT"] = school["LAT"]
    merge_schools.loc[mask, "LON"] = school["LON"]

###########################################################################

merge_schools_geo = gpd.GeoDataFrame(
    merge_schools,
    geometry=[Point(xy) for xy in zip(merge_schools['LON'], merge_schools['LAT'])],
    crs="EPSG:4326"  # WGS84 lat/lon
)

merge_schools_geo = merge_schools_geo.to_crs(epsg=3857)
merge_schools_geo = merge_schools_geo.to_crs(cook_clean.crs)


demo_aca_merge = gpd.sjoin(
    cook_clean,
    merge_schools_geo,
    how='right',
    predicate='intersects'
)
#demo_aca_merge.to_csv('merge_test_aca.csv')
###########################################################################

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


transport_merge = gpd.sjoin(cook_clean, transport_length, how='left', predicate='intersects')

transport_sum = (
    transport_merge.groupby(transport_merge.index)['transport_length']
    .sum()
)

transport_merge2 = cook_clean.copy()
transport_merge2['transport_length'] = transport_sum

######################################################################

desert['CensusTract']= desert['CensusTract'].astype(int)
desert_merge = pd.merge(cook_clean, desert, left_on='TRACTID', right_on='CensusTract')

######################################################################
#big merge
def clean_gdf_keys(gdf):

    cols_to_drop = ["geometry", "TRACTLABEL", "POP20"]
    existing_cols = [col for col in cols_to_drop if col in gdf.columns]
    gdf = gdf.drop(columns=existing_cols)

    if 'TRACTID' in gdf.columns:
        gdf['TRACTID'] = gdf['TRACTID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

    if 'year' in gdf.columns:
        gdf['year'] = gdf['year'].astype(int)

    return gdf

desert_drop = clean_gdf_keys(desert_merge)
income_drop = clean_gdf_keys(cook_income)
transport_drop = clean_gdf_keys(transport_merge2)
demo_aca_drop = clean_gdf_keys(demo_aca_merge)


cook_clean_unique = cook_clean.drop_duplicates(subset=["TRACTID"]).copy()
cook_clean_unique['TRACTID'] = cook_clean_unique['TRACTID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
if 'year' in cook_clean_unique.columns:
    cook_clean_unique['year'] = cook_clean_unique['year'].astype(int)

merged = cook_clean_unique.merge(demo_aca_drop, on="TRACTID", how="right")
merged = merged.merge(income_drop, on=["TRACTID", "year"], how="left")
merged = merged.merge(desert_drop, on="TRACTID", how="left")
merged = merged.merge(transport_drop, on="TRACTID", how="left")

merged_clean = merged.drop(columns=['TRACTID', 'TRACTLABEL', 'index_left', 
                                    'RCDTS', 'school_id', 'geometry', 
                                    'LAT', 'LON', 'OBJECTID','CensusTract', 'GEOID', 
                                    'TRACTLABEL','RCDTS', 'school_id', 'school_name', 
                                    'NCESSCH', 'LEAID', 'NAME', 'OPSTFIPS', 'STREET', 'CITY', 
                                    'STATE', 'ZIP', 'STFIP', 'CNTY', 'NMCNTY', 'LOCALE',
                                    'CBSA', 'NMCBSA', 'CBSATYPE', 'CSA', 
                                    'NMCSA', 'NECTA', 'NMNECTA', 'CD', 'SLDL', 'SLDU', 
                                    'SCHOOLYEAR', 'State','County'])
merged_clean.head(10)
len(merged)
merged_clean.columns
merged_test = merged_clean.groupby('year')['x_enrollment'].sum().reset_index()
merged_test

merged_clean.to_csv('final_merged.csv')
