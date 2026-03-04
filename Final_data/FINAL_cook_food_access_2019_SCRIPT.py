import pandas as pd
import zipfile
import re

# 1) cook county good accesss
zip_path = r"food_access_data_2019.zip"

with zipfile.ZipFile(zip_path) as z:
    with z.open("Food Access Research Atlas.csv") as f:
        food_access = pd.read_csv(f)

cook_food_access = food_access[
    (food_access["State"].str.strip().str.lower() == "illinois") &
    (food_access["County"].str.strip().str.lower() == "cook county")
].copy()

assert not cook_food_access.empty, "No Cook County rows found."
assert cook_food_access["State"].nunique() == 1
assert cook_food_access["County"].nunique() == 1

keep_cols = [
    "State", "County", "CensusTract", "Urban",
    "LILATracts_1And10", "LILATracts_halfAnd10", "LILATracts_1And20", "LILATracts_Vehicle",
    "LowIncomeTracts", "LA1and10", "LAhalfand10", "LA1and20",
]

cook_subset = cook_food_access[keep_cols].copy()

cook_subset["CensusTract"] = (
    cook_subset["CensusTract"].astype(str).str.replace(r"\.0$", "", regex=True)
)

assert set(cook_subset["Urban"].dropna().unique()).issubset({0, 1})

cook_subset.to_csv("cook_food_access_2019.csv", index=False)
