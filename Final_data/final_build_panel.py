import pandas as pd
import numpy as np
import os
import csv
import zipfile
import re
import pandas as pd
import numpy as np
import os
import csv
import pandas as pd
import numpy as np
import os
import re

########################################
#Dominick's food desert

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


#################################################################################
#ZZ's 3 files to create panel_xy
#!/usr/bin/env python3
"""Combine processed 2016-2017 and 2018-2025 data into final panel dataset."""


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'final_data')
#!/usr/bin/env python3
"""Process 2016 and 2017 Illinois Report Card text files for Cook County high schools."""

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_DIR), 'final_data')


def format_rcdts(rcdts):
    """Format 15-char RCDTS code as XX-XXX-XXXX-XX-XXXX."""
    r = rcdts.strip()
    return f"{r[0:2]}-{r[2:5]}-{r[5:9]}-{r[9:11]}-{r[11:]}"


def format_grades(grades_str):
    """Format grades from '9 10 11 12' to '9 - 12'."""
    parts = grades_str.strip().split()
    nums = []
    for p in parts:
        try:
            nums.append(int(p))
        except ValueError:
            pass
    if not nums:
        return grades_str.strip()
    if len(nums) == 1:
        return str(nums[0])
    return f"{min(nums)} - {max(nums)}"


def safe_float(val):
    """Convert field value to float, NaN if empty/invalid."""
    v = val.strip().replace(',', '')
    if v == '' or v == '.':
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def process_year(main_file, assessment_file, year, field_map, ela_field, math_field):
    """Process one year of text-format report card data."""
    rows = []
    with open(main_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row) < 13:
                continue
            county = row[6].strip()
            school_type = row[11].strip()
            if county == 'Cook' and school_type in ['HIGH SCHOOL', 'CHARTER SCH']:
                data = {
                    'school_id': format_rcdts(row[0]),
                    'school_name': row[3],
                    'district': row[4],
                    'county': row[6],
                    'school_type': 'High School',
                    'grades_served': format_grades(row[12]),
                    'year': year,
                }
                for col, idx in field_map.items():
                    if idx is not None and idx < len(row):
                        data[col] = safe_float(row[idx])
                    else:
                        data[col] = np.nan
                rows.append(data)

    # Read assessment file for ELA/Math proficiency
    assessment = {}
    with open(assessment_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row) <= max(ela_field, math_field):
                continue
            rcdts = row[0].strip()
            if rcdts:
                assessment[rcdts] = {
                    'y_ela_prof': safe_float(row[ela_field]),
                    'y_math_prof': safe_float(row[math_field]),
                }

    for data in rows:
        raw_rcdts = data['school_id'].replace('-', '')
        if raw_rcdts in assessment:
            data['y_ela_prof'] = assessment[raw_rcdts]['y_ela_prof']
            data['y_math_prof'] = assessment[raw_rcdts]['y_math_prof']
        else:
            data['y_ela_prof'] = np.nan
            data['y_math_prof'] = np.nan

    return pd.DataFrame(rows)


# RC16 field mappings (0-indexed)
RC16_FIELDS = {
    'y_chronic_abs': 133,
    'y_grad_4yr': 141,
    'x_ap_coursework': None,
    'x_attendance_rate': 69,
    'x_dropout_rate': 137,
    'x_enrollment': 20,
    'x_mobility_rate': 125,
    'x_pct_asian': 16,
    'x_pct_black': 14,
    'x_pct_el': 45,
    'x_pct_hispanic': 15,
    'x_pct_homeless': 57,
    'x_pct_iep': 49,
    'x_pct_low_income': 53,
    'x_pct_white': 13,
    'x_suspension_rate': None,
    'x_teacher_attendance': 1418,
    'x_teacher_retention': 571,
}

# RC17 field mappings (0-indexed)
RC17_FIELDS = {
    'y_chronic_abs': 181,
    'y_grad_4yr': 241,
    'x_ap_coursework': None,
    'x_attendance_rate': 69,
    'x_dropout_rate': 185,
    'x_enrollment': 20,
    'x_mobility_rate': 125,
    'x_pct_asian': 16,
    'x_pct_black': 14,
    'x_pct_el': 45,
    'x_pct_hispanic': 15,
    'x_pct_homeless': 57,
    'x_pct_iep': 49,
    'x_pct_low_income': 53,
    'x_pct_white': 13,
    'x_suspension_rate': None,
    'x_teacher_attendance': 1462,
    'x_teacher_retention': 615,
}


if __name__ == '__main__':
    df16 = process_year(
        os.path.join(DATA_DIR, 'rc16.txt'),
        os.path.join(DATA_DIR, 'rc16_assessment.txt'),
        2016, RC16_FIELDS,
        ela_field=258, math_field=262,
    )
    print(f"2016: {len(df16)} schools")

    df17 = process_year(
        os.path.join(DATA_DIR, 'rc17.txt'),
        os.path.join(DATA_DIR, 'rc17_assessment.txt'),
        2017, RC17_FIELDS,
        ela_field=262, math_field=266,
    )
    print(f"2017: {len(df17)} schools")

    df = pd.concat([df16, df17], ignore_index=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = os.path.join(OUTPUT_DIR, 'rc16_rc17_processed.csv')
    df.to_csv(outpath, index=False)
    print(f"Saved {len(df)} rows to {outpath}")
##################################################
#!/usr/bin/env python3
"""Process 2018-2025 Illinois Report Card XLSX files for Cook County high schools."""

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_DIR), 'final_data')

XLSX_FILES = {
    2018: '18-Report-Card-Public-Data-Set.xlsx',
    2019: '2019-Report-Card-Public-Data-Set.xlsx',
    2020: '2020-Report-Card-Public-Data-Set.xlsx',
    2021: '2021-RC-Pub-Data-Set.xlsx',
    2022: '2022-Report-Card-Public-Data-Set.xlsx',
    2023: '23-RC-Pub-Data-Set.xlsx',
    2024: '24-RC-Pub-Data-Set.xlsx',
    2025: '2025-Report-Card-Public-Data-Set.xlsx',
}


def format_rcdts(rcdts):
    """Standardize RCDTS to XX-XXX-XXXX-XX-XXXX format."""
    s = str(rcdts).strip()
    if '-' in s and len(s) >= 19:
        return s  # already formatted
    # Raw 15-char format
    s = s.replace('-', '').replace(' ', '')
    if len(s) >= 15:
        return f"{s[0:2]}-{s[2:5]}-{s[5:9]}-{s[9:11]}-{s[11:]}"
    return s


def find_col(df, patterns, exact=False):
    """Find first column matching any pattern (case-insensitive substring)."""
    for col in df.columns:
        cl = str(col).lower()
        for p in patterns:
            if exact:
                if cl == p.lower():
                    return col
            else:
                if p.lower() in cl:
                    return col
    return None


def find_sheet(xl, patterns):
    """Find sheet name matching patterns."""
    for s in xl.sheet_names:
        sl = s.lower()
        for p in patterns:
            if p.lower() in sl:
                return s
    return None


def get_school_level_mask(df, rcdts_col):
    """Filter to school-level rows (RCDTS school part != 0000)."""
    def is_school_level(rcdts):
        s = str(rcdts)
        parts = s.split('-')
        if len(parts) == 5:
            return parts[4] != '0000'
        # Handle unformatted 15-char codes
        if len(s) >= 15 and s.replace(' ', '').isalnum():
            return s[-4:] != '0000'
        return True
    return df[rcdts_col].apply(is_school_level)


def process_xlsx_year(filepath, year):
    """Process one year of XLSX report card data."""
    print(f"  Processing {year}...")
    xl = pd.ExcelFile(filepath)

    # --- Read General sheet ---
    gen_sheet = find_sheet(xl, ['general'])
    # Avoid picking General (2)
    for s in xl.sheet_names:
        if s.lower() == 'general':
            gen_sheet = s
            break

    gen = pd.read_excel(filepath, sheet_name=gen_sheet, header=0, dtype=str)

    rcdts_col = find_col(gen, ['rcdts'])
    county_col = find_col(gen, ['county'])
    stype_col = find_col(gen, ['school type'])
    level_col = find_col(gen, ['type'], exact=True) or find_col(gen, ['level'], exact=True)

    # Filter to school-level Cook County high schools
    mask = gen[county_col].str.strip().str.lower() == 'cook'
    mask &= gen[stype_col].str.strip().str.upper().isin(['HIGH SCHOOL', 'CHARTER SCH'])
    if level_col:
        mask &= gen[level_col].str.strip().str.lower() == 'school'
    else:
        mask &= get_school_level_mask(gen, rcdts_col)

    gen = gen[mask].copy()
    gen = gen.drop_duplicates(subset=[rcdts_col])

    # Build output dataframe
    result = pd.DataFrame()
    result['school_id'] = gen[rcdts_col].apply(format_rcdts)
    result['school_name'] = gen[find_col(gen, ['school name'])].values
    result['district'] = gen[find_col(gen, ['district'])].values

    # Read county as-is
    result['county'] = gen[county_col].values
    result['school_type'] = gen[stype_col].values

    # Grades served
    gc = find_col(gen, ['grades served', 'grades'])
    if gc:
        def format_grades_xlsx(g):
            """Format grades from '9 10 11 12' to '9 - 12'."""
            s = str(g).strip()
            if not s or s == 'nan':
                return s
            parts = s.split()
            nums = []
            has_letter = False
            for p in parts:
                try:
                    nums.append(int(p))
                except ValueError:
                    has_letter = True
            if has_letter:
                # Contains K, PK, etc — format as "K - max"
                if nums:
                    prefix = [p for p in parts if not p.isdigit()]
                    return f"{prefix[0]} - {max(nums)}"
                return s
            if not nums:
                return s
            if len(nums) == 1:
                return str(nums[0])
            return f"{min(nums)} - {max(nums)}"

        if year <= 2018:
            result['grades_served'] = gen[gc].apply(format_grades_xlsx)
        else:
            result['grades_served'] = gen[gc].values
    else:
        result['grades_served'] = np.nan

    result['year'] = year

    # --- Numeric columns from General sheet ---
    col_map = {
        'y_chronic_abs': ['chronic absenteeism'],
        'y_grad_4yr': ['4-year graduation rate - total', '4 year graduation rate - total',
                        '4-year grad rate', 'hs 4-year graduation rate'],
        'x_attendance_rate': ['student attendance rate'],
        'x_dropout_rate': ['dropout rate - total', 'high school dropout rate - total'],
        'x_enrollment': ['# student enrollment', 'student enrollment - total'],
        'x_mobility_rate': ['student mobility rate', 'mobility rate'],
        'x_pct_asian': ['% student enrollment - asian', 'enrollment - asian %'],
        'x_pct_black': ['% student enrollment - black', 'enrollment - black'],
        'x_pct_el': ['% student enrollment - el', 'enrollment - el %'],
        'x_pct_hispanic': ['% student enrollment - hispanic', 'enrollment - hispanic'],
        'x_pct_homeless': ['% student enrollment - homeless', 'enrollment - homeless %'],
        'x_pct_iep': ['% student enrollment - iep', 'enrollment - iep %'],
        'x_pct_low_income': ['% student enrollment - low income', 'enrollment - low income %'],
        'x_pct_white': ['% student enrollment - white', 'enrollment - white %'],
        'x_suspension_rate': ['% crdc in-school suspensions'],
        'x_teacher_attendance': ['teacher attendance rate'],
        'x_teacher_retention': ['teacher retention rate'],
    }

    for target_col, patterns in col_map.items():
        src = find_col(gen, patterns)
        if src:
            result[target_col] = pd.to_numeric(gen[src].values, errors='coerce')
        else:
            result[target_col] = np.nan

    # --- AP coursework (use CRDC column from General sheet) ---
    ap_col = find_col(gen, ['crdc advanced placement coursework'])
    if ap_col:
        result['x_ap_coursework'] = pd.to_numeric(gen[ap_col].values, errors='coerce')
    else:
        result['x_ap_coursework'] = np.nan

    # --- ELA / Math proficiency from ELAMathScience sheet ---
    ela_sheet = find_sheet(xl, ['elamath', 'ela and math', 'ela math'])
    if ela_sheet:
        ela_df = pd.read_excel(filepath, sheet_name=ela_sheet, header=0, dtype=str)
        ela_rcdts = find_col(ela_df, ['rcdts'])
        if ela_rcdts:
            ela_prof_col = find_col(ela_df, ['% ela proficiency', 'ela proficiency total %'])
            math_prof_col = find_col(ela_df, ['% math proficiency', 'math proficiency total %'])

            # Filter to school level
            ela_level = find_col(ela_df, ['type'], exact=True) or find_col(ela_df, ['level'], exact=True)
            if ela_level:
                ela_df = ela_df[ela_df[ela_level].str.strip().str.lower() == 'school']
            else:
                ela_df = ela_df[get_school_level_mask(ela_df, ela_rcdts)]

            ela_df['_rcdts_fmt'] = ela_df[ela_rcdts].apply(format_rcdts)
            ela_df = ela_df.drop_duplicates(subset=['_rcdts_fmt'])

            if ela_prof_col:
                ela_map = dict(zip(ela_df['_rcdts_fmt'],
                                   pd.to_numeric(ela_df[ela_prof_col], errors='coerce')))
                result['y_ela_prof'] = result['school_id'].map(ela_map)
            else:
                result['y_ela_prof'] = np.nan

            if math_prof_col:
                math_map = dict(zip(ela_df['_rcdts_fmt'],
                                    pd.to_numeric(ela_df[math_prof_col], errors='coerce')))
                result['y_math_prof'] = result['school_id'].map(math_map)
            else:
                result['y_math_prof'] = np.nan
        else:
            result['y_ela_prof'] = np.nan
            result['y_math_prof'] = np.nan
    else:
        result['y_ela_prof'] = np.nan
        result['y_math_prof'] = np.nan

    xl.close()
    print(f"    {year}: {len(result)} schools")
    return result


if __name__ == '__main__':
    frames = []
    for year, fname in sorted(XLSX_FILES.items()):
        fpath = os.path.join(DATA_DIR, fname)
        df = process_xlsx_year(fpath, year)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = os.path.join(OUTPUT_DIR, 'xlsx_2018_2025_processed.csv')
    combined.to_csv(outpath, index=False)
    print(f"\nSaved {len(combined)} rows to {outpath}")

##################################################

COLUMNS = [
    'school_id', 'school_name', 'district', 'county', 'school_type',
    'grades_served', 'year',
    'y_chronic_abs', 'y_ela_prof', 'y_grad_4yr', 'y_math_prof',
    'x_ap_coursework', 'x_attendance_rate', 'x_dropout_rate', 'x_enrollment',
    'x_mobility_rate', 'x_pct_asian', 'x_pct_black', 'x_pct_el',
    'x_pct_hispanic', 'x_pct_homeless', 'x_pct_iep', 'x_pct_low_income',
    'x_pct_white', 'x_suspension_rate', 'x_teacher_attendance',
    'x_teacher_retention',
]


if __name__ == '__main__':
    rc_path = os.path.join(DATA_DIR, 'rc16_rc17_processed.csv')
    xlsx_path = os.path.join(DATA_DIR, 'xlsx_2018_2025_processed.csv')

    df_rc = pd.read_csv(rc_path)
    df_xlsx = pd.read_csv(xlsx_path)

    print(f"RC16/17: {len(df_rc)} rows")
    print(f"XLSX: {len(df_xlsx)} rows")

    combined = pd.concat([df_rc, df_xlsx], ignore_index=True)

    # Define panel: schools present in the most recent year (2025)
    recent = combined[combined['year'] == 2025]
    panel_ids = set(recent['school_id'].unique())
    print(f"Panel schools (from 2025 data): {len(panel_ids)}")

    # Filter to panel schools only
    combined = combined[combined['school_id'].isin(panel_ids)].copy()

    # Ensure correct column order (exclude TRACTID, LAT, LON)
    for col in COLUMNS:
        if col not in combined.columns:
            combined[col] = np.nan
    combined = combined[COLUMNS]

    # Sort by school_id, year
    combined = combined.sort_values(['school_id', 'year']).reset_index(drop=True)

    # Ensure year is integer
    combined['year'] = combined['year'].astype(int)

    outpath = os.path.join(DATA_DIR, 'panel_yx_highschools_base_new.csv')
    combined.to_csv(outpath, index=False)
    print(f"\nSaved {len(combined)} rows to {outpath}")
    print(f"Unique schools: {combined['school_id'].nunique()}")
    print(f"Years: {sorted(combined['year'].unique())}")
    print(f"Schools per year:")
    print(combined.groupby('year')['school_id'].nunique())
