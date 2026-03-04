#!/usr/bin/env python3
"""Process 2016 and 2017 Illinois Report Card text files for Cook County high schools."""

import csv
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_DIR), 'data')


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
