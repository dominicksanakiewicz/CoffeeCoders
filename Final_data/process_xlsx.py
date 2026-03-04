#!/usr/bin/env python3
"""Process 2018-2025 Illinois Report Card XLSX files for Cook County high schools."""

import pandas as pd
import numpy as np
import os
import re

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_DIR), 'data')

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
