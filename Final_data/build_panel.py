#!/usr/bin/env python3
"""Combine processed 2016-2017 and 2018-2025 data into final panel dataset."""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

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
