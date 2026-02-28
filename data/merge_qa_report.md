# Merge QA Report — Cook County High Schools Panel

**Generated**: 2026-02-28 16:21:14

## Row Counts per Year

| Year | Rows | Schools |
|------|------|---------|
| 2016 | 151 | 151 |
| 2017 | 199 | 199 |
| 2018 | 199 | 199 |
| 2019 | 199 | 199 |
| 2020 | 201 | 201 |
| 2021 | 201 | 201 |
| 2022 | 201 | 201 |
| 2023 | 201 | 201 |
| 2024 | 201 | 201 |
| 2025 | 201 | 201 |

**Total**: 1,954 rows, 201 unique schools, 10 years

## Duplicate Key Check

✅ **(school_id, year) is unique** — no duplicates.

## Missingness by Variable

### Overall Missingness

| Variable | Missing % | Missing N |
|----------|-----------|-----------|
| x_ap_coursework | 41.1% | 804 |
| x_attendance_rate | 0.1% | 2 |
| x_dropout_rate | 10.9% | 213 |
| x_mobility_rate | 1.6% | 31 |
| x_pct_asian | 43.9% | 857 |
| x_pct_black | 3.4% | 66 |
| x_pct_el | 34.6% | 677 |
| x_pct_hispanic | 9.5% | 185 |
| x_pct_homeless | 13.3% | 259 |
| x_pct_iep | 0.8% | 15 |
| x_pct_white | 31.4% | 614 |
| x_suspension_rate | 41.1% | 804 |
| x_teacher_attendance | 62.7% | 1226 |
| x_teacher_retention | 6.7% | 130 |
| x_tract_la1and10 | 2.5% | 49 |
| x_tract_la1and20 | 2.5% | 49 |
| x_tract_lahalfand10 | 2.5% | 49 |
| x_tract_lilatracts_1and10 | 2.5% | 49 |
| x_tract_lilatracts_1and20 | 2.5% | 49 |
| x_tract_lilatracts_halfand10 | 2.5% | 49 |
| x_tract_lilatracts_vehicle | 2.5% | 49 |
| x_tract_lowincometracts | 2.5% | 49 |
| x_tract_median_hh_income | 5.5% | 107 |
| x_tract_pop20 | 0.5% | 10 |
| x_tract_pov_share_under_1_00 | 1.1% | 21 |
| x_tract_total_n_hh | 1.1% | 21 |
| x_tract_transport_length | 0.5% | 10 |
| x_tract_urban | 2.5% | 49 |
| y_chronic_abs | 0.2% | 3 |
| y_ela_prof | 18.8% | 368 |
| y_grad_4yr | 1.9% | 38 |
| y_math_prof | 21.0% | 410 |

### Missingness by Year (Key Variables)

| Year | y_chronic_abs | y_ela_prof | y_grad_4yr | y_math_prof | x_enrollment | x_pct_low_income | x_tract_median_hh_income | x_tract_pov_share_under_1_00 | x_tract_transport_length |
|------|------|------|------|------|------|------|------|------|------|
| 2016 | 0% | 2% | 3% | 3% | 0% | 0% | 0% | 0% | 0% |
| 2017 | 0% | 3% | 4% | 3% | 0% | 0% | 1% | 1% | 0% |
| 2018 | 0% | 2% | 2% | 2% | 0% | 0% | 3% | 3% | 0% |
| 2019 | 1% | 1% | 2% | 1% | 0% | 0% | 3% | 3% | 0% |
| 2020 | 0% | 100% | 1% | 100% | 0% | 0% | 0% | 0% | 0% |
| 2021 | 0% | 8% | 1% | 8% | 0% | 0% | 32% | 0% | 0% |
| 2022 | 0% | 1% | 2% | 1% | 0% | 0% | 2% | 1% | 1% |
| 2023 | 0% | 1% | 1% | 1% | 0% | 0% | 3% | 1% | 1% |
| 2024 | 0% | 39% | 1% | 45% | 0% | 0% | 5% | 1% | 1% |
| 2025 | 0% | 26% | 2% | 40% | 0% | 0% | 5% | 1% | 1% |

## Range Checks for Percent Variables

| Variable | Min | Max | Outliers (>100 or <0) |
|----------|-----|-----|----------------------|
| x_attendance_rate | 36.6 | 99.0 | ✓ 0 |
| x_dropout_rate | 0.0 | 66.2 | ✓ 0 |
| x_pct_asian | 0.0 | 36.8 | ✓ 0 |
| x_pct_black | 0.3 | 100.0 | ✓ 0 |
| x_pct_el | 0.1 | 80.5 | ✓ 0 |
| x_pct_hispanic | 0.1 | 100.0 | ✓ 0 |
| x_pct_homeless | 0.0 | 60.2 | ✓ 0 |
| x_pct_iep | 1.2 | 56.0 | ✓ 0 |
| x_pct_low_income | 1.1 | 100.0 | ✓ 0 |
| x_pct_white | 0.0 | 85.8 | ✓ 0 |
| x_teacher_retention | 0.0 | 100.0 | ✓ 0 |
| x_tract_pov_share_under_1_00 | 0.0 | 0.8 | ✓ 0 |
| y_chronic_abs | 0.0 | 100.0 | ✓ 0 |
| y_ela_prof | 0.0 | 100.0 | ✓ 0 |
| y_grad_4yr | 0.0 | 100.0 | ✓ 0 |
| y_math_prof | 0.0 | 97.2 | ✓ 0 |

## ACS Year Alignment

See [acs_year_alignment.md](acs_year_alignment.md) for the full rule.

| School Year | ACS End Year |
|-------------|-------------|
| 2016 | 2016 |
| 2017 | 2017 |
| 2018 | 2018 |
| 2019 | 2019 |
| 2020 | 2020 |
| 2021 | 2021 |
| 2022 | 2022 |
| 2023 | 2023 |
| 2024 | 2024 |
| 2025 | 2024 |

## Column Inventory

Total columns: 45

| Group | Columns |
|-------|---------|
| Y variables | y_chronic_abs, y_ela_prof, y_grad_4yr, y_math_prof |
| X school-level | x_ap_coursework, x_attendance_rate, x_dropout_rate, x_enrollment, x_mobility_rate, x_pct_asian, x_pct_black, x_pct_el, x_pct_hispanic, x_pct_homeless, x_pct_iep, x_pct_low_income, x_pct_white, x_suspension_rate, x_teacher_attendance, x_teacher_retention |
| X tract-level | x_tract_median_hh_income, x_tract_total_n_hh, x_tract_pov_share_under_1_00, x_tract_urban, x_tract_lilatracts_1and10, x_tract_lilatracts_halfand10, x_tract_lilatracts_1and20, x_tract_lilatracts_vehicle, x_tract_lowincometracts, x_tract_la1and10, x_tract_lahalfand10, x_tract_la1and20, x_tract_transport_length, x_tract_pop20 |
| Metadata | school_id, school_name, district, county, school_type, grades_served, year, acs_end_year, TRACTID, LAT, LON |