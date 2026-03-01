# ACS 5-Year Estimates → School Year Alignment Rule

**Generated**: 2026-02-28 16:21:13

## Rule

Each `school_year` maps to the ACS 5-year estimate whose **end year** equals
the school year. When no ACS data is available for a given school year
(e.g. 2025), we **carry forward** the most recent available ACS end year.

## Rationale

- ACS 5-year estimates (e.g., 2019–2023) are labeled by their **end year** (2023).
- The `cook_tract_income_v7_2016_2024.csv` file provides one row per
  (GEOID, year) where `year` already represents the ACS end year.
- For school year 2025, ACS data ending in 2025 is not yet released,
  so we carry forward the 2024 estimates. This is standard practice.

## Mapping Table

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
| 2025 | 2024 ← carry-forward |

## Impact

- School years 2016–2024: direct 1:1 match with ACS end year.
- School year 2025: uses ACS 2024 data (carry-forward).
- Each (tract_id, acs_end_year) pair is unique in the income table,
  guaranteeing no row multiplication during merge.
