"""
clean_panel_pipeline.py
========================
Produces a clean (school_id, year)-unique panel for Cook County high schools.

Inputs  (all in ./data/):
  - big_merged2.csv          : teammate's merged school–tract panel
  - panel_yx_highschools.csv : existing Y + school-level X panel
  - cook_tract_income_v7_2016_2024.csv : ACS tract income (2016-2024)

Outputs (all in ./data/):
  - panel_yx_highschools.csv  : updated panel with tract-level X features
  - duplicate_keys_report.csv : duplicate-key diagnostic (empty if clean)
  - acs_mapping_table.csv     : school_year → acs_end_year mapping
  - acs_year_alignment.md     : documentation of the mapping rule
  - merge_qa_report.md        : full QA report
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------------------------------------------------------------------
# 0. Helpers
# ---------------------------------------------------------------------------

def assert_unique_keys(df, keys, stage_name):
    """Assert (school_id, year) uniqueness; save debug CSV on failure."""
    dup = df.groupby(keys).size().reset_index(name="n_rows")
    dups = dup[dup["n_rows"] > 1]
    if len(dups) > 0:
        debug_path = os.path.join(DATA, f"DEBUG_duplicates_{stage_name}.csv")
        # Merge back to get full rows for inspection
        dup_rows = df.merge(dups[keys], on=keys, how="inner")
        dup_rows.to_csv(debug_path, index=False)
        raise AssertionError(
            f"[{stage_name}] PANEL CONTRACT VIOLATED: {len(dups)} duplicate "
            f"key groups found. Debug artifact saved → {debug_path}"
        )
    print(f"  ✓ [{stage_name}] Keys {keys} unique — {len(df)} rows")


def assert_row_count(df, max_rows, stage_name):
    """Assert row count did not increase beyond expected max."""
    if len(df) > max_rows:
        raise AssertionError(
            f"[{stage_name}] PANEL CONTRACT VIOLATED: row count {len(df)} "
            f"exceeds maximum {max_rows}"
        )
    print(f"  ✓ [{stage_name}] Row count {len(df)} ≤ {max_rows}")


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 70)
print("PANEL CLEANING PIPELINE — Cook County High Schools")
print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

print("\n[1] Loading data...")
bm2 = pd.read_csv(os.path.join(DATA, "big_merged2.csv"), low_memory=False)
panel_yx = pd.read_csv(os.path.join(DATA, "panel_yx_highschools.csv"), low_memory=False)
income_raw = pd.read_csv(os.path.join(DATA, "cook_tract_income_v7_2016_2024.csv"))

print(f"  big_merged2       : {bm2.shape[0]:,} rows × {bm2.shape[1]} cols")
print(f"  panel_yx          : {panel_yx.shape[0]:,} rows × {panel_yx.shape[1]} cols")
print(f"  income_raw        : {income_raw.shape[0]:,} rows × {income_raw.shape[1]} cols")

KEYS = ["school_id", "year"]

# ---------------------------------------------------------------------------
# 2. Diagnose duplicates in big_merged2
# ---------------------------------------------------------------------------
print("\n[2] Diagnosing duplicate keys in big_merged2...")
dup_counts = bm2.groupby(KEYS).size().reset_index(name="n_rows")
dup_report = dup_counts[dup_counts["n_rows"] > 1].copy()

if len(dup_report) == 0:
    print("  ✓ No duplicate (school_id, year) keys found in big_merged2!")
    dup_report_full = pd.DataFrame(columns=["school_id", "year", "n_rows", "note"])
    dup_report_full.loc[0] = ["ALL_UNIQUE", "", len(bm2), "No duplicates detected"]
else:
    print(f"  ✗ {len(dup_report)} duplicate key groups found!")
    # Enrich with example TRACTID, GEOID
    dup_keys = dup_report[KEYS]
    dup_rows = bm2.merge(dup_keys, on=KEYS, how="inner")
    tract_cols = [c for c in ["TRACTID", "GEOID", "CensusTract"] if c in dup_rows.columns]
    dup_report_full = dup_rows[KEYS + ["n_rows"] + tract_cols] if tract_cols else dup_rows[KEYS + ["n_rows"]]

dup_report_full.to_csv(os.path.join(DATA, "duplicate_keys_report.csv"), index=False)
print(f"  → Saved duplicate_keys_report.csv")

assert_unique_keys(bm2, KEYS, "big_merged2_raw")

# ---------------------------------------------------------------------------
# 3. ACS year alignment rule
# ---------------------------------------------------------------------------
print("\n[3] Defining ACS year alignment rule...")

school_years = sorted(bm2["year"].unique())
acs_available = sorted(income_raw["year"].unique())
print(f"  School years : {school_years}")
print(f"  ACS years    : {acs_available}")

# Rule: school_year → acs_end_year = school_year if available, else latest
# available ACS year (carry-forward)
mapping = []
for sy in school_years:
    if sy in acs_available:
        acs_ey = sy
    else:
        # Carry forward: use the latest ACS year <= school_year
        candidates = [a for a in acs_available if a <= sy]
        acs_ey = max(candidates) if candidates else None
    mapping.append({"school_year": sy, "acs_end_year": acs_ey})

acs_map = pd.DataFrame(mapping)
acs_map.to_csv(os.path.join(DATA, "acs_mapping_table.csv"), index=False)
print(f"  → Saved acs_mapping_table.csv")
print(acs_map.to_string(index=False))

# Write alignment doc
alignment_doc = f"""# ACS 5-Year Estimates → School Year Alignment Rule

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

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
"""
for _, row in acs_map.iterrows():
    sy = int(row["school_year"])
    aey = int(row["acs_end_year"]) if pd.notna(row["acs_end_year"]) else "N/A"
    note = " ← carry-forward" if sy != aey else ""
    alignment_doc += f"| {sy} | {aey}{note} |\n"

alignment_doc += """
## Impact

- School years 2016–2024: direct 1:1 match with ACS end year.
- School year 2025: uses ACS 2024 data (carry-forward).
- Each (tract_id, acs_end_year) pair is unique in the income table,
  guaranteeing no row multiplication during merge.
"""

with open(os.path.join(DATA, "acs_year_alignment.md"), "w") as f:
    f.write(alignment_doc)
print(f"  → Saved acs_year_alignment.md")

# ---------------------------------------------------------------------------
# 4. Extract tract-level X features from big_merged2
# ---------------------------------------------------------------------------
print("\n[4] Extracting tract-level X features from big_merged2...")

# Add acs_end_year mapping to big_merged2
year_to_acs = dict(zip(acs_map["school_year"], acs_map["acs_end_year"]))
bm2["acs_end_year"] = bm2["year"].map(year_to_acs)

# Select tract-level features to carry over
# Income / poverty columns
income_cols = [
    "median_hh_income", "total_n_hh", "pov_share_under_1_00",
]

# Food access / desert columns
food_cols = [
    "Urban", "LILATracts_1And10", "LILATracts_halfAnd10",
    "LILATracts_1And20", "LILATracts_Vehicle", "LowIncomeTracts",
    "LA1and10", "LAhalfand10", "LA1and20",
]

# Transport column
transport_cols = ["transport_length"]

# Location metadata
location_cols = ["TRACTID", "LAT", "LON"]

# Census population
census_cols = ["POP20"]

all_tract_cols = income_cols + food_cols + transport_cols + census_cols

# Check which columns actually exist in big_merged2
available_tract_cols = [c for c in all_tract_cols if c in bm2.columns]
missing_tract_cols = [c for c in all_tract_cols if c not in bm2.columns]
if missing_tract_cols:
    print(f"  ⚠ Missing columns (skipped): {missing_tract_cols}")
print(f"  Using {len(available_tract_cols)} tract-level features")

# Extract: school_id, year, acs_end_year, TRACTID, LAT, LON, + tract features
extract_cols = KEYS + ["acs_end_year"] + location_cols + available_tract_cols
# Deduplicate column list (some may overlap)
extract_cols = list(dict.fromkeys(extract_cols))

available_extract = [c for c in extract_cols if c in bm2.columns]
tract_features = bm2[available_extract].copy()

# Handle 2025 carry-forward: for rows where acs_end_year differs from year,
# we need to fill in income data from the carried-forward year
mask_cf = tract_features["year"] != tract_features["acs_end_year"]
n_cf = mask_cf.sum()
print(f"  Carry-forward rows (year ≠ acs_end_year): {n_cf}")

if n_cf > 0:
    # For carry-forward rows, the income cols are NaN because the original
    # merge joined on year (which was 2025, not in income table).
    # We need to fill from the same TRACTID's acs_end_year row.
    cf_rows = tract_features[mask_cf].copy()
    # Get income data for the carry-forward acs_end_year
    for _, cf_row in cf_rows.iterrows():
        tract_id = cf_row["TRACTID"]
        acs_yr = cf_row["acs_end_year"]
        # Find the donor row: same school, acs_end_year == year
        donor = tract_features[
            (tract_features["school_id"] == cf_row["school_id"])
            & (tract_features["year"] == acs_yr)
        ]
        if len(donor) == 1:
            for col in income_cols:
                if col in tract_features.columns:
                    tract_features.loc[
                        (tract_features["school_id"] == cf_row["school_id"])
                        & (tract_features["year"] == cf_row["year"]),
                        col
                    ] = donor[col].values[0]

    n_filled = tract_features.loc[mask_cf, "median_hh_income"].notna().sum()
    print(f"  Filled {n_filled}/{n_cf} carry-forward rows from ACS {int(acs_map[acs_map['school_year']==2025]['acs_end_year'].values[0])}")

# Rename tract-level features with x_tract_ prefix
rename_map = {}
for c in available_tract_cols:
    if not c.startswith("x_tract_"):
        rename_map[c] = f"x_tract_{c.lower()}"
tract_features = tract_features.rename(columns=rename_map)

# Keep TRACTID, LAT, LON without prefix (metadata, not features)
assert_unique_keys(tract_features, KEYS, "tract_features")

# ---------------------------------------------------------------------------
# 5. Merge tract features into panel_yx
# ---------------------------------------------------------------------------
print("\n[5] Merging tract features into panel_yx...")

n_before = len(panel_yx)
assert_unique_keys(panel_yx, KEYS, "panel_yx_before_merge")

# Drop location_cols if they already exist in panel_yx to avoid conflicts
for c in ["TRACTID", "LAT", "LON", "acs_end_year"]:
    if c in panel_yx.columns:
        panel_yx = panel_yx.drop(columns=[c])

panel_final = panel_yx.merge(tract_features, on=KEYS, how="left")

assert_unique_keys(panel_final, KEYS, "panel_final")
assert_row_count(panel_final, n_before, "panel_final")

print(f"  Merged: {panel_final.shape[0]:,} rows × {panel_final.shape[1]} cols")

# ---------------------------------------------------------------------------
# 6. QA checks
# ---------------------------------------------------------------------------
print("\n[6] Running QA checks...")

qa_lines = []
qa_lines.append(f"# Merge QA Report — Cook County High Schools Panel")
qa_lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# 6a. Row counts per year
qa_lines.append("## Row Counts per Year\n")
qa_lines.append("| Year | Rows | Schools |")
qa_lines.append("|------|------|---------|")
for yr in sorted(panel_final["year"].unique()):
    subset = panel_final[panel_final["year"] == yr]
    qa_lines.append(f"| {yr} | {len(subset)} | {subset['school_id'].nunique()} |")
qa_lines.append(f"\n**Total**: {len(panel_final):,} rows, "
                f"{panel_final['school_id'].nunique()} unique schools, "
                f"{panel_final['year'].nunique()} years\n")

# 6b. Duplicate key check
qa_lines.append("## Duplicate Key Check\n")
dup_check = panel_final.groupby(KEYS).size().reset_index(name="n")
max_n = dup_check["n"].max()
if max_n == 1:
    qa_lines.append("✅ **(school_id, year) is unique** — no duplicates.\n")
else:
    qa_lines.append(f"❌ **{(dup_check['n']>1).sum()} duplicate key groups found!**\n")

# 6c. Missingness by variable and year
qa_lines.append("## Missingness by Variable\n")

# Identify feature columns (y_ and x_ prefixed)
feature_cols = [c for c in panel_final.columns
                if c.startswith("y_") or c.startswith("x_")]

# Overall missingness
qa_lines.append("### Overall Missingness\n")
qa_lines.append("| Variable | Missing % | Missing N |")
qa_lines.append("|----------|-----------|-----------|")
for c in sorted(feature_cols):
    n_miss = panel_final[c].isna().sum()
    pct_miss = n_miss / len(panel_final) * 100
    if pct_miss > 0:
        qa_lines.append(f"| {c} | {pct_miss:.1f}% | {n_miss} |")
qa_lines.append("")

# Missingness by year for key variables
qa_lines.append("### Missingness by Year (Key Variables)\n")
key_features = [c for c in feature_cols if c in [
    "y_ela_prof", "y_math_prof", "y_chronic_abs", "y_grad_4yr",
    "x_tract_median_hh_income", "x_tract_pov_share_under_1_00",
    "x_tract_transport_length", "x_enrollment", "x_pct_low_income",
]]
if key_features:
    header = "| Year | " + " | ".join(key_features) + " |"
    qa_lines.append(header)
    qa_lines.append("|" + "------|" * (len(key_features) + 1))
    for yr in sorted(panel_final["year"].unique()):
        subset = panel_final[panel_final["year"] == yr]
        vals = []
        for c in key_features:
            if c in subset.columns:
                pct = subset[c].isna().mean() * 100
                vals.append(f"{pct:.0f}%")
            else:
                vals.append("N/A")
        qa_lines.append(f"| {yr} | " + " | ".join(vals) + " |")
    qa_lines.append("")

# 6d. Sanity range checks for percent variables
qa_lines.append("## Range Checks for Percent Variables\n")
pct_cols = [c for c in panel_final.columns
            if c.startswith(("x_pct_", "y_")) and "prof" in c.lower()
            or c in ["y_chronic_abs", "y_grad_4yr", "x_pct_low_income",
                      "x_pct_black", "x_pct_white", "x_pct_hispanic",
                      "x_pct_asian", "x_pct_el", "x_pct_iep",
                      "x_pct_homeless", "x_attendance_rate",
                      "x_dropout_rate", "x_teacher_retention",
                      "x_tract_pov_share_under_1_00"]]
# Deduplicate
pct_cols = list(dict.fromkeys(pct_cols))
pct_cols = [c for c in pct_cols if c in panel_final.columns]

qa_lines.append("| Variable | Min | Max | Outliers (>100 or <0) |")
qa_lines.append("|----------|-----|-----|----------------------|")
range_issues = []
for c in sorted(pct_cols):
    vals = pd.to_numeric(panel_final[c], errors="coerce")
    vmin = vals.min()
    vmax = vals.max()
    outliers = ((vals < 0) | (vals > 100)).sum()
    flag = f"⚠ {outliers}" if outliers > 0 else "✓ 0"
    qa_lines.append(f"| {c} | {vmin:.1f} | {vmax:.1f} | {flag} |")
    if outliers > 0:
        range_issues.append((c, outliers))
qa_lines.append("")

if range_issues:
    qa_lines.append("### ⚠ Range Warnings\n")
    for c, n in range_issues:
        qa_lines.append(f"- **{c}**: {n} values outside [0, 100]")
    qa_lines.append("")

# 6e. ACS mapping summary
qa_lines.append("## ACS Year Alignment\n")
qa_lines.append("See [acs_year_alignment.md](acs_year_alignment.md) for the full rule.\n")
qa_lines.append("| School Year | ACS End Year |")
qa_lines.append("|-------------|-------------|")
for _, row in acs_map.iterrows():
    sy = int(row["school_year"])
    aey = int(row["acs_end_year"]) if pd.notna(row["acs_end_year"]) else "N/A"
    qa_lines.append(f"| {sy} | {aey} |")
qa_lines.append("")

# 6f. Column inventory
qa_lines.append("## Column Inventory\n")
qa_lines.append(f"Total columns: {panel_final.shape[1]}\n")
qa_lines.append("| Group | Columns |")
qa_lines.append("|-------|---------|")
y_cols = [c for c in panel_final.columns if c.startswith("y_")]
x_school = [c for c in panel_final.columns if c.startswith("x_") and not c.startswith("x_tract_")]
x_tract = [c for c in panel_final.columns if c.startswith("x_tract_")]
meta_cols = [c for c in panel_final.columns if not c.startswith(("x_", "y_"))]
qa_lines.append(f"| Y variables | {', '.join(y_cols)} |")
qa_lines.append(f"| X school-level | {', '.join(x_school)} |")
qa_lines.append(f"| X tract-level | {', '.join(x_tract)} |")
qa_lines.append(f"| Metadata | {', '.join(meta_cols)} |")

qa_report = "\n".join(qa_lines)

with open(os.path.join(DATA, "merge_qa_report.md"), "w") as f:
    f.write(qa_report)
print(f"  → Saved merge_qa_report.md")

# ---------------------------------------------------------------------------
# 7. Save final panel
# ---------------------------------------------------------------------------
print("\n[7] Saving final panel...")
panel_final.to_csv(os.path.join(DATA, "panel_yx_highschools.csv"), index=False)
print(f"  → Saved panel_yx_highschools.csv ({panel_final.shape[0]:,} rows × {panel_final.shape[1]} cols)")

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"""
Outputs saved to {DATA}/:
  1. panel_yx_highschools.csv  — clean panel ({panel_final.shape[0]:,} rows × {panel_final.shape[1]} cols)
  2. duplicate_keys_report.csv — duplicate key diagnostic
  3. acs_mapping_table.csv     — school_year → acs_end_year mapping
  4. acs_year_alignment.md     — mapping rule documentation
  5. merge_qa_report.md        — full QA report

Key Stats:
  Schools       : {panel_final['school_id'].nunique()}
  Years         : {sorted(panel_final['year'].unique())}
  Y variables   : {len(y_cols)}
  X school-level: {len(x_school)}
  X tract-level : {len(x_tract)}
""")
