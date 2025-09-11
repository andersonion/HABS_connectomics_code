#!/usr/bin/env python3
"""
HABS Maestro metadata builder

- Compiles ordered runnos from connectome dirs
- Extracts subject (Med_ID without leading 'H') + visit (0/2 from _y*)
- Pulls Sex/Acq_Date from ADNI_HABS_dual_2years_2_05_2025.csv using (Subject, Visit->BL/M24)
- Merges Genomics*.csv on Med_ID (excluding 'Age'); logs no-matches
- Merges Clinical HD 1*.csv (visit 0) or Clinical HD 3*.csv (visit 2) on Med_ID
- Drops duplicate-identical columns and constant columns
- Writes CSV to /mnt/newStor/...; if that path isn’t writable/doesn’t exist, retries with $WORK
"""

from __future__ import annotations
import os
import re
import glob
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

# ----------------------------
# Hardcoded inputs (edit here)
# ----------------------------
DWI_DIR = Path("/mnt/newStor/paros/paros_WORK//harmonization/HABS/connectomes/DWI/plain/")
FMRI_DIR = Path("/mnt/newStor/paros/paros_WORK//harmonization/HABS/connectomes/fMRI/")

# UPDATED per your note:
METADATA_DIR = Path("/mnt/newStor/paros/paros_WORK/ADNI_HABS_request-545/")

HABS_DUAL_FILE = METADATA_DIR / "ADNI_HABS_dual_2years_2_05_2025.csv"  # first CSV

# Output (primary, then fallback using $WORK if needed)
PRIMARY_OUT_PATH = Path("/mnt/newStor/paros/paros_WORK//harmonization/HABS/metadata/HABS_maestro_metadata.csv")
FALLBACK_OUT_PATH = (Path(os.environ.get("WORK", "")) / "harmonization/HABS/metadata/HABS_maestro_metadata.csv") if os.environ.get("WORK") else None

LOG_LEVEL = logging.INFO
# ----------------------------

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("habs_maestro")

RUNNO_PATTERN = re.compile(r"^(H[^_]+)_y([02])($|[^0-9])")  # matches "H12345_y0..." or "H12345_y2..." at start
RUNNO_AT_END_PATTERN = re.compile(r"(H[^_/\\]+_y[02])(?=\.csv$)", re.IGNORECASE)  # matches "...H12345_y0.csv" at end


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV or XLSX with pandas."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def list_runnos_from_dir_start(dirpath: Path) -> List[str]:
    """
    Files begin with subject id and _y0/_y2 (e.g., H12345_y0_connectome.csv).
    Extract runno from the start: H12345_y0
    """
    runnos = set()
    for p in dirpath.glob("*"):
        if p.is_file():
            m = RUNNO_PATTERN.match(p.name)
            if m:
                runnos.add(f"{m.group(1)}_y{m.group(2)}")
    return sorted(runnos)


def list_runnos_from_dir_end(dirpath: Path) -> List[str]:
    """
    Files end with the pattern right before .csv (e.g., something.../whatever/H12345_y2.csv).
    Extract runno from the end: H12345_y2
    """
    runnos = set()
    for p in dirpath.glob("**/*.csv"):
        if p.is_file():
            m = RUNNO_AT_END_PATTERN.search(p.name)
            if m:
                runnos.add(m.group(1))
    return sorted(runnos)


def compile_ordered_runnos(dirs_start: Iterable[Path], dirs_end: Iterable[Path]) -> List[str]:
    runnos = set()
    for d in dirs_start:
        runnos.update(list_runnos_from_dir_start(d))
    for d in dirs_end:
        runnos.update(list_runnos_from_dir_end(d))
    ordered = sorted(runnos)  # lexicographic (H..., then _y0 before _y2)
    log.info("Compiled %d unique runnos", len(ordered))
    return ordered


def subject_visit_from_runno(runno: str) -> Tuple[str, str, str]:
    """
    Given runno like 'H12345_y0' return:
      - runno (as given)
      - subject (strip leading 'H' -> '12345')
      - visit ('0' or '2')
    """
    if "_y0" in runno:
        v = "0"
    elif "_y2" in runno:
        v = "2"
    else:
        raise ValueError(f"Runno missing visit: {runno}")
    subj = runno.split("_", 1)[0]
    if not subj.startswith("H"):
        raise ValueError(f"Runno subject does not start with H: {runno}")
    return runno, subj[1:], v


def map_visit_code(visit: str) -> str:
    """Map visit '0'->'BL', '2'->'M24'."""
    v = str(visit).strip()
    if v == "0":
        return "BL"
    elif v == "2":
        return "M24"
    else:
        raise ValueError(f"Unexpected visit code: {visit!r}")


def load_first_csv_table(path: Path) -> pd.DataFrame:
    df = read_table(path)
    # Normalize expected columns
    required_cols = {"Subject", "Visit", "Sex", "Acq_Date"}
    missing = required_cols - set(df.columns)
    if missing:
        log.warning("First CSV missing expected columns: %s", ", ".join(sorted(missing)))
    # String-normalize key columns
    for c in ["Subject", "Visit"]:
        if c in df.columns:
            df[c] = df[c].astype(str).strip()
    return df


def safe_concat_csvs(pattern: str) -> pd.DataFrame:
    """Concat all CSVs matching pattern (glob) into one DataFrame; empty if none."""
    paths = glob.glob(pattern)
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:
            log.warning("Failed to read %s: %s", p, e)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def build_lookup_dual(df_dual: pd.DataFrame) -> pd.DataFrame:
    """Key for lookups: ('Subject', 'Visit')."""
    df = df_dual.copy()
    if "Subject" in df.columns:
        df["Subject"] = df["Subject"].astype(str).str.strip()
    if "Visit" in df.columns:
        df["Visit"] = df["Visit"].astype(str).str.strip()
    return df


def build_lookup_on_med_id(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Med_ID column to string trimmed."""
    if df.empty:
        return df
    if "Med_ID" not in df.columns:
        log.warning("Expected Med_ID column not found in dataframe with columns: %s", list(df.columns))
        return df
    out = df.copy()
    out["Med_ID"] = out["Med_ID"].astype(str).str.strip()
    return out


def drop_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name in df.columns:
        return df.drop(columns=[name])
    return df


def merge_row_for_runno(runno: str,
                        df_dual: pd.DataFrame,
                        df_genomics_all: pd.DataFrame,
                        df_clin1_all: pd.DataFrame,
                        df_clin3_all: pd.DataFrame,
                        genomics_miss_log: List[str]) -> pd.Series:
    """Build a single row Series for one runno."""
    runno_str, subject, visit = subject_visit_from_runno(runno)
    visit_code = map_visit_code(visit)

    row = {"runno": runno_str, "Subject": subject}

    # First CSV: pull Sex, Acq_Date using (Subject, Visit)
    sel = df_dual[(df_dual["Subject"] == subject) & (df_dual["Visit"] == visit_code)]
    if sel.empty:
        log.warning("No match in first CSV for Subject=%s, Visit=%s (runno=%s)", subject, visit_code, runno_str)
        sex = pd.NA
        acq = pd.NA
    else:
        # If multiple, take first
        sex = sel.iloc[0].get("Sex", pd.NA)
        acq = sel.iloc[0].get("Acq_Date", pd.NA)
    row["Sex"] = sex
    row["Acq_Date"] = acq

    # Genomics: merge all fields (except Age) on Med_ID == subject
    gen_row = df_genomics_all[df_genomics_all.get("Med_ID", pd.Series([], dtype=object)) == subject]
    if gen_row.empty:
        genomics_miss_log.append(runno_str)
        genomics_values = {}
    else:
        gr = gen_row.iloc[0].drop(labels=[c for c in ["Age"] if c in gen_row.columns], errors="ignore")
        genomics_values = gr.to_dict()

    # Clinical: choose dataset by visit and merge on Med_ID
    clin_df = df_clin1_all if visit == "0" else df_clin3_all
    clin_row = clin_df[clin_df.get("Med_ID", pd.Series([], dtype=object)) == subject]
    clinical_values = {} if clin_row.empty else clin_row.iloc[0].to_dict()

    # Compose final row
    for k, v in genomics_values.items():
        if k not in row:
            row[k] = v
    for k, v in clinical_values.items():
        if k not in row:
            row[k] = v

    return pd.Series(row)


def remove_duplicate_identical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with identical values across all rows (true duplicates).
    Keep the first occurrence encountered.
    """
    keep_cols = []
    seen = []
    for col in df.columns:
        ser = df[col]
        signature = tuple((pd.isna(x), None if pd.isna(x) else x) for x in ser)
        if signature in seen:
            continue
        seen.append(signature)
        keep_cols.append(col)
    return df.loc[:, keep_cols]


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    mask = df.apply(lambda s: s.nunique(dropna=False) > 1)
    return df.loc[:, list(mask[mask].index)]


def main():
    # 1) Compile ordered runnos
    runnos = compile_ordered_runnos(
        dirs_start=[DWI_DIR],
        dirs_end=[FMRI_DIR],
    )
    if not runnos:
        log.error("No runnos found. Check your directories.")
        return

    # 2) Load first CSV
    df_dual_raw = load_first_csv_table(HABS_DUAL_FILE)
    df_dual = build_lookup_dual(df_dual_raw)

    # 3) Load Genomics (concat all Genomics*.csv)
    genomics_glob = str(METADATA_DIR / "Genomics*.csv")
    df_gen_all = safe_concat_csvs(genomics_glob)
    if df_gen_all.empty:
        log.warning("No Genomics*.csv files found under %s", METADATA_DIR)
    df_gen_all = build_lookup_on_med_id(df_gen_all)
    # Exclude Age field from genomics dataset itself (not from output base fields)
    if not df_gen_all.empty and "Age" in df_gen_all.columns:
        df_gen_all = df_gen_all.drop(columns=["Age"])

    # 4) Load Clinical (visit 0 -> 'Clinical HD 1'*.csv, visit 2 -> 'Clinical HD 3'*.csv)
    clin1_glob = str(METADATA_DIR / "Clinical HD 1*.csv")
    clin3_glob = str(METADATA_DIR / "Clinical HD 3*.csv")
    df_clin1_all = build_lookup_on_med_id(safe_concat_csvs(clin1_glob))
    df_clin3_all = build_lookup_on_med_id(safe_concat_csvs(clin3_glob))
    if df_clin1_all.empty:
        log.warning("No Clinical HD 1*.csv files found")
    if df_clin3_all.empty:
        log.warning("No Clinical HD 3*.csv files found")

    # 5) Build rows
    genomics_miss_log: List[str] = []
    rows = []
    for r in runnos:
        try:
            row = merge_row_for_runno(
                r, df_dual, df_gen_all, df_clin1_all, df_clin3_all, genomics_miss_log
            )
            rows.append(row)
        except Exception as e:
            log.error("Failed processing runno %s: %s", r, e)

    if not rows:
        log.error("No rows produced; aborting.")
        return

    df_out = pd.DataFrame(rows)

    # 6) Reorder: ensure base fields first if present
    base_order = ["runno", "Subject", "Sex", "Acq_Date"]
    remaining = [c for c in df_out.columns if c not in base_order]
    df_out = df_out[[c for c in base_order if c in df_out.columns] + remaining]

    # 7) Remove duplicate-identical columns
    before = df_out.shape[1]
    df_out = remove_duplicate_identical_columns(df_out)
    after_dupdrop = df_out.shape[1]
    log.info("Dropped %d duplicate-identical columns", before - after_dupdrop)

    # 8) Remove constant columns
    before2 = df_out.shape[1]
    df_out = remove_constant_columns(df_out)
    after_constdrop = df_out.shape[1]
    log.info("Dropped %d constant columns", before2 - after_constdrop)

    # 9) Write CSV (primary, then fallback to $WORK)
    out_path = PRIMARY_OUT_PATH
    out_dir = out_path.parent
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        log.info("Wrote %s (%d rows, %d cols)", out_path, df_out.shape[0], df_out.shape[1])
    except Exception as e:
        log.warning("Primary write failed: %s", e)
        if FALLBACK_OUT_PATH is None:
            log.error("No $WORK fallback available. Aborting.")
            return
        out_path = FALLBACK_OUT_PATH
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        log.info("Wrote (fallback) %s (%d rows, %d cols)", out_path, df_out.shape[0], df_out.shape[1])

    # 10) Report missing genomics
    if genomics_miss_log:
        log.warning("Genomics metadata missing for %d runnos", len(genomics_miss_log))
        miss_path = out_path.with_suffix(".genomics_missing.txt")
        with open(miss_path, "w") as fh:
            for r in genomics_miss_log:
                fh.write(r + "\n")
        log.info("Wrote genomics-missing log: %s", miss_path)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 200)
    main()
