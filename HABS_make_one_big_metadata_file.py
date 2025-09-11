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
    out
