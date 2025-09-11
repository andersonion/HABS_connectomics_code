#!/usr/bin/env python3
"""
HABS Maestro metadata builder (verbose, headers fixed, clinical date inference)

- Normalizes headers (spaces -> underscores) across ALL inputs.
- Compiles ordered runnos from connectome dirs (start- and end-patterns).
- Extracts subject (Med_ID without 'H') + visit (0/2 from _y*).
- Pulls Sex/Acq_Date from ADNI_HABS_dual_2years_2_05_2025.csv using (Subject, Visit->BL/M24).
- Loads Genomics*.csv and Clinical HD 1/2/3*.csv (headers normalized).
- Clinical selection per runno:
    1) Expect Visit_ID == 1 for _y0, Visit_ID == 3 for _y2.
    2) If not found, infer by date: choose Interview_Date closest to the imaging target date (BL/M24) from the dual file, searching across HD 1,2,3.
- Cleans values: strings with any whitespace -> 'CUT'; -9999 (numeric or string) -> empty.
- Drops duplicate-identical and constant columns.
- Logs missing Genomics/Clinical together in one CSV.
- Writes output to /mnt/newStor/... (fallback to $WORK if needed).
"""

from __future__ import annotations
import os
import re
import glob
import sys
import traceback
import logging
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Set, Optional

import pandas as pd
import numpy as np

# ----------------------------
# Hardcoded inputs (edit here)
# ----------------------------
DWI_DIR = Path("/mnt/newStor/paros/paros_WORK//harmonization/HABS/connectomes/DWI/plain/")
FMRI_DIR = Path("/mnt/newStor/paros/paros_WORK//harmonization/HABS/connectomes/fMRI/")

# Metadata folder (your corrected path)
METADATA_DIR = Path("/mnt/newStor/paros/paros_WORK/ADNI_HABS_request-545/")
HABS_DUAL_FILE = METADATA_DIR / "ADNI_HABS_dual_2years_2_05_2025.csv"  # first CSV

# Output (primary, then fallback using $WORK if needed)
PRIMARY_OUT_PATH = Path("/mnt/newStor/paros/paros_WORK//harmonization/HABS/metadata/HABS_maestro_metadata.csv")
FALLBACK_OUT_PATH = (Path(os.environ.get("WORK", "")) / "harmonization/HABS/metadata/HABS_maestro_metadata.csv") if os.environ.get("WORK") else None
# ----------------------------

# Loud logs to stdout
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logging.getLogger().handlers[:] = [handler]
logging.getLogger().setLevel(logging.DEBUG)
log = logging.getLogger("habs_maestro")

def _p(msg: str):
    print(msg, flush=True)

RUNNO_PATTERN = re.compile(r"^(H[^_]+)_y([02])($|[^0-9])")
RUNNO_AT_END_PATTERN = re.compile(r"(H[^_/\\]+_y[02])(?=\.csv$)", re.IGNORECASE)

# ---------- utilities ----------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace spaces with underscores in column names."""
    df = df.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df

def parse_date_cols(df: pd.DataFrame, prefer_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert date-like columns to datetime. First try 'prefer_cols' if provided,
    then any column containing 'date' (case-insensitive).
    """
    df = df.copy()
    tried: Set[str] = set()
    if prefer_cols:
        for c in prefer_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
                tried.add(c)
    # Parse any other date-ish columns
    for c in df.columns:
        if c in tried:
            continue
        lc = c.lower()
        if "date" in lc:
            df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
    return df

def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return normalize_columns(df)

def list_runnos_from_dir_start(dirpath: Path) -> List[str]:
    runnos = set()
    for p in dirpath.glob("*"):
        if p.is_file():
            m = RUNNO_PATTERN.match(p.name)
            if m:
                runnos.add(f"{m.group(1)}_y{m.group(2)}")
    return sorted(runnos)

def list_runnos_from_dir_end(dirpath: Path) -> List[str]:
    runnos = set()
    for p in dirpath.glob("**/*.csv"):
        if p.is_file():
            m = RUNNO_AT_END_PATTERN.search(p.name)
            if m:
                runnos.add(m.group(1))
    return sorted(runnos)

def compile_ordered_runnos(dirs_start: Iterable[Path], dirs_end: Iterable[Path]) -> List[str]:
    _p(f"[STEP] Scanning runnos…")
    runnos = set()
    for d in dirs_start:
        found = list_runnos_from_dir_start(d)
        _p(f"  • From {d}: {len(found)} runnos (start-pattern)")
        runnos.update(found)
    for d in dirs_end:
        found = list_runnos_from_dir_end(d)
        _p(f"  • From {d}: {len(found)} runnos (end-pattern)")
        runnos.update(found)
    ordered = sorted(runnos)
    _p(f"[OK] Total unique runnos: {len(ordered)}")
    if ordered:
        _p(f"     First 10: {ordered[:10]}")
    return ordered

def subject_visit_from_runno(runno: str) -> Tuple[str, str, str]:
    if "_y0" in runno:
        v = "0"
    elif "_y2" in runno:
        v = "2"
    else:
        raise ValueError(f"Runno missing visit: {runno}")
    subj = runno.split("_", 1)[0]
    if not subj.startswith("H"):
        raise ValueError(f"Runno subject does not start with H: {runno}")
    return runno, subj[1:], v  # strip leading H

def map_visit_code(visit: str) -> str:
    v = str(visit).strip()
    if v == "0":
        return "BL"
    elif v == "2":
        return "M24"
    else:
        raise ValueError(f"Unexpected visit code: {visit!r}")

def safe_concat_csvs(pattern: str, label: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    _p(f"[STEP] Loading {label}: pattern={pattern} (found {len(paths)} files)")
    frames = []
    for pth in paths:
        try:
            df = pd.read_csv(pth)
            df = normalize_columns(df)
            frames.append(df)
        except Exception as e:
            _p(f"[WARN] Failed to read {pth}: {e}")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    _p(f"[OK] {label}: rows={len(out)}, cols={len(out.columns)}")
    return out

# ---------- lookups ----------
def load_first_csv_table(path: Path) -> pd.DataFrame:
    _p(f"[STEP] Loading first CSV: {path}")
    df = read_table(path)
    df = parse_date_cols(df, prefer_cols=["Acq_Date"])
    _p(f"[OK] Rows: {len(df)}, Cols: {len(df.columns)} (Acq_Date dtype={df.get('Acq_Date', pd.Series()).dtype})")
    required = {"Subject", "Visit", "Sex", "Acq_Date"}
    missing = required - set(df.columns)
    if missing:
        _p(f"[WARN] First CSV missing expected columns: {sorted(missing)}")
    # string-normalize keys
    for c in ["Subject", "Visit"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def build_dual_date_map(df_dual: pd.DataFrame) -> Dict[str, Dict[str, pd.Timestamp]]:
    """
    Returns {subject: {'BL': date_or_NaT, 'M24': date_or_NaT}}
    """
    out: Dict[str, Dict[str, pd.Timestamp]] = {}
    if df_dual.empty:
        return out
    for _, row in df_dual.iterrows():
        subj = str(row.get("Subject", "")).strip()
        visit = str(row.get("Visit", "")).strip()
        acq = row.get("Acq_Date", pd.NaT)
        if not subj:
            continue
        if subj not in out:
            out[subj] = {"BL": pd.NaT, "M24": pd.NaT}
        if visit in ("BL", "M24"):
            out[subj][visit] = acq
    return out

def build_lookup_on_med_id(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        _p(f"[INFO] {label} is empty.")
        return df
    if "Med_ID" not in df.columns:
        _p(f"[WARN] {label} has no 'Med_ID' column. Columns: {list(df.columns)[:12]}…")
        return df
    out = df.copy()
    out["Med_ID"] = out["Med_ID"].astype(str).str.strip()
    # parse dates in clinical/genomics too (Interview_Date etc.)
    out = parse_date_cols(out, prefer_cols=["Interview_Date"])
    _p(f"[OK] {label}: unique Med_IDs={out['Med_ID'].nunique(dropna=True)}")
    return out

# ---------- cleaning ----------
def remove_duplicate_identical_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols, seen = [], []
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

def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Replace -9999 (numeric) and "-9999" (string) with NaN
    - Replace any string cell containing whitespace with 'CUT'
      (dates are parsed to datetime before this, so they won't be affected)
    """
    df = df.copy()

    # -9999 -> NaN
    df.replace(-9999, pd.NA, inplace=True)
    df.replace(to_replace=r'^\s*-9999\s*$', value=pd.NA, regex=True, inplace=True)

    # strings-with-whitespace -> 'CUT' (preserve NaNs)
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            s = df[col].astype("string")
            has_ws = s.str.contains(r"\s", na=False)
            s = s.mask(has_ws, "CUT")
            df[col] = s
    return df

# ---------- clinical selection ----------
def choose_clinical_row_for_run(
    subject: str,
    visit: str,  # "0" or "2"
    target_date: pd.Timestamp,
    clin1: pd.DataFrame,
    clin2: pd.DataFrame,
    clin3: pd.DataFrame
) -> Tuple[Optional[pd.Series], Optional[int]]:
    """
    Returns (selected_row, selected_hd_bucket) where bucket is 1,2,3.
    Preference:
      1) Rows with Visit_ID == expected (1 for y0, 3 for y2), choose nearest Interview_Date to target_date.
      2) Otherwise, across HD 1/2/3, choose the row with Interview_Date nearest to target_date.
    """
    expected_visit_id = 1 if visit == "0" else 3

    candidates: List[Tuple[pd.Series, int]] = []
    for bucket, df in ((1, clin1), (2, clin2), (3, clin3)):
        if df.empty or "Med_ID" not in df.columns:
            continue
        subdf = df[df["Med_ID"] == subject]
        if subdf.empty:
            continue
        # Normalize Visit_ID to numeric (coerce)
        if "Visit_ID" in subdf.columns:
            # Keep original but create a numeric helper
            visit_num = pd.to_numeric(subdf["Visit_ID"], errors="coerce")
            subdf = subdf.assign(_Visit_ID_num=visit_num)
        else:
            subdf = subdf.assign(_Visit_ID_num=pd.NA)

        # Add a date column we can compute distance on: prefer Interview_Date, else any *_Date col
        date_col = None
        if "Interview_Date" in subdf.columns:
            date_col = "Interview_Date"
        else:
            # Find any date-like column already parsed
            for c in subdf.columns:
                if "date" in c.lower():
                    date_col = c
                    break
        if date_col is None:
            # No usable date; still allow selection but distance is inf
            for _, r in subdf.iterrows():
                candidates.append((r, bucket))
        else:
            # Attach distance to target_date (NaT -> inf)
            dt = pd.to_datetime(subdf[date_col], errors="coerce", infer_datetime_format=True)
            if pd.isna(target_date):
                dist = pd.Series(np.inf, index=subdf.index)
            else:
                dist = (dt - target_date).abs()
                # NaT -> NaT.abs() -> NaT, set to inf
                dist = dist.fillna(pd.Timedelta.max)
            subdf = subdf.assign(_dist=dist)
            for _, r in subdf.iterrows():
                candidates.append((r, bucket))

    if not candidates:
        return None, None

    # 1) Try exact Visit_ID match
    exact = [(r, b) for (r, b) in candidates if not pd.isna(r.get("_Visit_ID_num")) and int(r["_Visit_ID_num"]) == expected_visit_id]
    if exact:
        # Among exact, pick smallest distance if available
        exact_with_dist = [ (r, b, r.get("_dist", pd.Timedelta.max)) for (r,b) in exact ]
        exact_with_dist.sort(key=lambda t: (pd.Timedelta.max if pd.isna(t[2]) else t[2]))
        r, b, _ = exact_with_dist[0]
        return r, b

    # 2) No exact Visit_ID → choose nearest by date across all
    with_dist = [ (r, b, r.get("_dist", pd.Timedelta.max)) for (r,b) in candidates ]
    with_dist.sort(key=lambda t: (pd.Timedelta.max if pd.isna(t[2]) else t[2]))
    r, b, _ = with_dist[0]
    return r, b

# ---------- main merge ----------
def main():
    try:
        _p("[START] HABS Maestro metadata build")

        # 1) Runnos
        runnos = compile_ordered_runnos(dirs_start=[DWI_DIR], dirs_end=[FMRI_DIR])
        if not runnos:
            _p("[FATAL] No runnos found. Check filename patterns/paths above.")
            return

        # 2) First CSV
        df_dual = load_first_csv_table(HABS_DUAL_FILE)
        # Build map of BL/M24 imaging dates per subject for inference
        subj_to_visit_dates = build_dual_date_map(df_dual)

        # 3) Genomics
        df_gen_all = safe_concat_csvs(str(METADATA_DIR / "Genomics*.csv"), "Genomics*.csv")
        if not df_gen_all.empty and "Age" in df_gen_all.columns:
            df_gen_all = df_gen_all.drop(columns=["Age"])
        df_gen_all = build_lookup_on_med_id(df_gen_all, "Genomics")

        # 4) Clinical (load 1, 2, and 3)
        df_clin1_all = build_lookup_on_med_id(safe_concat_csvs(str(METADATA_DIR / "Clinical HD 1*.csv"), "Clinical HD 1*.csv"), "Clinical HD 1")
        df_clin2_all = build_lookup_on_med_id(safe_concat_csvs(str(METADATA_DIR / "Clinical HD 2*.csv"), "Clinical HD 2*.csv"), "Clinical HD 2")
        df_clin3_all = build_lookup_on_med_id(safe_concat_csvs(str(METADATA_DIR / "Clinical HD 3*.csv"), "Clinical HD 3*.csv"), "Clinical HD 3")

        # 5) Build rows + track missing
        _p("[STEP] Building merged rows…")
        miss_genomics: Set[str] = set()
        miss_clinical: Set[str] = set()
        rows = []

        for i, runno in enumerate(runnos, 1):
            if i % 50 == 1 or i == len(runnos):
                _p(f"  • Processing {i}/{len(runnos)}: {runno}")

            try:
                runno_str, subject, visit = subject_visit_from_runno(runno)
                visit_code = map_visit_code(visit)

                # First CSV lookup (Sex, Acq_Date) using dual
                sel = df_dual[(df_dual.get("Subject") == subject) & (df_dual.get("Visit") == visit_code)]
                if sel.empty:
                    sex = pd.NA
                    acq_date = pd.NaT
                else:
                    sex = sel.iloc[0].get("Sex", pd.NA)
                    acq_date = sel.iloc[0].get("Acq_Date", pd.NaT)

                # Target date for clinical inference: prefer the visit-coded date from dual map
                target_date = subj_to_visit_dates.get(subject, {}).get(visit_code, pd.NaT)
                if pd.isna(target_date):
                    target_date = acq_date  # fallback to the imaging Acq_Date we pulled for the run

                row = {"runno": runno_str, "Subject": subject, "Sex": sex, "Acq_Date": acq_date}

                # ----- Genomics
                if not df_gen_all.empty and "Med_ID" in df_gen_all.columns:
                    gen_row = df_gen_all[df_gen_all["Med_ID"] == subject]
                    if gen_row.empty:
                        miss_genomics.add(runno_str)
                    else:
                        for k, v in gen_row.iloc[0].to_dict().items():
                            if k not in row:
                                row[k] = v
                else:
                    miss_genomics.add(runno_str)

                # ----- Clinical (choose from HD 1/2/3 with Visit_ID safety + date inference)
                selected, bucket = choose_clinical_row_for_run(subject, visit, target_date, df_clin1_all, df_clin2_all, df_clin3_all)
                if selected is None:
                    miss_clinical.add(runno_str)
                else:
                    row["Clinical_HD_Source"] = bucket  # Keep which bucket we chose (1/2/3)
                    for k, v in selected.to_dict().items():
                        if k not in row:
                            row[k] = v

                rows.append(pd.Series(row))

            except Exception as e:
                _p(f"[ERROR] runno={runno}: {e}")
                traceback.print_exc()

        if not rows:
            _p("[FATAL] No rows produced; aborting.")
            return

        df_out = pd.DataFrame(rows)

        # 5.5) Clean values BEFORE de-dup/constant-drop
        _p("[STEP] Cleaning values: strings with whitespace → 'CUT'; -9999 → empty")
        df_out = clean_values(df_out)

        # 6) Reorder base fields first
        base_order = ["runno", "Subject", "Sex", "Acq_Date", "Clinical_HD_Source"]
        remaining = [c for c in df_out.columns if c not in base_order]
        df_out = df_out[[c for c in base_order if c in df_out.columns] + remaining]
        _p(f"[OK] Raw merged shape (post-clean): {df_out.shape}")

        # 7) Deduplicate identical columns
        before = df_out.shape[1]
        df_out = remove_duplicate_identical_columns(df_out)
        after_dup = df_out.shape[1]
        _p(f"[CLEAN] Dropped {before - after_dup} duplicate-identical columns")

        # 8) Drop constant columns
        before2 = df_out.shape[1]
        df_out = remove_constant_columns(df_out)
        after_const = df_out.shape[1]
        _p(f"[CLEAN] Dropped {before2 - after_const} constant columns")
        _p(f"[OK] Final shape: {df_out.shape}")

        # 9) Write CSV
        out_path = PRIMARY_OUT_PATH
        out_dir = out_path.parent
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(out_path, index=False)
            _p(f"[DONE] Wrote CSV: {out_path}  (rows={df_out.shape[0]}, cols={df_out.shape[1]})")
        except Exception as e:
            _p(f"[WARN] Primary write failed: {e}")
            if FALLBACK_OUT_PATH is None:
                _p("[FATAL] No $WORK fallback available. Aborting.")
                return
            out_path = FALLBACK_OUT_PATH
            out_dir = out_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(out_path, index=False)
            _p(f"[DONE] Wrote CSV (fallback): {out_path}  (rows={df_out.shape[0]}, cols={df_out.shape[1]})")

        # 10) Missing metadata log (same filename as before, now CSV w/ both flags)
        if miss_genomics or miss_clinical:
            miss_path = out_path.with_suffix(".genomics_missing.txt")
            _p(f"[INFO] Writing missing metadata log → {miss_path}")
            all_runnos = sorted(miss_genomics | miss_clinical)
            with open(miss_path, "w") as fh:
                fh.write("runno,missing_genomics,missing_clinical\n")
                for r in all_runnos:
                    fh.write(f"{r},{int(r in miss_genomics)},{int(r in miss_clinical)}\n")
            _p(f"[INFO] Missing counts — Genomics: {len(miss_genomics)}, Clinical: {len(miss_clinical)}")

        _p("[END] Completed successfully.")

    except Exception as e:
        _p(f"[UNCAUGHT] {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 200)
    main()
