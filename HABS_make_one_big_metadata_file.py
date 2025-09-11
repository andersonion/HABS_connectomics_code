#!/usr/bin/env python3
"""
HABS Maestro metadata builder (verbose, with extra cleaning)

Tweaks:
- Log BOTH missing Genomics and missing Clinical to the SAME file.
- Replace any string cell that contains spaces with 'CUT'.
- Replace -9999 (numeric or string) with empty (NaN).
"""

from __future__ import annotations
import os
import re
import glob
import sys
import traceback
import logging
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Set

import pandas as pd

# ----------------------------
# Hardcoded inputs (edit here)
# ----------------------------
DWI_DIR = Path("/mnt/newStor/paros/paros_WORK//harmonization/HABS/connectomes/DWI/plain/")
FMRI_DIR = Path("/mnt/newStor/paros/paros_WORK//harmonization/HABS/connectomes/fMRI/")

# Your corrected metadata folder:
METADATA_DIR = Path("/mnt/newStor/paros/paros_WORK/ADNI_HABS_request-545/")

HABS_DUAL_FILE = METADATA_DIR / "ADNI_HABS_dual_2years_2_05_2025.csv"  # first CSV

# Output (primary, then fallback using $WORK if needed)
PRIMARY_OUT_PATH = Path("/mnt/newStor/paros/paros_WORK//harmonization/HABS/metadata/HABS_maestro_metadata.csv")
FALLBACK_OUT_PATH = (Path(os.environ.get("WORK", "")) / "harmonization/HABS/metadata/HABS_maestro_metadata.csv") if os.environ.get("WORK") else None
# ----------------------------

# Make logs visible on stdout AND force flush so you see them live.
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logging.getLogger().handlers[:] = [handler]
logging.getLogger().setLevel(logging.DEBUG)
log = logging.getLogger("habs_maestro")

def _p(msg: str):
    print(msg, flush=True)

RUNNO_PATTERN = re.compile(r"^(H[^_]+)_y([02])($|[^0-9])")
RUNNO_AT_END_PATTERN = re.compile(r"(H[^_/\\]+_y[02])(?=\.csv$)", re.IGNORECASE)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


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


def load_first_csv_table(path: Path) -> pd.DataFrame:
    _p(f"[STEP] Loading first CSV: {path}")
    df = read_table(path)
    _p(f"[OK] Rows: {len(df)}, Cols: {len(df.columns)}")
    required = {"Subject", "Visit", "Sex", "Acq_Date"}
    missing = required - set(df.columns)
    if missing:
        _p(f"[WARN] First CSV missing expected columns: {sorted(missing)}")
    for c in ["Subject", "Visit"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def safe_concat_csvs(pattern: str, label: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    _p(f"[STEP] Loading {label}: pattern={pattern} (found {len(paths)} files)")
    frames = []
    for pth in paths:
        try:
            frames.append(pd.read_csv(pth))
        except Exception as e:
            _p(f"[WARN] Failed to read {pth}: {e}")
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    _p(f"[OK] {label}: rows={len(df)}, cols={len(df.columns)}")
    return df


def build_lookup_dual(df_dual: pd.DataFrame) -> pd.DataFrame:
    df = df_dual.copy()
    for c in ("Subject", "Visit"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def build_lookup_on_med_id(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        _p(f"[INFO] {label} is empty (no files or no rows).")
        return df
    if "Med_ID" not in df.columns:
        _p(f"[WARN] {label} has no 'Med_ID' column. Columns: {list(df.columns)[:10]}…")
        return df
    out = df.copy()
    out["Med_ID"] = out["Med_ID"].astype(str).str.strip()
    nunique = out["Med_ID"].nunique(dropna=True)
    _p(f"[OK] {label}: unique Med_IDs={nunique}")
    return out


def merge_row_for_runno(runno: str,
                        df_dual: pd.DataFrame,
                        df_genomics_all: pd.DataFrame,
                        df_clin1_all: pd.DataFrame,
                        df_clin3_all: pd.DataFrame,
                        miss_genomics: Set[str],
                        miss_clinical: Set[str]) -> pd.Series:
    runno_str, subject, visit = subject_visit_from_runno(runno)
    visit_code = map_visit_code(visit)

    row = {"runno": runno_str, "Subject": subject}

    # First CSV
    sel = df_dual[(df_dual.get("Subject") == subject) & (df_dual.get("Visit") == visit_code)]
    if sel.empty:
        sex = pd.NA
        acq = pd.NA
    else:
        sex = sel.iloc[0].get("Sex", pd.NA)
        acq = sel.iloc[0].get("Acq_Date", pd.NA)
    row["Sex"] = sex
    row["Acq_Date"] = acq

    # Genomics (exclude Age)
    if not df_genomics_all.empty and "Med_ID" in df_genomics_all.columns:
        gen_row = df_genomics_all[df_genomics_all["Med_ID"] == subject]
        if gen_row.empty:
            miss_genomics.add(runno_str)
        else:
            gr = gen_row.iloc[0].drop(labels=[c for c in ["Age"] if c in gen_row.columns], errors="ignore")
            for k, v in gr.to_dict().items():
                if k not in row:
                    row[k] = v
    else:
        # If dataset missing entirely, count all as missing genomics
        miss_genomics.add(runno_str)

    # Clinical (by visit)
    clin_df = df_clin1_all if visit == "0" else df_clin3_all
    if not clin_df.empty and "Med_ID" in clin_df.columns:
        clin_row = clin_df[clin_df["Med_ID"] == subject]
        if clin_row.empty:
            miss_clinical.add(runno_str)
        else:
            for k, v in clin_row.iloc[0].to_dict().items():
                if k not in row:
                    row[k] = v
    else:
        # If the selected clinical dataset is empty, count as missing
        miss_clinical.add(runno_str)

    return pd.Series(row)


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
    - Replace any string cell containing one or more whitespace characters with 'CUT'
    - Replace numeric -9999 and string '-9999' with NaN
    """
    df = df.copy()

    # Replace -9999 (numeric) and "-9999" (string) with NaN
    df.replace(-9999, pd.NA, inplace=True)
    df.replace(to_replace=r'^\s*-9999\s*$', value=pd.NA, regex=True, inplace=True)

    # For string-like columns, replace any value containing whitespace with 'CUT'
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            s = df[col].astype("string")  # preserves NaN as <NA>
            has_space = s.str.contains(r"\s", na=False)
            # Only replace those with whitespace; leave NaNs alone
            s = s.mask(has_space, "CUT")
            df[col] = s
    return df


def main():
    try:
        _p("[START] HABS Maestro metadata build")

        # 1) Runnos
        runnos = compile_ordered_runnos(dirs_start=[DWI_DIR], dirs_end=[FMRI_DIR])
        if not runnos:
            _p("[FATAL] No runnos found. Check filename patterns/paths above.")
            return

        # 2) First CSV
        df_dual_raw = load_first_csv_table(HABS_DUAL_FILE)
        df_dual = build_lookup_dual(df_dual_raw)

        # 3) Genomics
        df_gen_all = safe_concat_csvs(str(METADATA_DIR / "Genomics*.csv"), "Genomics*.csv")
        if not df_gen_all.empty and "Age" in df_gen_all.columns:
            df_gen_all = df_gen_all.drop(columns=["Age"])
        df_gen_all = build_lookup_on_med_id(df_gen_all, "Genomics")

        # 4) Clinical
        df_clin1_all = safe_concat_csvs(str(METADATA_DIR / "Clinical HD 1*.csv"), "Clinical HD 1*.csv")
        df_clin3_all = safe_concat_csvs(str(METADATA_DIR / "Clinical HD 3*.csv"), "Clinical HD 3*.csv")
        df_clin1_all = build_lookup_on_med_id(df_clin1_all, "Clinical HD 1")
        df_clin3_all = build_lookup_on_med_id(df_clin3_all, "Clinical HD 3")

        # 5) Build rows + track missing
        _p("[STEP] Building merged rows…")
        miss_genomics: Set[str] = set()
        miss_clinical: Set[str] = set()
        rows = []
        for i, r in enumerate(runnos, 1):
            if i % 50 == 1 or i == len(runnos):
                _p(f"  • Processing {i}/{len(runnos)}: {r}")
            try:
                rows.append(
                    merge_row_for_runno(
                        r, df_dual, df_gen_all, df_clin1_all, df_clin3_all,
                        miss_genomics, miss_clinical
                    )
                )
            except Exception as e:
                _p(f"[ERROR] runno={r}: {e}")
                traceback.print_exc()

        if not rows:
            _p("[FATAL] No rows produced; aborting.")
            return

        df_out = pd.DataFrame(rows)

        # 5.5) Clean values BEFORE de-dup/constant-drop
        _p("[STEP] Cleaning values: spacey strings → 'CUT'; -9999 → empty")
        df_out = clean_values(df_out)

        # 6) Reorder
        base_order = ["runno", "Subject", "Sex", "Acq_Date"]
        remaining = [c for c in df_out.columns if c not in base_order]
        df_out = df_out[[c for c in base_order if c in df_out.columns] + remaining]
        _p(f"[OK] Raw merged shape (post-clean): {df_out.shape}")

        # 7) Drop duplicate-identical columns
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

        # 10) Missing metadata log (both genomics & clinical) to the SAME file as before
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
