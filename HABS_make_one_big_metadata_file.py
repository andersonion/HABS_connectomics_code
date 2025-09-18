#!/usr/bin/env python3
"""
HABS Maestro metadata builder — v3 (robust composite-key AB merge)

Drop-in replacement for earlier scripts. Produces debug CSVs and a timestamped backup.
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
from datetime import datetime

import pandas as pd
import numpy as np

# ---------- USER EDITABLE ----------
DWI_DIR = Path("/mnt/newStor/paros/paros_WORK//harmonization/HABS/connectomes/DWI/plain/")
FMRI_DIR = Path("/mnt/newStor/paros/paros_WORK//harmonization/HABS/connectomes/fMRI/")

METADATA_DIR = Path("/mnt/newStor/paros/paros_WORK/ADNI_HABS_request-545/")
HABS_DUAL_FILE = METADATA_DIR / "ADNI_HABS_dual_2years_2_05_2025.csv"

HABS_AB_FILE = Path("/mnt/newStor/paros/paros_WORK//harmonization/HABS/metadata/HABS_metadata_AB_enriched_v2.csv")
AB_FIELDS_WANTED = [
    "BAG_AB", "cBAG_AB", "PredictedAge_AB",
    "PredictedAge_corrected_AB", "Delta_BAG_AB", "Delta_cBAG_AB"
]

PRIMARY_OUT_PATH = Path("/mnt/newStor/paros/paros_WORK//harmonization/HABS/metadata/HABS_maestro_metadata.csv")
# -----------------------------------

# Logging to stdout
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logging.getLogger().handlers[:] = [handler]
logging.getLogger().setLevel(logging.DEBUG)
log = logging.getLogger("habs_maestro")

def _p(msg: str):
    print(msg, flush=True)

RUNNO_PATTERN = re.compile(r"^(H[^_]+)_y([02])($|[^0-9])")
RUNNO_AT_END_PATTERN = re.compile(r"(H[^_/\\]+_y[02])(?=\.csv$)", re.IGNORECASE)

# ----------------- I/O helpers -----------------
def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df

def safe_concat_csvs(pattern: str, label: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    _p(f"[STEP] Loading {label}: pattern={pattern} (found {len(paths)} files)")
    frames = []
    for pth in paths:
        try:
            df = pd.read_csv(pth, low_memory=False)
            df.columns = [c.strip().replace(" ", "_") for c in df.columns]
            frames.append(df)
        except Exception as e:
            _p(f"[WARN] Failed to read {pth}: {e}")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    _p(f"[OK] {label}: rows={len(out)}, cols={len(out.columns)}")
    return out

# ----------------- runno helpers -----------------
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
    for p in dirpath.glob("**/*"):
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

# ----------------- cleaning helpers -----------------
def _is_negative_repeated_int_scalar(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, (int, np.integer)) or (isinstance(x, (float, np.floating)) and float(x).is_integer()):
        xi = int(x)
        if xi >= 0:
            return False
        s = str(abs(xi))
        return len(s) >= 3 and len(set(s)) == 1
    if isinstance(x, str):
        s = x.strip()
        return re.fullmatch(r"-\s*([0-9])\1{2,}", s) is not None
    return False

def clean_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    df = df.copy()
    stats = {"minus9999": {}, "neg_repeated": {}, "space_strings": {}}

    # numeric -9999
    for col in df.select_dtypes(include=[np.number]).columns:
        mask = df[col] == -9999
        cnt = int(mask.sum())
        if cnt:
            df.loc[mask, col] = pd.NA
            stats["minus9999"][col] = stats["minus9999"].get(col, 0) + cnt
    # object -9999
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            s = df[col].astype("string")
            mask = s.str.fullmatch(r"\s*-9999\s*", na=False)
            cnt = int(mask.sum())
            if cnt:
                s = s.mask(mask, pd.NA)
                df[col] = s
                stats["minus9999"][col] = stats["minus9999"].get(col, 0) + cnt
    # neg repeated placeholders
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            s = df[col].astype("string")
            mask = s.str.fullmatch(r"\s*-\s*([0-9])\1{2,}\s*", na=False)
            cnt = int(mask.sum())
            if cnt:
                s = s.mask(mask, pd.NA)
                df[col] = s
                stats["neg_repeated"][col] = stats["neg_repeated"].get(col, 0) + cnt
    for col in df.select_dtypes(include=[np.number]).columns:
        mask = df[col].apply(_is_negative_repeated_int_scalar)
        cnt = int(mask.sum())
        if cnt:
            df.loc[mask, col] = pd.NA
            stats["neg_repeated"][col] = stats["neg_repeated"].get(col, 0) + cnt
    # whitespace -> CUT
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            s = df[col].astype("string")
            mask = s.str.contains(r"\s", na=False)
            cnt = int(mask.sum())
            if cnt:
                s = s.mask(mask, "CUT")
                df[col] = s
                stats["space_strings"][col] = stats["space_strings"].get(col, 0) + cnt
    return df, stats

def print_cleaning_summary(stats: Dict[str, Dict[str, int]], title: str):
    def _top(d: Dict[str, int], k=10):
        return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]
    _p(f"[CLEAN-SUMMARY] {title}")
    for cat in ("minus9999", "neg_repeated", "space_strings"):
        colmap = stats.get(cat, {})
        total = sum(colmap.values())
        _p(f"  - {cat}: total replaced = {total}")
        if total:
            for col, cnt in _top(colmap):
                _p(f"      • {col}: {cnt}")

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

# ----------------- AB helpers -----------------
def prepare_ab_lookup(path: Path) -> pd.DataFrame:
    if not path.exists():
        _p(f"[WARN] AB-enriched file not found: {path}")
        return pd.DataFrame()
    try:
        df = read_table(path)
    except Exception as e:
        _p(f"[WARN] Failed to read AB file: {e}")
        return pd.DataFrame()
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}
    # try med_id
    if "med_id" in cols:
        df["Subject"] = df[cols["med_id"]].astype(str).str.strip()
    # try runno-like
    runno_col = None
    for cand in ("runno", "run_no", "runnos", "id"):
        if cand in cols:
            runno_col = cols[cand]; break
    if runno_col and "Subject" not in df.columns:
        subj_list = []
        year_list = []
        for val in df[runno_col].astype(str).fillna(""):
            m = RUNNO_PATTERN.match(val)
            if m:
                subj_list.append(m.group(1)[1:])  # strip leading H
                year_list.append(int(m.group(2)))
            else:
                subj_list.append("")
                year_list.append(np.nan)
        df["Subject"] = pd.Series(subj_list)
        df["Year"] = pd.Series(year_list)
    # try visit -> year mapping
    if "Year" not in df.columns:
        if "visit" in cols:
            def visit_to_year(v):
                vstr = str(v).strip().upper()
                if vstr in ("BL", "BASELINE", "0", "Y0"): return 0
                if vstr in ("M24", "24", "2", "Y2"): return 2
                try:
                    ni = int(vstr)
                    if ni == 1: return 0
                    if ni == 3: return 2
                except Exception:
                    pass
                return np.nan
            df["Year"] = df[cols["visit"]].apply(visit_to_year)
        elif "visit_id" in cols:
            df["Year"] = pd.to_numeric(df[cols["visit_id"]], errors="coerce").map({1:0, 3:2})
    # final subject cleaning
    if "Subject" in df.columns:
        df["Subject"] = df["Subject"].astype(str).str.replace(r"^H", "", regex=True).str.strip()
    keep = [c for c in AB_FIELDS_WANTED if c in df.columns]
    if not keep:
        _p(f"[WARN] None of the requested AB fields found in {path}. Found columns: {list(df.columns)[:30]}")
        return pd.DataFrame()
    out = df.loc[:, ["Subject", "Year"] + keep].copy()
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    out = out[out["Subject"].astype(str) != ""].copy()
    _p(f"[OK] AB lookup prepared: rows={len(out)}, fields={keep}")
    return out

# ----------------- main -----------------
def main():
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    try:
        _p("[START] HABS Maestro metadata build (v3)")
        runnos = compile_ordered_runnos(dirs_start=[DWI_DIR], dirs_end=[FMRI_DIR])
        if not runnos:
            _p("[FATAL] No runnos found.")
            return

        df_dual = read_table(HABS_DUAL_FILE)
        if "Subject" in df_dual.columns:
            df_dual["Subject"] = df_dual["Subject"].astype(str).str.strip()
        subj_to_visit_dates = {}
        if not df_dual.empty:
            for _, r in df_dual.iterrows():
                s = str(r.get("Subject", "")).strip()
                v = str(r.get("Visit", "")).strip()
                acq = r.get("Acq_Date", pd.NaT)
                if s:
                    if s not in subj_to_visit_dates:
                        subj_to_visit_dates[s] = {"BL": pd.NaT, "M24": pd.NaT}
                    if v in ("BL", "M24"):
                        subj_to_visit_dates[s][v] = acq

        # loads
        df_gen_all = safe_concat_csvs(str(METADATA_DIR / "Genomics*.csv"), "Genomics*.csv")
        if not df_gen_all.empty and "Age" in df_gen_all.columns:
            df_gen_all = df_gen_all.drop(columns=["Age"])
        if "Med_ID" in df_gen_all.columns:
            df_gen_all["Med_ID"] = df_gen_all["Med_ID"].astype(str).str.strip()

        df_clin1_all = safe_concat_csvs(str(METADATA_DIR / "Clinical HD 1*.csv"), "Clinical HD 1*.csv")
        df_clin2_all = safe_concat_csvs(str(METADATA_DIR / "Clinical HD 2*.csv"), "Clinical HD 2*.csv")
        df_clin3_all = safe_concat_csvs(str(METADATA_DIR / "Clinical HD 3*.csv"), "Clinical HD 3*.csv")
        for d in (df_clin1_all, df_clin2_all, df_clin3_all):
            if "Med_ID" in d.columns: d["Med_ID"] = d["Med_ID"].astype(str).str.strip()

        df_bio1_all = safe_concat_csvs(str(METADATA_DIR / "Biomarker HD 1*.csv"), "Biomarker HD 1*.csv")
        df_bio2_all = safe_concat_csvs(str(METADATA_DIR / "Biomarker HD 2*.csv"), "Biomarker HD 2*.csv")
        df_bio3_all = safe_concat_csvs(str(METADATA_DIR / "Biomarker HD 3*.csv"), "Biomarker HD 3*.csv")
        for d in (df_bio1_all, df_bio2_all, df_bio3_all):
            if "Med_ID" in d.columns: d["Med_ID"] = d["Med_ID"].astype(str).str.strip()

        df_ab_lookup = prepare_ab_lookup(HABS_AB_FILE)

        # presence lists
        dwi_present = set(list_runnos_from_dir_start(DWI_DIR)) | set(list_runnos_from_dir_end(DWI_DIR))
        fmri_present = set(list_runnos_from_dir_start(FMRI_DIR)) | set(list_runnos_from_dir_end(FMRI_DIR))
        _p(f"[INFO] DWI runnos: {len(dwi_present)}; fMRI runnos: {len(fmri_present)}")

        # iterate runnos -> rows
        rows = []
        miss_genomics, miss_clinical, miss_biomarker = set(), set(), set()
        for i, runno in enumerate(runnos, 1):
            if i % 50 == 1 or i == len(runnos):
                _p(f"  • Processing {i}/{len(runnos)}: {runno}")
            try:
                runno_str, subject, visit = subject_visit_from_runno(runno)
                visit_code = map_visit_code(visit)
                expected_visit_id = 1 if visit == "0" else 3
                sel = df_dual[(df_dual.get("Subject") == subject) & (df_dual.get("Visit") == visit_code)]
                if sel.empty:
                    sex = pd.NA; acq_date = pd.NaT
                else:
                    sex = sel.iloc[0].get("Sex", pd.NA)
                    acq_date = sel.iloc[0].get("Acq_Date", pd.NaT)
                target_date = subj_to_visit_dates.get(subject, {}).get(visit_code, pd.NaT) if subj_to_visit_dates else acq_date
                if pd.isna(target_date):
                    target_date = acq_date
                row = {"runno": runno_str, "Subject": subject, "Year": int(visit), "Sex": sex, "Acq_Date": acq_date}
                row["DWI"] = runno_str if runno_str in dwi_present else pd.NA
                row["fMRI"] = runno_str if runno_str in fmri_present else pd.NA

                # genomics
                if not df_gen_all.empty and "Med_ID" in df_gen_all.columns:
                    g = df_gen_all[df_gen_all["Med_ID"] == subject]
                    if g.empty:
                        miss_genomics.add(runno_str)
                    else:
                        for k, v in g.iloc[0].to_dict().items():
                            if k not in row: row[k] = v
                else:
                    miss_genomics.add(runno_str)

                # clinical (simplified pick_best to avoid complexity)
                if visit == "0":
                    clin_allowed = [(1, df_clin1_all)]
                else:
                    clin_allowed = [(2, df_clin2_all), (3, df_clin3_all)]

                def pick_best(subj, allowed, exp_vid):
                    for bucket, dfc in allowed:
                        if dfc.empty or "Med_ID" not in dfc.columns: continue
                        sub = dfc[dfc["Med_ID"] == subj].copy()
                        if sub.empty: continue
                        # prefer exact Visit_ID match
                        if "Visit_ID" in sub.columns:
                            exact = sub[sub["Visit_ID"].astype(str).str.strip() == str(exp_vid)]
                            if not exact.empty:
                                return exact.iloc[0], bucket
                        # otherwise return first found row
                        return sub.iloc[0], bucket
                    return None, None

                clin_sel, clin_bucket = pick_best(subject, clin_allowed, expected_visit_id)
                if clin_sel is None:
                    miss_clinical.add(runno_str)
                else:
                    row["Clinical_HD_Source"] = clin_bucket
                    for k, v in clin_sel.to_dict().items():
                        if k not in row: row[k] = v

                # biomarker
                if visit == "0":
                    bio_allowed = [(1, df_bio1_all)]
                else:
                    bio_allowed = [(2, df_bio2_all), (3, df_bio3_all)]
                bio_sel, bio_bucket = pick_best(subject, bio_allowed, expected_visit_id)
                if bio_sel is None:
                    miss_biomarker.add(runno_str)
                else:
                    row["Biomarker_HD_Source"] = bio_bucket
                    for k, v in bio_sel.to_dict().items():
                        if k not in row: row[k] = v

                rows.append(pd.Series(row))
            except Exception as e:
                _p(f"[ERROR] runno={runno}: {e}")
                traceback.print_exc()

        if not rows:
            _p("[FATAL] No rows produced; aborting.")
            return

        df_out = pd.DataFrame(rows)

        # initial cleaning
        _p("[STEP] Cleaning values")
        df_out, clean_stats = clean_values(df_out)
        print_cleaning_summary(clean_stats, "Per-column replacements")

        # Initial AB merge (left). This may add columns with exact canonical names or suffixed copies.
        if not df_ab_lookup.empty:
            df_out["Subject"] = df_out["Subject"].astype(str).str.strip()
            df_out["Year"] = pd.to_numeric(df_out["Year"], errors="coerce").astype("Int64")
            df_ab_lookup["Subject"] = df_ab_lookup["Subject"].astype(str).str.strip()
            df_ab_lookup["Year"] = pd.to_numeric(df_ab_lookup["Year"], errors="coerce").astype("Int64")
            before_cols = set(df_out.columns)
            try:
                df_out = df_out.merge(df_ab_lookup, on=["Subject", "Year"], how="left", suffixes=("", "_absrc"))
                added = [c for c in df_out.columns if c not in before_cols]
                _p(f"[OK] Merged AB fields (raw added cols) = {added}")
            except Exception as e:
                _p(f"[WARN] AB merge failed: {e}")
            # premerge debug
            try:
                dbg_pre_base = PRIMARY_OUT_PATH.with_name(PRIMARY_OUT_PATH.stem + ".AB_debug_premerge.csv")
                dbg_cols = ["Subject", "Year"] + [c for c in AB_FIELDS_WANTED if c in df_out.columns or f"{c}_absrc" in df_out.columns]
                if len(dbg_cols) > 0:
                    df_out.loc[:, dbg_cols].to_csv(dbg_pre_base, index=False)
                    _p(f"[DEBUG] Wrote AB pre-merge debug -> {dbg_pre_base}")
            except Exception as e:
                _p(f"[WARN] Could not write AB premerge debug: {e}")
        else:
            _p("[WARN] AB lookup empty — skipping initial AB merge")

        # reorder base fields for readability
        base_order = ["runno", "Subject", "Year", "Sex", "Acq_Date", "DWI", "fMRI", "Clinical_HD_Source", "Biomarker_HD_Source"]
        remaining = [c for c in df_out.columns if c not in base_order]
        df_out = df_out[[c for c in base_order if c in df_out.columns] + remaining]
        _p(f"[OK] Raw merged shape (post-clean+AB merge): {df_out.shape}")

        # dedupe identical columns
        before = df_out.shape[1]
        df_out = remove_duplicate_identical_columns(df_out)
        after_dup = df_out.shape[1]
        _p(f"[CLEAN] Dropped {before - after_dup} duplicate-identical columns")

        # drop constant columns
        before2 = df_out.shape[1]
        df_out = remove_constant_columns(df_out)
        after_const = df_out.shape[1]
        _p(f"[CLEAN] Dropped {before2 - after_const} constant columns")
        _p(f"[OK] Shape after dedup/const-drop: {df_out.shape}")

        # low-pop drop (<10%) but protect AB fields and core identifiers
        nrows = len(df_out)
        protect = {c for c in ["runno", "Subject", "Year", "Sex", "Acq_Date"]} | set(AB_FIELDS_WANTED)
        cols_to_check = [c for c in df_out.columns if c not in protect]
        dropped_low = []
        for c in cols_to_check:
            non_empty = int(df_out[c].notna().sum())
            prop = non_empty / nrows if nrows else 0.0
            if prop < 0.10:
                df_out = df_out.drop(columns=[c])
                dropped_low.append((c, non_empty, prop))
                _p(f"[DROP_LOW_POP] Dropped column '{c}' — non-empty rows: {non_empty}/{nrows} ({prop:.1%})")
        _p(f"[INFO] Total low-pop columns dropped: {len(dropped_low)}")
        _p(f"[OK] Shape after low-pop drops: {df_out.shape}")

        # ---------------------------
        # FINAL AB RECONCILIATION (after all drops, BEFORE writing csv)
        # Uses robust composite string keys "Subject|Year"
        # ---------------------------
        _p("[STEP] Final AB reconciliation (post-drops)")

        # normalize keys
        df_out["Subject"] = df_out["Subject"].astype(str).str.strip()
        df_out["Year"] = pd.to_numeric(df_out["Year"], errors="coerce").astype("Int64")
        if not df_ab_lookup.empty:
            df_ab_lookup["Subject"] = df_ab_lookup["Subject"].astype(str).str.strip()
            df_ab_lookup["Year"] = pd.to_numeric(df_ab_lookup["Year"], errors="coerce").astype("Int64")

        if not df_ab_lookup.empty:
            # If suffixed columns exist and canonical does not, rename them
            for fld in AB_FIELDS_WANTED:
                suff = f"{fld}_absrc"
                if suff in df_out.columns and fld not in df_out.columns:
                    df_out.rename(columns={suff: fld}, inplace=True)
                    _p(f"[AB-FINAL] Renamed '{suff}' -> '{fld}'")

            # Build composite key strings on both sides and map
            def mk_key_series(df, subj_col="Subject", year_col="Year"):
                y = df[year_col].astype(object).where(df[year_col].notna(), "")
                return df[subj_col].astype(str).str.strip() + "|" + y.astype(str)

            df_out["_ab__key"] = mk_key_series(df_out, "Subject", "Year")
            ab_tmp = df_ab_lookup.copy()
            ab_tmp["_ab__key"] = mk_key_series(ab_tmp, "Subject", "Year")

            # build mapping dict per field (last occurrence wins)
            mapping_dicts = {}
            for fld in AB_FIELDS_WANTED:
                if fld in ab_tmp.columns:
                    # use the last entry for duplicate keys: take series and convert to dict (index -> value)
                    s = pd.Series(ab_tmp[fld].values, index=ab_tmp["_ab__key"].astype(str))
                    mapping_dicts[fld] = s.to_dict()
                else:
                    mapping_dicts[fld] = {}

            # fill/add using mapping dicts
            for fld in AB_FIELDS_WANTED:
                mapped = df_out["_ab__key"].map(mapping_dicts.get(fld, {}))
                if fld in df_out.columns:
                    before_nonnull = int(df_out[fld].notna().sum())
                    df_out[fld] = df_out[fld].where(df_out[fld].notna(), mapped)
                    after_nonnull = int(df_out[fld].notna().sum())
                    _p(f"[AB-FINAL-FILL] {fld}: filled {after_nonnull - before_nonnull} values from AB-lookup (now {after_nonnull}/{len(df_out)})")
                else:
                    df_out[fld] = mapped
                    _p(f"[AB-FINAL-ADD] {fld}: added from AB-lookup (non-empty={int(df_out[fld].notna().sum())}/{len(df_out)})")

            # cleanup
            df_out.drop(columns=["_ab__key"], inplace=True, errors="ignore")
        else:
            # ensure columns exist (empty) so it's obvious in final CSV
            for fld in AB_FIELDS_WANTED:
                if fld not in df_out.columns:
                    df_out[fld] = pd.NA
                    _p(f"[AB-FINAL-ENSURE] Created empty AB column '{fld}' (no AB lookup)")

        # final diagnostics + debug CSV
        for fld in AB_FIELDS_WANTED:
            non_empty = int(df_out[fld].notna().sum())
            _p(f"[AB-FINAL-COUNT] {fld}: non-empty={non_empty}/{len(df_out)} ({(non_empty/len(df_out) if len(df_out) else 0):.1%})")

        try:
            dbg_path = PRIMARY_OUT_PATH.with_name(PRIMARY_OUT_PATH.stem + f".AB_final_debug.{ts}.csv")
            dbg_cols = ["Subject", "Year"] + AB_FIELDS_WANTED
            df_out.loc[:, dbg_cols].to_csv(dbg_path, index=False)
            _p(f"[DEBUG] Wrote AB final debug file -> {dbg_path} (rows={len(df_out)})")
            _p(f"[DEBUG] Sample:")
            _p(str(df_out.loc[:, dbg_cols].head(6)))
        except Exception as e:
            _p(f"[WARN] Could not write AB final debug CSV: {e}")

        # ---------------------------
        # final write + timestamped backup
        # ---------------------------
        out_path = PRIMARY_OUT_PATH
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        _p(f"[DONE] Wrote CSV: {out_path}  (rows={df_out.shape[0]}, cols={df_out.shape[1]})")
        bak_path = out_path.with_name(out_path.stem + f".{ts}.bak.csv")
        df_out.to_csv(bak_path, index=False)
        _p(f"[DONE] Wrote timestamped backup CSV: {bak_path}")

        # missing metadata log
        if miss_genomics or miss_clinical or miss_biomarker:
            miss_path = out_path.with_suffix(".genomics_missing.txt")
            _p(f"[INFO] Writing missing metadata log → {miss_path}")
            all_runnos = sorted(miss_genomics | miss_clinical | miss_biomarker)
            with open(miss_path, "w") as fh:
                fh.write("runno,missing_genomics,missing_clinical,missing_biomarker\n")
                for r in all_runnos:
                    fh.write(f"{r},{int(r in miss_genomics)},{int(r in miss_clinical)},{int(r in miss_biomarker)}\n")
            _p(f"[INFO] Missing counts — Genomics: {len(miss_genomics)}, Clinical: {len(miss_clinical)}, Biomarker: {len(miss_biomarker)}")

        _p("[END] Completed successfully.")

    except Exception as e:
        _p(f"[UNCAUGHT] {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 240)
    main()
