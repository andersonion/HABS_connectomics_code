#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <HABS_maestro_metadata.csv> [--clinical <clinical.csv>]"
  exit 1
fi

MAESTRO_IN="$1"; shift || true
CLINICAL_IN=""
if [[ "${1-}" == "--clinical" ]]; then
  shift
  CLINICAL_IN="${1-}"; shift || true
fi

export MAESTRO="$MAESTRO_IN"
export CLINICAL="${CLINICAL_IN}"
export OUT_ROWS="healthy_rows.csv"
export OUT_SUBJ="healthy_subjects.txt"

python - <<'PY'
import csv, sys, os

MAESTRO = os.environ["MAESTRO"]
CLINICAL = os.environ.get("CLINICAL","")
OUT_ROWS = os.environ["OUT_ROWS"]
OUT_SUBJ = os.environ["OUT_SUBJ"]

# Column alias candidates (case-insensitive)
MEDID_CANDS = ["Med_ID","MedID","Subject","Subject_ID","subject_id","Med_ID.x","Med_ID.y"]
VISIT_CANDS = ["Visit_ID","VisitID","Visit","Visit_ID.x","Visit_ID.y"]
DIAB_CANDS  = ["Diabetes","Diabetes_Consensus","DiabetesYN","Diag_Diabetes"]
A1C_CANDS   = ["HbA1c","A1C","HBA1C","HbA1c_percent"]
NPDX_CANDS  = ["Neuropsych_Consensus_Diagnosis","NP_Dx","NP_Diagnosis","Cognitive_Diagnosis","Consensus_Diagnosis"]
CDR_CANDS   = ["CDR_Sum_of_Boxes","CDR_SOB","CDR_SumBoxes","CDR_Sum_Of_Boxes"]

def open_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        rows = list(r)
        header = r.fieldnames or []
    return header, rows

def find_col(header, cands):
    hmap = { (h or "").strip().lower(): h for h in header }
    for c in cands:
        if c.lower() in hmap:
            return hmap[c.lower()]
    return None

def norm(s): return (s or "").strip()
def low(s):  return norm(s).lower()
def to_float(s):
    try: return float(norm(s))
    except: return None

# Load maestro
mh, mrows = open_csv(MAESTRO)
mid_col   = find_col(mh, MEDID_CANDS)
v_col     = find_col(mh, VISIT_CANDS)
if not mid_col:
    print("ERROR: Med_ID column not found in maestro.", file=sys.stderr); sys.exit(2)

# Optionally load clinical (for join fields)
clin_idx = {}
diab_col = a1c_col = npdx_col = cdr_col = None
if CLINICAL:
    ch, crows = open_csv(CLINICAL)
    cmid_col  = find_col(ch, MEDID_CANDS) or "Med_ID"
    cv_col    = find_col(ch, VISIT_CANDS)
    diab_col  = find_col(ch, DIAB_CANDS)
    a1c_col   = find_col(ch, A1C_CANDS)
    npdx_col  = find_col(ch, NPDX_CANDS)
    cdr_col   = find_col(ch, CDR_CANDS)
    if cv_col:
        for r in crows:
            clin_idx[(norm(r.get(cmid_col)), norm(r.get(cv_col)))] = r
    else:
        for r in crows:
            clin_idx[(norm(r.get(cmid_col)), "")] = r

# Prefer fields on maestro if present
m_diab  = find_col(mh, DIAB_CANDS)
m_a1c   = find_col(mh, A1C_CANDS)
m_npdx  = find_col(mh, NPDX_CANDS)
m_cdr   = find_col(mh, CDR_CANDS)

def get_field(mrow, key):
    # choose maestro column first, else clinical-join column
    if key == "diab": col = m_diab or diab_col
    elif key == "a1c": col = m_a1c or a1c_col
    elif key == "npdx": col = m_npdx or npdx_col
    elif key == "cdr":  col = m_cdr  or cdr_col
    else: return None
    if col and col in mrow and norm(mrow.get(col)):
        return mrow.get(col)
    if not CLINICAL: return None
    mk = norm(mrow.get(mid_col))
    mv = norm(mrow.get(v_col)) if v_col else ""
    r = clin_idx.get((mk, mv)) or clin_idx.get((mk, ""))
    return (r or {}).get(col) if col else None

# Rules from the R6 Clinical Support Document:
# Diabetes present if A1c ≥ 6.5 or PMH diabetes; keep only Diabetes=No or A1c<6.5. :contentReference[oaicite:0]{index=0}
# Cognitively Unimpaired required; fallback to CDR sum of boxes == 0. :contentReference[oaicite:1]{index=1}
def diabetes_negative(mrow):
    v_diab = get_field(mrow, "diab")
    if low(v_diab) == "no":
        return True
    v_a1c = get_field(mrow, "a1c")
    if v_a1c is not None:
        x = to_float(v_a1c)
        if x is not None and x < 6.5:
            return True
    return False

def cognitively_unimpaired(mrow):
    v_np = get_field(mrow, "npdx")
    if v_np:
        t = low(v_np)
        if t.startswith("cognitively unimpaired") or t == "normal cognition" or t == "cu":
            return True
    v_cdr = get_field(mrow, "cdr")
    if v_cdr is not None:
        x = to_float(v_cdr)
        if x is not None and x == 0.0:
            return True
    return False

have_any_diab = bool(m_diab or diab_col or m_a1c or a1c_col)
have_any_cog  = bool(m_npdx or npdx_col or m_cdr or cdr_col)
if not (have_any_diab and have_any_cog):
    missing = []
    if not have_any_diab: missing.append("Diabetes/A1c")
    if not have_any_cog:  missing.append("Cognition (NP consensus / CDR)")
    print("WARNING: Missing columns: " + ", ".join(missing), file=sys.stderr)
    if not CLINICAL:
        print("Hint: pass --clinical <clinical.csv> so we can join those fields.", file=sys.stderr)

kept = [r for r in mrows if diabetes_negative(r) and cognitively_unimpaired(r)]

with open(OUT_ROWS, "w", newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=mh)
    w.writeheader(); w.writerows(kept)

seen = set()
with open(OUT_SUBJ, "w", encoding='utf-8') as f:
    for r in kept:
        mid = norm(r.get(mid_col))
        if mid and mid not in seen:
            seen.add(mid); f.write(mid+"\n")

print(f"Wrote {OUT_ROWS} ({len(kept)} rows) and {OUT_SUBJ} ({len(seen)} unique Med_IDs).")
PY
