#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <HABS_maestro_metadata.csv> [--clinical <clinical.csv>]"
  exit 1
fi

MAESTRO="$1"; shift || true
CLINICAL=""

if [[ "${1-}" == "--clinical" ]]; then
  shift
  CLINICAL="${1-}"; shift || true
fi

OUT_ROWS="healthy_rows.csv"
OUT_SUBJ="healthy_subjects.txt"

python - <<'PY'
import csv, sys, os

# ----- inputs from bash -----
maestro = os.environ.get("MAESTRO", None)
clinical = os.environ.get("CLINICAL", "")

# pull from parent env
for k in ("MAESTRO","CLINICAL","OUT_ROWS","OUT_SUBJ"):
    globals()[k] = os.environ.get(k)

# Column alias candidates (case-insensitive matching)
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
        k = c.strip().lower()
        if k in hmap:
            return hmap[k]
    return None

def norm(s):
    return (s or "").strip()

def low(s):
    return norm(s).lower()

def to_float(s):
    try:
        return float(norm(s))
    except:
        return None

# Load maestro
mh, mrows = open_csv(MAESTRO)
mid_col   = find_col(mh, MEDID_CANDS)
v_col     = find_col(mh, VISIT_CANDS)

if not mid_col:
    print("ERROR: Med_ID column not found in maestro.", file=sys.stderr)
    sys.exit(2)

# Optionally load clinical and index for join
clin_idx = {}
if CLINICAL:
    ch, crows = open_csv(CLINICAL)
    cmid_col  = find_col(ch, MEDID_CANDS) or "Med_ID"
    cv_col    = find_col(ch, VISIT_CANDS)
    # candidate clinical columns
    diab_col  = find_col(ch, DIAB_CANDS)
    a1c_col   = find_col(ch, A1C_CANDS)
    npdx_col  = find_col(ch, NPDX_CANDS)
    cdr_col   = find_col(ch, CDR_CANDS)

    # build index by (Med_ID, Visit_ID) if available, else by Med_ID
    if cv_col:
        for r in crows:
            clin_idx[(norm(r.get(cmid_col)), norm(r.get(cv_col)))] = r
    else:
        for r in crows:
            clin_idx[(norm(r.get(cmid_col)), "")] = r

else:
    diab_col = a1c_col = npdx_col = cdr_col = None

# If maestro itself carries usable columns, prefer them
m_diab  = find_col(mh, DIAB_CANDS)
m_a1c   = find_col(mh, A1C_CANDS)
m_npdx  = find_col(mh, NPDX_CANDS)
m_cdr   = find_col(mh, CDR_CANDS)

# Helper: fetch a value from maestro first, else from clinical join
def get_field(mrow, key):
    if key == "diab":
        col = m_diab or diab_col
    elif key == "a1c":
        col = m_a1c or a1c_col
    elif key == "npdx":
        col = m_npdx or npdx_col
    elif key == "cdr":
        col = m_cdr or cdr_col
    else:
        return None
    if col and col in mrow and norm(mrow.get(col)):
        return mrow.get(col)
    # clinical fallback
    if not CLINICAL: return None
    mk = norm(mrow.get(mid_col))
    mv = norm(mrow.get(v_col)) if v_col else ""
    r = clin_idx.get((mk, mv)) or clin_idx.get((mk, ""))
    if not r: return None
    return r.get(col)

def diabetes_negative(mrow):
    v_diab = get_field(mrow, "diab")
    if low(v_diab) == "no":
        return True
    # fallback to A1c if we have it
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

# Decide if we even have enough signal to filter
have_any_diab = (m_diab or diab_col or m_a1c or a1c_col)
have_any_cog  = (m_npdx or npdx_col or m_cdr or cdr_col)

if not (have_any_diab and have_any_cog):
    msg = []
    if not have_any_diab:
        msg.append("Diabetes/A1c")
    if not have_any_cog:
        msg.append("Cognition (NP consensus / CDR)")
    print("WARNING: Missing columns: " + ", ".join(msg), file=sys.stderr)
    if not CLINICAL:
        print("Hint: provide a clinical export via --clinical <clinical.csv> so we can join in those fields.", file=sys.stderr)

# Filter
kept = []
for r in mrows:
    if diabetes_negative(r) and cognitively_unimpaired(r):
        kept.append(r)

# Write healthy_rows.csv (with maestro header order)
with open(OUT_ROWS, "w", newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=mh)
    w.writeheader()
    for r in kept:
        w.writerow(r)

# Write healthy_subjects.txt
seen = set()
with open(OUT_SUBJ, "w", encoding='utf-8') as f:
    for r in kept:
        mid = norm(r.get(mid_col))
        if mid and mid not in seen:
            seen.add(mid)
            f.write(mid + "\n")

print(f"Wrote {OUT_ROWS} ({len(kept)} rows) and {OUT_SUBJ} ({len(seen)} unique Med_IDs).")
PY
