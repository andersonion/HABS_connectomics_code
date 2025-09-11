#!/usr/bin/env bash
set -euo pipefail

# Input (can be overridden): a CSV focused on subjects with connectomics
INPUT_CSV="${1:-HABS_maestro_metadata.csv}"

# Outputs
OUT_ROWS="healthy_rows.csv"
OUT_SUBJ="healthy_subjects.txt"

# Column name candidates (case-insensitive matching)
# Tweak/add aliases here if your headers differ.
MEDID_CANDS=("Med_ID" "MedID" "Subject" "Subject_ID" "subject_id" "Med_ID.x" "Med_ID.y")
VISIT_CANDS=("Visit_ID" "VisitID" "Visit" "Visit_ID.x" "Visit_ID.y")
DIAB_CANDS=("Diabetes" "Diabetes_Consensus" "DiabetesYN" "Diag_Diabetes")
A1C_CANDS=("HbA1c" "A1C" "HBA1C" "HbA1c_percent")
NPDX_CANDS=("Neuropsych_Consensus_Diagnosis" "NP_Dx" "NP_Diagnosis" "Cognitive_Diagnosis" "Consensus_Diagnosis")
CDRSOB_CANDS=("CDR_Sum_of_Boxes" "CDR_SOB" "CDR_SumBoxes" "CDR_Sum_Of_Boxes")

# Helper: find a header column by trying a list of candidates (case-insensitive).
find_col_ci() {
  local header="$1"; shift
  for cand in "$@"; do
    # anchor to whole field, case-insensitive
    if echo "$header" | awk -vRS=',' -vIGNORECASE=1 '{print}' | grep -qiE "^${cand}$"; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

HEADER="$(head -n1 "$INPUT_CSV")"

MED_ID="$(find_col_ci "$HEADER" "${MEDID_CANDS[@]}")"        || { echo "ERROR: Med_ID column not found."; exit 1; }
# Visit not strictly required for this filter, but we try to find it for completeness
VISIT_ID="$(find_col_ci "$HEADER" "${VISIT_CANDS[@]}")"      || true
DIAB_COL="$(find_col_ci "$HEADER" "${DIAB_CANDS[@]}")"       || true
A1C_COL="$(find_col_ci "$HEADER" "${A1C_CANDS[@]}")"         || true
NP_DX_COL="$(find_col_ci "$HEADER" "${NPDX_CANDS[@]}")"      || true
CDR_SOB_COL="$(find_col_ci "$HEADER" "${CDRSOB_CANDS[@]}")"  || true

echo "Detected columns:"
echo "  Med_ID:        ${MED_ID}"
echo "  Visit_ID:      ${VISIT_ID:-<none>}"
echo "  Diabetes:      ${DIAB_COL:-<none>}"
echo "  HbA1c:         ${A1C_COL:-<none>}"
echo "  NP Diagnosis:  ${NP_DX_COL:-<none>}"
echo "  CDR SumBoxes:  ${CDR_SOB_COL:-<none>}"
echo

# Prefer Miller (mlr) for CSV correctness
if command -v mlr >/dev/null 2>&1; then
  echo "Using Miller (mlr) for CSV filtering…"

  # Build Miller filter expression:
  # Healthy if:
  #   (Diabetes == "No" OR (missing) OR (A1c present and < 6.5))
  # AND
  #   (NP Diagnosis indicates cognitively unimpaired OR CDR_Sum_of_Boxes == 0)
  #
  # We treat missing Diabetes as unknown; then rely on A1c if present.
  # For cognition: prefer NP consensus text when present; else fall back to CDR sum boxes == 0.

  # Escape column names that might contain spaces by using ${...} in Miller.
  d_diab=''
  if [[ -n "${DIAB_COL:-}" ]]; then
    d_diab="(is_present(\$${DIAB_COL}) && tolower(\$${DIAB_COL}) == \"no\")"
  else
    d_diab="false"
  fi

  d_a1c=''
  if [[ -n "${A1C_COL:-}" ]]; then
    d_a1c="(is_present(\$${A1C_COL}) && is_numeric(\$${A1C_COL}) && \$${A1C_COL} < 6.5)"
  else
    d_a1c="false"
  fi

  c_np=''
  if [[ -n "${NP_DX_COL:-}" ]]; then
    c_np="(is_present(\$${NP_DX_COL}) && matches(tolower(\$${NP_DX_COL}), \"^cognitively[ _-]*unimpaired|^normal cognition|^cu$\"))"
  else
    c_np="false"
  fi

  c_cdr=''
  if [[ -n "${CDR_SOB_COL:-}" ]]; then
    c_cdr="(is_present(\$${CDR_SOB_COL}) && is_numeric(\$${CDR_SOB_COL}) && \$${CDR_SOB_COL} == 0)"
  else
    c_cdr="false"
  fi

  mlr --csv filter " ( (${d_diab}) || (${d_a1c}) ) && ( (${c_np}) || (${c_cdr}) ) " \
    then sort -f "${MED_ID}" \
    "$INPUT_CSV" > "$OUT_ROWS"

  # Unique subject list
  mlr --csv cut -f "${MED_ID}" "$OUT_ROWS" \
    | mlr --csv uniq -a \
    | tail -n +2 > "$OUT_SUBJ"

else
  echo "Miller not found. Falling back to gawk (robust CSV FPAT)…"

  # gawk CSV parser with quoted fields
  # Build index lookup for columns
  gawk -v IGNORECASE=1 -v OFS=',' \
       -v MED_ID="${MED_ID}" -v DIAB_COL="${DIAB_COL}" -v A1C_COL="${A1C_COL}" \
       -v NP_DX_COL="${NP_DX_COL}" -v CDR_SOB_COL="${CDR_SOB_COL}" '
    BEGIN{
      FPAT = "([^,]+)|(\"([^\"]|\"\")*\")"  # parse CSV fields incl. quotes
      healthy_rows_file = "'"$OUT_ROWS"'"
      subj_file = "'"$OUT_SUBJ"'"
    }
    NR==1{
      # header -> map names to indices (case-insensitive)
      for(i=1;i<=NF;i++){
        h=toupper(gensub(/^"|"$/,"","g",$i))
        idx[h]=i
      }
      midx = idx[toupper(MED_ID)]
      d_idx = (DIAB_COL==""?0:idx[toupper(DIAB_COL)])
      a1_idx = (A1C_COL==""?0:idx[toupper(A1C_COL)])
      np_idx = (NP_DX_COL==""?0:idx[toupper(NP_DX_COL)])
      cdr_idx = (CDR_SOB_COL==""?0:idx[toupper(CDR_SOB_COL)])


      # write header to healthy_rows.csv
      header_line=$0
      print header_line > healthy_rows_file
      next
    }
    {
      # helpers
      function trimq(s){ gsub(/^"|"$/,"",s); return s }
      function tol(s){ s=trimq(s); return tolower(s) }
      function num(s,  t){ t=trimq(s); gsub(/[^0-9.+-eE]/,"",t); return t }

      # Diabetes rule
      diab_ok = 0
      if(d_idx>0){
        dval = tol($d_idx)
        if(dval=="no") diab_ok=1
      }
      # A1c rule
      a1c_ok = 0
      if(a1_idx>0){
        a1 = num($a1_idx)
        if(a1!="" && a1+0 < 6.5) a1c_ok=1
      }
      diabetes_pass = (diab_ok || a1c_ok)

      # Cognition rule
      cog_ok = 0
      if(np_idx>0){
        nd = tol($np_idx)
        if(nd ~ /^cognitively[ _-]*unimpaired/ || nd ~ /^normal cognition$/ || nd=="cu") cog_ok=1
      }
      if(!cog_ok && cdr_idx>0){
        cdr = num($cdr_idx)
        if(cdr!="" && cdr+0==0) cog_ok=1
      }

      if(diabetes_pass && cog_ok){
        print $0 >> healthy_rows_file
        mids[trimq($(midx))]=1
      }
    }
    END{
      # write unique Med_IDs
      for(m in mids) if(m!="") print m > subj_file
    }
  ' "$INPUT_CSV"
fi

echo "Wrote:"
echo "  $OUT_ROWS"
echo "  $OUT_SUBJ"
	