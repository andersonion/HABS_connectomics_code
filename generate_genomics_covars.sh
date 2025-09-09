#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./concat_genomics_by_connectomes.sh /path/to/connectomes \
#     "Genomics HD 1 African American 50+ Request 545.csv" \
#     "Genomics HD 1 Mexican American 50+ Request 545.csv" \
#     "Genomics HD 1 Non-Hispanic White 50+ Request 545.csv" \
#     /path/to/output/genomics_by_subject.csv
#
# Output:
#   - /path/to/output/genomics_by_subject.csv
#   - /path/to/output/genomics_by_subject.unmatched_connectomes.txt

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 CONN_DIR GENO1.csv GENO2.csv GENO3.csv OUT.csv" >&2
  exit 1
fi

CONN_DIR=$1; shift
GENO1=$1; GENO2=$2; GENO3=$3; shift 3
OUTCSV=$1; shift || true

if [[ ! -d "$CONN_DIR" ]]; then
  echo "ERROR: Connectome directory not found: $CONN_DIR" >&2
  exit 1
fi
for f in "$GENO1" "$GENO2" "$GENO3"; do
  [[ -f "$f" ]] || { echo "ERROR: Missing genomics CSV: $f" >&2; exit 1; }
done

OUTDIR=$(dirname "$OUTCSV")
mkdir -p "$OUTDIR"
SIDECAR="${OUTCSV%.csv}.unmatched_connectomes.txt"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

GENO_ALL="$tmpdir/genomics_concat.csv"
SUBJECTS="$tmpdir/subjects_pairs.csv"

# 1) Concatenate genomics CSVs, keep only the first header
awk 'FNR==1 && NR>1 {next} {print}' "$GENO1" "$GENO2" "$GENO3" > "$GENO_ALL"

# 2) Find the Med_ID column index (1-based)
MID_COL=$(awk -F',' 'NR==1{
  for(i=1;i<=NF;i++){
    gsub(/^"|"$/,"",$i);
    if($i=="Med_ID"){ print i; exit }
  }
}' "$GENO_ALL")

if [[ -z "${MID_COL:-}" ]]; then
  echo "ERROR: Could not find Med_ID column in genomics CSVs." >&2
  exit 1
fi

# 3) Build unique list of connectome subject prefixes and map to Med_ID
#    subject = H########_y0 or H########_y2
#    med_id  = digits between H and _y
find "$CONN_DIR" -maxdepth 1 -type f -printf '%f\n' \
| awk 'match($0,/^(H[0-9]+_y[02])/,m){print m[1]}' \
| sort -u \
| awk -F',' '{
    subj=$0
    med=subj
    sub(/^H/,"",med);      # drop leading H
    sub(/_.*/,"",med);     # drop _y0/_y2 and beyond
    print subj","med
}' > "$SUBJECTS"

# 4) Join: for each subject, print subject + genomics row (matched by Med_ID).
#    Also record subjects with no match to sidecar.
#    We keep the first occurrence of each Med_ID's genomics row.
awk -v OFS=',' -F',' -v MID="$MID_COL" -v OUT="$OUTCSV" -v MISS="$SIDECAR" '
  NR==FNR {
    if (FNR==1) { header=$0; next }            # store header
    # Key = Med_ID (trim spaces and quotes)
    id=$MID
    gsub(/^[ \t"]+|[ \t"]+$/,"",id)
    if (!(id in geno)) { geno[id]=$0 }         # first occurrence wins
    next
  }
  FNR==1 {
    # subjects_pairs.csv has no header; write output header now
    print "subject", header > OUT
    # truncate sidecar
    close(MISS); system("> " MISS)
  }
  {
    # each line: subject,medid
    split($0, a, /,/)
    subj=a[1]; id=a[2]
    if (id in geno) {
      print subj, geno[id] >> OUT
    } else {
      print subj >> MISS
    }
  }
' "$GENO_ALL" "$SUBJECTS"

echo "Wrote: $OUTCSV"
echo "Unmatched connectome prefixes (no Med_ID in genomics): $SIDECAR"
