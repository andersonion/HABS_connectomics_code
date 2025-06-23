#! /bin/env python3
import json
import os
from pathlib import Path

WORK = os.environ['WORK']

# --- CONFIGURE THESE ---
bids_root = Path(f"{WORK}/human/HABS/HABS_BIDS")
reference_json = Path(f"{WORK}/human/HABS/HABS_BIDS/sub-H4980y0/func/sub-H4980y0_task-rest_bold.json")

# --- Load SliceTiming from reference ---
with open(reference_json, 'r') as f:
    ref_data = json.load(f)

if "SliceTiming" not in ref_data:
    raise ValueError(f"'SliceTiming' not found in {reference_json}")

slice_timing = ref_data["SliceTiming"]

# --- Loop through all JSON files ---
for json_file in bids_root.rglob("*.json"):
    with open(json_file, 'r') as f:
        data = json.load(f)

    if "SliceTiming" not in data:
        print(f"Inserting SliceTiming into: {json_file}")
        data["SliceTiming"] = slice_timing

        # Optional: backup the original file
        backup_path = json_file.with_suffix(".json.bak")
        os.rename(json_file, backup_path)

        # Save modified file
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)
