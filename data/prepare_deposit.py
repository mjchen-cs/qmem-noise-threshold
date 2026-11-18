# -*- coding: utf-8 -*-
"""Copy the experiment folders used in the paper from a working run directory into a clean deposit folder
(for Figshare), keeping manifest.json, raw.json and jobs.json and removing manifest fields that the analysis does
not use. Usage: python prepare_deposit.py SOURCE_RUNS_DIR DEST_DIR"""
import os, sys, json, shutil

RUNS = (["MS_origin_pairs_raw", "MS_origin_pairs", "MS_origin_code_raw", "MS_origin_code_ro", "RO_origin_ro",
         "MS_origin_pilot", "MS_origin_cal", "MS_origin_ref", "MSI_origin_iso", "MS_origin_par5"]
        + [f"MS_origin_scr{g}" for g in range(35)] + [f"MS_origin_scrp{g}" for g in range(7)])
FILES = ["chip.json", "screen_groups.json", "screen_groupsp.json", "screen_result.json", "screen_resultp.json",
         "zgate_test_20261117_1818.json", "codez_test_20261117_1842.json", "codez_test_20261117_1901.json",
         "codez_test_20261118_0140.json", "cancel_test_20261118_0200.json"]
KEEP = {"step", "mode", "layout", "records", "chip_fetched", "n_circuits", "channels_scan", "task", "phys_override", "amend", "args"}

src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
for r in RUNS:
    s = os.path.join(src, r)
    if not os.path.isdir(s):
        print("missing", r); continue
    d = os.path.join(dst, r); os.makedirs(d, exist_ok=True)
    man = json.load(open(os.path.join(s, "manifest.json")))
    json.dump({k: v for k, v in man.items() if k in KEEP}, open(os.path.join(d, "manifest.json"), "w"), indent=1)
    for f in ("raw.json", "jobs.json"):
        if os.path.exists(os.path.join(s, f)):
            shutil.copy(os.path.join(s, f), os.path.join(d, f))
for f in FILES:
    if os.path.exists(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    else:
        print("missing", f)
print("deposit written to", dst)
