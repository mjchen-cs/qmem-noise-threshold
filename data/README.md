# Data

The raw data are deposited on Figshare: https://figshare.com/s/113a67455bf9eacda354 . Download them into this
folder (or set the environment variable `DATA_DIR`). Every experiment folder contains

* `manifest.json` – qubit layout (`layout.physical`, `system`, `m0`, `syndrome3`), and one record per circuit
  (`key`, letters `P` per site, `meta.policy` = memory and noise setting, `meta.active` = sites in use);
* `raw.json` – outcome counts per circuit (`counts`: outcome index → count, logical qubit 0 = most significant bit)
  and the cloud job identifier;
* `jobs.json` – job identifiers, submission and completion times.

| Folder | Content | Used in |
|---|---|---|
| `MS_origin_pairs_raw` | 9 Bell pairs, a = 1.00, 0.40, 0.56, 0.64, 0.72, 0.80, 0.88; 4,000 circuits × 40 shots each; raw counts | Fig. 2 |
| `MS_origin_pairs` | the same with the platform's readout correction (amend) | Supplementary S9 |
| `MS_origin_code_raw` | chain 111-120-129-138, bare and rep3, a = 1.00, 0.48–0.80; 3,000 circuits per setting | Fig. 3 (run 1) |
| `RO_origin_ro` | readout calibration for run 1 | Fig. 3 (run 1) |
| `MS_origin_code_ro` | same chain, a = 1.00, 0.52–0.64, 4,000 circuits per setting, interleaved readout calibration | Fig. 3 (run 2) |
| `MS_origin_pilotch` | nine pairs, second day: pilot (a = 1.00, 0.28; eta = 0.62, 0.78), 400 circuits | site check |
| `MS_origin_phase` | nine pairs, second day: threshold scan, dephasing a = 0.20–0.34 and depolarizing eta = 0.55–0.94, 4,000 circuits per setting | Fig. 3, Supplementary S16 |
| `MS_origin_codepilot` | coding chain, second day: pilot | chain check |
| `MS_origin_bias`, `MS_origin_bias2` | coding chain, second day: bias scans (t, t, lambda_z), lambda_z = 0.95, 0.90 and 1.00, 0.98; 4,000 circuits per setting, interleaved readout calibration | Fig. 4, Supplementary S16 |
| `mcm_test_20261119_0058.json` | mid-circuit measurement test (rejected by the cloud service) | Supplementary S9 |
| `RSP_origin` | second qubit set (system 119), unencoded / rep3 / rep5 memories, a = 0.40–0.80; 1,800 circuits × 10 shots per setting; platform readout correction (single-site quantities only) | Supplementary S14 |
| `MS_summary_*.json`, `MSI_summary_origin_iso.json`, `RSP_summary_origin.json`, `*_mitigate.json` | analysis summaries (outputs of `hardware/analyze.py`, `hardware/mitigate.py`) | figures, Supplementary S11–S15 |
| `MS_origin_scr0`–`34`, `MS_origin_scrp0`–`6` | site screening of 104 chains and 86 pairs | Methods, S9 |
| `MS_origin_pilot`, `MS_origin_cal`, `MS_origin_ref`, `MSI_origin_iso`, `MS_origin_par5` | diagnosis of sites selected from calibration data | S9, S15 |
| `chip.json` | calibration snapshot of the processor | site screening |
| `screen_*.json` | screening groups and results | Methods |
| `zgate_test_*.json`, `codez_test_*.json`, `cancel_test_*.json` | compilation control tests | S9 |

Experiments were run on the Wukong 180 superconducting processor (Origin Quantum) on 17–19 November 2026.
