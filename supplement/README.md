# Supplementary analysis (Supplementary Sections S10–S16)

All scripts read the deposited raw data from `DATA_DIR` (default `../data`).

| Step | Command | Output |
|---|---|---|
| 1. Full result tables and robustness checks | `python supp_compute.py` | `supp_results.json` |
| 2. Validation on simulated data (run several seeds; each call writes `sim_<test>_<seed>.json`) | `python supp_simulate.py --test V1 --reps 75 --seed 0` (seeds 0, 1)<br>`python supp_simulate.py --test V2 --reps 50 --seed 0`<br>`python supp_simulate.py --test V3 --reps 13 --seed 0` (seeds 0–3) | `sim_*.json` |
| 3. Aggregate the simulations | `python supp_simulate.py --test summary` | `sim_summary.json` |
| 4. Figures | `python supp_figures.py` | `figs/sfig_*.pdf`, `rsp_fits.json`, `diag_results.json` |
| 5. LaTeX tables | `python supp_tables.py` | `supp_tables.tex`, `supp_table_letters.tex` |

`supp_core.py` contains the estimators (identical definitions to `../hardware/analyze.py` and `../hardware/mitigate.py`,
vectorized). Runtime: step 1 about 1 min, step 2 about 10 min on two cores.
