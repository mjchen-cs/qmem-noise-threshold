# Sharp noise threshold for learning with quantum memory

Code for the paper *Sharp noise threshold for learning with quantum memory*:
theory and numerics for the threshold $\lambda_X^2+\lambda_Y^2+\lambda_Z^2=1$, the experiments on the
Wukong 180 superconducting processor, their analysis, and the figures. Raw data: https://figshare.com/s/113a67455bf9eacda354 (see [`data/README.md`](data/README.md)).

```
git clone https://github.com/mjchen-cs/qmem-noise-threshold.git
```

```
pip install -r requirements.txt
```

## Theory (`theory/`)

| Script | Paper | What it does |
|---|---|---|
| `kfunction.py` | Methods, Supp. S5–S6 | $K$ from its definition, the reduction formula $K(W)$ and the closed form for one memory qubit; cross-checks them |
| `maximize_K.py` | Supp. S8 | maximizes $K$ over all $m$-qubit memory marginals, e.g. `python maximize_K.py 4 0.577,0.577,0.577 2` |
| `stabilizer_K.py` | Supp. S7 | classical formula $K=2^k2^{-H_2(L\mid E)}$ for stabilizer-code memories, checked against the general formula |
| `stabilizer_scan.py` | Supp. S8 | random stabilizer codes, `python stabilizer_scan.py 7` (300 codes) |
| `stabilizer_depolarizing.py` | Supp. S7 | depolarizing noise at $1/\sqrt3$: averaged versus syndrome-conditioned values |
| `code_thresholds.py` | Fig. 1a, Supp. S4 | per-site factors and thresholds of coded memories (table of 13 codes) |

## Hardware (`hardware/`)

Requires `pyqpanda3` and an Origin Quantum API key in the environment variable `ORIGIN_API_KEY`
(never written to disk). Every command also runs offline with `--mode dryrun` (exact simulation plus readout flips).

| Script | What it does |
|---|---|
| `circuits.py` | gate lists, exact simulator, memory encoders, native compilation (RPhi pulses, virtual Z through CZ), pyqpanda3 translation |
| `cloud.py` | cloud back end (cloud optimization and server-side readout correction off by default) and dry-run back end |
| `run.py` | experiments: `MS` (all sites at once), `MSI` (one site at a time), `RO` (readout calibration); resumable |
| `screen.py` | empirical screening of chains and pairs |
| `analyze.py` | per-site factors $\nu_j$, factorization ratio $R_w$, joint advantage $\log_2A_w$ with bootstrap intervals; frame alignment from the interleaved noiseless circuits |
| `mitigate.py` | per-site factors with and without linear tensor-product readout mitigation |
| `drift.py` | common-mode drift check across submission batches |
| `compile_tests/` | compilation control tests (cloud optimization, single-$Z$ test, gate cancellation) |

Reproducing the analysis from the deposited data:

```
cd hardware
python analyze.py  --step MS --mode origin_pairs_raw --B 200        # Fig. 2 numbers
python drift.py    --mode origin_pairs_raw
python mitigate.py --run MS_origin_code_raw --ro RO_origin_ro        # Fig. 3, run 1
python mitigate.py --run MS_origin_code_ro                           # Fig. 3, run 2 (interleaved readout calibration)
python screen.py   --size 2 --collect --qmin 0.7                     # site selection
```

Repeating the main experiment (about 450 s of processor time):

```
python run.py --fetch_chip
python run.py --step MS --mode origin --paths 94-104,92-102,119-128,131-141,129-139,112-121,79-88,40-49,105-114 \
              --kinds bare --a 1.00,0.40,0.56,0.64,0.72,0.80,0.88 --circ 4000 --shots 40 --tag _pairs_raw
```

## Figures (`figures/`)

`fig1_threshold.py`, `fig2_scaling.py` (needs `data/MS_summary_origin_pairs_raw.json` from `analyze.py`),
`fig3_code.py` (needs the two `*_mitigate.json` files from `mitigate.py`).

## License

MIT (see `LICENSE`).
