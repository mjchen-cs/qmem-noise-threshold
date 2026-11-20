# CSS codes under independent bit and phase noise (Supplementary Section S7)

Numerical and symbolic checks for the reduction of the threshold conjecture, for CSS codes with
λ_Y = λ_X λ_Z, to the classical inequality F(A,B;b)·F(B^⊥,A^⊥;a) ≤ 2^k, and for the remaining
lemma (Conjecture S7.x, a Ginibre-type correlation inequality).

| script | what it checks |
|---|---|
| `cdlib.py` | helper functions (row reduction, dual space, exact F(A,B;b)) |
| `check_css_dual.py` | the inequality on random pairs (A,B), m ≤ 8, on the boundary (1+a²)(1+b²)=2 |
| `check_L1_symbolic.py` | the closed-form identity of lemma L1 (sympy) |
| `check_case3_endpoints.py`, `check_case3_interpolation.py` | endpoint identities and interpolation bounds of induction case 3 |
| `check_L3b_product.py` | the remaining inequality D ≥ 0 for random product measures |
| `check_L3b_priors.py` | D for ferromagnetic Ising priors (holds) and comparison classes |
| `counterexample_gks2.py` | a law satisfying the first and second Griffiths inequalities with D < 0 |
| `check_ginibre_cone.py` | adversarial minimisation of D over the cone cut out by Ginibre inequalities |
| `search_fastF.py` | fast exact evaluator of F(A,B;b) (bit-vector enumeration, up to ~20 qubits) |
| `search_families.py` | structured codes up to 18 qubits (repetition, Shor-type, quantum Hamming, Reed–Muller, surface, toric) |
| `search_adv.py` | per-qubit biases optimised on each qubit's boundary curve (structured and random codes) |
| `search_climb.py` | hill-climbing over code pairs (`python search_climb.py seed n seconds`) |
| `search_rand14.py` | random dense and sparse pairs with 14–16 qubits |
| `search_l3b_adv.py` | the remaining inequality with 4–5 syndrome bits, biases optimised to minimise D (`python search_l3b_adv.py seed seconds 4,5`) |
| `certificate_lp.py` | Fourier-multiplier LP certificates (`python certificate_lp.py d seed trials`) |

Requirements: numpy, scipy, sympy. All scripts run in seconds to minutes on a laptop.
