# qPathFinder — Quantum Algorithms for Multi-Constrained Path Finding

A research codebase implementing and comparing three quantum approaches to the **multi-constrained path (MCP)** problem on directed acyclic graphs (DAGs): finding a source-to-sink path whose multiple additive arc metrics all fall within specified ranges.

## The Problem

Given a DAG with source `s`, sink `t`, and per-arc additive metrics (e.g. cost, delay, bandwidth), find an `s-t` path whose metric totals simultaneously satisfy given constraints. This is a classic network optimization problem that is NP-hard in general, and becomes exponentially harder as the number of constraints grows — because the intersection of several independent constraint sets shrinks rapidly.

## Approaches Implemented

### 1. GM-QAOA (Grover-Mixer QAOA) — `qpathfinder/`

A variant of QAOA with a Grover-style mixer. The circuit encodes every arc as a qubit ("arc-indicator" encoding) and uses:

- **State preparation** (`U_S`): three modes — uniform over all paths, equal-split (a random walk), or Gibbs (amplitude proportional to `exp(-tau × path_length)`).
- **Decoder (Grover mixer)**: phase separator with separate cost and delay phase gates, followed by `U_S` reflection.
- **Classical baselines**: exact enumeration, LARAC Lagrangian relaxation, and Boltzmann (Gibbs) path sampling.

Key results: the decoder with tuned angles amplifies the optimal constrained path. With `p` rounds (complex-weight phase terms), success probability grows as `sin²((2p+1)θ)` where `θ = arcsin(√(M/N))` and `M` is the number of feasible paths out of `N` total.

**Key files:**

| File | Description |
|---|---|
| `qpathfinder.py` | Core module — `PathDAG`, state prep circuits, `StateVector` simulator, `SubspaceGMQAOA`, classical baselines, `score_counts()` |
| `instances.py` | Five test DAGs (I1–I5) with cost/delay metrics, fixed seeds |
| `run_checks.py` | Re-analysis of the 2020 figures, GM-QAOA correctness checks, classical simulation |
| `extra_checks.py` | Tuned-phase amplitude amplification (one round) on uniform and Gibbs start states |
| `make_experiments.py` | Builds `experiments.json` — 15 hardware circuits (E0, E1, E3, E4, E5) with ideal distributions |
| `experiments.json` | Pre-computed circuit definitions for hardware deployment |

**Test instances (I1–I5):** small-to-medium DAGs from 4 to 18 nodes, 5 to 56 arcs, 3 to 256 paths. I5 (layered 4×4) is the largest — 56 qubits, 256 paths.

### 2. Grover Search with Range Constraints — `grover_mcp/`

Full register-level and path-subspace simulations of Grover search where **K additive metrics each have an allowed range** `[L_k, U_k]`.

- **Oracle**: controlled addition of each metric into accumulator registers, range test, and un-computation.
- **Two simulators**: register-level (every gate applied explicitly, small instances) and path-subspace (amplitudes over enumerated paths, any size).
- **Third metric**: adds a third constraint beyond cost and delay, turning a two-constraint problem into a harder three-conjunction problem.

Key result: Grover rounds scale as `O(√(N/M))` against `O(N/M)` for uniform path sampling. That is not an advantage over good classical methods: exact meet-in-the-middle reaches the same exponent in the worst case, sub-sampled meet-in-the-middle scales like `√(N/M)` on the layered instances, and exact dynamic programming decides every tested instance with far fewer operations (see `baselines.py`). The register-level simulator confirms the path-subspace analysis for the smallest instances; it applies U_S gate by gate and the adders and range tests as register arithmetic.

**Key files:**

| File | Description |
|---|---|
| `grover_mcp.py` | Instances with 3 metrics and ranges, register-level simulator, path-subspace simulator, scaling series |
| `figures.py` | Circuit diagram (`fig_grover_circuit.png`) and results (`fig_grover_results.png`) |
| `results.txt`, `results.json` | Output from 10 Nov 2026 runs |

**Scaling experiments:** layered DAGs from `2×4` to `4×10` (up to 262,144 paths), showing log-log scaling of Grover rounds vs `√(N/M)` and classical walks vs `N/M`.

### 3. Decoded Quantum Interferometry (DQI) — `dqi_paths/` + `dqi_cirq/`

DQI solves the **many-metric path problem** where:

- A fabric has `n` stages with `p` parallel ports each (`p` is prime), yielding `pⁿ` total paths.
- There are `m = p - 1` modular metrics indexed `y = 1..p-1`. Port `a` at stage `j` weighs `a · y^j (mod p)` under metric `y`.
- Each metric has a set of `r` allowed residues `F_y`. Goal: find paths whose metric totals land in the allowed sets for as many metrics as possible.

This maps to the **Optimal Polynomial Intersection** problem: the path is the coefficient vector of a polynomial `Q`, and metric `y` evaluates `Q(y) mod p`.

**The DQI pipeline (5 steps):**

1. Prepare a superposition over small subsets of metrics with amplitudes from Fourier coefficients.
2. Write the syndrome `s = Bᵀe` into the path register (Vandermonde encoding).
3. **Decode**: `s → e`, which empties the metric register (syndromes of patterns with ≤ l nonzero entries are distinct).
4. Fourier transform over `ℤₚ` on each of the `n` digits.
5. Measure the path register — the distribution concentrates on paths satisfying many metrics.

**Key files:**

| File | Description |
|---|---|
| `dqi_paths.py` | Core module — `Instance` class, 5 DQI pipeline steps, closed-form comparison |
| `closed_form.py` | Finite-size eigenvalue formula (tridiagonal), checked against simulation; evaluated at sizes too large to simulate |
| `figures.py` | Three figures: problem illustration, pipeline diagram, score comparison |
| `results.txt` | Output of both scripts — six instances from `p=5` to `p=61` |

### 4. DQI on Real Hardware — `dqi_cirq/` + `hardware/`

Practical implementations of DQI (`l = 1`) as circuits for current quantum hardware:

- **Qudit toy** (`qudit_toy.py`): 6 five-level qudits (2 stages × 5 ports). Demonstrates the full DQI pipeline with qudit operations.
- **Binary instances** (`binary_toys.py`): B1 (5 qubits) and B2 (11 qubits) — the smallest nontrivial cases suitable for current noisy hardware.
- **OpenQASM 2** (`qasm/b1.qasm`, `qasm/b2.qasm`): Hardware-ready circuit files written by an own QASM writer, validated by parsing back and re-simulating.
- **Noise analysis** (`hardware/score_runs.py`): Scores hardware runs against ideal distributions, pre-run noise models, and survival estimates from calibration data.

**Key files:**

| File | Description |
|---|---|
| `ops.py` | Backend-neutral circuit format — gate list with matrices, numpy simulators, Cirq builder, OpenQASM writer/parser |
| `qudit_toy.py` | Full DQI pipeline on 6 qudits (dimension 5), numpy checks + Cirq interface |
| `binary_toys.py` | B1 (5 qubits, 2 stages) and B2 (11 qubits, 3 stages), noise simulation, hardware scoring |
| `verify_numpy.py` | 35/35 checks pass — unitary, noise, QASM round-trip validation |
| `independent_check.py` | Independent verification — a separate QASM reader/simulator confirming the results share no code with the primary implementation |
| `hardware/score_runs.py` | Hardware run analysis: scores `runs.json` against ideal, noise model, and uniform baselines |
| `hardware/fig4_hardware.png` | Hardware figure: B1 on 5 qubits preserves the pattern; B2 on 11 qubits loses most signal |

## Repository Structure

```
q_path_finder/
├── README.md                     # This file
├── LICENSE
├── ieee_paper/                   # The paper
│   ├── qpath_finder.tex          # LaTeX source (IEEEtran)
│   ├── qpath_finder.pdf          # Compiled paper
│   └── figures/                  # Figures, calibration screenshots, transpiled circuits
├── qpathfinder/                  # State preparation, DAG instances, GM-QAOA
│   ├── qpathfinder.py            # Core module
│   ├── instances.py              # 5 test DAGs
│   ├── run_checks.py             # Re-analysis + correctness checks
│   ├── extra_checks.py           # Tuned-phase amplitude amplification
│   ├── make_experiments.py       # Hardware circuit builder
│   ├── experiments.json          # Pre-computed circuits (15 total)
│   └── results.txt               # Check output
├── grover_mcp/                   # Grover search with range constraints
│   ├── grover_mcp.py             # Register-level + path-subspace simulators
│   ├── baselines.py              # DP, meet-in-the-middle, resource estimate
│   ├── figures.py                # Circuit + results figures
│   ├── results.txt, results.json # Run output
│   ├── baselines.txt, baselines.json
│   └── fig_grover_*.png          # Figures
├── dqi_paths/                    # DQI — state-vector simulation
│   ├── dqi_paths.py              # Core: Instance class, 5 DQI steps
│   ├── closed_form.py            # Finite-size eigenvalue formula
│   ├── figures.py                # 3 figures
│   └── results.txt
├── dqi_cirq/                     # DQI — hardware circuits
│   ├── ops.py                    # Backend-neutral gate format + simulators
│   ├── qudit_toy.py              # 6-qudit DQI demonstration
│   ├── binary_toys.py            # B1 (5 qubits) + B2 (11 qubits)
│   ├── verify_numpy.py           # 35/35 numpy checks pass
│   ├── independent_check.py      # Independent verification (no shared code)
│   ├── fig_dqi_circuit.py        # Gate-level drawing of the B1 circuit
│   ├── qasm/b1.qasm, qasm/b2.qasm
│   └── results.txt
└── hardware/                     # Hardware run analysis
    ├── score_runs.py             # Scores real runs vs ideal/noise model
    ├── reanalysis.py             # Weight choices, z-scores, error per CZ
    ├── fig_hardware.py           # Hardware comparison figure
    ├── runs.json                 # Raw hardware counts
    └── fig4_hardware.png         # B1 vs B2 hardware comparison
```

## The Paper

The paper is `ieee_paper/qpath_finder.tex` (compiled: `ieee_paper/qpath_finder.pdf`), with its figures in `ieee_paper/figures/`. Appendix A maps the paper to the code in this repository. `ieee_paper/figures/` also holds the IBM backend calibration screenshots (`fig7_*`, `fig8_*`) and the transpiled circuits that were run (`fig9_*`, `fig10_*`), which the paper cites but does not reproduce.

To rebuild it with TeX Live or MacTeX (`brew install --cask mactex`), run pdflatex three times to resolve cross-references:

```bash
cd ieee_paper
pdflatex qpath_finder.tex
pdflatex qpath_finder.tex
pdflatex qpath_finder.tex
```

## Algorithm Comparison

| Aspect | GM-QAOA | Grover MCP | DQI |
|---|---|---|---|
| **Encoding** | Arc-indicator (1 qubit per arc) | Arc + accumulator registers | Vandermonde polynomial encoding |
| **Constraints** | 2 metrics (cost + delay) | 3+ metrics with ranges | Many metrics (p−1 for prime p) |
| **Goal** | Exact satisfaction (optimal path) | Range constraints | Maximize satisfied metrics |
| **Speedup** | None: classically simulable at fixed depth | Quadratic over sampling only; matched by classical meet-in-the-middle | Inherited from DQI on OPI, not proven |
| **Best for** | Small DAGs (≤24 arcs on simulator) | Multi-range feasibility | Large path spaces with modular structure |
| **Classical analogue** | Boltzmann sampler at the LARAC multiplier (beats 2 rounds on I4, I5) | Dynamic programming (pseudo-polynomial), meet-in-the-middle, LARAC | Interpolation; brute force at simulated sizes |

## Getting Started

### Prerequisites

- Python 3.10+
- `numpy`, `scipy`, `matplotlib`

### Running Each Module

**GM-QAOA (qpathfinder):**
```bash
cd qpathfinder
python3 run_checks.py      # Re-analyses + correctness checks
python3 extra_checks.py    # Tuned-phase amplitude amplification
python3 make_experiments.py # Builds experiments.json
```

**Grover MCP:**
```bash
cd grover_mcp
python3 grover_mcp.py      # All instances + scaling series
python3 figures.py         # Generates figures
```

**DQI (state-vector):**
```bash
cd dqi_paths
python3 dqi_paths.py       # Six instances with varying sizes
python3 closed_form.py     # Closed-form formula + larger-size evaluation
python3 figures.py         # Three figures
```

**DQI (circuits + hardware):**
```bash
cd dqi_cirq
python3 verify_numpy.py    # 35/35 numpy checks (no Cirq needed)
python3 qudit_toy.py       # Numpy checks + Cirq if installed
python3 binary_toys.py     # Numpy + Cirq simulation, writes QASM
```

### Dependencies

Core: `numpy`, `scipy` (for GM-QAOA angle optimization).
Visualization: `matplotlib`.
Optional (Cirq-based): `pip install cirq` to run QuDIT and binary DQI on Cirq simulators.
Optional (Qiskit): `pip install qiskit` to convert `experiments.json` circuits for IBM hardware.

## Key Results Summary

### GM-QAOA
- `p`-round decoder achieves success `sin²((2p+1)θ)` vs uniform sampling `1/N`.
- Gibbs preparation + decoder achieves the same with `p = 1` (single round).
- Classical simulation of the decoder is exact — the classical rejection sampler matches the quantum distribution to `TVD < 0.004`.
- On I5 (56 qubits, 256 paths), 12 rounds achieve `P(optimum) = 0.9999`.
- With realistic noise (ε = 10⁻³), GM-QAOA with p=1 outperforms uniform on I1–I4 but loses on I5 due to circuit depth.

### Grover Search
- Grover rounds scale as `O(√(N/M))` (log-log slope ≈ 0.5) vs random walks `O(N/M)` (log-log slope ≈ 1.0).
- Confirmed across 17 scaling instances from `2×4` to `4×10` layered DAGs.
- The register-level simulator (U_S gate by gate, arithmetic as register operations) matches the path-subspace formula to `5e-16`.
- `baselines.py`: dynamic programming, exact and sub-sampled meet-in-the-middle, evaluations to 90 % success, and a logical Toffoli estimate per oracle call (`baselines.txt`).

### DQI
- DQI expectation `E[metrics]` exceeds uniform random and classical interpolation.
- Closed-form eigenvalue formula agrees with simulation to `1e-14` on the instances with `2l+1 < n+1` (the hypothesis of the DQI finite-size theorem).
- On hardware: B1 (5 qubits) keeps 67 % of the ideal gain, B2 (11 qubits) 15 %; one effective error of about 1 % per transpiled CZ fits both runs (`hardware/reanalysis.py`).

## Notes

- All simulations are ideal (noise-free) unless otherwise noted.
- The hardware figures (Fig. 4) use real IBM hardware runs with calibration snapshots.
- Cirq and Qiskit code is included but was not executed during development (libraries not installed in the authoring environment). API versions were verified against current documentation.
- The decoder angles in `experiments.json` were optimized with knowledge of the optimum — they fix the circuits for hardware runs and represent a best case, not a deployable procedure.
