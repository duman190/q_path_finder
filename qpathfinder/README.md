# qpathfinder

Reference code for the path-superposition state preparation, the DAG instances and GM-QAOA. Needs Python 3 with numpy and scipy.
Everything here was executed on 10 Nov 2026 except `to_qiskit()` (qiskit was not installed
where the code was written), so treat that one function as unverified until its first run.

## Files

| File | What it is |
|---|---|
| `qpathfinder.py` | The DAG class, the state-preparation circuit U_S (uniform, equal-split, Boltzmann), a small gate-level simulator, the Grover-mixer decoder, the classical simulation of that decoder, classical baselines, `score_counts()` for measured data, and the `to_qiskit()` adapter. |
| `instances.py` | The five test instances I1..I5 (fixed seeds) and their delay bounds. |
| `run_checks.py`, `extra_checks.py` | The checks behind the numbers of this module. Output is in `results.txt`. |
| `make_experiments.py` | Writes `experiments.json`: the 15 hardware circuits of experiments E0, E1, E3, E4, E5 as gate lists, each with its ideal output distribution and the optimum bit string. |
| `experiments.json` | Output of the above. Circuits with up to 24 qubits were replayed gate by gate and matched their ideal distribution to 1e-9. |

## Gate lists

A circuit is a list of gates on qubits 0..n-1, one qubit per arc of the DAG:

- `["x", q]`
- `["cx", control, target]`
- `["cry", theta, control, target]`
- `["p", angle, q]`: diag(1, exp(-i*angle))
- `["refl0", beta]`: phase exp(-i*beta) on the all-zeros state (an n-qubit multi-controlled phase)

Bit strings are written q_{n-1} ... q_0, the same order IBM uses for counts.

## From a gate list to a hardware run

```python
import json
from qpathfinder import to_qiskit, score_counts
from instances import suite

exps = json.load(open("experiments.json"))["circuits"]
dags = {g.name: g for g, _ in suite()}

e = exps[1]                                   # E1, I1, uniform prep
qc = to_qiskit([tuple(g) for g in e["gates"]], e["n_qubits"])
# transpile and run qc with IBM's current Sampler (see their "sampler" guide), get `counts`
# print(score_counts(dags[e["instance"]], counts, e["ideal"], e["optimum"]))
```

For every run, save alongside the raw counts: backend name, date, the calibration snapshot,
transpiler optimisation level and seed, the physical-qubit layout, and the two-qubit gate count
and depth of the transpiled circuit.

## Known gaps

- `to_qiskit()` builds the reflection with `mcp` and no ancillas. That is fine up to about a
  dozen qubits. For I4 and I5 use an ancilla-assisted multi-controlled gate instead, and record
  which synthesis was used.
- The decoder angles in `experiments.json` were tuned with knowledge of the optimum. They fix the
  circuits for the hardware runs; they are a best case, not a deployable procedure.
- The 56-qubit circuits (I5) could not be replayed gate by gate here. Their ideal distributions
  come from the classical formula, which matched the gate-level simulator on I1..I4.
