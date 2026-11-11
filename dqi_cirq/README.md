# dqi_cirq

Decoded quantum interferometry (DQI, l = 1) on the many-metric path problem of `../dqi_paths`,
written as circuits: one 5-level-qudit toy (simulation only) and two qubit instances small enough
for current hardware. Every circuit is a single backend-neutral op list (`ops.py`); the numpy
simulator, the Cirq builder and the OpenQASM writer all read that same list.

## Files

| File | What it is |
|---|---|
| `ops.py` | Op-list format (gate name or explicit matrix + targets), numpy state-vector and density-matrix appliers, 2-qubit depolarising channel, readout flips, gate counts, the Cirq builder `to_cirq`, own OpenQASM 2 writer and parser. |
| `qudit_toy.py` | Deliverable A: `Instance(5, 2, 2, seed=3)` (2 stages x 5 ports, 4 metrics) on 6 qudits of dimension 5. `build_ops`, numpy checks, and (Cirq) `build_qudit_circuit`, `simulate_qudit_with_cirq`, `sample_qudit_with_cirq`. |
| `binary_toys.py` | Deliverable B: instances B1 (5 qubits) and B2 (11 qubits), gate lists, direct-definition reference, numpy noise prediction, `score_counts`, and (Cirq) `build_binary_circuit`, `simulate_ideal`, `simulate_noisy`, `export_qasm`. |
| `verify_numpy.py` | Runs every numpy check, prints a PASS/FAIL table, writes `results.txt` and `qasm/*.qasm`. |
| `qasm/b1.qasm`, `qasm/b2.qasm` | Hardware-ready OpenQASM 2 (gates h, x, z, ry, cx, ccx), written by our own writer and validated by parsing back and re-simulating. |
| `results.txt` | Output of `verify_numpy.py`. |

## What was executed here and what was not

- Executed (Python 3.13, numpy 2.5): everything in `verify_numpy.py` and the `__main__` of
  `qudit_toy.py` and `binary_toys.py` up to the Cirq part. 35/35 checks pass (`results.txt`).
- **Not executed: any Cirq code.** Cirq is not installed in the environment where this was written.
  All `import cirq` lines sit inside functions; the self-tests print "Cirq self-test skipped".
- Cirq API confirmed on quantumai.google: `cirq.MatrixGate(matrix, *, name=None, qid_shape=None,
  unitary_check=True, ...)`; `cirq.depolarize(p, n_qubits=1)` with
  rho -> (1-p) rho + p/(4^n-1) sum_{P != I} P rho P (our numpy channel implements exactly this);
  `cirq.LineQid(i, dimension=d)` and qudit simulation with `cirq.Simulator`. The docs pages did not
  state MatrixGate's index order; the big-endian convention used below is Cirq's documented
  general convention, not confirmed on that page. Everything else (`cirq.ry`, `cirq.CCX`,
  `cirq.CCXPowGate`, `cirq.bit_flip`, `cirq.is_measurement`, `DensityMatrixSimulator`,
  `circuit.to_qasm()`) is used from memory of the long-stable API.

## Conventions

- Matrix index order: for an op on targets (t0, t1, ...), the row/column index is big-endian
  over (t0, t1, ...): t0 is the most significant digit. Same for the full state (register 0 most
  significant), which is what `cirq.Simulator(...).simulate(c, qubit_order=qids)` returns.
- Qudit toy registers: 0..3 metric digits (metric y = i+1), 4..5 path digits (stage 0, 1).
  Fourier step F[x, s] = exp(-2 pi i s x / 5) / sqrt(5), the sign of `dqi_paths.py`.
- Qubit instances: qubits 0..m-1 metric, m..m+n-1 path (qubit m+j = stage j), ancilla last (B2).
- Bitstrings: from Cirq (`result.measurements["path"]`, key "path") character j = stage j.
  In the QASM files classical bit c[j] = stage j, and Qiskit/IBM count strings print c[n-1]
  first, so **stage j = character n-1-j**. `score_counts(..., bit_order="qiskit")` reverses it.
  A QASM file produced by Cirq's `to_qasm` names the register `m_path`; the same rule holds.
- Sign check for the binary case: g_i(t) = (-1)^(t+v_i)/sqrt 2, so ghat_i(1) = (-1)^(v_i): a Z on
  metric qubit i when v_i = 1, as in the brief. The check compares amplitudes, not only
  probabilities, against w0 P0 + w1 P1(x) (max deviation 2e-16).

## Running with Cirq

    pip install cirq
    python3 verify_numpy.py      # numpy checks (no Cirq needed)
    python3 qudit_toy.py         # numpy checks + Cirq simulate/run vs numpy, PASS/FAIL
    python3 binary_toys.py       # Cirq ideal + noisy runs vs numpy prediction, writes qasm/*_from_cirq.qasm

## Real hardware

1. **IBM (free tier)**: use `qasm/b1.qasm` / `qasm/b2.qasm` (or the Cirq export) and load them in
   Qiskit with `qiskit.qasm2.load(path)` or `qiskit.qasm2.loads(text)`; transpile for the
   chosen backend and run with the Sampler primitive following IBM's current Qiskit Runtime
   guide (the API changes; not reproduced here). Transpiling turns each ccx into at least six cx,
   more where the qubits are not adjacent. B1 is the realistic first target.
2. IonQ, AQT, Pasqal and Azure Quantum have Cirq interfaces (`cirq-ionq`, `cirq-aqt`,
   `cirq-pasqal`, Azure Quantum's Cirq support); see their docs.
3. Google's quantum processors are not publicly accessible.

Feeding counts back:

    from binary_toys import B1, score_counts
    r = score_counts(B1(), counts, bit_order="qiskit")   # counts = {"01": 812, ...} from IBM
    r["dist"], r["tv"], r["mean_satisfied"]

Use `bit_order="cirq"` for counts from `simulate_ideal` / `simulate_noisy` or other Cirq results.

## Noise model (crude)

Two-qubit depolarising error p2 after every cx; each ccx applied ideally and then charged with six
2-qubit depolarising errors on the pairs of the textbook 6-cx decomposition; independent readout
flip p_ro on each measured qubit. No single-qubit errors, no T1/T2, no crosstalk, no routing
(SWAP) overhead, no coherent errors. Real devices will usually do worse than predicted,
especially for B2.
