# grover_mcp

Grover search for a multi-constrained path with range constraints on a DAG. Needs Python 3 with numpy and matplotlib,
and the `qpathfinder` folder (state-preparation circuit and test instances) next to this one.

| File | What it is |
|---|---|
| `grover_mcp.py` | Instances with three metrics, range selection, a register-level simulator (arc qubits and one total register per metric, every gate applied), a path-subspace simulator, and the scaling series. Writes `results.txt` and `results.json`. |
| `figures.py` | Draws `fig_grover_circuit.png` and `fig_grover_results.png` from `results.json`. |
| `results.txt`, `results.json` | Output of the run of 10 Nov 2026. |

All results are ideal (noise-free) simulations. Nothing here was run on hardware.
