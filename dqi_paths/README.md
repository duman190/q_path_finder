# dqi_paths

State-level simulation of decoded quantum interferometry (DQI) for the many-metric path problem (Section on DQI in `ieee_paper/qpath_finder.tex`).
Needs Python 3 with numpy and matplotlib. Everything here was run on 10 Nov 2026.

| File | What it is |
|---|---|
| `dqi_paths.py` | The path problem (n stages, p ports per stage, p-1 modular metrics) and a state-level simulation of the five DQI steps. `python3 dqi_paths.py` prints the six instances of the paper. |
| `closed_form.py` | The finite-size eigenvalue expression for the expected score, checked against the simulation and evaluated at larger sizes. Also tests where unique decoding stops, and prints the toy instance in full. |
| `figures.py` | Draws three figures: problem, pipeline and scores. |
| `results.txt` | Output of the two scripts above. |
| `fig1_problem.png`, `fig2_pipeline.png`, `fig3_scores.png` | The figures. |

Notes

- The simulation works on state vectors, not gates. Decoding is a lookup table; the code checks
  that every pattern with at most n//2 nonzero entries has its own syndrome.
- Allowed sets are random subsets of the residues (fixed seeds).
- The closed form is a reconstruction of the DQI paper's finite-size theorem. It agrees with the
  simulation to 1e-14 on the instances with an odd number of stages; check it against the
  paper's text before quoting it.
