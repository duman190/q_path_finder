"""Scores the hardware runs in runs.json against the ideal distribution, the pre-run noise model
(1% two-qubit error, 2% readout error), uniform output, and a survival estimate from calibration.
Run from this folder:  python3 score_runs.py   (needs numpy; reads ../dqi_cirq/binary_toys.py)"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dqi_cirq"))
from binary_toys import B1, B2, noisy_distribution, score_counts

def analyse(path=os.path.join(HERE, "runs.json")):
    rec = json.load(open(path)); out = {}
    for run in rec["runs"]:
        inst = {"B1": B1, "B2": B2}[run["instance"]]()
        counts, N = run["counts"], run["shots"]
        hw = np.asarray(score_counts(inst, counts, bit_order="qiskit")["dist"])
        ideal = np.asarray(inst.reference()["prob"]); s = np.asarray(inst.scores())
        model = np.asarray(noisy_distribution(inst, 0.01, 0.02)).reshape(hw.shape)
        uni = np.full(hw.shape, 1 / hw.size); best = s == s.max()
        mean = lambda d: float((d * s).sum()); tv = lambda a, b: float(0.5 * np.abs(a - b).sum())
        c, g = rec["calibration"][run["backend"]], run["transpiled_ops"]
        common = (1 - c["sx_error_median"]) ** (g["sx"] + g["x"]) * (1 - c["readout_error_median"]) ** hw.ndim * (1 - c["init_error_median"]) ** inst.N
        order = sorted(np.ndindex(hw.shape), key=lambda i: (-ideal[i], i))
        out[run["instance"]] = dict(
            backend=run["backend"], shots=N, gates=g, qubits=inst.N,
            paths=["".join(map(str, i)) for i in order], score=[int(s[i]) for i in order],
            ideal=[float(ideal[i]) for i in order], model=[float(model[i]) for i in order], hw=[float(hw[i]) for i in order],
            se=[math.sqrt(hw[i] * (1 - hw[i]) / N) for i in order], uniform=1 / hw.size, m=int(inst.m),
            mean=dict(random=mean(uni), ideal=mean(ideal), model=mean(model), hw=mean(hw)),
            best=dict(random=float(uni[best].sum()), ideal=float(ideal[best].sum()), model=float(model[best].sum()), hw=float(hw[best].sum())),
            tv=dict(model=tv(model, ideal), hw=tv(hw, ideal), random=tv(uni, ideal), hw_random=tv(hw, uni)),
            signal_fraction=(mean(hw) - mean(uni)) / (mean(ideal) - mean(uni)),
            survival_median_cz=(1 - c["cz_error_median"]) ** g["cz"] * common,
            survival_layered=(1 - c["two_qubit_error_layered"]) ** g["cz"] * common,
            port1=[float(hw.sum(axis=tuple(a for a in range(hw.ndim) if a != j))[1]) for j in range(hw.ndim)],
            port1_ideal=[float(ideal.sum(axis=tuple(a for a in range(hw.ndim) if a != j))[1]) for j in range(hw.ndim)])
    return out

if __name__ == "__main__":
    for k, v in analyse().items():
        print(k, v["backend"], "mean metrics", {a: round(b, 4) for a, b in v["mean"].items()}, "best-path share", {a: round(b, 4) for a, b in v["best"].items()})
        print("   distance from ideal", {a: round(b, 4) for a, b in v["tv"].items()}, "signal fraction", round(v["signal_fraction"], 3),
              "survival estimates", round(v["survival_median_cz"], 3), round(v["survival_layered"], 3))
        print("   paths", v["paths"], "hardware", [round(x, 4) for x in v["hw"]])
