import os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "acc_data")
SUBDIRS = ("golden", "simulator", "runtime")

# tensor display name  ->  snapshot filename (same in the three subdirs)
TENSORS = [
    ("input_padding_mask",            "input_padding_mask.txt"),
    ("flows.2/Split_output_0",        "_duration_predictor_flows_2_Split_output_0-rs.txt"),
    ("flows.2/Split_output_1",        "_duration_predictor_flows_2_Split_output_1-rs.txt"),
    ("flows.2/Concat_57",             "_duration_predictor_flows_2_Concat_57_output_0-rs.txt"),
    ("flows.2/Mul_81",                "_duration_predictor_flows_2_Mul_81_output_0-rs.txt"),
    ("flows.2/Softplus",              "_duration_predictor_flows_2_Softplus_output_0-rs.txt"),
    ("flows.2/Pow",                   "_duration_predictor_flows_2_Pow_output_0-rs.txt"),
    ("flows.2/Add_39",                "_duration_predictor_flows_2_Add_39_output_0-rs.txt"),
    ("flows.2/Add_40",                "_duration_predictor_flows_2_Add_40_output_0-rs.txt"),
    ("flows.3/Split_output_0",        "_duration_predictor_flows_3_Split_output_0-rs.txt"),
    ("flows.3/Split_output_1",        "_duration_predictor_flows_3_Split_output_1-rs.txt"),
    ("flows.3/Concat_57",             "_duration_predictor_flows_3_Concat_57_output_0-rs.txt"),
    ("flows.3/Mul_81",                "_duration_predictor_flows_3_Mul_81_output_0-rs.txt"),
    ("flows.0/Sub",                   "_duration_predictor_flows_0_Sub_output_0-rs.txt"),
    ("flows.0/Mul",                   "_duration_predictor_flows_0_Mul_output_0-rs.txt"),
    ("flows.0/Mul_1",                 "_duration_predictor_flows_0_Mul_1_output_0-rs.txt"),
    ("log_duration (rs)",             "log_duration-rs.txt"),
    ("log_duration (fp32)",           "log_duration.txt"),
    ("prior_means",                   "prior_means-rs.txt"),
    ("prior_log_variances",           "prior_log_variances-rs.txt"),
]


def load_tensor(sub, fname):
    return np.loadtxt(os.path.join(DATA, sub, fname))


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def summarize(rows):
    lines = []
    lines.append("=== MMS-TTS RKNN NPU collapse: golden | simulator | runtime ===\n")
    lines.append(f"{'tensor':24s} {'src':10s} {('n'):>6s} {'min':>9s} {'max':>9s} "
                 f"{'uniq(4dp)':>10s}   first 6 values")
    lines.append("-" * 100)
    for name, fname in TENSORS:
        arrs = {}
        for sub in SUBDIRS:
            f = os.path.join(DATA, sub, fname)
            if os.path.isfile(f):
                arrs[sub] = load_tensor(sub, fname)
        for sub in SUBDIRS:
            a = arrs.get(sub)
            if a is None:
                lines.append(f"{name:24s} {sub:10s}  MISSING")
                continue
            n = a.size
            rnd = np.round(a, 4)
            uniq = len(np.unique(rnd))
            first6 = ", ".join(f"{v:.4f}" for v in a[:6])
            lines.append(f"{name:24s} {sub:10s} {n:6d} {a.min():9.4f} {a.max():9.4f} "
                         f"{uniq:10d}   [{first6}]")
        if "golden" in arrs and "simulator" in arrs and "runtime" in arrs:
            g, s, r = arrs["golden"], arrs["simulator"], arrs["runtime"]
            c_gs, c_gr = cos_sim(g, s), cos_sim(g, r)
            e_gs, e_gr = np.linalg.norm(g - s), np.linalg.norm(g - r)
            r_uniq = len(np.unique(np.round(r, 4)))
            collapsed = r_uniq <= 5 and r_uniq < len(np.unique(np.round(g, 4))) / 10
            lines.append(f"{name:24s} {'---':10s} {'':6s} {'':9s} {'':9s} {'':10s}"
                         f"   cos(g,s)={c_gs:7.5f} euc(g,s)={e_gs:8.4f}  "
                         f"cos(g,r)={c_gr:7.5f} euc(g,r)={e_gr:8.4f}"
                         f"  {'<== CONSTANT ON NPU' if collapsed else 'ok'}")
        lines.append("")
    return "\n".join(lines)


def main(out_path):
    os.makedirs(DATA, exist_ok=True)
    report = summarize(TENSORS)
    with open(out_path, "w") as fh:
        fh.write(report + "\n")
    print(report)
    missing = [n for n, fn in TENSORS if not any(
        os.path.isfile(os.path.join(DATA, s, fn)) for s in SUBDIRS)]
    if missing:
        print("NOTE: no data found yet for:", ", ".join(missing))
        print("Run `python scripts/fetch_acc_data.py` on the Orange Pi and copy acc_data/ here.")


if __name__ == "__main__":
    main(os.path.join(BASE, "illustrate_findings_history.txt"))