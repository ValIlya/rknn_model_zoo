import os
import sys
import argparse
import numpy as np

SUBDIRS = ('golden', 'simulator', 'runtime')

TENSORS = [
    ('log_duration (rs)',       'log_duration-rs.txt'),
    ('log_duration (fp32)',     'log_duration.txt'),
    ('flows.2/Softplus',        '_duration_predictor_flows_2_Softplus_output_0-rs.txt'),
    ('flows.2/Pow',             '_duration_predictor_flows_2_Pow_output_0-rs.txt'),
    ('flows.2/Add_37',          '_duration_predictor_flows_2_Add_37_output_0-rs.txt'),
    ('flows.2/Add_38',          '_duration_predictor_flows_2_Add_38_output_0-rs.txt'),
    ('flows.2/Add_39',          '_duration_predictor_flows_2_Add_39_output_0-rs.txt'),
    ('flows.2/Add_40',          '_duration_predictor_flows_2_Add_40_output_0-rs.txt'),
    ('flows.2/Concat_57',       '_duration_predictor_flows_2_Concat_57_output_0-rs.txt'),
    ('flows.2/Split_output_0',  '_duration_predictor_flows_2_Split_output_0-rs.txt'),
    ('flows.2/Split_output_1',  '_duration_predictor_flows_2_Split_output_1-rs.txt'),
    ('flows.2/Mul_81',          '_duration_predictor_flows_2_Mul_81_output_0-rs.txt'),
    ('flows.3/Concat_57',       '_duration_predictor_flows_3_Concat_57_output_0-rs.txt'),
    ('flows.3/Mul_81',          '_duration_predictor_flows_3_Mul_81_output_0-rs.txt'),
]


def load(dir, fname):
    p = os.path.join(dir, fname)
    if os.path.isfile(p):
        return np.loadtxt(p)
    return None


def summarize(outdir):
    rows = []
    for name, fname in TENSORS:
        arrs = {s: load(os.path.join(outdir, s), fname) for s in SUBDIRS}
        if all(a is None for a in arrs.values()):
            rows.append('%-22s ALL MISSING' % name)
            continue
        sizes = {s: (a.size if a is not None else None) for s, a in arrs.items()}
        line = '%-22s' % name
        for s in SUBDIRS:
            a = arrs[s]
            if a is None:
                line += '  %-10s MISSING' % s
            else:
                u = len(np.unique(np.round(a, 4)))
                line += '  %-10s n=%-5d [%9.5f,%9.5f] uniq=%-6d' % (s, a.size, a.min(), a.max(), u)
        rows.append(line)
        g, r = arrs['golden'], arrs['runtime']
        if g is not None and r is not None:
            if g.size != r.size:
                rows.append('    !!! SHAPE MISMATCH golden n=%d vs runtime n=%d' % (g.size, r.size))
            else:
                maxd = float(np.max(np.abs(g - r)))
                runiq = len(np.unique(np.round(r, 4)))
                ok = runiq > 5 and maxd < 0.1
                rows.append('    max|rt-gold| = %.6f   runtime_uniq = %d   %s'
                            % (maxd, runiq, 'PASS(non-const,<0.1)' if ok else 'FAIL'))
    return '\n'.join(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('outdir')
    args = p.parse_args()
    print(summarize(args.outdir))


if __name__ == '__main__':
    main()