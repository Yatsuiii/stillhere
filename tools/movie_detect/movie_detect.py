"""movie_detect: per-person detection of movie-driven network tracking, with calibrated false alarms.

Input: one TSV per run (rows = TRs, columns = regions), named <subject>[_view-<k>].tsv, already cleaned
(nuisance regressed) and cut to the movie. All runs must have the same length.
Optional: extract.py builds these TSVs from MNI-space BOLD (e.g. fMRIPrep) + sphere coordinates.

For each subject: network series = mean of z-scored regions (averaged over the subject's views).
Reference = leave-one-subject-out mean of the reference group (default: all subjects).
Null = phase-randomised surrogates of the subject's series. The cutoff is calibrated so that tests against
mismatched references (time-reversed, circularly shifted) are detected at --fa (default 5%).
Outputs: per-subject table, group rate with Wilson CI, minutes-needed curve, receipt.json with input hashes.
"""
import argparse, glob, hashlib, json, os, re, sys, platform
import numpy as np

def zz(x):
    x = np.asarray(x, float); return (x - x.mean()) / x.std()

def surrogates(x, n, rng):
    X = np.fft.rfft(x - x.mean()); T = len(x)
    ph = rng.uniform(0, 2 * np.pi, (n, len(X))); ph[:, 0] = 0
    if T % 2 == 0: ph[:, -1] = 0
    S = np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=T)
    return (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True)

def pval(x, r, S):
    x, r = zz(x), zz(r); o = float((x * r).mean()); nl = (S * r).mean(1)
    return o, float((1 + (nl >= o).sum()) / (len(nl) + 1))

def mismatched(r, k):
    T = len(r); return [np.roll(r[::-1], int(o)) for o in np.linspace(T * .05, T * .95, k)]

def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return (c - h, c + h)

def sha256(f):
    h = hashlib.sha256()
    with open(f, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def load(indir, regions):
    subj = {}
    files = sorted(glob.glob(os.path.join(indir, '*.tsv')))
    if not files: sys.exit(f'no .tsv files in {indir}')
    for f in files:
        m = re.match(r'(.+?)(?:_view-(\d+))?\.tsv$', os.path.basename(f))
        hdr = open(f).readline().rstrip('\n').split('\t')
        a = np.loadtxt(f, skiprows=1, ndmin=2)
        cols = [hdr.index(r) for r in regions] if regions else list(range(a.shape[1]))
        net = zz(np.mean([zz(a[:, c]) for c in cols], 0))
        subj.setdefault(m.group(1), []).append((int(m.group(2) or 1), net))
    T = {len(v[1]) for vs in subj.values() for v in vs}
    if len(T) != 1: sys.exit(f'runs differ in length: {sorted(T)}')
    return {s: [n for _, n in sorted(v)] for s, v in subj.items()}, files

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('indir'); ap.add_argument('--out', default='movie_detect_out')
    ap.add_argument('--regions', default='', help='comma list of TSV columns forming the network (default: all)')
    ap.add_argument('--reference-subjects', default='', help='file with one subject per line (default: all)')
    ap.add_argument('--tr', type=float, required=True); ap.add_argument('--views', type=int, default=0, help='use first k views per subject (0 = all)')
    ap.add_argument('--fa', type=float, default=0.05); ap.add_argument('--surrogates', type=int, default=2000)
    ap.add_argument('--mismatched', type=int, default=20); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--minutes', default='1,2,3,4', help='window lengths (min) for the minutes-needed curve; full length always added')
    a = ap.parse_args()
    regions = [r for r in a.regions.split(',') if r]
    data, files = load(a.indir, regions)
    refset = [l.strip() for l in open(a.reference_subjects)] if a.reference_subjects else list(data)
    T = len(next(iter(data.values()))[0])
    lens = sorted({int(round(m * 60 / a.tr)) for m in map(float, a.minutes.split(',')) if m * 60 / a.tr < T} | {T})
    rng = np.random.default_rng(a.seed)
    def series(s):
        v = data[s][:a.views] if a.views else data[s]; return zz(np.mean(v, 0))
    full = {s: series(s) for s in data}
    per = {s: {} for s in data}; fa = {L: [] for L in lens}
    for s in data:
        others = [o for o in refset if o != s and o in data]
        ref = zz(np.mean([full[o] for o in others], 0))
        for L in lens:
            for st in (range(0, T - L + 1, max(1, L // 3)) if L < T else [0]):
                x, r = full[s][st:st + L], ref[st:st + L]; S = surrogates(zz(x), a.surrogates, rng)
                o, p = pval(x, r, S); per[s].setdefault(L, []).append((o, p))
                fa[L] += [pval(x, m, S)[1] for m in mismatched(r, a.mismatched)]
    cut = {L: float(np.quantile(fa[L], a.fa)) for L in lens}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'per_subject.tsv'), 'w') as f:
        f.write('subject\tn_views\tr_full\tp_full\tdetected_full\t' + '\t'.join(f'share_detected_{L * a.tr / 60:.1f}min' for L in lens) + '\n')
        for s in sorted(data):
            o, p = per[s][T][0]
            f.write(f"{s}\t{len(data[s][:a.views] if a.views else data[s])}\t{o:.3f}\t{p:.4f}\t{int(p < cut[T])}\t" +
                    '\t'.join(f"{np.mean([q < cut[L] for _, q in per[s][L]]):.3f}" for L in lens) + '\n')
    k = sum(per[s][T][0][1] < cut[T] for s in data); n = len(data); lo, hi = wilson(k, n)
    curve = {f'{L * a.tr / 60:.2f}': float(np.mean([np.mean([q < cut[L] for _, q in per[s][L]]) for s in data])) for L in lens}
    summary = {'detected': k, 'n': n, 'rate': k / n, 'wilson95': [lo, hi], 'cutoffs': {str(L): c for L, c in cut.items()},
               'false_alarm_at_cutoff': {str(L): float(np.mean(np.array(fa[L]) < cut[L])) for L in lens}, 'minutes_curve': curve}
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    receipt = {'tool': 'movie_detect', 'args': vars(a), 'python': platform.python_version(), 'numpy': np.__version__,
               'inputs': {os.path.basename(f): sha256(f) for f in files},
               'outputs': {f: sha256(os.path.join(a.out, f)) for f in ('per_subject.tsv', 'summary.json')}}
    json.dump(receipt, open(os.path.join(a.out, 'receipt.json'), 'w'), indent=1)
    print(f"detected {k}/{n} = {k / n:.0%} (95% CI {lo:.0%}-{hi:.0%}); minutes curve {curve}")

if __name__ == '__main__': main()
