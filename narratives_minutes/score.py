# narratives_minutes/PREREG.md (sha256 05ee8a452926ab31...). Scoring, run once on the extracted series.
import glob, json, zlib, sys
import numpy as np
sys.path.insert(0, '/home/claude/stillhere/partly_cloudy')
from core import zz, surr, pval, wilson
W = '/home/claude/young'
STORIES = ['black', 'forgot', 'merlin', 'sherlock', 'tunnel', '21styear', 'slumlord']
KS = [1, 2, 3, 5, 7, 10, 13]; TR = 1.5; NREF = 20
D = {}
for st in STORIES:
    for f in sorted(glob.glob(f'{W}/ts/{st}_sub-*.npz')):
        d = np.load(f); D.setdefault(st, {})[f.split('_')[-1][:-4]] = {n: d[n] for n in ('LANG', 'AUD')}
print({s: len(v) for s, v in D.items()}, 'runs', sum(len(v) for v in D.values()))
rng = np.random.default_rng(20261126)
def draw(st, excl):
    pool = [s for s in sorted(D[st]) if s != excl]
    return pool if len(pool) <= NREF else sorted(rng.choice(pool, NREF, replace=False).tolist())
REF = {}   # (listener story, listener, ref story) -> ref ids ; drawn once, fixed order
for st in STORIES:
    for s in sorted(D[st]):
        for rs in STORIES:
            REF[(st, s, rs)] = draw(rs, s)
def refser(rs, ids, net, T): return zz(np.mean([D[rs][i][net][:T] for i in ids], 0))
rows = []
for st in STORIES:
    for s in sorted(D[st]):
        for net in ('LANG', 'AUD'):
            x = D[st][s][net]
            for k in KS:
                T = int(round(k * 60 / TR))
                if len(x) < T: continue
                seed = zlib.crc32(f'{st}|{s}|{net}|{k}'.encode())
                S = surr(x[:T], 1000, seed)
                r, p = pval(x[:T], refser(st, REF[(st, s, st)], net, T), S)
                wrong = []
                for rs in STORIES:
                    if rs == st: continue
                    if min(len(D[rs][i][net]) for i in REF[(st, s, rs)]) < T: continue
                    wrong.append(pval(x[:T], refser(rs, REF[(st, s, rs)], net, T), S)[1])
                rows.append(dict(story=st, sub=s, net=net, k=k, r=float(r), p=float(p), wrong=[float(w) for w in wrong]))
json.dump(rows, open(f'{W}/rows.json', 'w'))
out = {}
for net in ('LANG', 'AUD'):
    for k in KS:
        R = [x for x in rows if x['net'] == net and x['k'] == k]
        fa = np.concatenate([x['wrong'] for x in R]); c = float(np.quantile(fa, 0.05))
        det = np.array([x['p'] <= c for x in R]); n = len(R); kd = int(det.sum())
        subs = sorted(set(x['sub'] for x in R)); by = {u: [i for i, x in enumerate(R) if x['sub'] == u] for u in subs}
        brng = np.random.default_rng(1000 + k)
        bs = []
        for _ in range(2000):
            idx = np.concatenate([by[u] for u in brng.choice(subs, len(subs))]); bs.append(det[idx].mean())
        lo, hi = wilson(kd, n); blo, bhi = np.quantile(bs, [.025, .975])
        unc = np.mean([x['p'] < .05 for x in R]); unc_fa = float(np.mean(fa < .05))
        per = {st: f"{np.mean([x['p'] <= c for x in R if x['story'] == st]):.0%} of {sum(x['story'] == st for x in R)}" for st in STORIES if any(x['story'] == st for x in R)}
        out[f'{net} {k}min'] = dict(n=n, people=len(subs), cutoff=c, detect=kd / n, miss=1 - kd / n, wilson_detect=[lo, hi], person_boot_detect=[float(blo), float(bhi)], uncal_detect=float(unc), uncal_wrong_story=unc_fa, wrong_story_n=int(len(fa)), median_r=float(np.median([x['r'] for x in R])), per_story=per)
        print(f"{net:4s} {k:2d} min  n={n:3d} ({len(subs)} people)  detect {kd/n:5.1%}  miss {1-kd/n:5.1%}  Wilson [{1-hi:.1%}, {1-lo:.1%}] miss  person-boot miss [{1-bhi:.1%}, {1-blo:.1%}]  cutoff p<={c:.4f}  uncal p<.05: {unc:.0%} (wrong story {unc_fa:.1%})  median r {np.median([x['r'] for x in R]):.2f}", flush=True)
        print('      per story:', per, flush=True)
json.dump(out, open(f'{W}/summary.json', 'w'), indent=1)
