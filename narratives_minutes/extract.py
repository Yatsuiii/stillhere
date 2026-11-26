# narratives_minutes/PREREG.md (sha256 05ee8a452926ab31...): extraction only (no scoring).
import json, os, sys, tempfile, urllib.request, io, csv, concurrent.futures as cf
import numpy as np, nibabel as nib
sys.path.insert(0, '/home/claude/stillhere')
from stillhere import heldout as H
W = '/home/claude/young'; B = 'https://fcp-indi.s3.amazonaws.com/data/Projects/narratives'
E = json.load(open(f'{W}/event_meta.json')); T = json.load(open(f'{W}/task_meta.json')); X = json.load(open(f'{W}/scan_exclude.json'))
SETS = H.parcel_sets(f'{W}/atlas/lh.Schaefer2018_400Parcels_17Networks_order.annot', f'{W}/atlas/rh.Schaefer2018_400Parcels_17Networks_order.annot')
STORIES = [('black', 'black'), ('forgot', 'forgot'), ('merlin', 'merlin'), ('sherlock', 'sherlock'), ('tunnel', 'tunnel'), ('21styear', '21styear'), ('slumlordreach', 'slumlord')]
TRIM = 6
def get(url, dst):
    last = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(url, timeout=600) as r: open(dst, 'wb').write(r.read()); return
        except Exception as e: last = e
    raise RuntimeError(f'{url}: {last}')
def excluded(task, sub, fname):
    ex = X.get(task, {}).get(sub)
    if ex is None: return False
    return any(e in fname for e in ex) if isinstance(ex, list) and ex else True
def one(task, sub_task, sub):
    out = f'{W}/ts/{sub_task}_{sub}.npz'
    if os.path.exists(out): return 'cached'
    meta = T[task][sub]; f6 = meta['bold']['fsaverage6']['preproc']
    paths = sorted(p.replace('/jukebox/hasson/snastase/narratives', B).replace('fsaverage6', 'fsaverage5') for p in f6)
    Lp = [p for p in paths if 'hemi-L' in p]; Rp = [p for p in paths if 'hemi-R' in p]
    if len(Lp) != 1 or len(Rp) != 1: return f'skip runs {len(Lp)}'
    if excluded(task, sub, os.path.basename(Lp[0]).split('_space')[0]): return 'excluded'
    conf = meta['confounds'][0].replace('/jukebox/hasson/snastase/narratives', B)
    with tempfile.TemporaryDirectory() as td:
        get(Lp[0], f'{td}/L.gii'); get(Rp[0], f'{td}/R.gii'); get(conf, f'{td}/c.tsv')
        D = np.vstack([np.column_stack([d.data for d in nib.load(f'{td}/{h}.gii').darrays]) for h in 'LR']).astype(np.float64)
        rows = list(csv.DictReader(open(f'{td}/c.tsv'), delimiter='\t'))
    on = E[task][sub_task]['onset']; du = E[task][sub_task]['duration']
    if D.shape[1] < on + du: return f'short {D.shape[1]} < {on+du}'
    sl = slice(on + TRIM, on + du)
    names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    names += [n + '_derivative1' for n in names[:6]] + [f'a_comp_cor_0{i}' for i in range(5)] + [k for k in rows[0] if k.startswith('cosine')]
    C = np.array([[float(r[n]) if r[n] not in ('n/a', '') else 0.0 for n in names] for r in rows])[sl]
    C = np.column_stack([np.ones(len(C)), (C - C.mean(0)) / (C.std(0) + 1e-12)])
    Y = D[:, sl].T
    Y = Y - C @ np.linalg.lstsq(C, Y, rcond=None)[0]
    sd = Y.std(0); ok = sd > 1e-8
    Y[:, ok] = (Y[:, ok] - Y[:, ok].mean(0)) / sd[ok]; Y[:, ~ok] = np.nan
    z = lambda x: (x - x.mean()) / x.std()
    res = {net: z(np.nanmean(Y[:, np.concatenate([np.asarray(p) for p in SETS[net]])], 1)) for net in ('LANG', 'AUD')}
    fd = np.array([float(r['framewise_displacement']) if r['framewise_displacement'] != 'n/a' else 0 for r in rows])[sl]
    np.savez(out, LANG=res['LANG'], AUD=res['AUD'], fd=fd, nvert_bad=int((~ok).sum()))
    return f'ok {len(res["LANG"])}'
jobs = [(t, st, s) for t, st in STORIES for s in sorted(T[t])]
print(len(jobs), 'jobs', flush=True)
with cf.ThreadPoolExecutor(6) as ex:
    fut = {ex.submit(one, *j): j for j in jobs}
    for f in cf.as_completed(fut):
        j = fut[f]
        try: print(j[1], j[2], f.result(), flush=True)
        except Exception as e: print(j[1], j[2], 'ERR', e, flush=True)
