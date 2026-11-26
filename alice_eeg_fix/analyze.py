"""alice_eeg_fix/PROTOCOL.md (sha256 72d008404350d827...). usage: python analyze.py CLONE DATA PER_SEGMENT_CSV OUT"""
import json, os, re, sys
import mne, numpy as np, pandas as pd
clone, data, segcsv, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True)
SHIFTED = ['sub-002', 'sub-026', 'sub-027', 'sub-030', 'sub-032', 'sub-034', 'sub-035', 'sub-036']
NOAUD = ['sub-005', 'sub-038']; FAINT = ['sub-009', 'sub-022', 'sub-023']
PRED = ['NGRAM', 'LogFreq', 'LogFreq_Prev', 'LogFreq_Next', 'SndPower', 'Length', 'Position', 'Sentence']
story = pd.read_csv(f'{data}/sourcedata/AliceChapterOne-EEG.csv').set_index('Order')
seg = pd.read_csv(segcsv)
part = pd.read_csv(f'{clone}/participants.tsv', sep='\t').set_index('participant_id')
subs = sorted(d for d in os.listdir(clone) if re.fullmatch(r'sub-\d+', d) and os.path.exists(f'{clone}/{d}/eeg/{d}_task-alicelistening_events.tsv'))
TM = None; res = {}
def epochs_for(raw, onsets):  # manual epoching, one epoch per onset (no merging or silent drops)
    sf = raw.info['sfreq']; X = raw.get_data() * 1e6; i0 = int(round(-0.3 * sf)); i1 = int(round(1.0 * sf))
    times = np.arange(i0, i1 + 1) / sf; b = (times >= -0.2) & (times <= 0)
    S = np.round(onsets * sf).astype(int)
    Y = np.stack([X[:, s + i0: s + i1 + 1] for s in S]); Y -= Y[:, :, b].mean(2, keepdims=True)
    return Y, times
def betas(X, Y):  # X trials x p (with intercept), Y trials x ch x t
    B = np.linalg.lstsq(X, Y.reshape(len(Y), -1), rcond=None)[0]; return B.reshape((X.shape[1],) + Y.shape[1:])
for sub in subs:
    vhdr = f'{data}/{sub}/eeg/{sub}_task-alicelistening_eeg.vhdr'
    if not os.path.exists(vhdr) or not os.path.exists(f'{data}/sourcedata/proc/S%02d.mat' % int(sub.split('-')[1])): print(sub, 'no eeg or no proc'); continue
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose='error')
    from scipy.io import loadmat
    pm = loadmat(f'{data}/sourcedata/proc/S%02d.mat' % int(sub.split('-')[1]), squeeze_me=True, simplify_cells=True)['proc']
    # DEVIATION 2: his ICA eye-component removal, his channel list and his final retained trials
    ica = pm['ica']; lab = [str(c) for c in ica['topolabel']]
    eegch = [c for c in raw.ch_names if c.isdigit()]; raw.pick(eegch)
    if str(pm['implicitref']) not in raw.ch_names: raw = mne.add_reference_channels(raw, str(pm['implicitref']), copy=False)
    raw.set_eeg_reference([str(c) for c in pm['refchannels']], verbose='error')   # his linked-mastoid reference
    raw.pick(lab).filter(0.1, 30, verbose='error')
    Xr = raw.get_data(); U = np.asarray(ica['unmixing'], float); A = np.linalg.pinv(U); rej = np.atleast_1d(ica['rejcomp']).astype(int) - 1
    Sx = U @ Xr; Xr = Xr - A[:, rej] @ Sx[rej]
    raw._data[:] = Xr
    fin = pm['rejections']['final']
    keepch = [c for c in lab if c in set(str(x) for x in np.atleast_1d(fin['chanpicks']))]
    raw.pick(keepch)
    picks_trl = np.atleast_1d(fin['trialpicks']).astype(int) - 1
    ev = pd.read_csv(f'{clone}/{sub}/eeg/{sub}_task-alicelistening_events.tsv', sep='\t')
    trl = np.asarray(pm['trl'], float); order = trl[picks_trl, 3 + list(pm['varnames']).index('Order')].astype(int)   # his final retained trials
    ev = ev[ev['word_index'].isin(story.index) & ev['word_index'].isin(order)].copy()
    ev['csv_onset'] = story.loc[ev['word_index'], 'onset'].values
    rel = ev['onset'].values.astype(float); cor = rel.copy()
    if sub in SHIFTED:
        ss = seg[seg['subject'] == sub].set_index('segment'); s = ss['measured_fine_s']
        cor = np.array([s[int(k)] + o for k, o in zip(ev['segment'], ev['csv_onset'])])
        und = ev['segment'].map(ss['verdict']).eq('undetermined').values   # DEVIATION 1: segment not placeable -> drop from both
        cor[und] = -1.0
    dur = raw.n_times / raw.info['sfreq']
    keep = (rel > 0.3) & (rel < dur - 1.0) & (cor > 0.3) & (cor < dur - 1.0)
    ev = ev[keep]; rel = rel[keep]; cor = cor[keep]
    Xd = story.loc[ev['word_index'], PRED].astype(float).values; Xd = (Xd - Xd.mean(0)) / Xd.std(0)
    Xc = np.column_stack([Xd, ((lambda c: (c - c.mean()) / c.std())(story.loc[ev['word_index'], 'CFG'].astype(float).values))])
    r = {}
    for lab, on in [('released', rel), ('corrected', cor)]:
        if lab == 'corrected' and sub not in SHIFTED: r[lab] = r['released']; continue
        Y, t = epochs_for(raw, on); TM = t
        ok = np.abs(Y).max((1, 2)) < 100   # DEVIATION 3: same amplitude rejection for both versions
        Y = Y[ok]; Xd_, Xc_ = Xd[ok], Xc[ok]
        ALL = [str(i) for i in range(1, 62)]
        def full(a):  # channels -> fixed 1..61 rows, NaN where the channel was dropped
            o = np.full((61,) + a.shape[1:], np.nan); idx = [ALL.index(c) for c in raw.ch_names]; o[idx] = a; return o
        r[lab] = dict(erp=full(Y.mean(0)), ngram=full(betas(np.column_stack([np.ones(len(Y)), Xd_]), Y)[1]), cfg=full(betas(np.column_stack([np.ones(len(Y)), Xc_]), Y)[-1]), n=len(Y))
    res[sub] = r; print(sub, 'words', len(ev), 'shifted' if sub in SHIFTED else '', flush=True)
t = TM; cons = [s for s in res if s not in SHIFTED + NOAUD + FAINT]
w1 = (t >= -0.1) & (t <= 0.6); w2 = (t >= 0) & (t <= 1.0)
def sim(sub, key, lab, win):
    others = np.nanmean([res[o]['released'][key][:, win] for o in cons if o != sub], 0)
    a = res[sub][lab][key][:, win].ravel(); b = others.ravel(); ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])
out_j = {'consistent_n': len(cons)}
for key, win, name in [('erp', w1, 'M1 word ERP'), ('ngram', w2, 'M2 NGRAM beta map')]:
    cs = [sim(s, key, 'released', win) for s in cons]
    print(f'\n{name}: consistent subjects (leave-self-out) median {np.median(cs):.2f}, 10-90% [{np.quantile(cs, .1):.2f}, {np.quantile(cs, .9):.2f}]')
    out_j[name] = {'consistent': cs, 'shifted': {}}
    for s in SHIFTED:
        if s in res:
            a, b = sim(s, key, 'released', win), sim(s, key, 'corrected', win); out_j[name]['shifted'][s] = [a, b]
            print(f'  {s} ({part.loc[s, "analyzed"]}): released {a:+.2f}  corrected {b:+.2f}')
ana = [s for s in res if part.loc[s, 'analyzed'] == 'yes']; ana_cons = [s for s in ana if s not in SHIFTED]
print(f'\nM3: analyzed n={len(ana)}, analyzed consistent n={len(ana_cons)}')
bins = [(b, b + 0.05) for b in np.arange(0.2, 0.7, 0.05)]
for key in ('ngram', 'cfg'):
    best = None
    for (a, b) in bins:
        m = (t >= a) & (t < b); M = np.nanmean([res[s]['released'][key][:, m].mean(1) for s in ana_cons], 0)
        ch = int(np.nanargmax(np.abs(M)))
        if best is None or abs(M[ch]) > abs(best[2]): best = (ch, (a, b), M[ch])
    ch, (a, b), _ = best; m = (t >= a) & (t < b)
    from scipy import stats
    for lab in ('released', 'corrected'):
        v = np.array([res[s][lab][key][ch, m].mean() for s in ana]); v = v[np.isfinite(v)]; tt, p = stats.ttest_1samp(v, 0)
        out_j[f'M3 {key} {lab}'] = dict(channel=ch, window=[a, b], t=float(tt), p=float(p), mean=float(v.mean()))
        print(f'  {key.upper()} ch {ch+1} {a*1000:.0f}-{b*1000:.0f} ms, {lab}: mean beta {v.mean():+.3f} uV, t({len(v)-1}) = {tt:+.2f}, p = {p:.4f}')
json.dump(out_j, open(f'{out}/results.json', 'w'), indent=1)
