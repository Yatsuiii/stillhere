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
def epochs_for(raw, onsets):
    sf = raw.info['sfreq']; ev = np.column_stack([np.round(onsets * sf).astype(int), np.zeros(len(onsets), int), np.ones(len(onsets), int)])
    ep = mne.Epochs(raw, ev, tmin=-0.3, tmax=1.0, baseline=(-0.2, 0), preload=True, reject=None, flat=None, event_repeated='merge', verbose='error')
    return ep.get_data(copy=False) * 1e6, ep.times
def betas(X, Y):  # X trials x p (with intercept), Y trials x ch x t
    B = np.linalg.lstsq(X, Y.reshape(len(Y), -1), rcond=None)[0]; return B.reshape((X.shape[1],) + Y.shape[1:])
for sub in subs:
    vhdr = f'{data}/{sub}/eeg/{sub}_task-alicelistening_eeg.vhdr'
    if not os.path.exists(vhdr): print(sub, 'no eeg'); continue
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose='error')
    eeg = [c for c in raw.ch_names if c not in ('AUD', 'Aux5', 'VEOG', 'HEOG', 'Aux1', 'Aux2', 'Aux3', 'Aux4')][:61]
    raw.pick(eeg).set_eeg_reference('average', verbose='error').filter(0.1, 30, verbose='error')
    ev = pd.read_csv(f'{clone}/{sub}/eeg/{sub}_task-alicelistening_events.tsv', sep='\t')
    ev = ev[ev['word_index'].isin(story.index)].copy()
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
        r[lab] = dict(erp=Y.mean(0), ngram=betas(np.column_stack([np.ones(len(Y)), Xd]), Y)[1], cfg=betas(np.column_stack([np.ones(len(Y)), Xc]), Y)[-1], n=len(Y))
    res[sub] = r; print(sub, 'words', len(ev), 'shifted' if sub in SHIFTED else '', flush=True)
t = TM; cons = [s for s in res if s not in SHIFTED + NOAUD + FAINT]
w1 = (t >= -0.1) & (t <= 0.6); w2 = (t >= 0) & (t <= 1.0)
def sim(sub, key, lab, win):
    others = np.mean([res[o]['released'][key][:, win] for o in cons if o != sub], 0)
    return float(np.corrcoef(res[sub][lab][key][:, win].ravel(), others.ravel())[0, 1])
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
        m = (t >= a) & (t < b); M = np.mean([res[s]['released'][key][:, m].mean(1) for s in ana_cons], 0)
        ch = int(np.argmax(np.abs(M)))
        if best is None or abs(M[ch]) > abs(best[2]): best = (ch, (a, b), M[ch])
    ch, (a, b), _ = best; m = (t >= a) & (t < b)
    from scipy import stats
    for lab in ('released', 'corrected'):
        v = np.array([res[s][lab][key][ch, m].mean() for s in ana]); tt, p = stats.ttest_1samp(v, 0)
        out_j[f'M3 {key} {lab}'] = dict(channel=ch, window=[a, b], t=float(tt), p=float(p), mean=float(v.mean()))
        print(f'  {key.upper()} ch {ch} {a*1000:.0f}-{b*1000:.0f} ms, {lab}: mean beta {v.mean():+.3f} uV, t({len(v)-1}) = {tt:+.2f}, p = {p:.4f}')
json.dump(out_j, open(f'{out}/results.json', 'w'), indent=1)
