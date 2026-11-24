"""Build movie_detect TSVs from MNI-space BOLD (e.g. fMRIPrep *_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz).

usage: extract.py --bold sub-01.nii.gz [--confounds conf.tsv] --spheres spheres.tsv --drop 10 --out tsv/sub-01.tsv
spheres.tsv: columns name, x, y, z (MNI mm); radius set by --radius (default 9 mm).
"""
import argparse, numpy as np
from nilearn.maskers import NiftiSpheresMasker
ap = argparse.ArgumentParser(); ap.add_argument('--bold', required=True); ap.add_argument('--confounds')
ap.add_argument('--confound-cols', default='trans_x,trans_y,trans_z,rot_x,rot_y,rot_z')
ap.add_argument('--spheres', required=True); ap.add_argument('--radius', type=float, default=9)
ap.add_argument('--drop', type=int, default=0); ap.add_argument('--keep', type=int, default=0); ap.add_argument('--out', required=True)
a = ap.parse_args()
rows = [l.rstrip('\n').split('\t') for l in open(a.spheres)][1:]
names = [r[0] for r in rows]; coords = [tuple(map(float, r[1:4])) for r in rows]
conf = None
if a.confounds:
    import csv
    R = list(csv.DictReader(open(a.confounds), delimiter='\t')); cols = a.confound_cols.split(',')
    conf = np.array([[float(r[c]) if r[c] not in ('n/a', '') else 0. for c in cols] for r in R])
m = NiftiSpheresMasker(coords, radius=a.radius, detrend=True, standardize='zscore_sample')
X = m.fit_transform(a.bold, confounds=conf)[a.drop:]
if a.keep: X = X[:a.keep]
np.savetxt(a.out, X, delimiter='\t', header='\t'.join(names), comments='', fmt='%.6f')
print(a.out, X.shape)
