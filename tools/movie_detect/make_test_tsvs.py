# builds the ds000228 adult TSVs used in the README check from the stillhere extraction (partly_cloudy/extract.py output)
import numpy as np, glob, csv, sys, os
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
P={r['participant_id']:r for r in csv.DictReader(open(sys.argv[3]),delimiter='\t')}
TOM=['DMPFC','MMPFC','VMPFC','PC','LTPJ','RTPJ']
for f in sorted(glob.glob(f'{src}/*.npz')):
    s=os.path.basename(f)[:-4]; d=np.load(f); R=d['R']; X=np.column_stack([R,np.ones(168),np.linspace(-1,1,168)])
    if P[s]['Child_Adult']!='adult': continue
    Y=np.column_stack([d[r]-X@np.linalg.lstsq(X,d[r],rcond=None)[0] for r in TOM])[10:]
    np.savetxt(f'{out}/{s}.tsv',Y,delimiter='\t',header='\t'.join(TOM),comments='',fmt='%.6f')
