"""R2's data check (cloud rulings of Oct 31, 21:30 UTC, R2 item 1), before R2's rule: headers and sidecars only, no brain value read.
Paranoia (Finn et al. 2018), OpenNeuro ds001338 v1.0.0 raw data, and its fMRIPrep derivative (OpenNeuroDerivatives ds001338-fmriprep,
s3://openneuro-derivatives/fmriprep/ds001338-fmriprep/). Reads: the derivative's dataset_description.json (GeneratedBy); for every
listener, story run and hemisphere, the fsaverage5 GIFTI's XML header up to its first <Data> element (a ranged request, so no data array
is downloaded; the root's NumberOfDataArrays is the volume count, the first DataArray's Dim0 the vertex count) and its JSON sidecar; the
raw dataset's participants.tsv, task-story{1,2,3}_bold.json and _events.tsv (the story onset per run). Writes r2/data_check.json.
usage: python r2/data_check.py"""
import csv
import io
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

RAW = "https://s3.amazonaws.com/openneuro.org/ds001338/"
DER = "https://openneuro-derivatives.s3.amazonaws.com/"; DPRE = "fmriprep/ds001338-fmriprep/"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
TASKS = ("story1", "story2", "story3")


def get(url, rng=None) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    return urllib.request.urlopen(req, timeout=60).read()


def listing(base, prefix):
    keys, token = {}, None
    while True:
        q = f"?list-type=2&prefix={urllib.parse.quote(prefix)}" + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        r = ET.fromstring(get(base + q))
        keys.update({c.find("s:Key", NS).text: int(c.find("s:Size", NS).text) for c in r.findall("s:Contents", NS)})
        t = r.find("s:NextContinuationToken", NS)
        if t is None:
            return keys
        token = t.text


def gifti_header(url) -> dict:
    """Root and first-DataArray attributes from the bytes before the first <Data> element; refuses if that is not within 64 KB."""
    b = get(url, (0, 65535)); i = b.find(b"<Data>")
    if i < 0:
        raise SystemExit(f"refused: no <Data> within 64 KB of {url}")
    h = b[:i].decode()
    root = dict(re.findall(r'(\w+)="([^"]*)"', re.search(r"<GIFTI ([^>]*)>", h).group(1)))
    da = dict(re.findall(r'(\w+)="([^"]*)"', re.search(r"<DataArray\s([^>]*)>", h).group(1)))
    md = dict(re.findall(r"<Name>(?:<!\[CDATA\[)?([^<\]]*)(?:\]\]>)?</Name><Value>(?:<!\[CDATA\[)?([^<\]]*)(?:\]\]>)?</Value>", h))
    return {"volumes": int(root["NumberOfDataArrays"]), "vertices": int(da["Dim0"]), "dimensionality": int(da["Dimensionality"]),
            "data_type": da.get("DataType"), "intent": da.get("Intent"), "structure": md.get("AnatomicalStructurePrimary"),
            "header_bytes_read": i}


def main():
    desc = json.loads(get(DER + DPRE + "dataset_description.json"))
    raw_keys, der_keys = listing("https://s3.amazonaws.com/openneuro.org", "ds001338/"), listing(DER.rstrip("/"), DPRE)
    subs = sorted(r["participant_id"] for r in csv.DictReader(io.StringIO(get(RAW + "participants.tsv").decode()), delimiter="\t"))
    task_json = {t: json.loads(get(RAW + f"task-{t}_bold.json")) for t in TASKS}
    events = {t: list(csv.DictReader(io.StringIO(get(RAW + f"task-{t}_events.tsv").decode()), delimiter="\t")) for t in TASKS}
    raw_bold = {k: v for k, v in raw_keys.items() if k.endswith("_bold.nii.gz")}
    runs, problems = {}, []
    for s in subs:
        for t in TASKS:
            r = {}
            for hemi in ("L", "R"):
                stem = f"{DPRE}{s}/func/{s}_task-{t}_hemi-{hemi}_space-fsaverage5_bold"
                if f"{stem}.func.gii" not in der_keys:
                    problems.append(f"{s} {t} hemi-{hemi}: no fsaverage5 GIFTI"); continue
                r[hemi] = {**gifti_header(DER + stem + ".func.gii"), "bytes": der_keys[f"{stem}.func.gii"],
                           "sidecar": json.loads(get(DER + stem + ".json"))}
            if set(r) == {"L", "R"}:
                if r["L"]["volumes"] != r["R"]["volumes"]:
                    problems.append(f"{s} {t}: hemispheres differ in volumes")
                for hemi, want in (("L", "CortexLeft"), ("R", "CortexRight")):
                    if r[hemi]["vertices"] != 10242 or r[hemi]["structure"] != want:
                        problems.append(f"{s} {t} hemi-{hemi}: {r[hemi]['vertices']} vertices, {r[hemi]['structure']}")
                    if r[hemi]["sidecar"].get("RepetitionTime") != 1:
                        problems.append(f"{s} {t} hemi-{hemi}: RepetitionTime {r[hemi]['sidecar'].get('RepetitionTime')}")
            r["raw_bold_present"] = f"ds001338/{s}/func/{s}_task-{t}_bold.nii.gz" in raw_bold
            runs[f"{s} {t}"] = r
    vols = {t: sorted({r["L"]["volumes"] for k, r in runs.items() if k.endswith(t) and "L" in r}) for t in TASKS}
    out = {"label": "R2 data check: headers and sidecars only, no brain value read",
           "derivative": {"url": f"s3://openneuro-derivatives/{DPRE}", "generated_by": desc.get("GeneratedBy"),
                          "source": desc.get("SourceDatasets"), "license": desc.get("License")},
           "listeners": len(subs), "participants": subs, "tasks": list(TASKS),
           "fsaverage5": {"files": sum(1 for k in der_keys if "space-fsaverage5" in k and k.endswith(".func.gii")),
                          "bytes": sum(v for k, v in der_keys.items() if "space-fsaverage5" in k and k.endswith(".func.gii"))},
           "volumes_per_task": vols, "raw_task_sidecars": task_json,
           "story_onset_per_task": {t: [{k: e[k] for k in e} for e in ev] for t, ev in events.items()},
           "vertex_order": "TRIBE's: left hemisphere (CortexLeft, 10,242) then right (CortexRight, 10,242); files are per hemisphere",
           "slice_timing_corrected": sorted({str(r[h]["sidecar"].get("SliceTimingCorrected")) for r in runs.values() for h in ("L", "R") if h in r}),
           "runs": runs, "problems": problems}
    Path(__file__).with_name("data_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("listeners", "volumes_per_task", "slice_timing_corrected", "problems")}),
          json.dumps(out["derivative"]["generated_by"]), json.dumps(out["story_onset_per_task"])[:600])


if __name__ == "__main__":
    main()
