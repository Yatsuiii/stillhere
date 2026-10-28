# Changes to frozen code

## picker-pred input paths (Oct 28, 16:10 UTC)

picker-pred's input paths were changed after the freeze because Kaggle moved attached notebook outputs; the analysis is unchanged; the diff is in the repo.

- Why: Kaggle now mounts attached notebook outputs at `/kaggle/input/notebooks/<owner>/<slug>/`, not `/kaggle/input/<slug>/` (seen with a probe kernel that lists directory names only). The frozen lookup therefore found 0 of 42 cached feature files, and the first run (pushed Oct 28 05:10 UTC) rebuilt every clip, which does not fit Kaggle's 12 h session limit. That run is not used; the rerun uses this version.
- What changed (3 lines, the same in the template and the rendered kernel): the two lookups of the kernel's own cached features (the feature kernels' driver logs and feature files) find the slug at any depth; the forbidden-input guard checks every attached directory, since at the top level it now sees only `notebooks`. Nothing else changed: the same 42 clips, model, stages and checks. D26's checks still run: every source file is re-fetched and matched, and the LPP-FR section 8 + pieman self-check must pass. The three clips with no cached features (LPP-EN 1, LPP-FR 1 and 8) are rebuilt in the kernel as before.
- Made before any picker brain data was downloaded (the download pass starts Thu Oct 29 01:01 UTC). The picker code bundle (9d70b4b0...) does not contain this kernel and is unchanged.
- Hashes (SHA-256): rendered kernel `stillhere_picker_pred.py` 682c2a01db932ddac95bd17c0c2d6e172f761a3bef485c0ac4d99dca8e96dedf -> 1a9f18128ed582fe0dc8f8afe239d3669d5df07b20a4a687ed1d12787e2b41c9; template `driver_tpl.py` 3fef361070856f3b8fd05189784a115bbbc98c3c6c293a0bee6ce0c5eb279614 -> fcd34fbdb6debf40ce44373421db654aa76cbafc8647a8ed4dfbf7f1ac967390.
- Checks: the kernel's local checks pass (9 of 9). On the probed layout with the feature kernels' real driver logs, the frozen lookup finds 0 of 42 clips and the fixed one 39 of 42; a planted nested `ds003171` folder is refused by the fixed guard and missed by the frozen one.

```diff
diff -ru a/d22/picker_pred/driver_tpl.py b/d22/picker_pred/driver_tpl.py
--- a/d22/picker_pred/driver_tpl.py	2026-10-28 21:36:11.539464400 +0530
+++ b/d22/picker_pred/driver_tpl.py	2026-10-28 21:36:11.543813580 +0530
@@ -120,7 +120,7 @@
 if __name__ == "__main__":
     OUT.mkdir(parents=True, exist_ok=True); IN.mkdir(exist_ok=True)
     # 0. guards, secrets, environment -----------------------------------------------------------------------------------------------
-    bad = [p for p in glob.glob("/kaggle/input/*") if any(x in p.lower() for x in ("stillhere-conf", "ds003171", "stillhere-pred", "gate1", "notthefall"))]
+    bad = [p for p in glob.glob("/kaggle/input/**/", recursive=True) if any(x in p.lower() for x in ("stillhere-conf", "ds003171", "stillhere-pred", "gate1", "notthefall"))]
     if bad:
         log("refused: a forbidden input is attached"); raise SystemExit(1)
     tok = None
@@ -143,13 +143,13 @@
     # 1. cached features: SHA-256 against the feature kernels' RESULT lines, stock checkpoint ----------------------------------------
     logged = {}
     for slug in CACHE_SLUGS:
-        for p in glob.glob(f"/kaggle/input/{slug}/**/driver.log", recursive=True):
+        for p in glob.glob(f"/kaggle/input/**/{slug}/**/driver.log", recursive=True):
             for line in open(p):
                 if line.startswith("RESULT "):
                     r = json.loads(line[7:]); logged[r["clip"]] = (r["file_sha256"], r["audio_end_s"], r["ckpt_sha256"], slug)
     cache_ok = {}
     for st in stems:
-        hits = [p for slug in CACHE_SLUGS for p in glob.glob(f"/kaggle/input/{slug}/**/feats/{st}.npz", recursive=True)]
+        hits = [p for slug in CACHE_SLUGS for p in glob.glob(f"/kaggle/input/**/{slug}/**/feats/{st}.npz", recursive=True)]
         if st in logged and len(hits) == 1 and logged[st][2] == PIN:
             h = hashlib.sha256(open(hits[0], "rb").read()).hexdigest()
             if h == logged[st][0]:
diff -ru a/d22/picker_pred/kernels/picker_pred/stillhere_picker_pred.py b/d22/picker_pred/kernels/picker_pred/stillhere_picker_pred.py
--- a/d22/picker_pred/kernels/picker_pred/stillhere_picker_pred.py	2026-10-28 21:36:11.540852731 +0530
+++ b/d22/picker_pred/kernels/picker_pred/stillhere_picker_pred.py	2026-10-28 21:36:11.545235844 +0530
@@ -120,7 +120,7 @@
 if __name__ == "__main__":
     OUT.mkdir(parents=True, exist_ok=True); IN.mkdir(exist_ok=True)
     # 0. guards, secrets, environment -----------------------------------------------------------------------------------------------
-    bad = [p for p in glob.glob("/kaggle/input/*") if any(x in p.lower() for x in ("stillhere-conf", "ds003171", "stillhere-pred", "gate1", "notthefall"))]
+    bad = [p for p in glob.glob("/kaggle/input/**/", recursive=True) if any(x in p.lower() for x in ("stillhere-conf", "ds003171", "stillhere-pred", "gate1", "notthefall"))]
     if bad:
         log("refused: a forbidden input is attached"); raise SystemExit(1)
     tok = None
@@ -143,13 +143,13 @@
     # 1. cached features: SHA-256 against the feature kernels' RESULT lines, stock checkpoint ----------------------------------------
     logged = {}
     for slug in CACHE_SLUGS:
-        for p in glob.glob(f"/kaggle/input/{slug}/**/driver.log", recursive=True):
+        for p in glob.glob(f"/kaggle/input/**/{slug}/**/driver.log", recursive=True):
             for line in open(p):
                 if line.startswith("RESULT "):
                     r = json.loads(line[7:]); logged[r["clip"]] = (r["file_sha256"], r["audio_end_s"], r["ckpt_sha256"], slug)
     cache_ok = {}
     for st in stems:
-        hits = [p for slug in CACHE_SLUGS for p in glob.glob(f"/kaggle/input/{slug}/**/feats/{st}.npz", recursive=True)]
+        hits = [p for slug in CACHE_SLUGS for p in glob.glob(f"/kaggle/input/**/{slug}/**/feats/{st}.npz", recursive=True)]
         if st in logged and len(hits) == 1 and logged[st][2] == PIN:
             h = hashlib.sha256(open(hits[0], "rb").read()).hexdigest()
             if h == logged[st][0]:
```
