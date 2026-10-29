"""Order-map go/no-go (cloud plan v2, item 1; exploratory, forecast only, no brain data). Rule: docs/rules/order_map_gonogo.md.

Pure functions (pieces, shuffle, timeline mapping, kept seconds, change measure, verdict) plus two stages:
  prepare: cut the three LibriVox clips, take local faster-whisper word timings, build intact / sentence_shuffle / reversed and the
           word table each condition gives TRIBE's text lane; writes OUT_DIR/<clip>/ and OUT_DIR/prepare_manifest.json.
  score:   read the forecasts scripts/order_map_run.py wrote, un-shuffle, score, and write results/order_map_gonogo.json.
usage: HF_HUB_OFFLINE=1 HF_HUB_CACHE=<faster-whisper cache> PYTHONPATH=.:<asr site-packages> python3.11 scripts/order_map.py prepare SRC OUT
       PYTHONPATH=.:<site-packages with nibabel> python3 scripts/order_map.py score OUT ATLAS_DIR"""
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stillhere import conditions as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SR = 16000
START_S, DUR_S = 60.0, 40.0                           # each clip: 40 s from 60 s of its chapter recording (past every intro)
CLIPS = {"alice_ch01": "wonderland_ch_01_64kb.mp3", "alice_ch02": "wonderland_ch_02_64kb.mp3", "alice_ch03": "wonderland_ch_03_64kb.mp3"}
SHORT_SENTENCE_S = 6.0                                # any sentence shorter than this: shuffle blocks of whole sentences instead
BLOCK_MIN_S = 15.0
FALLBACK_MIN_S = 10.0                                 # two-block fallback: each block keeps >= 5 s after the 5 s drop
DROP_S = 5.0                                          # seconds dropped after every cut
SEED = C.SEED                                         # the Pilot's default seed (20261028)
CONDITIONS = ("intact", "sentence_shuffle", "reversed")
LABEL = "exploratory; forecast only, no brain data"


# ---- pieces and the shuffle ------------------------------------------------------------------------------------------------------
def piece_spans(words: list, groups: list, n: int, sr: int) -> list:
    """(a, b) in seconds of each group of consecutive word indices, on the word-boundary tiling of stillhere.conditions."""
    cuts = C.cut_points(words, n, sr)
    return [(cuts[g[0]] / sr, cuts[g[-1] + 1] / sr) for g in groups]


def best_blocks(durs: list, min_s: float) -> list:
    """Partition consecutive items (durations durs) into the most blocks that each last >= min_s; ties: the smallest longest block, then
    the first in enumeration order. Returns lists of item indices. One block when no split qualifies."""
    n = len(durs); best = None
    for k in range(n):                                                     # k cut positions among the n - 1 gaps
        for cut in itertools.combinations(range(1, n), k):
            edges = (0, *cut, n); blocks = [list(range(a, b)) for a, b in zip(edges[:-1], edges[1:])]
            lens = [sum(durs[i] for i in b) for b in blocks]
            if min(lens) < min_s:
                continue
            key = (-len(blocks), max(lens))
            if best is None or key < best[0]:
                best = (key, blocks)
    return best[1] if best else [list(range(n))]


def midpoint_split(durs: list, min_s: float) -> list:
    """Two blocks split at the sentence end nearest the clip's midpoint among those leaving >= min_s on both sides; one block if none."""
    tot = sum(durs); ends = np.cumsum(durs)[:-1]
    ok = [i for i, e in enumerate(ends) if e >= min_s and tot - e >= min_s]
    if not ok:
        return [list(range(len(durs)))]
    i = min(ok, key=lambda j: (abs(ends[j] - tot / 2), j))
    return [list(range(i + 1)), list(range(i + 1, len(durs)))]


def pieces(words: list, n: int, sr: int) -> tuple:
    """(groups of word indices, kind): whole sentences if every sentence lasts >= SHORT_SENTENCE_S; else blocks of whole sentences,
    the most blocks of >= BLOCK_MIN_S each; if no two such blocks exist, two blocks split at the sentence end nearest the midpoint, each
    >= FALLBACK_MIN_S. One piece (no shuffle possible) only when none of these exists."""
    sents = C.sentences(words)
    durs = [b - a for a, b in piece_spans(words, sents, n, sr)]
    if min(durs) >= SHORT_SENTENCE_S:
        return sents, "sentences"
    blocks, kind = best_blocks(durs, BLOCK_MIN_S), "blocks"
    if len(blocks) < 2:
        blocks, kind = midpoint_split(durs, FALLBACK_MIN_S), "two blocks at the midpoint"
    return [[w for s in blk for w in sents[s]] for blk in blocks], kind


def shuffle_order(n_pieces: int, seed: int = SEED) -> list:
    """Non-identity permutation of the pieces (stillhere.conditions' rule); identity only when there is one piece."""
    return C._permutation(n_pieces, np.random.default_rng(seed))


def shuffled_starts(spans: list, order: list) -> dict:
    """Start (s) of each piece in the shuffled audio: the pieces tile it in `order` with their original lengths."""
    out, t = {}, 0.0
    for k in order:
        out[k] = t; t += spans[k][1] - spans[k][0]
    return out


def moved_words(words: list, groups: list, spans: list, order: list) -> list:
    """The shuffled condition's word table: every word keeps its piece-relative time; pieces in shuffled order."""
    s2 = shuffled_starts(spans, order); out = []
    for k in order:
        for i in groups[k]:
            w = words[i]; d = s2[k] - spans[k][0]
            out.append({"word": w["word"], "start": w["start"] + d, "end": w["end"] + d})
    return out


def build(x: np.ndarray, sr: int, words: list, seed: int = SEED) -> dict:
    """The three conditions and their word tables. Reversed gets no words: no word reaches the text lane."""
    groups, kind = pieces(words, len(x), sr)
    order = shuffle_order(len(groups), seed)
    cuts = C.cut_points(words, len(x), sr); wspans = list(zip(cuts[:-1], cuts[1:]))
    spans = piece_spans(words, groups, len(x), sr)
    return {"audio": {"intact": np.asarray(x, float).copy(),
                      "sentence_shuffle": C.assemble(x, wspans, [i for k in order for i in groups[k]], sr),
                      "reversed": np.flip(np.asarray(x, float)).copy()},
            "words": {"intact": [dict(w) for w in words], "sentence_shuffle": moved_words(words, groups, spans, order), "reversed": []},
            "meta": {"piece_kind": kind, "pieces": groups, "order": order, "piece_spans_s": [[round(a, 4), round(b, 4)] for a, b in spans],
                     "seed": seed}}


# ---- un-shuffle and kept seconds -------------------------------------------------------------------------------------------------
def kept_seconds(spans: list, order: list, n_seconds: int, drop_s: float = DROP_S) -> list:
    """(shuffled second s, original-time centre) for every 1 s forecast sample [s, s+1) of the shuffled condition that lies inside one
    piece and is not in the first drop_s seconds of a piece whose predecessor in the shuffled audio differs from its predecessor in
    the intact (the clip start counts as a predecessor). Its original-time centre is a_k + (s + 0.5 - a'_k)."""
    s2 = shuffled_starts(spans, order); prev = {k: (order[i - 1] if i else None) for i, k in enumerate(order)}
    out = []
    for s in range(n_seconds):
        for k in order:
            a2, b2 = s2[k], s2[k] + spans[k][1] - spans[k][0]
            if s >= a2 - 1e-9 and s + 1 <= b2 + 1e-9:
                moved = prev[k] != (k - 1 if k else None)
                if not moved or s >= a2 + drop_s - 1e-9:
                    out.append((s, spans[k][0] + (s + 0.5 - a2)))
                break
    return out


def align_pair(I: np.ndarray, S: np.ndarray, kept: list) -> tuple:
    """Intact forecast I (seconds x parcels, sample s covers [s, s+1)) linearly read at the kept original-time centres, and the shuffled
    forecast S at the matching shuffled seconds. Centres outside the intact samples' centres are dropped. Returns (I_k, U_k, kept)."""
    c = np.arange(len(I)) + 0.5
    ok = [(s, t) for s, t in kept if c[0] - 1e-9 <= t <= c[-1] + 1e-9 and s < len(S)]
    tt = np.array([t for _, t in ok]); ss = np.array([s for s, _ in ok], int)
    Ik = np.stack([np.interp(tt, c, I[:, j]) for j in range(I.shape[1])], 1)
    return Ik, S[ss], ok


def parcel_change(Ik: np.ndarray, Uk: np.ndarray) -> np.ndarray:
    """Per parcel: RMS of (un-shuffled - intact) over the kept seconds divided by the SD (ddof 0) of the intact over the same seconds.
    Scale-free: multiplying a parcel's forecast by any constant leaves it unchanged."""
    return np.sqrt(((Uk - Ik) ** 2).mean(0)) / Ik.std(0)


def region_series(P: np.ndarray, parcels: list) -> np.ndarray:
    """(seconds, parcels): parcel-mean forecast, raw TRIBE units."""
    return np.stack([P[:, v].mean(1) for v in parcels], 1)


def clip_call(lang: float, aud: float) -> bool:
    return bool(lang > aud)


def verdict(calls: list) -> str:
    """GO only when the language-area change is larger than the sound-area change on 3 of 3 clips."""
    return "GO" if len(calls) == 3 and all(calls) else "NO-GO"


# ---- prepare --------------------------------------------------------------------------------------------------------------------
def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cut_clip(mp3, out_wav) -> str:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3), "-ss", str(START_S), "-t", str(DUR_S), "-ac", "1", "-ar", str(SR),
                    "-c:a", "pcm_s16le", "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact", str(out_wav)], check=True)
    return sha256_file(out_wav)


def whisper_words(wav) -> list:
    """Local faster-whisper base.en word timings (CPU, int8, offline), the source of the existing Pilot assets' words."""
    from faster_whisper import WhisperModel
    m = WhisperModel("Systran/faster-whisper-base.en", device="cpu", compute_type="int8", local_files_only=True)
    x, _ = C.read_wav(wav)
    segs, _ = m.transcribe(x.astype(np.float32), language="en", word_timestamps=True)
    return [{"word": w.word.strip(), "start": round(float(w.start), 3), "end": round(float(w.end), 3)} for s in segs for w in s.words]


def write_tsv(words: list, path) -> int:
    """<wav>.tsv in the table TRIBE's ExtractWordsFromAudio reads instead of running its own speech-to-text."""
    lines = ["text\tstart\tduration\tsequence_id\tsentence"]
    lines += [f"{w['word']}\t{w['start']:.4f}\t{w['end'] - w['start']:.4f}\t0\t" for w in words]
    Path(path).write_text("\n".join(lines) + "\n")
    return len(words)


def prepare(src: Path, out: Path) -> dict:
    man = {"label": LABEL, "start_s": START_S, "dur_s": DUR_S, "seed": SEED, "clips": {}}
    for clip, mp3 in CLIPS.items():
        d = out / clip; d.mkdir(parents=True, exist_ok=True)
        src_wav = d / "source.wav"; man_c = {"source_file": mp3, "source_wav_sha256": cut_clip(src / mp3, src_wav)}
        words = whisper_words(src_wav); (d / "source.words.json").write_text(json.dumps({"words": words}, indent=1) + "\n")
        x, sr = C.read_wav(src_wav); c = build(x, sr, words)
        man_c.update(n_words=len(words), n_sentences=len(C.sentences(words)), **c["meta"], files={})
        for name in CONDITIONS:
            wav = d / f"{name}.wav"
            man_c["files"][name] = {"sha256": C.write_wav(wav, c["audio"][name], sr), "samples": int(len(c["audio"][name])),
                                    "text_lane_words": write_tsv(c["words"][name], wav.with_suffix(".tsv"))}
        man["clips"][clip] = man_c
    (out / "prepare_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    return man


# ---- score ----------------------------------------------------------------------------------------------------------------------
def score(out: Path, atlas: Path) -> dict:
    from stillhere import heldout
    sets = heldout.parcel_sets(atlas / "lh.Schaefer2018_400Parcels_17Networks_order.annot",
                               atlas / "rh.Schaefer2018_400Parcels_17Networks_order.annot")
    man = json.loads((out / "prepare_manifest.json").read_text()); res = {"label": LABEL, "rule": "docs/rules/order_map_gonogo.md",
                                                                         "regions": {"LANGUAGE": "LANG", "HEARING": "AUD"}, "clips": {}}
    calls = []
    for clip, m in man["clips"].items():
        d = out / clip; P = {c: np.load(d / f"pred_{c}.npz") for c in CONDITIONS}
        info = {c: json.loads((d / f"run_{c}.json").read_text()) for c in CONDITIONS}
        spans = [tuple(s) for s in m["piece_spans_s"]]
        kept = kept_seconds(spans, m["order"], len(P["sentence_shuffle"]["P"]))
        per, tc = {}, {}
        for reg, key in res["regions"].items():
            R = {c: region_series(P[c]["P"], sets[key]) for c in CONDITIONS}
            Ik, Uk, ok = align_pair(R["intact"], R["sentence_shuffle"], kept)
            per[reg] = float(parcel_change(Ik, Uk).mean())
            tc[reg] = {c: [round(float(v), 5) for v in R[c].mean(1)] for c in CONDITIONS}
            tc[reg]["unshuffled_kept"] = {"original_time_s": [round(t, 3) for _, t in ok], "value": [round(float(v), 5) for v in Uk.mean(1)]}
        call = clip_call(per["LANGUAGE"], per["HEARING"]); calls.append(call)
        res["clips"][clip] = {"source": m["source_file"], "source_wav_sha256": m["source_wav_sha256"], "piece_kind": m["piece_kind"],
                              "n_pieces": len(m["pieces"]), "order": m["order"], "kept_seconds": len(ok), "change": per,
                              "language_change_larger": call, "runs": info, "timecourses_raw_units": tc}
    res["n_clips_language_larger"] = int(sum(calls)); res["verdict"] = verdict(calls)
    res["consequence"] = ("order map allowed on the page and in the video" if res["verdict"] == "GO" else
                          "no order map: the page and video use time-course panels plus the reversed variant")
    (REPO / "results/order_map_gonogo.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        m = prepare(Path(sys.argv[2]), Path(sys.argv[3]))
        for k, v in m["clips"].items():
            print(f"{k}: {v['n_words']} words, {v['n_sentences']} sentences, {len(v['pieces'])} {v['piece_kind']}, order {v['order']}, "
                  f"text-lane words {[f['text_lane_words'] for f in v['files'].values()]}")
    elif sys.argv[1] == "score":
        r = score(Path(sys.argv[2]), Path(sys.argv[3]))
        for k, v in r["clips"].items():
            print(f"{k}: kept {v['kept_seconds']} s; change LANGUAGE {v['change']['LANGUAGE']:.3f} vs HEARING {v['change']['HEARING']:.3f}")
        print(f"verdict {r['verdict']} ({r['n_clips_language_larger']} of 3)")
