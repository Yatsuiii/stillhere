"""Builds the four exploratory stimulus conditions from the public LibriVox clip (assets/librivox_alice_ch01_75s.wav and its local
faster-whisper word timings) into assets/conditions/, with a manifest of orders, lengths and SHA-256s. usage: python scripts/build_conditions.py"""
from pathlib import Path

from stillhere import conditions as C

A = Path(__file__).resolve().parents[1] / "assets"

if __name__ == "__main__":
    man = C.build_files(A / "librivox_alice_ch01_75s.wav", A / "librivox_alice_ch01_75s.words.json", A / "conditions", "alice_ch01_75s")
    for name, f in man["files"].items():
        print(f'{f["sha256"]}  {name}: assets/conditions/{f["file"]} ({f["samples"]} samples)')
    print(f'{man["n_words"]} words, {man["n_sentences"]} sentences, sentence order {man["sentence_order"]}')
