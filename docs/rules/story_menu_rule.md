# Story menu rule: fresh public stories for the live menu (exploratory)

Committed before any story in the pool is downloaded, transcribed, screened or read (cloud plan of Oct 29, 19:20 UTC, item 1b). It
reads no brain data. Everything below is fixed by this commit; a change is a new commit that says what changed and why, made before the
step it changes runs. Results are labelled exploratory.

## 1. Pool

- Source: the LibriVox catalogue entries "Short Story Collection Vol. 001" and "Short Story Collection Vol. 002", in that order. If
  either title does not resolve to exactly one LibriVox project, stop and report; no other volume is substituted.
- A section (one story) enters the pool only if, per the LibriVox catalogue: its language is English; it lists exactly one reader;
  its listed length is from 4:00 to 30:00 (m:ss) inclusive. (Amended Oct 29, cloud ruling of 19:55 UTC: the first commit said
  4:00 to 12:00, under which only 2 stories of Vol. 001-002 qualify; a 30-40 s window does not need a short story, and the bar only
  limited screen cost.)
- Order: catalogue order (Vol. 001 sections in section order, then Vol. 002). The first 25 sections that meet the conditions above and
  are not excluded below are the pool; later sections are never read.
- Excluded: known public story-fMRI stimuli. A section is excluded if its title, lower-cased with punctuation removed, equals a
  stimulus title of: Narratives (OpenNeuro ds002345, every story in its stimuli list), Le Petit Prince (ds003643), the Alice Datasets
  (ds002322, Alice's Adventures in Wonderland chapter 1), the Moth stories of LeBel et al. 2023 (ds003020), Harry Potter and the
  Philosopher's Stone chapter 9 (Wehbe et al. 2014) and the Natural Stories texts (Futrell et al. 2018; fMRI in Shain et al. 2020).
  Also excluded by name: The Little Match Girl, whose adapted text ("The Matchstick Seller") is in the Natural Stories set (amended
  Oct 29, 19:55 UTC ruling; keeps every menu story's "No public scan" line true without a footnote).
- Before a story goes on the menu, its title is searched on the web with "fMRI" and with "OpenNeuro"; the queries and what they found
  are recorded with the story. On any hit naming a scanned stimulus of that story, it is excluded (or, if it is already on the menu,
  its public_scan field records the dataset). (Amended Oct 29, 19:55 UTC ruling.)
- The title exclusion and this search are applied to every clip on stillhere/live_menu.json, and their result is that clip's
  "public_scan" field (null, or the dataset). The page shows "No public scan" only for a clip whose public_scan is null.
- Audio: the section's 64 kb MP3 from the LibriVox archive.org item, decoded to 16 kHz mono WAV with ffmpeg (bit-exact flags, as
  scripts/order_map.py cut_clip). Word table: local faster-whisper base.en word timings with the settings of scripts/order_map.py
  whisper_words. All of this runs on the laptop CPU.

## 2. Trim (LibriVox preamble and closing)

Words are normalised with stillhere/manipulation_check.words (lower case, punctuation stripped, hyphens split).
- Start: find, within the first 90 s of the section, the last occurrence of any of these phrases in the word table: "librivox org",
  "librivox dot org", "libravox org", "libravox", "read by", "recorded by", "reading by". ("libravox org" and "libravox" added Oct 29
  under the cloud ruling of 20:15 UTC: faster-whisper spells LibriVox "LibraVox" in nearly every story.) The trimmed story starts at the first sentence (stillhere/conditions.py
  sentences()) that begins after that phrase. No phrase found: the section fails the screen ("no LibriVox preamble found").
- End: find, within the last 60 s, the first occurrence of any of "end of", "this recording is in the public domain", "the end". The
  trimmed story ends at the last sentence that ends before it. No phrase found: it ends at the end of the last complete sentence.

## 3. Screen (bars)

Voice Transcribe (Meta) runs with diarization on consecutive 60 s chunks of the trimmed story, from its start; the last chunk may be
shorter. Its use as a speaker gate was cleared by results/diarization_validation.json (2 labels, agreement 0.985, changes 5 of 5). A
story passes only if all three hold:
1. One voice: in every chunk, one speaker label covers at least 95% of the speech time inside Voice Transcribe's turns.
2. Silence: the longest gap between consecutive words of the faster-whisper table inside the trimmed story is at most 5.0 s.
3. Agreement: the longest common subsequence (stillhere/manipulation_check.lcs) of Voice Transcribe's words (chunks in order) with
   the faster-whisper table's words, over the number of faster-whisper words, is at least 0.95. (The faster-whisper table is what
   TRIBE's text lane reads; this checks Voice Transcribe confirms it.)

Spend: the Voice Transcribe screen stops before a call that would take its batch past $35 (amended Oct 29, 19:55 UTC ruling; the
first commit said $25). A story that is not fully screened when the
cap is reached does not pass. The real cost is reported.

## 4. Content notes (Muse Spark Open)

- One call per passing story, on Fireworks; Baseten only if the Fireworks call fails (an error or no valid output after 2 attempts).
  A story's notes come from exactly one host, recorded with them.
- Input: the trimmed story's faster-whisper text, sent as allowlisted public text through stillhere/egress.py. Nothing else.
- Output (json_schema): a list of notes, each a tag from the list below and a word-for-word quote from the text, plus a description of
  exactly two sentences (stored, not shown).
- Tags (fixed list, shown as written): death or dying; injury or illness; falls or head injury; being trapped or unable to move or
  speak; drowning; suicide or self-harm; grief or loss; medical care; violence; cruelty to animals; sexual content; alcohol or drugs;
  frightening scenes; slurs or demeaning language.
- Code keeps a note only if its quote's normalised words occur as a contiguous run in the story's normalised words, and times it at
  the start of that run, in seconds of the original LibriVox file. If the egress guard blocks a quote, the quote is withheld and the
  tag and time stay.
- Content notes do not decide which stories reach the menu.

## 5. Window (the live clip)

- Candidates start at each sentence start of the trimmed story, earliest first. From a start, sentences are added in order while the
  clip (first word start minus 0.3 s, to last word end plus 0.3 s) is at most 40.0 s (worker/recipe.py MAX_SECONDS).
- A candidate is valid if that clip is at least 30.0 s long and holds at least 6 sentences, and starts and ends on whole sentences
  (by construction). The story's window is its earliest valid candidate; a story with none does not reach the menu.
- Keep the first 10 stories in catalogue order that pass the screen and have a window. Each gets a descriptive id
  (story_<short title>), its wav and word-table SHA-256 on the allowlists, and the recipe allowlist lines.

## 6. Display lines (word for word; nothing else describes these stories)

PLAN card for a live route, and the same card on BUILD:

"{title}, read by {reader}: one of {n} screened public stories · 1 voice heard · {k} sentences · notes: {tags with times, or "none of
the listed tags found"}. Content notes come from Muse Spark Open reading the transcript. Not a full review; listen before you use a
story."

- {n}: the number of stories this rule put on the menu (at most 10). {k}: sentences in the window. {tags with times}: each kept note as
  "{tag} ({m:ss})", in time order, joined by ", "; each timed note links to the archive.org file at #t={seconds}.
- The word "safe" never appears in these lines or near them.
- Stories that fail are counted and greyed out; their only text is their title and the number of the bar they failed (section 3,
  "1", "2" or "3"; or "trim").

## 7. Order and stop

Fetch, trim and word tables (sections 1-2) on the laptop; the screen (3); notes (4) for passing stories only; windows (5). The live
menu changes only after at least 3 windows have all 4 forecasts cached with the fixed engine and checked. Hard stop for live parts:
Sat Oct 31 12:00 UTC; after it, the menu ships as Part 1 left it (alice_ch02 first).
