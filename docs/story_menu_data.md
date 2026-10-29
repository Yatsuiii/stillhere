# Story menu data (the PLAN and BUILD story card)

The fields the page reads to show docs/rules/story_menu_rule.md section 6's card, agreed Oct 29 by the web and worker forks. The worker
fork writes them into `stillhere/live_menu.json` (the planner rebuilds its clip enum from the same file) when a story passes the screen.

Per screened clip, beside the clip's existing fields (`id` = `story_<short title>`, `title` = the section's title as the LibriVox
catalogue lists it, `reader` = the section reader's display name, `author`, `source`, `start_s`, `dur_s`, `intact_wav_sha256`,
`public_scan`):

```json
"screen": {"archive_url": "https://archive.org/download/<item>/<file>.mp3",
           "sentences": 7,
           "notes": [{"tag": "grief or loss", "t": 65.4, "quote": "word for word, or null when the guard withheld it"}],
           "notes_host": "fireworks"}
```

- `archive_url`: the original LibriVox section MP3 (64 kb) the story was decoded from; `t`: seconds in that file (the word tables are
  file times).
- `sentences`: {k}, sentences in the window. A story on the menu passed bar 1, so "1 voice heard" is fixed text.
- `notes`: the notes code kept (quote found as a contiguous run, timed at its start); tags from `stillhere/story_tags.py` only.
- `notes_host`: the one host whose call produced the notes.

At the top of `live_menu.json`:

```json
"screen_summary": {"n_on_menu": 10, "failed": [{"title": "...", "bar": "2"}], "rule": "docs/rules/story_menu_rule.md"}
```

- `n_on_menu`: {n}, stories the rule put on the menu (at most 10). `failed`: each screened story that did not reach the menu, with
  only its title and the bar it failed ("1", "2", "3" or "trim"); the page greys them out with nothing else.

How the page shows it (`web/stations.py` `story_card`): the rule's line word for word, {tags with times} as "{tag} ({m:ss})" in time
order joined by ", ", each linking to `archive_url#t={whole seconds}`, or "none of the listed tags found". A clip with no `screen`
(alice_ch01 to ch03) shows no card: the PLAN and BUILD views are as before.
