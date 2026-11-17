# Speech validation Round 2 — collection in progress

**Historical progress snapshot, superseded by the completed
[Round 2 report](ROUND2_VALIDATION_REPORT.md).** Its partial metrics and reference
review state below describe the earlier collection stage.

16 November 2026. **This round is incomplete. Keep Medium opt-in and retain the
existing global Small default.** No general multilingual accuracy claim follows
from these results.

## Current corpus status

| Group | Captured independent takes | Eligible verified references | Outstanding |
| --- | ---: | ---: | --- |
| English | 5 | 4 | `english_01` needs another playback/reference check |
| Hindi | 2 | 1 verified five-second excerpt | Full takes remain unscored; the other four planned utterances are not captured |
| Retained Hinglish | 1 | 1 | Existing playback-verified recording retained unchanged |

Five English takes were initially confirmed by the user after playback. The
first was later quarantined when both models returned a substantially different
utterance from the four-word reference. This is a reason to request another
human check, not permission to treat either model's output as ground truth.
Its original confirmation and audio are retained; active WER/CER is withheld.

The user paused before Hindi capture. After resumption, the first Hindi take
was rejected as different words, subsequently identified as the user's voice.
The user reported that the system beep was inaudible. A WAV chime was tested and
confirmed audible, and only the session helper's start cue was changed.
The new take reached the 60-second safety limit with active end audio. The user
identified their voice in a five-second preview, but a word-for-word reference
is still required. A preview is not a separate independent accuracy recording.
The user subsequently typed “आज मौसम अच्छा है।” as the exact words heard in
that five-second preview. A lossless NPY of the same excerpt is now bound to that
human reference, with source sample coordinates and original capture hash.
This is one utterance from one source capture, not another independent recording.
The full take remains unscored. The short-take safety cap was reduced to 20 s in
the session helper only; the normal application settings remain unchanged.
Hindi 2 is waiting for the user's Ready response before capture.

Session manifest, raw float32 NPY recordings, playback WAVs, suggested prompts,
separate reference files and setup audit are preserved under
`benchmark-recordings/round2-20261116/`. Suggested prompts are never references
until the actual captured words are manually confirmed. Rejected takes and
retries have separate files and are not overwritten.

## Available-sample comparison

The unchanged Medium candidate and Small baseline were each loaded in a
separate sequential CUDA process and reused for 30 calls: five passes on each
of the five English takes and the retained Hinglish recording. The disputed
English reference was subsequently withheld from active accuracy summaries.
The resumed Hindi takes and newly verified excerpt were not part of these GPU runs.

| Corpus | Eligible independent samples | Small raw WER / CER | Medium raw WER / CER | Small normalized WER / CER | Medium normalized WER / CER |
| --- | ---: | --- | --- | --- | --- |
| English | 4 | 13.51% / 6.71% | 14.86% / 6.41% | 10.81% / 6.71% | 9.46% / 6.41% |
| Hindi | 0 | Unavailable | Unavailable | Unavailable | Unavailable |
| Retained Hinglish | 1 | 78.95% / 58.33% | 21.05% / 15.28% | 78.95% / 58.33% | 21.05% / 15.28% |

WER/CER aggregate error counts over the verified reference lengths and remain
separate by corpus language and sample. Repeated passes are not additional
independent recordings. Vocabulary normalization and raw output are scored
separately; no correction was added for these recordings.

| Sample | Duration s | Small raw WER / CER | Medium raw WER / CER | Small / Medium warm median s |
| --- | ---: | --- | --- | --- |
| `english_01` | 8.544 | Withheld | Withheld | .434 / .850 |
| `english_02`, technical terms | 7.904 | 18.18% / 0% | 27.27% / 2.04% | .411 / .825 |
| `english_03`, numbers/negation | 6.944 | 33.33% / 34.69% | 33.33% / 34.69% | .380 / .848 |
| `english_04`, short pause | 9.440 | 0% / 0% | 0% / 0% | .399 / .898 |
| `english_05`, paragraph | 17.376 | 10.26% / 3.23% | 10.26% / 2.15% | .649 / 1.553 |
| Retained `hinglish_technical` | 12.832 | 78.95% / 58.33% | 21.05% / 15.28% | 1.030 / 1.909 |

The pause sample was captured in full according to the user and decoded without
word/character errors by both models. The longer English paragraph is 17.376 s;
it does not establish behavior on complete long sessions or speech exceeding a
30-second decoding window. A longer Hindi prompt is prepared but unrecorded.

The number sample penalizes `27` versus the reference's `twenty seven` under the
existing literal scoring policy; both models also differed elsewhere from the
confirmed reference. No post-hoc numeric equivalence rule was introduced to
reduce the errors. Technical spelling remains model-dependent: Medium rendered
DBMS incorrectly in its English technical-term take, while Small's normalized
output matched that reference. Medium does not clearly dominate these English
samples, despite its previously measured Hinglish advantage.

[Full per-pass raw/normalized comparison](ROUND2_COMPARISON_PARTIAL.md) is derived
from `stt-round2-medium-partial.json` and `stt-round2-small-partial.json`.
Withheld historical reference metrics remain explicitly archived in those JSONs,
not included in active WER/CER or eligible-reference counts.

## Timing, memory and implementation scope

| Configuration | Fresh-process model load s | Sampled total GPU MiB |
| --- | ---: | ---: |
| Small CUDA / float16 | 2.379 | 787 |
| Medium CUDA / int8_float16 | 5.532 | 1235 |

Warm medians exclude pass 1 per sample. Timings may be reported for an unverified
reference because decoding actually ran, but that does not make its accuracy
score valid. Loading is a fresh model instance with potentially warm Windows file
cache; GPU memory is sampled total device usage, including other processes.
No fallback or OOM occurred. The microphone was not opened during benchmarking.

Medium remains the same opt-in preset: automatic language, beam 5, no prompt,
temperature 0 and the existing VAD. Production speech modules, global defaults,
private environment configuration, notes/reminders/tasks, router and storage
were not modified. No wake-word detection was implemented. Hash checks confirmed
the production source and database remained unchanged.

```text
python -m pytest -q       295 passed, 1 deselected
python -m ruff check .    All checks passed!
```

**Broader-use recommendation is deferred.** Complete the exact-reference review
or replace English 1, then obtain at least five verified Hindi recordings with
controlled capture. Retain this partial comparison as history and produce a new
full-round report after those recordings are verified. Even that small,
single-speaker corpus will support only limited conclusions.
