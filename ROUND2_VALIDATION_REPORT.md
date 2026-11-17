# VoiceOS speech validation — Round 2

16 November 2026. **Primary collection and testing are complete: five full
English recordings, five full Hindi recordings, and the retained verified
Hinglish recording.** All references were supplied or explicitly confirmed by
the user after playback. Each model received the same preserved waveform for
five passes: 55 CUDA calls per model, 110 calls total.

**Recommendation: keep Medium int8_float16 opt-in for limited, reviewed trials.
It is not ready to be described as generally accurate multilingual STT.** Hindi
and the retained Hinglish sample improved substantially over Small, but Hindi
still has 38.24% raw WER, English aggregate WER is tied, and Medium costs more
latency and memory. The global Small default was not changed.

## Language-specific accuracy

WER/CER below are percentages. Errors are aggregated over verified reference
lengths within each corpus language. Repeated passes are not independent
recordings; the corpus has 11 recordings, not 110 independent samples.

| Corpus | Independent recordings | Small raw WER / CER | Medium raw WER / CER | Small normalized WER / CER | Medium normalized WER / CER |
| --- | ---: | --- | --- | --- | --- |
| English | 5 | 16.47% / 8.46% | 16.47% / 7.69% | 14.12% / 8.46% | 14.12% / 7.69% |
| Hindi | 5 | 61.03% / 44.56% | **38.24% / 14.99%** | 61.03% / 44.56% | 38.24% / 14.99% |
| Retained Hinglish | 1 | 78.95% / 58.33% | **21.05% / 15.28%** | 78.95% / 58.33% | 21.05% / 15.28% |

The distinct references contain 85 English words, 136 Hindi words and 19 Hinglish
words under the declared scoring policy. The 94-word Hindi paragraph accounts
for much of the Hindi aggregate; per-sample results matter alongside that number.
All five passes produced identical raw text for each model/sample pairing.

[Complete comparison table](ROUND2_COMPARISON.md) includes every pass's exact
audio path and SHA-256, model/compute, language, beam/prompt/VAD settings, raw and
normalized transcripts, human reference, WER/CER, latency and sampled GPU memory.
Machine-readable sources are `stt-round2-medium.json` and
`stt-round2-small.json`.

## Per-recording results

| Recording / condition | Audio s | Small raw WER / CER | Medium raw WER / CER | Small warm median s | Medium warm median s |
| --- | ---: | --- | --- | ---: | ---: |
| `english_01`, introduction/name, reference rechecked | 8.544 | 36.36% / 21.28% | 27.27% / 17.02% | .397 | .865 |
| `english_02`, technical terms | 7.904 | 18.18% / 0% | 27.27% / 2.04% | .412 | .834 |
| `english_03`, numbers/negation | 6.944 | 33.33% / 34.69% | 33.33% / 34.69% | .362 | .811 |
| `english_04`, short internal pause | 9.440 | 0% / 0% | 0% / 0% | .376 | .847 |
| `english_05`, English paragraph | 17.376 | 10.26% / 3.23% | 10.26% / 2.15% | .610 | 1.456 |
| `hindi_01_retry02`, short weather sentence | 6.208 | 25% / 7.69% | **0% / 0%** | .400 | .992 |
| `hindi_02`, name/project | 8.320 | 14.29% / 8.16% | 21.43% / 10.20% | .660 | 1.429 |
| `hindi_03`, numbers/negation | 7.744 | 57.14% / 18.37% | 35.71% / 12.24% | .668 | 1.464 |
| `hindi_04`, short internal pause | 7.744 | 50% / 24.39% | 40% / 9.76% | .621 | 1.403 |
| `hindi_05`, long Hindi passage | 34.048 | 71.28% / 57.61% | 42.55% / 17.31% | 3.657 | 11.134 |
| `hinglish_technical`, retained mixed-language sample | 12.832 | 78.95% / 58.33% | 21.05% / 15.28% | .854 | 1.833 |

The complete comparison also contains normalized scores per sample. Medium was
not better on every recording: Small's Hindi name/project result and normalized
English technical-term result were better. Both models matched the English pause
reference, but Hindi pause recognition still had errors even though the user
confirmed the complete sentence was captured.

Examples of actual Medium output:

- Short Hindi: `आज मौसम अच्छा है।` — matched the confirmed reference.
- Name/project: `मेरा नाम केवल मिन्डा है और मैं अपने प्रोजेक पर काम कर रहा हूं।`
  — spelling/orthography differed from the confirmed reference.
- Hindi numbers: `मुझे आज तेनो किताबे पढनी हैं लेकिन में अभी घर नहीं जा रहा हूं।`
  — the intended `तीन` and other words were not faithfully written.
- Retained Hinglish: `आज मैं voiceOS प्रोजेक पे Python और Gemini इस्तमाल कर रहा हूँ और BBMS का काम अभी बाकी है।`
  — Hindi/English mixing remained, but DBMS and other words were still wrong.

Small's long Hindi output degenerated into repeated combining marks and failed
to transcribe the complete passage faithfully. Its raw output is preserved in
the full table and JSON, without a cleanup that would conceal the failure.
Medium continued the passage but still made 40 word errors against its 94-word
reference. Its lower error rate does not establish dependable long dictation.

## Reference integrity and preserved recordings

Raw mono 16 kHz float32 NPY arrays, PCM16 playback WAVs, reference files, prompts,
capture metadata and audits are under `benchmark-recordings/round2-20261116/`.
The frozen **primary-only** manifest is `primary-session.json`. It contains 11
different waveform hashes and no derived excerpts or unverified takes.

The live collection manifest `session.json` also retains the auxiliary takes.
They are explicitly excluded from the primary comparison:

- `hindi_01`: user identified different captured words; no exact reference.
- `hindi_01_retry01`: 59.904 s, ended at the recording safety limit with active
  end audio; the full recording has no verified reference and remains unscored.
- `hindi_01_verified_excerpt`: a five-second lossless excerpt of that retry.
  The user typed `आज मौसम अच्छा है।` after hearing that preview. Its reference
  applies only to the excerpt; it was not counted as another independent full
  recording or used to meet the five-Hindi-recording minimum.

The original English 1 suggested sentence was not what the recording contained.
When the model outputs strongly disagreed with the initial confirmation, its
scores were withheld and the user was asked to transcribe the saved audio again.
The user supplied this actual captured wording independently:

> hi, i am keval minda. currently working on project voice os

That verbatim reference is now hash-bound to English 1. The prior suggested
reference, confirmation and withheld historical metrics remain preserved. No
recognizer output became ground truth, and the raw transcript was not corrected
to match the new reference.

The other new references were confirmed after complete playback, before their
recognition outputs were shown. These are user-verified references, not an
independent professional annotation or a second-listener audit. A reference
recheck materially changed the English evaluation; that process limitation
should be considered when interpreting this small corpus.

## Scoring and normalization

The existing policy was retained: NFC, case folding, punctuation-to-whitespace,
collapsed word spacing; CER omits spaces and counts Unicode code points. Literal
word boundaries, number rendering and Devanagari marks can affect scores.

For example, `27` versus `twenty seven` contributes to the English number-sample
error. The rechecked English 1 reference uses `voice os`, while the existing
normalizer produces `VoiceOS`; joining the brand worsens that sample's normalized
WER under the literal reference policy. CER omits spaces, so that join has no
corresponding character penalty. No post-hoc scoring exception or number/script
conversion was added to make either model look better.

Raw and normalized text remain separate. Vocabulary aliases, name guards and
decoding were not changed. Uncertain names such as Medium's `Gable` were not
rewritten to Keval. Successful normalization on some samples is not evidence
that the recognizer heard every word correctly.

## Latency and GPU memory

RTX 3050 Laptop GPU, 4096 MiB. Each model was loaded once in its own sequential
process and reused for 55 transcriptions. No simultaneous model residency,
CUDA error, OOM or CPU fallback occurred. Every scored row was checked to confirm
actual CUDA execution and the expected compute type.

| Configuration | Fresh-process model load s | Sampled total-device GPU peak MiB |
| --- | ---: | ---: |
| Small / float16 | 4.003 | 787 |
| Medium / int8_float16 | 5.759 | 1363 |

| Corpus | Small warm median s | Medium warm median s |
| --- | ---: | ---: |
| English, 20 warm calls | .392 | .847 |
| Hindi, 20 warm calls | .649 | 1.429 |
| Hinglish, 4 warm calls | .854 | 1.833 |

Warm latency excludes pass 1 for each sample. The long Hindi sample's Medium
warm range was 10.289–11.446 s, median 11.134 s; the language median does not
represent that long utterance. Its VAD-retained input was 31.384 s. The earlier
roughly 1235 MiB figure was not a fixed upper bound: this larger corpus reached
1363 MiB sampled device usage.

Loading is a fresh model instance in a new process, with potentially warm OS
file cache. GPU usage includes other processes and is sampled, not an exact
allocation peak. Benchmarks used cached weights and `--require-cuda`; CPU fallback
was not timed as GPU performance.

## Recording and language preservation

All 11 primary recordings ended through the existing silence endpoint, had no
clipped samples and no end-truncation flag, and were fully playback-verified.
The pause samples include the words after their intended short pauses. The long
Hindi recording exercises more than 30 seconds of VAD-retained speech. These
facts describe the measured samples, not a general guarantee about quiet syllables
or all recording environments.

An inaudible `winsound.Beep` session cue was replaced with a user-confirmed WAV
chime in the diagnostic helper. Its chime can be present in raw recordings and
is retained identically for both models. The helper's short-take safety cap was
set to 20 s; the long-take cap remained 60 s. Automatic silence stopping stayed
enabled, and the application's recording logic/settings were not changed.
The separate minute-long unverified retry demonstrates why capture quality and
reference identity still need checking even when signal diagnostics look usable.

Both models retained Hindi-script output on the Hindi corpus and English on
English; the retained Hinglish sentence kept Hindi plus Latin technical words.
No entirely English translation of Hindi appeared in these primary runs.
`task="transcribe"`, automatic language, multilingual windows, beam 5, no prompt,
temperature 0 and the existing ASR VAD were retained. Language probabilities were
not used as accuracy scores. No translation/LLM rewriting was introduced.

## Scope, checks and recommendation

Production speech source, model defaults, private environment settings,
normalization, notes/reminders/tasks, router and storage were not modified.
No wake-word detector was implemented. Existing action validation and transcript
confirmation remain in place; this recording/benchmark workflow sends no speech
to actions or Gemini. Model reuse and architecture remain as previously validated.

The full offline suite and Ruff were run during this round:

```text
python -m pytest -q       295 passed, 1 deselected
python -m ruff check .    All checks passed!
```

The session-only helper was formatted/linted separately. Afterwards, all
production source/database hashes were compared with the pre-round audit and
remained unchanged. All 110 WER/CER rows were independently recomputed from their
human references; waveform hashes, reference files, repeatability and model reuse
checks passed. `completion-audit.json` records the checks.

Medium is the stronger candidate for this speaker's tested Hindi and retained
Hinglish, with substantial residual errors and a latency cost. It supports
**limited broader opt-in trials with transcript review/editing**. Do not treat
these results as general multilingual readiness, change the requested global
default, or infer reliability for unseen speakers, microphones, noise, spontaneous
code-switching or commands sensitive to names/numbers. Five recordings per
language and one known Hinglish sentence are a small, prompted, single-speaker
pilot. More independent mixed-language and real-use recordings are needed for
a broader accuracy claim.

## Reproduce

Use the frozen primary-only manifest to avoid accidentally counting diagnostic
excerpts or unverified takes. Run models sequentially and choose new output
filenames to preserve these measurements:

```powershell
python -X utf8 -m audio.benchmark run --manifest benchmark-recordings/round2-20261116/primary-session.json --stt-profile medium-candidate --profiles baseline --repeat 5 --gpu-memory --require-cuda --vad-audit --output stt-round2-medium-repeat.json
python -X utf8 -m audio.benchmark run --manifest benchmark-recordings/round2-20261116/primary-session.json --stt-profile medium-candidate --model small --compute-type float16 --profiles baseline --repeat 5 --gpu-memory --require-cuda --vad-audit --output stt-round2-small-repeat.json
python -X utf8 -m audio.comparison_report --reports stt-round2-medium-repeat.json stt-round2-small-repeat.json --output ROUND2_COMPARISON_REPEAT.md
```
