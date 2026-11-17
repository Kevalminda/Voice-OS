# Medium candidate validation — 15 November 2026

Rechecked on 16 November without source or decoding changes:
[current measurements and recommendation](MEDIUM_CANDIDATE_RECHECK_20261116.md).

**Recommendation: retain Medium int8_float16 as an opt-in candidate; do not make
it the global default yet.** The confirmed Hinglish improvement reproduced, the
candidate runs through the actual application, and observed memory fits the
4 GB GPU. English/Hindi accuracy is still unmeasured, the one Hinglish sentence
still has four word errors, and Medium's warm latency varies substantially.

No new decoding search, prompt tuning, vocabulary substitutions, translation
step, wake-word implementation, notes/reminders/tasks, or action-routing changes
were introduced. The candidate's settings are the settings already measured in
the investigation.

## Selection and application configuration

The plain application was audited by loading its effective speech settings,
including `.env`, without displaying credentials. It still selected `small`,
device/compute `auto`, language auto, beam 5, empty prompt and the existing VAD.
Its temperature fallback list was `0,.2,.4,.6,.8,1`, whereas the controlled
candidate used `0`. Thus an unqualified `python main.py --voice` did **not**
already use the Medium candidate. The private `.env` and global defaults remain
unchanged pending validation.

The shared opt-in profile `medium-candidate` now selects:

| Setting | Value |
| --- | --- |
| Model | Existing local Faster-Whisper Medium cache, or model name `medium` |
| Execution | CUDA / int8_float16, with the application's existing CPU fallback |
| Language | Automatic; multilingual decoding windows enabled |
| Task | `transcribe` in the existing decoder |
| Beam / prompt / temperature | 5 / none / 0 |
| ASR VAD | Enabled; threshold .5, minimum silence 500 ms, speech padding 300 ms |
| Capture | Existing automatic recording; current defaults: 2 s silence, 8 s initial timeout, 60 s safety maximum |

Both `main.py` and `audio.benchmark` use `audio/stt_profiles.py`, so the candidate
definition does not depend on copying a sequence of environment assignments.
Explicit CLI overrides still take precedence and can intentionally depart from
the candidate; the startup display shows the effective model, actual execution
device/compute, language, beam, VAD status and whether a prompt is configured.

```powershell
.\.venv\Scripts\python.exe -X utf8 main.py --voice --stt-profile medium-candidate --save-recordings benchmark-recordings/voice-captures
```

The existing local Medium weights were used; validation required cached files
and performed no model download. `-X utf8` also supports redirected Hindi output.
One initial application replay with the Windows pipe's default encoding stopped
when printing Hindi; its already saved audio/transcript remain in the archive.
The successful replay used UTF-8. This was an output-encoding failure, not an
empty recognition result or a changed waveform.

## Actual application replay

`python -X utf8 -m audio.validation` runs the real application voice loop with
the saved waveform substituted for microphone input. It uses real Whisper,
Silero, normalization, archiving and transcript review. Terminal answers decline
both transcripts; database initialization is stubbed and the command boundary
is guarded. This is explicitly **not** a new physical microphone test or a
measurement of microphone capture latency.

Measured artifact: `stt-medium-validation-app.json`.

- Actual execution: **CUDA / int8_float16**, one model creation, two transcriptions.
- Model load: **5.590 s**; decode calls **2.176 s / 1.944 s**.
- Sampled total-device GPU peak: **1233 MiB**.
- Command submissions: **0**; both displayed transcripts were declined.
- Both unique saved NPY archives equal the original waveform's hash; raw and
  normalized texts are retained separately in their JSON metadata.
- Replay archives are labeled `saved_waveform_replay`, avoiding a false claim
  that repeated model comparisons are fresh microphone recordings.

The actual model output for both calls was:

> आज मैं voiceOS प्रोजेक पे Python और Gemini इस्तमाल कर रहा हूँ और BBMS का काम अभी बाकी है।

The normalizer only canonicalized `voiceOS` to `VoiceOS`. It did not repair the
remaining errors or rewrite the Hindi sentence.

## Corpus and independently repeated comparison

Manifest: `benchmark-recordings/medium-validation/session.json`. Only the existing
human-verified, playback-confirmed Hinglish reference was copied. The source audio
was referenced at its preserved path and was never overwritten. As requested,
English and Hindi remain empty/unscored. The withdrawn background-Hindi recording
was not used to calculate accuracy.

| Corpus language / exact sample | Verified recordings | Small raw WER / CER | Medium raw WER / CER |
| --- | ---: | --- | --- |
| English / pending | 0 | Unavailable | Unavailable |
| Hindi / pending | 0 | Unavailable | Unavailable |
| Hinglish / `hinglish_technical.npy`, 12.832 s | 1 | 78.95% / 58.33% | **21.05% / 15.28%** |

Verified reference:

> आज मैं VoiceOS project में Python और Gemini इस्तेमाल कर रहा हूँ, और DBMS का काम अभी बाकी है।

Decoded float32 SHA-256:
`1adf58c7904830c398a02ca2145e9a4514a51968b4c0691e950f56cfc6add5c2`.

Both models received this exact waveform five times, in separate sequential
processes. Every pass reproduced its model's prior WER/CER. Normalized WER/CER
equaled raw WER/CER for these baseline results. Five repeats are still **one
independent sample**, not five verified recordings. No model-generated output
was accepted as a reference.

The benchmark now includes `summary_by_language`, `summary_by_sample` and
`coverage_by_language`. Corpus language labels come from manifest categories,
not Whisper's detected language; Hinglish is not incorrectly aggregated into
Hindi merely because the decoder selected `hi`. WER/CER aggregate error counts
against reference lengths rather than averaging percentages. Missing references
produce null metrics, including in the generated comparison table.

[Complete per-pass table, raw/normalized text and coverage](MEDIUM_CANDIDATE_COMPARISON.md).
JSON sources: `stt-medium-validation-candidate.json` and
`stt-medium-validation-small.json`.

| Model / compute | New-process model load s | First decode s | Passes 2–5 median s | Passes 2–5 range s | Sampled GPU MiB |
| --- | ---: | ---: | ---: | --- | ---: |
| Small / float16 | 2.246 | 2.058 | 0.997 | 0.983–1.005 | 785 |
| Medium / int8_float16 | 5.345 | 2.219 | 3.637 | 1.872–5.368 | 1265 |

These are fresh-process model initialization timings, not guaranteed cold-disk
measurements: Windows may cache model files. Warm latency excludes pass 1 for
each sample/profile; model loading and VAD audit time are separate. GPU figures
are sampled total device memory, include other processes and may miss a brief
peak. No CUDA fallback/OOM occurred in these validation runs. The `--require-cuda`
benchmark flag rejects CPU fallback rather than reporting it as GPU performance.
The hardware was the RTX 3050 Laptop GPU with 4096 MiB VRAM. No latency improvement
over Small is claimed.

## Recording, language preservation and architecture

Real streaming Silero plus the existing endpoint state machine were replayed on
the preserved waveform: speech started, stopping reason `silence`, retained
duration **12.832 s**, trailing silence **2.016 s**. Nine seconds of digital zero
input reached the initial silence timeout after **8 s**, with speech never
started. This digital signal is labeled synthetic and is not an English/Hindi
accuracy sample. Existing offline stream tests also cover short pauses, noise,
overflow/stall handling, final sample retention and the maximum-duration limit.
No physical microphone was opened during this follow-up.

The decoder still passes `task="transcribe"`, language `None` for automatic mode,
and no initial prompt. There is no LLM-based transcript rewriting. All candidate
outputs in this validation retained Hindi grammar and Latin technical words;
none became an entirely English translation. The prior forced-English control
did paraphrase in English despite transcription mode, so `task="transcribe"`
alone cannot guarantee preservation on unseen code-switching speech. Detection
probability remains diagnostic and is not used as an accuracy score or automatic
reason to force a language.

Wake-word integration remains architecturally possible: activation is separate
from reusable `SpeechToText`, `record_audio`, `transcribe(audio)` and transcript
review. A future trigger can activate capture, or a shared-stream adapter can
hand a waveform to the existing transcriber. Continuous listening would require
deliberate microphone ownership, buffering and handoff; opening competing streams
must not be assumed to work. No wake detector, always-listening process, new
dependency or bypass of transcript/action confirmation was implemented.

## Repeat the validation or add later verified samples

Use the existing three-language manifest. Future English/Hindi recording is
optional and requires an operator: record once into the empty slot, wait for the
local `Listening` cue, speak, then allow silence. Listen to the saved WAV before
writing a UTF-8 reference containing **every actual word**, including restarts.
Only then run `verify`. Suggested text is never pre-populated as ground truth.
Use different slots/new manifests for further independent recordings; record
commands refuse to overwrite existing samples.

```powershell
# Optional later capture; change english to hindi for the other empty slot:
python -X utf8 -m audio.benchmark record --automatic --manifest benchmark-recordings/medium-validation/session.json --case english
# After playback and checking the exact words in an independently written file:
python -X utf8 -m audio.benchmark verify --manifest benchmark-recordings/medium-validation/session.json --case english --reference-file english-reference.txt

# Run sequentially; keep the same manifest and profiles for both models:
python -X utf8 -m audio.benchmark run --manifest benchmark-recordings/medium-validation/session.json --stt-profile medium-candidate --profiles baseline --repeat 5 --gpu-memory --require-cuda --vad-audit --output stt-medium-validation-candidate.json
python -X utf8 -m audio.benchmark run --manifest benchmark-recordings/medium-validation/session.json --stt-profile medium-candidate --model small --compute-type float16 --profiles baseline --repeat 5 --gpu-memory --require-cuda --vad-audit --output stt-medium-validation-small.json
python -X utf8 -m audio.comparison_report --reports stt-medium-validation-candidate.json stt-medium-validation-small.json --output MEDIUM_CANDIDATE_COMPARISON.md

# Replay through the actual reviewed application, with microphone/actions substituted:
python -X utf8 -m audio.validation --manifest benchmark-recordings/medium-validation/session.json --case hinglish_technical --archive benchmark-recordings/medium-validation/application-captures --output stt-medium-validation-app.json
```

Before default promotion, collect independently verified English/Hindi and more
natural Hinglish samples, including names, technical terms, different speakers,
noise and pauses. Agree on acceptable command-word errors and latency, then
evaluate those recordings without tuning against their references. Current
evidence supports using the candidate for reviewed trials, not a general accuracy
or no-translation guarantee.

## Scope and verification

Changed: `audio/stt_profiles.py` (shared measured preset), `main.py` (opt-in
selection and effective execution display), `audio/benchmark.py` (selection,
language/sample coverage and warm timing), `audio/comparison_report.py` (coverage
table), `audio/validation.py` (explicit saved-audio application/endpoint replay),
speech regression tests and documentation. Architecture, recording/decoder
implementation, normalizer, action router and storage logic were retained.

The complete offline suite and Ruff were run separately from manual GPU replay:

```text
python -m pytest -q       295 passed, 1 deselected
python -m ruff check .    All checks passed!
```

Targeted formatting and Python 3.11 compilation checks also passed. Original
database and supplied WAV hashes remained unchanged. Pytest uses mocked recognition and
temporary databases; it is not evidence of live multilingual accuracy. Existing
review, deletion confirmation, allowlisted actions and unknown-command rejection
remain covered; raw transcription gains no shell, arbitrary SQL or system-action
execution capability.
