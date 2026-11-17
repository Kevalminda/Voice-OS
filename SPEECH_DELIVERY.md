# VoiceOS speech module — delivery

17 November 2026. **Ready for a supervised speech demonstration.** VoiceOS is
middleware on an existing operating system. The delivered scope is recording,
multilingual STT, conservative normalization, review and safe integration.
Recognition is not generally error-free; correction/rejection remains essential.

## Implemented and frozen

| Component | Actual implementation |
| --- | --- |
| Capture and silence | `audio/recording.py`: sounddevice stream, cached Silero VAD, sample-based endpoint, initial timeout and safety maximum |
| STT and reuse | `audio/speech_to_text.py`: reusable Faster-Whisper instance, `task="transcribe"`, CUDA selection and CPU float32 fallback |
| Language/settings | `audio/audio_config.py`, `audio/stt_profiles.py`: auto/en/hi and the opt-in Medium profile |
| Normalization | `audio/transcript_normalizer.py`: explicit configurable vocabulary, separate raw/normalized text, guarded names and review hints |
| Application/review | `main.py`: load once, activate, capture, display, yes/no/edit, then the existing command boundary |
| Capture preservation | `audio/capture_archive.py`: unique lossless NPY and JSON before inference when `--save-recordings` is configured |

Small remains the global default. Medium remains opt-in: CUDA/int8_float16,
automatic language, multilingual windows, beam 5, no prompt, temperature 0 and
the validated VAD. Default capture remains automatic with two seconds of
silence, eight seconds of initial silence timeout and a 60-second safety cap.
[Configuration snapshot](SPEECH_CONFIGURATION_FREEZE.json) records the freeze;
runtime still reads the existing configuration/CLI.

## Delivery changes

- `audio/transcript_normalizer.py`: informational review flags for name
  introductions/ambiguous configured terms and detected numbers. Values and
  uncertain spellings are not repaired. Explicit vocabulary behavior is retained.
- `audio/speech_to_text.py`: serializable review metadata in the existing result
  and preserved transcript; no decoding/capture configuration change.
- `main.py`: display review hints, use UTF-8 for redirected voice output, and add
  explicit `--speech-only` demonstration mode. Normal command mode retains the
  existing allowlist, validation and deletion confirmation.
- `tests/test_transcript_normalizer.py`, `tests/test_speech_to_text.py`,
  `tests/test_voice_integration.py`: review/value preservation, serialization,
  Unicode output, action rejection, microphone retry and shutdown regression tests.
- [Demo procedure](SPEECH_DEMO.md), this delivery note, the configuration snapshot
  and a README link. A session-only live test wrapper and its artifacts remain
  under `benchmark-recordings/delivery-20261117/`.

The baseline passed before changes. Only the gaps above were implemented; working
recording, model selection, VAD, routing and action modules were retained. No
wake-word, notes, reminders, tasks or new system-control action was implemented.

## Actual hardware check

One real production voice loop used the frozen Medium profile on the RTX 3050
Laptop GPU. The test wrapper supplied an audible WAV start cue and guarded the
action/storage boundaries; microphone input and inference were real. It used a
20-second per-launch safety cap for these short checks, leaving global defaults
unchanged. Operator choices exercised the actual yes/no/edit prompts.

| Live input | Captured s | Decode s | Observed result |
| --- | ---: | ---: | --- |
| English | 6.272 | 2.479 | `Restart the computer in 27 minutes.`; number flagged; user confirmed; action rejected |
| Hindi | 5.344 | 1.256 | Raw `मेरा नाम केवल मिंडा है।`; configured normalization `मेरा नाम Keval Minda है।`; name flagged; operator exercised edit; action rejected |
| Hinglish | 7.648 | 1.321 | Raw `आज में voiceOS में python और Gemini इस्तमाल कर रहा हूं।`; technical spellings normalized; operator declined |
| Quiet input after cue | 8.000 | 0 | Initial silence timeout; `no_speech`; no confirmation or action submission |

All three spoken captures ended on silence. The model loaded once in **5.441 s**
and was reused across all four activations. Quiet input skipped inference. Sampled
total-device GPU peak was **1235 MiB**, including other processes; this is not an
exact allocation peak or a universal upper bound. The microphone closed after
each capture; normal `exit` ended the app and released model resources.

Two confirmed/edited transcripts received explicit local `[rejected]` results;
the declined and quiet cases submitted nothing. Observed command-processor calls
and database initialization calls were both **zero**. This demonstrates the
speech-only capability gate, not offline semantic intent classification.

Evidence: `stt-delivery-live.json`, unique NPY/JSON captures under
`benchmark-recordings/delivery-20261117/live-captures/`, and `delivery-checks.json`.
These are functional smoke results, not a new WER/CER benchmark. No reference was
inferred from the recognizer output. Hinglish spelling differences remain visible
and unchanged; no entire-sentence translation appeared in these live samples.

## Verification and preservation

```text
python -m pytest -q       308 passed, 1 deselected
python -m ruff check .    All checks passed!
```

Targeted formatting and Python 3.11 compilation passed. The existing error/noise,
empty/long input, safety timeout, CUDA fallback, validation and confirmation tests
remain, with new retry/shutdown tests. CPU fallback is covered by offline
regressions; a physical GPU failure was not induced during the live demo.

Hashes verified **468 existing benchmark artifacts unchanged**. The new normalizer
produced exactly the same normalized text on all **110 archived Round 2 outputs**;
only review metadata was added. No ASR comparison was rerun. Both frozen runtime
configurations match the pre-change snapshot. Action, NLU, router and storage
sources were unchanged, and the original backup project was not modified.

## Launch and demonstrate

From the existing project root:

```powershell
# Reliable local speech demonstration with the opt-in Medium candidate:
.\.venv\Scripts\python.exe main.py --voice --speech-only --stt-profile medium-candidate --save-recordings benchmark-recordings/submission-demo-medium

# Use the unchanged Small default instead:
.\.venv\Scripts\python.exe main.py --voice --speech-only --save-recordings benchmark-recordings/submission-demo-small
```

Follow [SPEECH_DEMO.md](SPEECH_DEMO.md) for English, Hindi, Hinglish, automatic
termination, review/editing, unsupported-action rejection, quiet input and exit.
Every application action is disabled in `--speech-only`; Gemini and storage are
not needed. Omit that flag only when deliberately using the existing configured
Gemini-backed command processor. Its service availability is an external
dependency and was not exercised by this local speech demonstration.

## Known limits and submission readiness

- Review flags are conservative, non-exhaustive hints, not confidence scores or
  a reliable named-entity/number parser. Ordinary ambiguous words can be flagged;
  absence of a flag never bypasses confirmation.
- Hindi spelling, names, numbers and code-switching remain imperfect. Historical
  [Round 2](ROUND2_VALIDATION_REPORT.md) measured Medium WER of 16.47% English,
  38.24% Hindi and 21.05% on one retained Hinglish recording. Those small-corpus
  results remain frozen and do not establish general multilingual accuracy.
- Background voices may prolong capture until the safety cap. Safety-limited or
  potentially truncated captures are withheld from command submission. Inspect
  the selected input and use a quiet demonstration environment.
- Longer dictation costs more latency; the earlier long-Hindi warm median was
  11.134 s and corpus GPU peak 1363 MiB. This delivery makes no accuracy or speed
  improvement claim beyond the functional fixes.

The speech module is ready to submit as a **working, supervised multilingual
interaction demonstration with explicit limitations and preserved evidence**.
It is not an autonomous operating system, a wake-word assistant, a new task
manager, or a guarantee of accurate unattended application control.
