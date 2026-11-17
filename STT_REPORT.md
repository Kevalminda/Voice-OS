# STT audit and verification — 15 November 2026

Follow-up language selection audit and current verification:
[STT_LANGUAGE_REPORT.md](STT_LANGUAGE_REPORT.md). The measurements below remain
the original STT subsystem baseline.

## Diagnosis before implementation

All project Python source, existing tests, requirements, configuration and project
documentation were inspected before edits. No AGENTS.md was found in the workspace
or its ancestor directories. The private .env contents, notes database and backup
project were not inspected or modified for this work.

The original audio class used multilingual small, eager CUDA/float16 model loading,
16 kHz mono float32 arrays, device ID 1 and five-second capture. It flattened audio
but did not check input capabilities, silence, failures, levels or cut-off speech.
There was no CPU fallback, timing, structured status or normalization. Each instance
loaded its own model; creating an instance per command would reload weights.

The root module test forced Hindi with no prompt; the other root script used auto
language and an English technical-word prompt. Both captured and loaded a model at
import time, so pytest collection could trigger microphones, GPU loads and (for an
uncached model) downloads. They did not share test audio, so their results were not
a controlled comparison. The larger standalone script rewrote test_audio.wav.

Both requested transcription (one through the library default); no explicit
translation or normalization implementation existed. VAD was enabled with default
parameters and no diagnostics. Silence/VAD removal, short recordings, ambiguous
language selection and limited model/code-switching accuracy are plausible failure
sources. The exact cause of every historical empty result cannot be established
without those recordings. Forcing a language changes decoding behavior; neither
`task=transcribe` nor language probability guarantees the desired script or words.
Technical-word errors are visible on the supplied clip. CUDA affects execution
performance, not an automatic accuracy upgrade.

Initial lint: 2 unsorted import blocks. Existing regression baseline (excluding the
two unsafe audio scripts): **78 passed, 1 live test deselected**. The sandbox could
not access the Store Python executable; running the existing interpreter outside
the sandbox confirmed Python 3.11.9. The virtual environment was not rebuilt.

## Implementation

- Validated environment/dataclass configuration, OS default input, enumeration and
  configurable durations (12-second default, maximum 120); strict 16 kHz mono float32.
- RMS/peak/clipping/end-activity diagnostics; empty/near-silent audio bypasses model
  inference. Failed, short and overflowed recordings return useful errors.
- One lazy reusable model per instance; automatic CUDA/CPU compute selection,
  bounded CPU fallback for initialization/lazy CUDA failures, cached permanent
  initialization errors. No loop retries missing weights or model initialization.
- Explicit en/hi/auto with transcribe task and configurable vocabulary-only prompt,
  VAD padding/silence settings and no previous-text conditioning.
- Conservative Unicode phrase correction with raw transcript and correction audit;
  ambiguous names require adjacent configured full-name context and a name-introduction
  cue; phrases meaning "only Minda" are preserved. No Gemini fallback.
- One record/replay diagnostic, shared immutable samples across modes, separate
  timing fields, optional GPU memory sampling and JSON reports. Both old manual
  filenames forward to it and are safe to import.
- Optional CLI voice mode reviews raw/normalized text; empty/truncated/declined
  input is not submitted. Accepted text enters the same process_text_command().
  Existing schemas, NLU, validation, router, SQL and deletion controls are unchanged.
- Speech dependencies match installed versions; no dependency upgrade was performed.
  Added .env.example without reading/replacing private values. Recordings/reports
  are ignored by git. No external messages, API calls or application actions were
  executed by the hardware diagnostics.

See README for the final directory tree, API, settings, installation and exact run
commands. Historical AUDIT.md/VERIFICATION.md describe the previous text milestone.

## Actual hardware execution, separate from mocked tests

Windows, Python 3.11.9; faster-whisper 1.2.1, CTranslate2 4.8.5, sounddevice 0.5.6,
NumPy 2.4.6. NVIDIA RTX 3050 Laptop GPU, 4096 MiB, driver 610.78. CUDA reported one
device and float16/int8_float16 support. Actual inference succeeded on CUDA float16,
CUDA int8_float16 and CPU int8. This exercises working inference libraries, beyond
merely detecting the GPU. Microphones were enumerated, including input and output
devices; no new microphone recording or live Gemini request was made.

The existing test_audio.wav was replayed without overwriting it or downloading
models: 80000 samples, 5 seconds, PCM16, mono, 16000 Hz. RMS 0.014684, peak 0.459015,
no clipping and silent final 200 ms. Decoded float32 sample SHA-256:
`0e2c91edbfac75cc290dc41582173199627cfd7fe2094bcccc72b27d085f6f74`.

Original code was measured before modification on this same array and cached small
model with CUDA float16, beam 5, default VAD and no prompt. Each mode ran once:

| Original measurement | Seconds |
| --- | ---: |
| Model constructor | 0.892 |
| Forced English decode | 0.625 |
| Forced Hindi decode | 0.502 |
| Automatic decode | 0.636 |

Original forced English returned “My name is Cable Minda. I am working on a
voice-over project.” Hindi/auto used Devanagari with name/technical-word errors.
Both forced modes returned language probability 1.0; auto selected hi at 0.95264.
There is no independently annotated reference corpus, so no WER/accuracy figure
is asserted.

Updated pipeline used the same small model and audio with configured VAD/prompt,
three passes on GPU and two on CPU. First call loaded the model; later calls reused
it. Recorded reports remain locally as stt-small-fp16.json, stt-small-cuda.json
(int8_float16 profile), and stt-small-cpu.json. All reported times are measured:

| Updated profile | Load seconds | First en decode seconds | Warm en seconds | Warm hi seconds | Warm auto seconds | Sampled total GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA float16 | 1.560 | 0.738 | 0.251–0.255 | 0.543–0.562 | 0.651–0.720 | 795 |
| CUDA int8_float16 | 3.591 | 2.067 | 0.270–0.300 | 0.638–0.642 | 0.717–0.734 | 507 |
| CPU int8 | 1.646 | 1.737 | 1.515 | 2.147 | 3.393 | not sampled |

Float16 was preferred for automatic CUDA selection after measuring these profiles:
small fits within the 4 GB budget and generally decoded faster here. int8_float16
remains explicit/configurable for memory experiments. Automatic CPU selection is
int8. Medium/distil-small.en were not cached, downloaded or measured. No larger
default was selected; README gives opt-in commands to benchmark them on the same
audio, with sampled memory and warm passes. distil-small.en is English-only.

Updated normalization took 0.082–0.205 ms on the float16 run. Warm float16 total STT
wall time was 0.251–0.256 s (en), 0.544–0.562 s (hi), 0.652–0.721 s (auto). First
English total was 2.299 s including loading. File replay recording_time is 0, while
recording_duration is 5 s; microphone capture latency was not measured. Interactive
capture waits the configured fixed duration plus device overhead. Review/Gemini/
actions are outside these STT timings. The int8_float16 trial initially used auto
compute selection; the final auto policy was then changed to measured float16.

These are small-sample observations, not a rigorous speedup study. Original load
timing excluded imports already performed at module import; updated lazy loading
includes them. Runs also differ in prompt/VAD/conditioning, warm state, monitoring
overhead and generated text length. The original baseline had only one pass per
mode. No overall before/after speedup or accuracy improvement is claimed. GPU
memory was sampled at intervals and includes other processes; transient peaks may
be missed and these values are not model weight sizes.

The updated English result still contains “Cable Minda”; Hindi still contains
incorrect name/project spellings. A configured VoiceOS alias/case is normalized,
but uncertain names and malformed words are preserved. This is intentional
evidence of remaining model limitations, not a claim of natural mixed-script
recognition for arbitrary speech. Gather reference recordings before tuning the
vocabulary or switching models.

## Automated verification

Offline tests mock model/microphone/LLM calls and isolate SQLite. Coverage includes
empty/near-silent/invalid/valid audio, bad input devices and formats, overflow/short
recordings, model reuse and initialization failures, CUDA initialization/lazy
fallback, explicit/automatic languages, empty model output, identical audio across
modes, vocabulary boundaries/context/raw preservation, voice submission review,
existing processor compatibility, clarification/rejection and delete approval/
cancellation. Existing tests were retained unchanged.

Final verification with the workspace interpreter:

```text
.\.venv\Scripts\python.exe -m pytest -q
158 passed, 1 deselected in 3.36s

.\.venv\Scripts\ruff.exe check .
All checks passed!
```

The deselected test is the existing opt-in live Gemini test. Ruff formatting check
passed for all 12 changed/new Python files. Compilation of audio/, tests/ and CLI
entry points passed. CLI --help and --list-devices worked; both manual audio modules
imported without capture or model load. No fresh mic speech was supplied.
Database and original WAV SHA-256 hashes matched before/after final verification:
storage/voiceos.db `d59667a69e74ff737354c0eab356231592733ac0b45c4681a90de2c6871ae219`;
test_audio.wav `8b9074714649e3829a57d441c99ba73b184662b4b692ee8ac2eac96dc0d515ab`.

## Manual completion and limitations

Fresh microphone recognition, separate English/Hindi/Hinglish accuracy, medium/
distilled memory/latency, live Gemini integration and human reference scoring remain
unverified. No physical speech can be supplied autonomously. Follow the README
record-once commands and compare modes against your own written references.

Fixed windows can still cut speech off; end-RMS detection only warns of likely
cutoff and can false-positive on noise. Leave trailing silence and extend duration.
RMS threshold and VAD may suppress quiet speech; compare settings on identical WAVs.
Normalization only corrects configured full phrases. It is not semantic validation,
translation, guaranteed name recognition, or a replacement for review. Longer model
prompts can bias/hallucinate vocabulary; keep hints short and evaluate them. Hindi
script and English word retention remain model-dependent. Native DLL failures
that terminate Python cannot be caught by Python fallback; use explicit CPU mode
and repair the runtime if the process exits abruptly. Local OS user access remains
the authorization boundary; no new authentication layer was introduced.

Primary setup references: [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper),
[Distil-Whisper](https://github.com/huggingface/distil-whisper).
