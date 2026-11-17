# VoiceOS

VoiceOS is a multilingual voice interaction **middleware layer**, not an operating system.
It supports the existing text-to-note pipeline plus local microphone transcription
with reviewed voice commands. Reminders, web search and TTS remain deferred.

For submission, use the [speech delivery notes](SPEECH_DELIVERY.md) and
[repeatable local demonstration](SPEECH_DEMO.md). `--voice --speech-only` demonstrates
the real recording, STT and review flow while rejecting all application actions
without Gemini or database use. The Small default and Medium opt-in remain unchanged.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Only create .env if it does not already exist:
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Set `GEMINI_API_KEY` privately in `.env`. Run `python list_models.py` to discover
models available to your account, then set `GEMINI_MODEL` to a model that supports
structured JSON. No model name is assumed to be available. Discovery and normal
NLU use network access and your existing Gemini account/quota; no live request
was made during this delivery. A free quota is not guaranteed by this software.

On Linux/macOS use `python3.11 -m venv .venv` and `source .venv/bin/activate`.
Run from the extracted project root (the directory containing `main.py`):

```text
python main.py
```

The database initializes automatically. Enter `exit`, Ctrl+C or EOF to stop.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| GEMINI_API_KEY | required | Your existing Gemini credential; never commit it |
| GEMINI_MODEL | required | Model selected from your account's available models |
| GEMINI_TIMEOUT_MS | 15000 | Per-request timeout; 100–120000 milliseconds |
| GEMINI_MAX_ATTEMPTS | 3 | Total attempts including first request; 1–5 |
| VOICEOS_DATABASE_PATH | storage/voiceos.db beside source | Optional SQLite path; relative overrides resolve from working directory |
| VOICEOS_LOG_LEVEL | INFO | Python application logging level |

The SDK is pinned to the inspected version, 1.75.0. It is configured for one SDK
attempt, so application retries do not multiply SDK retries. Temporary HTTP
408/429/500/502/503/504 and transport failures retry with bounded exponential
waits. Authentication and invalid-request failures do not retry. Exhausted quota
can also return 429; retries stop at the configured cap. No unbounded loop exists.

## Supported commands

- `Save a note that my DBMS exam is tomorrow.`
- `Show my notes.`
- `Update note 1 to buy groceries on Friday.`
- `Delete note 1.`
- `एक नोट बनाओ कि कल मेरी परीक्षा है।`
- `Mera note 1 delete karo.`

Language understanding depends on Gemini. IDs must be positive SQLite integers.
Low confidence, missing details, and unknown commands do not execute. A deletion
checks existence, displays the ID and up to 120 preview characters, and asks for
`yes`/`y`. Anything else cancels. The command is parsed only once. If the note's
content changes during confirmation, deletion fails and must be requested again.

## Architecture

Current path: terminal text → NLU service → Gemini structured JSON → strict parser
and Pydantic schema → confidence/required-field validation → controlled router →
parameterized SQLite note actions → user response.

- `core/schemas.py`: command/result types and entity normalization.
- `core/validation.py`: confidence, required details and ID range checks.
- `core/action_router.py`: allowlisted actions and deletion confirmation boundary.
- `core/command_processor.py`: orchestration and sanitized user errors.
- `nlu/`: extraction, SDK configuration and validated parsing.
- `actions/notes.py`: local note CRUD and atomic checked deletion.
- `storage/database.py`: configurable connections and initialization.

Integration callers pass `confirm_action(pending: ActionResult) -> bool` to the
processor/router. Only the literal boolean `True` approves deletion. This callback
signature now supplies the preview; old zero-argument callbacks must be updated.
Never expose low-level storage functions directly as LLM tools. They are trusted
internal CRUD primitives and deliberately remain usable in unit tests.

Voice commands use the same `process_text_command()` entry point and all existing
validation/confirmation controls. The prototype relies on the local user's OS
account; application authentication and TTS are not implemented.

## Speech setup and use

Use Python 3.11 and the workspace `.venv`. Requirements include the already
installed Faster-Whisper 1.2.1, CTranslate2 4.8.5, sounddevice 0.5.6 and NumPy.
No PyTorch installation is needed. The voice CLI calls `load_model()` once at
startup, prepares the cached streaming Silero VAD, and reuses both. Library callers
can load explicitly or retain lazy loading on their first nonsilent clip. An uncached model can
download from Hugging Face; set `VOICEOS_STT_LOCAL_FILES_ONLY=true` to prohibit it.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --list-devices
python main.py --voice --language auto
# Optional explicit language/device selection (use an ID from your own device list):
python main.py --voice --language en --microphone "Microphone Array"
python main.py --voice --language hi
# Existing text interface:
python main.py
```

Press Enter to activate the microphone. Wait for **Listening**, then speak naturally.
Capture continues through pauses shorter than two seconds and stops after two
seconds of continuous VAD-confirmed silence. The initial silence timeout is eight
seconds and the safety maximum is sixty seconds. These are independent settings.
The stream keeps all samples from activation, including leading/trailing audio.
Review the raw transcript and any spelling corrections;
choose `yes`, `no`, or `edit`. Only reviewed text reaches NLU. Delete commands
still require the separate existing deletion preview and confirmation. Empty
speech and captures stopped by the safety maximum are not submitted. VAD-confirmed
silence completion takes precedence over the RMS tail heuristic, so background
noise is not automatically interpreted as a cut-off sentence. VAD can still miss
quiet speech or treat background voices as speech. Configure a longer silence
threshold for thoughtful pauses:

```powershell
python main.py --voice --silence-seconds 3 --max-duration 90 --initial-timeout 10
```

`--duration` is retained as an alias for `--max-duration` in the voice CLI; it now
sets a safety limit. `VOICEOS_RECORD_SECONDS` applies to fixed diagnostic capture.
The maximum is at most 120 seconds and initial timeout must not exceed it.
See [AUTOMATIC_RECORDING_REPORT.md](AUTOMATIC_RECORDING_REPORT.md) for current
results and a microphone validation procedure. Live validation remains pending.

English (`en`), Hindi (`hi`) and automatic (`auto`, Python `None`) are supported.
Hinglish uses multilingual `small` or `medium`; it is not a Whisper language code.
Start with `auto` and compare `en`/`hi` against the actual speech rather than
forcing every Hinglish utterance into one language. Short/code-switched recordings can confuse language
detection. `task="transcribe"` requests original-language output but cannot
guarantee its script or prevent model-level translation-like errors. Language
probability is not transcript accuracy; forced modes may report 1.0.

The native defaults are auto language, per-segment language detection enabled,
and an empty initial prompt. This removes the previous implicit English vocabulary
hint. Explicit settings can override them. Hindi words may be written in Devanagari;
consistent Romanized Hinglish is not guaranteed. No sentence translation or whole-
sentence transliteration postprocessor is used. These settings reduce imposed
language bias but cannot guarantee that Whisper will preserve every word/meaning.

Language selection follows one deterministic policy: an explicit library-call
mode overrides `AudioConfig.language`; otherwise that configured preference is
used, defaulting to `auto`. CLI `--language` (alias `--preferred-language`) overrides
`VOICEOS_STT_LANGUAGE`. Explicit `auto`/Python `None` restores detection even when
the configuration prefers English or Hindi. There is one setting, without a
second competing preference variable. For example:

```powershell
# Default decoder preference for this terminal/session:
$env:VOICEOS_STT_LANGUAGE = "en"
python main.py --voice
# Explicitly restore automatic detection for this run:
python main.py --voice --language auto
# Or deliberately choose Hindi:
python main.py --voice --preferred-language hi
Remove-Item Env:VOICEOS_STT_LANGUAGE
```

This preference selects the decoder language, not a translation service or
desired output-script converter. Forced English can still change Hindi meaning;
review against the speech. Auto's winning language token does not prove that
its words or script are correct. No probability/fluency heuristic automatically
chooses another transcript. See [STT_LANGUAGE_REPORT.md](STT_LANGUAGE_REPORT.md)
for the language audit and results on both available recordings.

### GPU and model configuration

The inspected RTX 3050 has 4096 MiB VRAM. Keep `small` as the measured baseline.
Automatic execution prefers CUDA with a supported `float16` compute type; CPU
prefers `float32` after int8 produced corrupt output on a measured clip. Explicit
CPU int8 remains available for comparison. CUDA setup/resource failures trigger at most one CPU fallback,
including errors raised while iterating lazy segments. Missing models and
permanent initialization failures do not trigger repeated downloads/reloads.
Each instance owns one reusable model; create one instance per sequential CLI
session. Concurrent calls on one instance are not part of this interface.

CTranslate2 GPU execution needs cuBLAS for CUDA 12 and cuDNN 9 for CUDA 12,
available to the Python process through the Windows DLL search path. A working
`nvidia-smi` alone does not verify those DLLs. Follow the
[Faster-Whisper GPU instructions](https://github.com/SYSTRAN/faster-whisper#gpu)
and [NVIDIA cuDNN installation guide](https://docs.nvidia.com/cudnn/installation/backend).
Use compatible installed libraries; do not blindly upgrade or downgrade the
existing environment. Restart VS Code after changing PATH. Set
`VOICEOS_STT_DEVICE=cpu` to use speech without CUDA.

| Model | Use | Evaluation status on this machine |
| --- | --- | --- |
| small | Multilingual baseline | CUDA float16/int8_float16 and CPU float32/int8 exercised on saved WAVs |
| medium | Multilingual candidate | Downloaded under .models/faster-whisper-medium; CUDA int8_float16 measured at 1233 MiB sampled total GPU memory; accuracy unverified |
| distil-small.en | English-only latency comparison | Not downloaded/measured; diagnostic runs English only; Hindi/auto calls rejected |

The [Distil-Whisper model documentation](https://github.com/huggingface/distil-whisper)
identifies the `.en` model as English-only. No model can guarantee the requested
mixed-script spelling. GPU acceleration does not itself improve accuracy.
See [STT_REPORT.md](STT_REPORT.md) for actual resource/timing measurements and
the limits of the before/after comparison.

The local medium model is optional and is not the default. It occupied approximately
1.53 GB on disk and decoded slower in the measured samples. To evaluate it in the
same integrated voice interface:

```powershell
python main.py --voice --language auto --model .models/faster-whisper-medium --device cuda --compute-type int8_float16
```

### Speech settings

CLI arguments override corresponding `.env` settings. Library callers can pass
`AudioConfig(...)` to avoid environment configuration. `.env` is loaded by CLI
entry points; `AudioConfig.from_env()` reads the current process environment.

| Variable | Default | Meaning |
| --- | --- | --- |
| VOICEOS_STT_MODEL | small | Multilingual model name or local converted model directory |
| VOICEOS_STT_DEVICE | auto | auto, cuda or cpu |
| VOICEOS_STT_COMPUTE_TYPE | auto | CUDA float16 preferred; int8_float16 optional; CPU float32 preferred; explicit int8 available |
| VOICEOS_STT_LANGUAGE | auto | Preferred decoder mode: auto, en or hi; explicit CLI choice overrides |
| VOICEOS_MICROPHONE | blank | OS default input; otherwise ID or unambiguous sounddevice name |
| VOICEOS_RECORDING_MODE | automatic | Automatic endpointing; fixed is retained for controlled diagnostics |
| VOICEOS_SILENCE_SECONDS | 2 | Continuous VAD-confirmed silence after speech before stopping |
| VOICEOS_MAX_RECORD_SECONDS | 60 | Automatic recording safety maximum, 1–120 seconds |
| VOICEOS_INITIAL_SILENCE_SECONDS | 8 | Timeout waiting for speech; must not exceed maximum |
| VOICEOS_CAPTURE_VAD_THRESHOLD | 0.5 | Streaming speech-onset probability threshold, not a transcript score |
| VOICEOS_SPEECH_START_MS | 96 | Consecutive speech evidence required to start the silence timer |
| VOICEOS_RECORD_SECONDS | 12 | Fixed diagnostic capture window only, greater than 0 and at most 120 seconds |
| VOICEOS_SILENCE_RMS | 0.001 | Reject low-level audio before inference; adjust for soft speech/noise |
| VOICEOS_STT_BEAM_SIZE | 5 | Decode beam size, 1–10 |
| VOICEOS_STT_VAD | true | Enable Silero VAD; diagnostic `--no-vad` allows comparison |
| VOICEOS_VAD_MIN_SILENCE_MS | 500 | VAD minimum silence |
| VOICEOS_VAD_SPEECH_PAD_MS | 300 | VAD speech boundary padding |
| VOICEOS_VAD_THRESHOLD | 0.5 | Silero speech threshold; compare gentler settings on saved audio |
| VOICEOS_STT_MULTILINGUAL | true | Language detection per model segment in auto mode; explicit en/hi stays forced |
| VOICEOS_STT_TEMPERATURES | 0,0.2,0.4,0.6,0.8,1 | Bounded decoding fallback schedule; benchmark single-temperature decoding before choosing it |
| VOICEOS_STT_PROMPT | blank | Optional evaluated spelling hint; no implicit English prompt |
| VOICEOS_STT_LOCAL_FILES_ONLY | false | Require cached/local models, prohibit download |
| VOICEOS_VOCABULARY_PATH | blank | Optional trusted JSON vocabulary replacing built-in corrections |

Capture validates mono float32 input at 16000 Hz. Device errors, unsupported
formats, overflows and incomplete buffers return actionable errors. Input is
not blindly amplified: RMS/peak/clipping diagnostics expose low gain without
turning silence into noise. Array callers must supply 16 kHz floating-point
samples in [-1, 1]; other rates and PCM integers require conversion first.
Normalization and inference never call Gemini. Reviewed command text still
uses the existing Gemini NLU integration.

### Pipeline and library API

```text
microphone -> float32/16 kHz mono -> validation + audio diagnostics
           -> reusable Faster-Whisper (task=transcribe)
           -> conservative mixed-script vocabulary correction
           -> raw/normalized transcript review
           -> process_text_command() -> existing NLU -> schema + validation
           -> allowlisted router + existing deletion confirmation -> SQLite actions
```

```python
from audio import AudioConfig, SpeechToText

stt = SpeechToText(config=AudioConfig(language="hi", duration=12))
load_seconds = stt.load_model()  # Optional for library use; the voice CLI does this at startup.
vad_load_seconds = stt.load_vad()
audio = stt.record_audio()  # Once; can reuse for multiple transcriptions.
result = stt.transcribe(audio, language=None, recording_info=stt.last_recording_info)
# Or: result = stt.listen_and_transcribe(language="en")
print(result["raw_text"], result["normalized_text"])
```

Results retain `text` as a compatibility alias for `normalized_text` and add
`raw_text`, `language`, `language_probability`, `status`, `empty_transcription`,
`audio_diagnostics`, `warnings`, `corrections`, actual device/compute type, and
timings in seconds. `recording_duration` is the audio length; `recording_time`
is capture wall time (0 when transcribing an existing array). Model loading,
transcription (including segment consumption), normalization and total STT wall
time are separate. A later call has zero model load time unless fallback loads
CPU once. First inference includes lazy library/VAD setup. These timings exclude
terminal review, Gemini and action execution.

`language_mode` and `language_selection_source` describe the requested decoder
mode and whether it came from configuration or an explicit library call.
`detected_language` is populated only for auto detection. `language_probability_kind`
is `detection`, `forced`, or `unavailable`; the existing `language_probability`
field retains the original library value for compatibility. `language_candidates`
preserves auto's language-token distribution for inspection, without using it to
rank transcript accuracy. `raw_text` is never overwritten by normalization.

`segments` retains original segment text, timestamps, temperature, average token
log probability and no-speech probability for diagnosis. These are not accuracy
scores. `speech_duration_after_vad` makes retained audio duration measurable.
`has_lexical_content=false` marks punctuation/combining-mark-only output unusable
and blocks it in the voice loop while preserving raw text. Text with letters can
still be wrong; this check does not establish fidelity.

`recording_info` describes automatic/fixed capture, stop reason, speech onset,
captured duration, trailing silence and capture wall time. Initial timeout without
speech bypasses Whisper even if background RMS is high. Maximum-duration captures
retain raw text for inspection but do not enter command processing. Silero runs on
512-sample frames; the two-second threshold is rounded to whole frames (normally
2.016 seconds), plus model hangover and device buffering in live operation.

Direct `record_audio(duration=...)` or `AudioConfig(recording_mode="fixed")` explicitly
retains fixed capture for compatibility. The normal voice loop supplies no duration
argument and uses automatic endpointing.

### Mixed-script normalization

`audio/transcript_normalizer.py` preserves Hindi words, punctuation and raw text.
It matches complete Unicode word/phrase spans, including Devanagari combining
marks, against a small explicit vocabulary. It does not translate, fuzzy-match,
globally substitute substrings, or infer missing command words/IDs. Technical
spellings such as Python, Gemini and VoiceOS can be corrected. Ambiguous `केवल`
remains unchanged unless the configured adjacent full-name group Keval Minda is
recognized after a name cue such as `मेरा नाम` or `my name is`. Even `केवल Minda`
without that cue is preserved because it could mean "only Minda". Incorrect sounds
such as "Cable" are intentionally not guessed.
DBMS, microphone and Faster-Whisper now have explicit vocabulary entries. Literal
hyphens in configured technical names are matched deliberately; punctuation
between unrelated words is never swallowed. `audio/vocabulary.example.json` shows
how to configure confirmed spellings. No LLM normalization fallback is implemented.

A custom UTF-8 JSON file can replace the vocabulary:

```json
{
  "vocabulary": {
    "VoiceOS": ["Voice OS", "वॉइस ओएस"],
    "Python": ["पाइथन", "पायथन"],
    "Keval": ["केवल"],
    "Minda": ["मिंडा", "मिन्दा"]
  },
  "ambiguous_terms": ["Keval", "Minda"],
  "name_groups": [["Keval", "Minda"]],
  "name_contexts": ["मेरा नाम", "नाम", "my name is", "I am"]
}
```

Set `VOICEOS_VOCABULARY_PATH` to that file. Candidate aliases are explicit
spelling decisions; add homonyms to `ambiguous_terms` or omit them. Conflicting
aliases are rejected. Unknown/uncertain phrases retain the original transcript.
Review corrections before submitting commands, especially dictated note content.

### Controlled microphone and model diagnostics

Both root `test_*.py` audio scripts are manual entry points to the same workflow,
with execution guards. Importing them or running pytest does not record or load
a model. Diagnostics never import the command processor or execute actions.

```powershell
python test_speech_to_text.py --list-devices
# Record ONCE, then reuse the same samples in en, hi and auto:
python test_speech_to_text.py --duration 15 --save-audio stt-mixed.wav --repeat 3 --gpu-memory --output stt-small.json
# Replay without opening the microphone:
python test_speech_to_text.py --input stt-mixed.wav --local-files-only --repeat 3 --output stt-replay.json
# Compare VAD/prompt on identical audio:
python test_speech_to_text.py --input stt-mixed.wav --no-vad --prompt= --output stt-no-vad.json
# Opt-in larger model comparison (may download model and consume substantial disk/VRAM):
python test_speech_to_text.py --input stt-mixed.wav --model medium --device cuda --compute-type int8_float16 --repeat 3 --gpu-memory --output stt-medium.json
# English-only comparison, using a separately recorded English clip:
python test_speech_to_text.py --input stt-english.wav --model distil-small.en --repeat 3 --gpu-memory --output stt-distil.json
```

Reports include the shared sample SHA-256, mode, actual returned language and
probability, raw/normalized text, empty status, audio diagnostics and timings.
GPU memory is sampled total device usage (other processes included), not an
exact process peak. Compare warm passes separately from model load/first decode.
Saved audio uses PCM16; replay hashes can differ from original float32 samples
due to quantization. All modes within each run still receive identical samples.
Existing WAVs are never overwritten by `--save-audio`.

Omitting diagnostic `--language` compares all three modes independently of the
environment preference. To evaluate just the selected mode or repeat stress cases:

```powershell
python test_speech_to_text.py --input stt-mixed.wav --preferred-language en --local-files-only --repeat 3 --output stt-english-preference.json
python test_speech_to_text.py --input stt-mixed.wav --language auto --local-files-only --output stt-auto.json
python test_speech_to_text.py --input stt-mixed.wav --stress --local-files-only --repeat 2 --gpu-memory --output stt-stress.json
```

`--stress` requires a saved WAV and adds reproducible derived cases: two seconds
of digital silence, two seconds of seeded Gaussian noise, speech plus noise,
the first two seconds of the source, and speech tiled/trimmed to 45 seconds.
Each case is labeled and hashed; every mode uses the same samples for that case.
These signals test edge conditions and runtime behavior. They are not independent
English/Hindi/Hinglish recordings or evidence of natural long-speech accuracy.
Noise rejection by RMS alone is not reliable; VAD/model output and user review
remain relevant. No diagnostic result enters command processing.

For manual accuracy evaluation, record separate English, Hindi and Hinglish
clips with known reference text, including names, negation, note IDs and trailing
silence. Compare each language mode/model on each same WAV. Review raw errors
and meaning before adding vocabulary; track latency, empties and memory together
with word/character errors against references. Do not choose the transcript with
the highest language probability. If VAD removes speech, compare `--no-vad` on
the same clip. Save English, Hindi and Hinglish reference recordings under distinct
new filenames and run the identical replay command on each. Keep written references
separate from recognized text; never infer ground truth from the best-looking result.
See [HINDI_HINGLISH_INVESTIGATION.md](HINDI_HINGLISH_INVESTIGATION.md) for the newer
preserved microphone recordings, playback-checked reference status and actual
model/VAD/prompt comparisons. Earlier reports retain their original test scope.

Current speech-related structure:

```text
audio/
    __init__.py
    audio_config.py
    speech_to_text.py
    transcript_normalizer.py
    diagnostics.py
    benchmark.py
    recording.py
    vocabulary.example.json
tests/
    test_audio_config.py
    test_speech_to_text.py
    test_transcript_normalizer.py
    test_voice_integration.py
    test_speech_benchmark.py
    test_automatic_recording.py
    ... existing offline safety/NLU/storage tests unchanged
main.py                      # text CLI and optional reviewed voice input
test_speech_to_text.py        # manual diagnostic wrapper
test_audio_module.py          # compatible alias of the same diagnostic
core/  nlu/  actions/  storage/  # existing command architecture
STT_REPORT.md                 # audit and actual measurements
SPEECH_BENCHMARK.md           # reference-gated accuracy benchmark and latest measurements
```

### Reference-verified accuracy benchmark

The [completed Round 2 report](ROUND2_VALIDATION_REPORT.md) compares five verified
English recordings, five verified Hindi recordings and the retained Hinglish
sample. [Earlier candidate validation](MEDIUM_CANDIDATE_VALIDATION.md) records
the application replay and configuration audit. Medium remains an explicit
opt-in candidate after this small single-speaker pilot:

```powershell
python -X utf8 main.py --voice --stt-profile medium-candidate --save-recordings benchmark-recordings/voice-captures
```

The same `--stt-profile medium-candidate --profiles baseline` selects the frozen
candidate in `python -m audio.benchmark run`. Explicit CLI options override the
profile. No global default or private `.env` promotion is implied. `-X utf8`
supports Hindi output when Windows redirects stdout.

See [SPEECH_BENCHMARK.md](SPEECH_BENCHMARK.md) for the latest audit, configuration
comparisons, limits and a guided English/Hindi/Hinglish recording procedure.
The benchmark has no NLU/action integration and no Gemini calls.

```powershell
python -m audio.benchmark init --manifest benchmark-recordings/my-session/session.json
python -m audio.benchmark run --manifest benchmark-recordings/my-session/session.json --output stt-benchmark.json --repeat 2 --gpu-memory
```

Initialization includes both available saved files as **unverified** and eight
empty recording slots. WER/CER and exact vocabulary/name recall are calculated
only after a reference is explicitly marked human-verified. Raw and normalized
text are scored separately. Missing categories and references stay visible.
Model loading is measured once and inference is repeated on identical samples.
Saved raw float32 `.npy` files avoid PCM16 quantization during subsequent comparisons.
The benchmark forbids model downloads by default; use --allow-download only when
you deliberately want to evaluate an uncached model.

To preserve future VoiceOS captures before inference, including when inference
fails, opt in with a private local directory:

```powershell
python main.py --voice --language auto --save-recordings benchmark-recordings/voice-captures
```

`VOICEOS_RECORDING_DIRECTORY` sets the same preference; the CLI overrides it.
Each activation saves a unique lossless float32 NPY and JSON metadata with the
waveform hash, capture settings and separately retained raw/normalized results.
Saved model output is never a reference. Recording persistence uses local disk
and has no automatic retention cleanup; choose a directory suitable for private
audio. The existing transcript review and action confirmation still apply.

For controlled accuracy tests, record one sample per empty manifest slot, then
listen to the saved WAV **before** confirming its exact words. A spoken prompt or
an initial confirmation alone does not prove the intended speaker was captured.

```powershell
python -m audio.benchmark record --automatic --manifest benchmark-recordings/my-session/session.json --case hinglish_technical
# Play the saved WAV, check speaker and every word, then create reference.txt as UTF-8:
python -m audio.benchmark verify --manifest benchmark-recordings/my-session/session.json --case hinglish_technical --reference-file reference.txt
# Compare 19 explicitly recorded, one-variable decoding/VAD/prompt profiles:
python -m audio.benchmark run --manifest benchmark-recordings/my-session/session.json --cases hinglish_technical --model small --device cuda --compute-type float16 --investigation --repeat 2 --gpu-memory --require-cuda --vad-audit --export-vad benchmark-recordings/my-session/discarded-small --output stt-investigation-small.json
```

Run medium only after the small process exits, using the same manifest/case with
`--model medium --compute-type int8_float16` and a new output/export directory.
The benchmark defaults to cached weights; an uncached medium needs an explicitly
authorized download. `--require-cuda` rejects CPU fallback during GPU comparisons.
The investigation profiles use automatic per-window language detection, no prompt,
beam 5 and temperature 0 as their controlled base. Other rows vary VAD on/off,
silence duration, padding, threshold, beam, temperature fallback, general Hindi/
mixed/technical prompts, or explicit en/hi. Every run uses `task="transcribe"`.

`--vad-audit` reports discarded sample intervals and signal levels. `--export-vad`
also writes lossless discarded excerpts and playback WAVs without overwriting
existing excerpts. Listen before labeling discarded signal as lost speech.
Per-call GPU figures are sampled total device usage, including other processes;
very short calls may contain no sample. Model loading is reported separately.

The `import` subcommand can attach a VoiceOS capture JSON to an empty slot via
`--capture PATH --case ID --manifest PATH`; it checks the saved waveform hash and
leaves its reference unavailable. Generate the complete comparison table with:

```powershell
python -m audio.comparison_report --reports stt-investigation-small.json stt-investigation-medium.json --output comparison.md
```

## Verification

```text
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m compileall -q .
```

Ordinary tests use a fresh temporary SQLite database per test and block socket
connections. Gemini, microphones and Whisper models are mocked. No physical
microphone, downloaded weights, GPU or live API is needed for pytest.
No production DB setup is needed.
An optional live test is excluded by default. To deliberately allow API quota use:

```powershell
$env:VOICEOS_LIVE_TEST = "1"
python -m pytest -m live test_gemini_client.py -q
Remove-Item Env:VOICEOS_LIVE_TEST
```

## Privacy and limitations

User command text is sent to Gemini. Notes are local plaintext SQLite records,
not encrypted. API keys, command bodies and exception bodies are not logged by
application code. CLI suppresses SDK/HTTP diagnostic logs; embedding applications
must apply equivalent logging policy. Do not use real secrets as test inputs.

No arbitrary SQL, shell or Python execution is available to the model. This does
not make model interpretation infallible: review the deletion preview carefully.
The current API returns generic safe errors; inspect configuration and reproduce
with synthetic data for debugging. Existing reminder/search enums are retained
for compatibility, but their placeholder routes now reject instead of pretending
an action succeeded. The CLI is synchronous and may wait through configured API
attempts. Content comparison protects against changed content, not identical-content
edit history. No multi-user authorization or session-bound remote confirmation
protocol is provided.

See `AUDIT.md`, `CHANGELOG.md`, and `VERIFICATION.md` for this delivery's findings.
