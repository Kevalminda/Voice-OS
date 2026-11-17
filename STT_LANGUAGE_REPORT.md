# Language handling follow-up — 15 November 2026

## Audit and root cause

Inspected audio configuration, capture/inference, language selection, normalization,
diagnostics, CLI integration, existing speech/voice tests and command safety
boundaries. Baseline verified: **158 passed, 1 deselected**; Ruff passed.

The existing configuration already supported auto/en/hi, but diagnostics always
compared all three modes and results did not distinguish forced language from
detected language. A user preference cannot be inferred from acoustic language
detection. The follow-up makes this boundary explicit without adding a competing
configuration variable or a second transcription/NLU pipeline.

Faster-Whisper 1.2.1 extracts audio features after optional VAD, selects a language
token when language=None, and uses that language in the decoder tokenizer. The
default detector considers the initial audio window, up to approximately 30 seconds;
the current pipeline does not enable per-segment multilingual detection. Detection
therefore makes one language choice even for code-switched speech. The vocabulary
prompt is used in subsequent decoding; it does not override acoustic language
detection. Forced modes skip language detection and the library sets their language
probability to 1. The [versioned Faster-Whisper source](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/transcribe.py)
and installed source were checked for this behavior.

On the user's latest recorded-buffer report, auto selected Hindi at 0.93945.
Replaying its PCM16 WAV selected Hindi at 0.94824, ahead of Urdu 0.04300 and English
0.00582. That explains the decoder's choice, not whether the words are correct.
The acoustic classification may reflect Hindi or code-switched pronunciation;
without a human reference, the precise linguistic cause is unverified. No clipping
or boundary activity was detected on either original saved sample. The source of
the language/script disagreement is model decoding/selection rather than evidence
of an audio-format, CUDA, clipping or capture-window failure on these two samples.

Forced English gives coherent English and clearer technical terms for the stated
English-output preference. It is appropriate to select en explicitly when that is
the actual speech/decoding preference. Coherence alone cannot establish faithful
recognition: the latest English output says “Yesterday I had an exam” while the
Hindi output starts “कल ... है”. Turning Hindi into fluent English can change
meaning or behave like translation even with task=transcribe. A written reference
was requested; none was available at verification time. No all-Hinglish-to-English
policy, accuracy score, transcript ranking or automatic translation was added.

## Language-selection strategy

1. An explicit library-call language (en, hi, auto or None) takes precedence.
2. Otherwise AudioConfig.language / VOICEOS_STT_LANGUAGE supplies the preference.
3. Default auto passes None to Whisper; no automatic English/Hindi fallback is used.
4. Main CLI --language and its new --preferred-language alias override environment
   configuration. Explicit --language auto clears an en/hi environment preference.
5. Diagnostic CLI without --language retains the controlled three-mode comparison.
   With --language/--preferred-language it evaluates only that explicitly selected
   mode. This comparison behavior is independent of environment preference.

Existing text/language/probability fields remain compatible. New fields identify
language_selection_source (configuration/call), detected_language (auto only),
language_probability_kind (detection/forced/unavailable), and language_candidates
(library language-token distribution, when available). No distribution is used to
score accuracy or silently rerun/rerank a transcript. Silence has no detected
language/probability. Existing raw_text remains unchanged by normalization.

The vocabulary and normalization implementation were not modified. There are no
new recording-specific aliases, script converters or LLM normalization calls.
The multilingual small model, CUDA float16 selection and existing configurable
prompt/VAD defaults were retained. Prompt-off control results below show a tradeoff
that needs reference-based evaluation before changing global defaults.

## Actual saved audio results

These are real cached-model GPU executions, separate from mocked unit tests.
No new microphone capture, model download, live Gemini call or application action
was performed. Both WAVs were decoded at 16000 Hz, mono float32 from PCM16.

| Saved recording | Length | RMS / peak | Audio checks | Auto result |
| --- | ---: | --- | --- | --- |
| test_audio.wav | 5 s | 0.014684 / 0.459015 | No clipping; final 200 ms silent | hi, 0.95508 |
| stt-mixed.wav | 15 s | 0.008873 / 0.142242 | No clipping; final 200 ms silent | hi, 0.94824 |

test_audio.wav, three passes per mode:

- Forced en raw: `My name is Cable Minda. I am working on a voiceOS project.`
  The existing normalizer changes voiceOS casing to VoiceOS; Cable is preserved.
- Forced hi and auto raw: `मेरा नाम केबल मिन्डा है, मैं एक वोई सो इस प्रुजिक पे काम कर रहा हूं`
  The existing VoiceOS phrase alias is corrected; uncertain name/project wording
  remains. Both modes returned the same text in all three passes.

stt-mixed.wav, two passes per mode:

- Forced en raw: `Yesterday I had an exam for my DbMS. I want to add a reminder. I am working on the VoiceOS project.`
- Forced hi and auto raw: `कल मेरे DBMS के इजाम है, मुझे रिमान्टर सेट कना है, मैं VoiceOS परजेक्ते काम कर रहा है।`
- These outputs repeated within this run. The previous recorded-buffer report had
  `VoiceOS project` in Hindi/auto instead of `VoiceOS परजेक्ते`. Saving PCM16
  quantizes the original float32 recording, so replay is not bit-identical to
  that buffer. The raw baseline reports were preserved; no improvement is inferred.

Saved-WAV decoded sample hashes:

- test_audio.wav: `0e2c91edbfac75cc290dc41582173199627cfd7fe2094bcccc72b27d085f6f74`
- stt-mixed.wav: `fc146e1a0466c189d8bf16a5a1ea59bab5c6725bc9d16d8a6c26be612d2f1d7f`

Detailed local reports: stt-language-original.json and stt-language-mixed.json.
The latter also contains every raw stress-test output and case hash. The filename
stt-mixed.wav does not independently establish its actual spoken language. There
are no separate human-labeled English, Hindi and Hinglish reference recordings,
so per-language WER/accuracy and a recognition fix cannot be claimed.

## Actual derived/synthetic stress results

The diagnostic uses seed 42 and labels each generated signal. All three modes
receive identical audio for each case, with two passes. These are not new human
recordings or natural long-form speech.

| Case derived from stt-mixed.wav | Definition | Observed result |
| --- | --- | --- |
| Silence | 2 s digital zeros | All six calls returned no_speech, empty raw text, zero inference time |
| Noise only | 2 s Gaussian noise, target RMS 0.02 | All six calls returned empty_output; auto's en label did not imply speech or accuracy |
| Noisy speech | Full saved clip plus Gaussian noise, target RMS 0.02 | Degraded texts; auto selected Urdu in both passes; raw results retained |
| Short | First 2 s, which may cut words | English returned “tomorrow is my day”; Hindi/auto varied, with prompt echo in one auto pass |
| Long | Source tiled/trimmed to 45 s | Unstable text in all modes; Hindi/auto sometimes generated unrelated scripts, punctuation or gibberish |
| Ambiguous selection | Same original clips decoded under en/hi/auto | Different scripts/wording; no automatic transcript winner chosen |

In particular, status=ok means nonempty model output, not correct recognition.
Language probability does not make the noisy/short/long outputs trustworthy.
Auto remains capable of choosing languages outside en/hi; no covert remapping
of Urdu to Hindi was introduced. User review remains essential.

A limited prompt-disabled control reused the same source/short/long case definitions
and model settings (stt-language-no-prompt.json, one pass per mode). On that pass,
short auto returned `कल मेरे देमीम` instead of echoing the prompt, and long auto
returned readable Hindi with remaining errors. Short auto decode was 0.340 s and
long auto 0.887 s, versus approximately 7.8 s and 8.0–9.2 s in the prompted stress
run. However the full source lost the VoiceOS spelling and English changed from
“Yesterday...” to “Tomorrow...”. This demonstrates prompt sensitivity and a local
latency observation; it does not validate meaning, completeness or general accuracy.
Keep the existing configurable prompt under evaluation. Use --prompt= to compare
without it on the exact same WAV/cases. No global prompt default was changed from
these unannotated samples.

## GPU performance

Python 3.11.9, faster-whisper 1.2.1, CTranslate2 4.8.5, RTX 3050 Laptop 4096 MiB;
all measured inference used CUDA float16 and the cached small model. Model loading
occurred once per diagnostic invocation and was reused across its modes/cases.

| Source | Model load | First en decode | Warm en | Warm hi | Warm auto |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original 5 s | 2.029 s | 1.478 s | 0.251–0.252 s | 0.529–0.535 s | 0.638–0.653 s |
| Saved 15 s | 1.357 s | 0.707 s | 0.323 s | 0.607 s | 0.757 s |

Sampled total GPU memory peaked at 785 MiB for the original replay and 1009 MiB
for the mixed/stress workflow. Other processes are included and transient peaks
can be missed. The prompted 36-call source/stress workflow took 92.045 s; problematic
short/long decodes dominated it (long forced hi reached 18.308 s). This is measured
failure behavior, not a low-latency guarantee. Existing baseline measurements are
in STT_REPORT.md. No hardware/accuracy improvement is inferred from timing variance.

## Integration and verification

Only reviewed normalized text reaches the existing process_text_command(). Empty,
truncated or declined input remains blocked. The existing strict NLU schema,
confidence/required-field checks, allowlisted router, deletion preview/confirmation,
cancellation and unknown-command rejection remain intact. No router, SQL, NLU or
action implementation was rewritten. Speech has no shell/SQL execution interface.

New tests exercise selection precedence, explicit auto overrides, forced/detected
metadata, English/Hindi/Hinglish output preservation with labeled mock responses,
silent/noisy/short/long waveform handling, ambiguous detection without reranking,
saved-WAV loading with mocked inference, CLI preference selection and rejection of
unrestricted action requests through the existing pipeline. Mock texts are contract
fixtures and are never presented as actual recognition results. Existing tests
were retained. Actual GPU diagnostics remain manual entry points outside pytest.

Final verification with the existing Python 3.11.9 interpreter:

```text
python -m pytest -q
193 passed, 1 deselected in 4.48s

python -m ruff check .
All checks passed!
```

The deselected test is the existing opt-in live Gemini test. Formatting checks
passed for all seven modified Python files. Compilation and both CLI help commands
passed. The existing database and both source WAV hashes matched their pre-edit
values; they were not overwritten. No dependency versions were changed.

## Files changed in this follow-up

- audio/audio_config.py: central deterministic language selection and input validation.
- audio/speech_to_text.py: selection/probability provenance and detected-language metadata.
- audio/diagnostics.py: selected-mode replay and labeled reproducible stress cases.
- main.py: preferred-language CLI alias and active mode display.
- tests/test_audio_config.py, tests/test_speech_to_text.py, tests/test_voice_integration.py:
  policy, waveform, preservation and safety coverage.
- .env.example and README.md: preference semantics and exact evaluation commands.
- STT_REPORT.md: link to this follow-up; original baseline measurements preserved.
- STT_LANGUAGE_REPORT.md: audit, actual outcomes, performance and limitations.

## Exact run and repeat commands

```powershell
.\.venv\Scripts\Activate.ps1
python main.py --voice --preferred-language en --duration 15
python main.py --voice --language hi --duration 15
python main.py --voice --language auto --duration 15
python test_speech_to_text.py --input test_audio.wav --local-files-only --device cuda --compute-type float16 --repeat 3 --gpu-memory --output stt-language-original-repeat.json
python test_speech_to_text.py --input stt-mixed.wav --local-files-only --device cuda --compute-type float16 --repeat 2 --stress --gpu-memory --output stt-language-mixed-repeat.json
python test_speech_to_text.py --input stt-mixed.wav --local-files-only --prompt= --stress --output stt-language-control-repeat.json
python -m pytest -q
python -m ruff check .
```

To close accuracy gaps, record new English, Hindi and Hinglish clips under distinct
filenames and keep the words actually spoken as separate human references. Replay
each saved file in all three modes; include technical terms, negation and note IDs.
Use a sufficiently long recording window with trailing silence. Do not use Whisper
output as the reference. No new microphone test or independently scored language
corpus was available for this follow-up. Hindi/Hinglish recognition remains limited
by the small model, acoustics, code-switching, prompts and decoder behavior.
