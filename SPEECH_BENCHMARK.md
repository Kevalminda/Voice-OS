# Speech fidelity and performance — 15 November 2026

This is the pre-endpointing baseline. Current automatic recording/native defaults
and the measured medium-model comparison are in
[AUTOMATIC_RECORDING_REPORT.md](AUTOMATIC_RECORDING_REPORT.md). References for the
two original clips remain unavailable.

## Diagnosis and scope

Audited STT configuration, microphone capture, inference, normalization, voice CLI,
audio tests and dependency versions. Existing 16 kHz mono float32 capture validates
devices, silence, clipping and possible end truncation. It does not automatically
amplify, resample, denoise or otherwise transform microphone samples. Existing
raw/normalized transcript separation and conservative name context were retained.

The model was reused per instance but loaded lazily on the first command; the
voice CLI now initializes it once before the recording loop. Application actions,
NLU, validation, router, SQLite and existing confirmation mechanisms are unchanged.
No notes, reminders, tasks, database or system-control features were implemented.

Automatic language classification chooses a dominant language token, not a desired
output script. Forcing en can produce translation-like output for Hindi/Hinglish;
forcing hi can phoneticize English terms. task=transcribe is necessary but does
not guarantee word-level code-switching fidelity. Configurable per-segment language
detection is now available for auto mode; explicit en/hi remains explicit. Model
segments are not individual words, so this option cannot guarantee natural Hinglish.
The [Faster-Whisper 1.2.1 implementation](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/transcribe.py)
was checked against the installed source for these options.

Prior real stress results showed prompt echo, wrong-script output and slow fallback
decoding. This phase exposes temperatures, segment details and VAD-retained duration
to make those failures measurable. Language/token probabilities are diagnostic
metadata, never direct accuracy scores or automatic transcript selectors. Empty
audio bypasses inference; non-silent audio may still return empty text after VAD
or model no-speech decisions. No blanket retry without VAD or forced language was
added, because it can hallucinate or change the speech.

## Changes

- audio/audio_config.py: validated per-segment detection, VAD threshold and bounded
  temperature schedule; existing auto/en/hi selection remains intact.
- audio/speech_to_text.py: public idempotent load_model(), original segment details,
  retained VAD duration and beginning-level diagnostics. Punctuation/combining-mark
  only output is retained but marked unusable instead of entering command processing.
- main.py: one startup load, optional --multilingual/--no-multilingual and --prompt,
  with the same reviewed normalized-text and deletion confirmation path.
- audio/transcript_normalizer.py and audio/vocabulary.example.json: exact DBMS,
  microphone and Faster-Whisper entries plus safe literal-hyphen phrase matching.
  Name correction still requires configured full-name/context matches; Cable is
  not guessed to be Keval. Ordinary Hindi words and command intent are not translated.
- audio/diagnostics.py: exposes multilingual decoding alongside existing replay.
- audio/benchmark.py: repeatable profile comparisons, human-reference-gated WER/CER,
  term/name recall, raw/normalized metrics, recording/hash provenance, lossless raw
  capture, and visible missing categories. It never imports NLU/actions.
- Audio/voice/benchmark tests, .env.example, .gitignore and README.md updated.

No dependencies were upgraded or added. Python 3.11 compatibility is retained.
No backup project, original recordings or production database was modified.

## Reference policy and metrics

The user confirmed that exact manually verified references are unavailable for
test_audio.wav and stt-mixed.wav. Both exist in this workspace and are included
as unverified samples. Whisper output is never copied into a reference.
reference_verified=true plus reference_source=human and actual reference text
are required for scoring. The verify command binds the reference to the decoded
audio SHA-256; changed samples are rejected before scoring.

WER is word-level Levenshtein distance divided by reference words. CER uses Unicode
characters after NFC, casefold, Unicode punctuation-to-space conversion and collapsed
whitespace; spaces are omitted from CER. The same policy applies to all profiles,
and no transliteration is applied for evaluation. Empty reference denominators
produce null rates while insertion/error counts remain available. Rates can exceed
1 when there are many insertions. Raw and normalized transcripts are scored
separately; aggregate rates weight by reference lengths across repeated calls.

Term/name recall counts exact whole phrases present in the verified reference and
recovered in the hypothesis, under the declared text policy. It does not guess
phonetic equivalence or claim semantic/name accuracy from a single match. Case-only
spelling fixes are covered by unit tests because these rates intentionally ignore
case. Language preservation and translation errors require human review per profile;
their fields stay null until reviewed. Script counts describe text, not meaning.

**Before/after WER, CER, vocabulary/name recall and translation-error rates are
unavailable for these two clips. No recognition-accuracy improvement is claimed.**
Fresh microphone testing was offered with a read/beep/confirm procedure, then
deferred at the user's request. No new live recording was made in this phase.

## Actual GPU comparisons on identical saved audio

Python 3.11.9, Faster-Whisper 1.2.1, CTranslate2 4.8.5, RTX 3050 Laptop 4096 MiB;
cached multilingual small, CUDA float16. One model load served all **32 calls**:
2 saved recordings × 8 profiles × 2 passes. The benchmark default forbids downloads.
Cold model initialization measured **1.429 s**; sampled total GPU usage peaked at
**785 MiB**, including other processes and excluding any unsampled transient peak.
Per-call model-load duration was zero after startup. GPU timing includes lazy
runtime setup on the first transcription, separately from constructor loading.

| Profile | Configuration difference | Median decode over four calls |
| --- | --- | ---: |
| baseline | Existing auto, vocabulary hint, VAD, fallback temperatures | 0.762 s |
| auto_multilingual | Per-segment detection in auto | 0.691 s |
| auto_no_prompt | Per-segment detection, empty prompt | 0.723 s |
| auto_no_vad | Empty prompt, per-segment detection, VAD disabled | 0.711 s |
| auto_gentle_vad | Empty prompt, threshold 0.35, 400 ms pad, 700 ms silence | 0.741 s |
| auto_deterministic | Empty prompt, per-segment detection, temperature 0 only | 0.715 s |
| english | Explicit en, existing vocabulary prompt/VAD | 0.291 s |
| hindi | Explicit hi, existing vocabulary prompt/VAD | 0.560 s |

These aggregate medians include mixed clip lengths and the first cold-runtime call;
they are observations, not accuracy or causal speedup claims. Full per-pass timings,
raw text, segment temperatures and settings are in stt-accuracy-candidates.json.

test_audio.wav: 5 seconds, RMS 0.014684, peak 0.459015, no clipping/end activity.
Baseline VAD retained 4.436 seconds; gentler VAD retained 4.568; disabled retained 5.
Auto returned Hindi. Baseline raw:

> मेरा नाम केबल मिन्डा है, मैं एक वोई सो इस प्रुजिक पे काम कर रहा हूं

Forced English raw:

> My name is Cable Minda. I am working on a voiceOS project.

stt-mixed.wav: 15 seconds, RMS 0.008873, peak 0.142242, no clipping/end activity.
Baseline retained 5.496 seconds; gentle VAD 5.760; disabled VAD 15.000 seconds.
These durations quantify removal; reference-free analysis cannot prove that all
removed samples were unneeded silence. Auto baseline raw:

> कल मेरे DBMS के इजाम है, मुझे रिमान्टर सेट कना है, मैं VoiceOS परजेक्ते काम कर रहा है।

Forced English raw:

> Yesterday I had an exam for my DbMS. I want to add a reminder. I am working on the VoiceOS project.

Empty prompt/per-segment raw on the same file:

> कल मेरे DbMS के अजाम है, मुझे रिमान्टर सेट कन रहा है, मैं वोई सो यस परज़िक ते काम कर रहा है।

Per-segment detection alone returned the same raw text as baseline for both clips.
Prompt-off and VAD variations changed words. None established a reliable original-
language solution or a quality winner. Vocabulary normalization preserves raw text
and now standardizes explicit DbMS casing to DBMS, without guessing unclear words.

## Selected configuration and CPU behavior

Retained small/CUDA float16, auto language, beam 5, existing prompt/VAD and bounded
temperature defaults as the compatibility baseline. Per-segment detection, empty
prompt, gentle VAD and deterministic decoding remain explicit candidates until
verified-reference comparisons support selecting one. Forcing English is a user
choice, not the default remedy for Hinglish. User dictionaries can opt into precise
technical spellings; no recording-specific correction rules were added.

CPU automatic/fallback precision now prefers float32 following the actual int8
corruption described below. Explicit --device cpu --compute-type int8 remains
available. This trades quantized memory savings for higher numeric precision on
the 16 GB RAM machine; it does not constitute a verified WER improvement.

The prior small/float16 versus int8_float16 GPU comparison used approximately 795
versus 507 MiB of sampled total GPU memory; float16 generally decoded faster there.
This phase retained the measured float16 choice. Medium and English-only distilled
models were not cached/downloaded or promoted automatically. To evaluate medium
deliberately, use --model medium --compute-type int8_float16 --allow-download and
measure VRAM, latency and verified accuracy on the same corpus. English-only models
cannot validate Hindi/Hinglish; explicit en is required for those model variants.

Actual CPU int8 baseline load was **1.566 s**. Original clip decodes were **3.687 s**
and **3.515 s**. The 15-second clip triggered temperature-1 fallback, took **29.987 s**
and **32.075 s**, and produced repeated combining marks or unrelated-script text.
This is an observed fallback failure, not a claim of usable CPU accuracy. The new
nonlexical-output check blocks combining-mark-only text in the voice loop; arbitrary
letter-containing gibberish still requires review. No speech goes directly to actions.

A CPU int8 no-prompt/single-temperature control took **3.959 s** and **4.056 s** on
the two files, returning readable Hindi with remaining spelling errors. No reference
exists to validate its meaning or choose it globally merely for speed. Reports:
stt-accuracy-cpu.json and stt-accuracy-cpu-deterministic.json.

The CPU float32 control loaded in **1.749 s** and decoded the original/15-second
clips in **6.367 s / 6.187 s**. The longer clip returned readable Hindi instead of
the degenerate int8 outputs:

> कल मेरे DBMS के इजाम है, मुझे रिमान्टर सेट कना है, मैं VoiceOS project काम कर रहा है।

This supports the more conservative float32 CPU default for this machine, without
establishing general Hindi/Hinglish accuracy. Details: stt-accuracy-cpu-float32.json.

## Recording procedure for verified references

Create a new session directory so previous recordings cannot be overwritten:

```powershell
.\.venv\Scripts\Activate.ps1
python main.py --list-devices
python -m audio.benchmark init --manifest benchmark-recordings/fresh/session.json
```

Set VOICEOS_MICROPHONE to an ID/name from your own device list if needed. The OS
default remains valid when blank. The recorder checks 16 kHz mono float32 support;
no gain increase or denoising is applied. Original samples are saved losslessly as
.npy plus a convenience PCM16 WAV; benchmark replay uses the unquantized .npy.

The following are **suggested read prompts, never pre-verified references**:

| Case slot | Suggested prompt or condition | Window |
| --- | --- | ---: |
| english | I am testing VoiceOS with Python, Gemini, and my DBMS project. | 12 s |
| hindi | कल मेरी परीक्षा है और मुझे आज पढ़ाई पूरी करनी है। | 12 s |
| hinglish_technical | आज मैं VoiceOS project में Python और Gemini के साथ speech recognition test कर रहा हूँ। | 15 s |
| hinglish_names | नाम Keval Minda है और project का नाम VoiceOS है। | 12 s |
| short | Python और Gemini। | 6 s |
| long | Read a natural paragraph mixing Hindi and English for about 20 seconds. | 25 s |
| trailing_silence | यह microphone test है। Then remain silent for at least 5 seconds. | 12 s |
| noisy | Repeat a known sentence with ordinary room/fan noise. | 12 s |

For EACH case:

1. Run the recording command below, changing --case/--duration. Press Enter when
   ready. After the countdown and start beep, speak naturally; leave a silent pause
   at the end. A lower beep marks completion.
2. Before reading any Whisper output, confirm the exact words actually spoken.
   Include mistakes/restarts/omissions. Listen to the saved WAV if uncertain.
3. Write those human-checked words to a UTF-8 reference file. If the intended prompt
   was not spoken exactly, use the actual words. If unsure, leave it unverified.
4. Only then run verify. Never copy a model result as ground truth.

```powershell
python -m audio.benchmark record --manifest benchmark-recordings/fresh/session.json --case english --duration 12
# Create reference-english.txt manually from the words you actually spoke, then:
python -m audio.benchmark verify --manifest benchmark-recordings/fresh/session.json --case english --reference-file benchmark-recordings/fresh/reference-english.txt
```

The assistant-guided equivalent asks you to confirm the words after each capture.
--no-wait --delay 3 is available only when the speaker is already ready for a timed
assistant-guided recording. No case is marked verified automatically. The user
deferred these live captures during this phase.

Add expected exact terms/names to that case's terms/names lists only when they occur
in its verified reference. Optionally add human reviews per profile:

```json
"reviews": {
  "baseline": {"language_preserved": true, "translation_error": false, "notes": "Human checked"}
}
```

These are manual annotations, not automatically inferred from script or probability.

## Repeat comparisons and verification

```powershell
python -m audio.benchmark run --manifest benchmark-recordings/fresh/session.json --output stt-accuracy.json --repeat 2 --gpu-memory
python -m audio.benchmark run --manifest benchmark-recordings/fresh/session.json --output stt-candidates.json --profiles baseline auto_multilingual auto_no_prompt auto_no_vad auto_gentle_vad auto_deterministic english hindi --repeat 2 --gpu-memory
python main.py --voice --language auto --multilingual --prompt= --duration 15
python -m pytest -q
python -m ruff check .
```

Final offline verification: **232 passed, 1 deselected** (the existing opt-in live
Gemini test). Ruff lint and changed-file formatting checks passed; Python compilation
and CLI-help verification passed. Source recordings and production DB hashes matched
before/after checks. Mock microphone/model responses test contracts, metrics,
raw preservation, context/word boundaries, reuse and safety; they do not establish
live recognition accuracy. Actual hardware runs above use saved real audio and
cached weights. Fresh mic accuracy, verified per-language WER/CER, natural long-form
recognition and robust code-switching remain open until confirmed recordings exist.
