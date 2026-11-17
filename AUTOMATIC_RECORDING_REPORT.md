# Automatic capture and native transcription — 15 November 2026

## Diagnosis

The previous recorder used sounddevice.rec(int(seconds * 16000)) followed by wait().
It was designed as a simple fixed-window diagnostic and then reused by the voice
loop; it had no streaming speech endpoint detector. Increasing the window reduced
cut-off speech but imposed unnecessary waiting. It is now replaced in the normal
voice interface. Fixed capture remains explicit for controlled reference collection.

The complete pre-display path was inspected: microphone -> Faster-Whisper
SpeechToText._decode -> raw_text -> TranscriptNormalizer -> display -> existing
review -> process_text_command -> Gemini NLU -> existing validated router. There
is no sentence translation utility or LLM transcript-rewriting step before display.
Gemini is only called downstream after transcript approval for command understanding.
The normalizer matches configured word/phrase spellings and cannot produce the
observed complete English paraphrases from Hindi grammar.

The translation-like wording is present in Whisper's own raw segments, before
normalization. It was reproduced by forced English decoding on the same saved
audio: medium returned “My name is Keval Minda and I am working on a voice-over
project.” and “Tomorrow is my DBMS exam. I have to set reminders. I am working on
this project.” The client explicitly requested task=transcribe. Thus the reproduced
failure originates at model decoding, not the router, Gemini or post-normalizer.
English forcing and vocabulary-prompt bias are demonstrated risk factors; without
the user's exact new utterance recording, the trigger of every historical failure
cannot be identified. Auto mode can also hallucinate or choose a wrong language.

## Automatic recording implementation

The normal voice loop now:

1. Loads one Whisper model and prepares the cached bundled Silero ONNX VAD at
   startup. The same model/session is reused; only recurrent VAD state resets.
2. Opens a 16000 Hz mono float32 InputStream after microphone activation.
3. Displays Listening only after the stream is open, so the user does not start
   speaking before microphone initialization completes.
4. Copies 512-sample blocks in the PortAudio callback into a bounded queue. Neural
   inference runs outside the callback, with no network/LLM request.
5. Retains every captured sample, including initial audio and sentence endings.
   Three consecutive 32 ms speech frames (96 ms) establish speech onset.
6. After onset, waits for two seconds of continuous VAD-confirmed silence. Possible
   speech resets that timer with a lower release threshold to protect quiet tails.
   Short pauses do not reset the entire recording or trigger a new model load.
7. Closes the stream on normal silence, initial timeout, safety maximum, interruption
   or error. Overflow, malformed blocks, lost callbacks and device failures are
   reported; a stalled callback stream has a bounded one-second wait.

Defaults: silence 2 s, initial silence timeout 8 s, maximum 60 s, capture speech
threshold 0.5. The 512-sample frame quantization makes a two-second timeout normally
2.016 s. Real device buffering and neural speech hangover add latency. The maximum
is a safety guard, not the expected end mechanism. Initial timeout without detected
speech skips Whisper even if background RMS is high. A maximum-limit capture is
displayed for diagnosis but does not reach the command processor. VAD-confirmed
silence is allowed to complete despite elevated background RMS; the older RMS-tail
heuristic is still exposed for diagnosis rather than vetoing that result.
Confirmed quiet speech with a non-flat signal is also not rejected solely because
leading/trailing silence dilutes whole-recording RMS. No gain is added; zero-level
audio remains rejected.

Settings: VOICEOS_RECORDING_MODE, VOICEOS_SILENCE_SECONDS,
VOICEOS_MAX_RECORD_SECONDS, VOICEOS_INITIAL_SILENCE_SECONDS,
VOICEOS_CAPTURE_VAD_THRESHOLD and VOICEOS_SPEECH_START_MS. Main CLI options are
--silence-seconds, --max-duration and --initial-timeout. --duration is retained as
an alias for the safety maximum in main.py. Initial timeout must not exceed it.
Direct record_audio(duration=...) and recording_mode=fixed retain explicit fixed
capture for compatibility; diagnostic/reference tools use that deliberately.

The adapter reuses Faster-Whisper's bundled ONNX asset, so no new VAD dependency or
online VAD download was introduced. The installed v6 h/c state interface was checked
directly. Streaming and bundled batch probabilities matched exactly on all 157 and
469 frames of the two saved files (maximum difference 0.0). Reference APIs:
[sounddevice InputStream](https://python-sounddevice.readthedocs.io/en/0.5.6/api/streams.html),
[bundled Silero implementation](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/vad.py).

## Native-language defaults and spelling

Defaults now use language=auto/None, task=transcribe, per-segment multilingual
detection enabled, and an empty initial prompt. This removes the previous implicit
English vocabulary hint. Forced en/hi and custom vocabulary hints remain explicit
options. An inspected effective configuration confirmed auto/true/empty prompt;
no private API credentials were read or printed. Temperature/VAD decoding controls
remain configurable, and language probability is never used as transcript accuracy.

Raw text and segment details are preserved. A separate normalized result retains
the existing configurable vocabulary and conservative full-name context rules.
Keval Minda, VoiceOS, Python, Gemini, DBMS and other exact configured technical
aliases can be matched; Cable/voice-over/unrecognized names are not guessed into
project-specific names. No recording-specific transcript or new fuzzy correction
was hardcoded. The existing audio/vocabulary.example.json remains available.

Preserving Hindi expressions is different from guaranteeing Romanized writing.
Whisper commonly writes Hindi in Devanagari, with Latin technical terms when it
recognizes them. No whole-sentence translation, Romanization or grammatical rewrite
is performed after inference. The requested exact Romanized Hinglish script and
reliable code-switching recognition remain model limitations, not solved claims.

## Actual VAD replay, separate from microphone tests

Both existing WAVs were used unchanged. Each replay explicitly appended three
seconds of digital silence to let the endpoint finish. This was offline model
execution without a microphone or simulated wall-clock waiting.

| Source | Original length | Endpoint capture length | Final silence | VAD processing |
| --- | ---: | ---: | ---: | ---: |
| test_audio.wav | 5 s | 6.656 s | 2.016 s | 0.0289 s |
| stt-mixed.wav | 15 s | 8.160 s | 2.016 s | 0.0396 s |

Streaming VAD initialization measured 0.371 s in that process and was reused across
both replays. Capture lengths are replay endpoint positions, not measurements of
physical microphone latency. The longer recording's unused trailing window was
not awaited in the replay. No sentence-ending words were trimmed by the controller;
recognizer fidelity still needs a human reference.

Additional explicitly derived/synthetic cases:

- Seed-42 Gaussian background noise at target RMS 0.02 stopped at the 8 s initial
  timeout with no speech onset (0.0361 s offline processing).
- A saved-speech region was cropped to high-VAD frames with 128 ms context, repeated
  with a one-second gap, and followed by silence. Both speech regions were retained;
  the endpoint was 10.944 s, with 2.016 s final silence (0.0425 s processing).
- An earlier naive concatenation of complete recordings plus a one-second added
  gap ended after the first clip. Those files already contain leading/trailing
  silence, so their actual gap exceeded the intended short-pause condition. That
  result is retained and labeled in the replay report; the corrected fixture above
  makes the pause condition explicit rather than misreporting it as a success.

Details: stt-automatic-vad-replay.json. No synthetic signal is presented as a new
English/Hindi/Hinglish microphone recording.

## Actual recognition/model comparison

Python 3.11.9; Faster-Whisper 1.2.1, CTranslate2 4.8.5; RTX 3050 Laptop 4096 MiB.
The public multilingual medium model was downloaded into this workspace at
.models/faster-whisper-medium (1,527,906,378-byte model.bin, approximately 1.53 GB),
using the [Systran model repository](https://huggingface.co/Systran/faster-whisper-medium/tree/main)
revision 08e178d. It is ignored by git and is optional, not the default model.
No dependency upgrade or original backup modification was performed.

Each run loaded one model and reused identical decoded samples across profiles
and passes. legacy explicitly reproduces auto + prior vocabulary hint + single
initial language choice; native is auto + no prompt + per-segment detection.
Forced English/Hindi were evaluated separately, not treated as Hinglish remedies.

| Model / compute | Cold model load | Sampled total GPU peak | Legacy median | Native median |
| --- | ---: | ---: | ---: | ---: |
| small / CUDA float16 | 1.983 s | 785 MiB | 0.736 s | 0.747 s |
| medium / CUDA int8_float16 | 4.858 s | 1233 MiB | 1.619 s | 1.665 s |

Medians use four decodes per profile (two files, two passes). GPU memory includes
other processes and sampling can miss transient peaks. Cold runtime work on the
first decode is separate from model constructor timing. No measured accuracy
winner or overall speedup is asserted. Medium fit the measured 4 GB environment
but decoded slower and still made errors. Small remains the default; medium can
be selected explicitly in the same integrated voice interface.

Actual examples, all unverified against human references:

| Source / configuration | Raw output |
| --- | --- |
| 5 s / small native | मेरा नाम केबल मिन्दा है, मैं एक वोईस हो इस प्रुजिक पे काम कर रहा हूं |
| 5 s / medium legacy | मेरा नाम केवल मिंडा है, मैं एक VoiceOS प्रूजेक पे काम कर रहा हूं। |
| 5 s / medium native | मेरा नाम केवल मिंडा है, मैं एक वोईस वोईस प्रूजेक पे काम कर रहा हूं। |
| 15 s / small native | कल मेरे DbMS के अजाम है, मुझे रिमान्टर सेट कन रहा है, मैं वोई सो यस परज़िक ते काम कर रहा है। |
| 15 s / medium legacy | कल मेरा DBMS का exam हैं, मुझे reminder से act करना हैं, मैं VoiceOS इस project पर काम कर रहा हूं। |
| 15 s / medium native | कल मेरा DBMS का exam है, मुझे reminder से act करना है, मैं voice over इस project पर काम कर रहा हूं। |

Medium's exact configured केवल मिंडा name phrase can be normalized to Keval Minda
in the explicit name-introduction context. The small model's uncertain केबल is
preserved. Medium still misrecognized VoiceOS in the prompt-free result and other
wording is uncertain. Existing configured casing changes DbMS to DBMS separately.
These observations do not establish Hindi/Hinglish accuracy improvement.

The final small/CPU float32 native run loaded in 1.824 s and had median decode
6.609 s. The 5 s source took 6.290/6.456 s; the 15 s source 6.762/6.766 s. Text
remained Devanagari-dominant with spelling errors. CPU fallback is functional but
has no real-time guarantee. No Gemini call occurs during STT. The model and VAD
session are not reinitialized per voice command.

Reports: stt-automatic-small.json, stt-automatic-medium.json and stt-automatic-cpu.json.
The user's exact references for the saved WAVs remain unavailable. WER/CER, name/
term accuracy rates, semantic language preservation and translation-error rates
remain null/unverified. English forced-decoder outputs on these clips are not
independent English microphone tests. Fresh labeled English/Hindi/Hinglish tests
and live endpoint timing remain pending; prior microphone deferral was respected.

## Verification and integration

The integrated path is activation -> automatic capture -> transcribe -> separate
vocabulary normalization -> raw/normalized display -> existing review interface ->
existing command processor. No new actions, SQL, shell execution, notes, reminders
or router features were introduced. Existing rejection/clarification and destructive-
action confirmation remain intact. Safety-limit/empty/unusable/declined speech
does not bypass those controls or enter NLU. Accepted text uses the same allowlist.

Tests cover language-independent endpointing, pauses below two seconds, silence
after speech, onset near the initial deadline, speech-like continuous noise, callback
overflow/stall/bad blocks, exact maximum lengths, stream closing, ready notification
after open, model/VAD reuse, native output preservation with mocked text, vocabulary
matching and existing command safety. Actual Silero/model replays above are separate
from mocked tests. Final verification with the workspace Python 3.11.9 interpreter:

```text
python -m pytest -q
277 passed, 1 deselected in 5.19s

python -m ruff check .
All checks passed!
```

The deselected test is the existing opt-in live Gemini test. Formatting and Python
compilation checks passed. CLI help exposes automatic capture, model selection and
action-free automatic reference collection. Original WAV and production DB hashes
matched their pre-edit values. Hardware figures above use actual saved audio/model
execution; no new physical microphone recording or live Gemini request was made.

Changed modules: audio/recording.py, audio/audio_config.py, audio/speech_to_text.py,
audio/benchmark.py, audio/diagnostics.py, main.py, speech/config/voice tests and the
new tests/test_automatic_recording.py, plus configuration/documentation/.gitignore.
The transcript normalizer, NLU, action router, action modules and database code were
audited but not changed in this phase.

## Run and validate

```powershell
.\.venv\Scripts\Activate.ps1
python main.py --list-devices
python main.py --voice --language auto
python main.py --voice --silence-seconds 2 --max-duration 60 --initial-timeout 8
# Optional measured medium model:
python main.py --voice --model .models/faster-whisper-medium --device cuda --compute-type int8_float16 --language auto
```

To perform a live automatic test without invoking application actions, use the same
capture implementation through the existing benchmark recording tool:

```powershell
python -m audio.benchmark init --manifest benchmark-recordings/automatic-live/session.json
python -m audio.benchmark record --automatic --manifest benchmark-recordings/automatic-live/session.json --case english --max-duration 60
```

Wait for Listening, speak naturally, pause briefly inside a sentence, then remain
silent. Automatic capture uses no beep while the microphone is open. Raw float32
samples, PCM16 playback WAV and stop metadata are saved. It performs no transcription
or action. Repeat for hindi and hinglish_technical, using the prompts/procedure in
SPEECH_BENCHMARK.md as suggestions only. After EACH recording, confirm the exact
words actually spoken, before reading Whisper output. Only manually checked words
belong in a UTF-8 reference file:

```powershell
python -m audio.benchmark verify --manifest benchmark-recordings/automatic-live/session.json --case english --reference-file benchmark-recordings/automatic-live/english-reference.txt
python -m audio.benchmark run --manifest benchmark-recordings/automatic-live/session.json --output stt-automatic-verified.json --profiles legacy native english hindi --repeat 2 --gpu-memory
python -m pytest -q
python -m ruff check .
```

Repeat the reference confirmation for Hindi/Hinglish and score only verified cases.
Keep longer pauses in mind: a pause exceeding the configured threshold deliberately
ends capture. Very quiet speech or background voices can confuse VAD, continuous
speech-like noise reaches the safety limit, and native decoding can still produce
wrong language/script/meaning. No perfect or live-validated Hinglish claim is made.
