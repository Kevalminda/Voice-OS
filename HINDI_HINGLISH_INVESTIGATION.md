# Hindi and Hinglish investigation — 15 November 2026

Medium substantially improved the **one playback-confirmed Hinglish recording**.
Under the same automatic-language baseline, raw WER changed from 78.95% with
small to 21.05% with medium; raw CER changed from 58.33% to 15.28%. Both compute
types reproduced those scores in five further baseline passes. This is a measured
improvement on one sentence, not a claim that multilingual recognition is fixed.

The newly captured Hindi file contained background speech, as the user confirmed
after playback. Its initially associated reference was withdrawn. Its historical
WER/CER values are invalid and excluded from all accuracy comparisons. A Hindi
replacement slot exists. The user chose to use the confirmed Hinglish sample
only for now, so no further Hindi recording was made.

The complete tables show every actual transcript, normalized transcript,
reference status, model/compute, VAD parameters, language, beam, prompt, WER/CER,
decode latency and sampled GPU memory:

- [Verified Hinglish: 104 calls](HINDI_HINGLISH_COMPARISON.md).
- [Unverified older files and withdrawn Hindi association: 252 calls](HINDI_UNVERIFIED_COMPARISON.md).
- Machine-readable results: `stt-hinglish-verified-*.json`,
  `stt-hinglish-repeat-*.json`, `stt-hindi-*.json`.
- Recording manifest: `benchmark-recordings/hindi-investigation/session.json`.
  These private artifacts are retained locally and excluded by `.gitignore`.

## Findings from the audit

The decoder explicitly uses `task="transcribe"`. There is no translation utility,
LLM transcript rewriting, or command execution in STT or the benchmark. Raw text
is retained before the separate deterministic vocabulary normalizer runs.
Automatic mode enables Faster-Whisper's per-decoding-window language detection;
this is not word-by-word code-switching or an independent mixed-language model.

The exact confirmed Hinglish waveform demonstrates where unintended English
output can arise: explicitly forcing `en` made both models produce an English
paraphrase in their **raw decoder output**, despite the transcription task.
Small returned: “Today I am using Python and Gemina on this project and the work
of BbMS is still pending.” Medium similarly returned English. The normalizer did
not introduce this change. Forced Hindi matched the automatic baseline on this
clip and did not improve its errors.

Automatic detection selecting Hindi is compatible with a sentence containing
Hindi grammar and English names. A language probability such as 0.97 estimates
the model's language choice; it does not measure word accuracy, spelling,
speaker identity, or preservation of all languages. A coherent forced-English
result alone cannot establish that English is the correct choice for an older
unverified recording. Use explicit English for known English speech; do not
apply it to all Hinglish.

The small model's phonetic and technical-term errors persisted with VAD disabled.
Medium's improvement under otherwise identical settings supports model capacity
as a contributor on this clip. VAD boundaries and prompts also changed output,
but no universal VAD or prompt improvement was demonstrated. The Hindi
background capture was a recording/reference association problem; its results
cannot diagnose recognition of the intended Hindi sentence.

## Preserved recordings and references

| Exact sample | Duration | Reference status | Actual baseline result / disposition |
| --- | ---: | --- | --- |
| `test_audio.wav`, decoded SHA `0e2c91…` (full SHA in table) | 5.000 s | Unavailable | Small: “मेरा नाम केबल मिन्दा है, मैं एक वोईस हो इस प्रुजिक पे काम कर रहा हूं”; medium: “मेरा नाम केवल मिंडा है, मैं एक वोईस वोईस प्रूजेक पे काम कर रहा हूं।” No WER/CER. |
| `stt-mixed.wav`, decoded SHA `fc146e…` (full SHA in table) | 15.000 s | Unavailable | Small: “कल मेरे DbMS के अजाम है, मुझे रिमान्टर सेट कन रहा है, मैं वोई सो यस परज़िक ते काम कर रहा है।” Medium: “कल मेरा DBMS का exam है, मुझे reminder से act करना है, मैं voice over इस project पर काम कर रहा हूं।” No accuracy ranking. |
| `hindi.npy` / playback `hindi.wav`, SHA `c88807b6f2b61f4d0f61771a7637afc4fdc254e63c189cdce7e6841aa3f6de9f` | 13.184 s | **Withdrawn: background speech** | User heard the saved WAV and rejected the speaker/content association. Raw audio/results retained; active scores unavailable. |
| `hinglish_technical.npy` / playback WAV, SHA `1adf58c7904830c398a02ca2145e9a4514a51968b4c0691e950f56cfc6add5c2` | 12.832 s | **Human verified, reconfirmed after playback** | Scored against the exact sentence below. All configurations reused this same read-only float32 waveform. |
| Reported earlier 5.632 s / 6.208 s live tests | — | Waveforms unavailable | User-reported transcripts retained in `reported-tests.json`; these are different recordings and cannot support a controlled model/VAD comparison. |

Verified Hinglish reference:

> आज मैं VoiceOS project में Python और Gemini इस्तेमाल कर रहा हूँ, और DBMS का काम अभी बाकी है।

Small float16, automatic baseline, raw:

> आज में वोईस हो इस प्रुजेक पे पाइटन और जेमिना इस तमाल कर रहा हों और भी भी एमस का काम अभी बाखी है

Medium, both compute types, automatic baseline, raw:

> आज मैं voiceOS प्रोजेक पे Python और Gemini इस्तमाल कर रहा हूँ और BBMS का काम अभी बाकी है।

Medium's normalized output only canonicalizes `voiceOS` to `VoiceOS`; it does not
repair `BBMS`, `प्रोजेक`, `पे`, or `इस्तमाल`. Raw and normalized metrics are equal
under case-insensitive scoring. DBMS remains an actual recognition error.

The two new captures used the existing automatic recorder with 16 kHz mono
float32 input, a readiness cue after stream opening, and a 2.016-second observed
silence endpoint. Both have zero clipped samples and no end-truncation flag.
Their wall capture times were 13.282 s and 12.969 s respectively. Those signal
checks do not establish speaker identity or transcription accuracy. The Hindi
playback check is why the incorrect reference association was discovered.

NPY files retain the exact samples used for inference; WAVs are PCM16 playback
copies. Every run verifies the sample hash before scoring. The withdrawn Hindi
reports retain obsolete values only under `invalidated_reference_metrics` for
audit history; active metrics and summaries have zero scored calls and null
WER/CER. No model output was used as ground truth.

## VAD comparisons on the verified Hinglish sample

| VAD experiment | Discarded signal | Small raw WER | Medium int8_float16 raw WER |
| --- | ---: | ---: | ---: |
| On: threshold .5, silence 500 ms, pad 300 ms | 6.472 s | 78.95% | 21.05% |
| Off | 0 s | 73.68% | 63.16% |
| Silence 250 / 1000 / 2000 ms | 6.472 s each | 78.95% each | 21.05% each |
| Pad 0 ms | 7.072 s | 73.68% | 52.63% |
| Pad 150 ms | 6.772 s | 63.16% | 57.89% |
| Pad 600 ms | 5.872 s | 78.95% | 52.63% |
| Threshold .35 | 6.376 s | 68.42% | 68.42% |
| Threshold .65 | 6.536 s | 68.42% | 63.16% |

Base VAD retains sample interval `[76096, 177856)` = 4.756–11.116 s.
It discards the first 4.756 s (RMS .000332, peak .00497) and final 1.716 s
(RMS .0000368, peak .000244). There are no discarded interior gaps in this case.
Zero padding shifts the leading boundary to 5.056 s and includes a peak .06726
in the discarded opening, so zero padding can remove substantial onset signal.
It must not be adopted just because one model's score happened to improve.

The medium baseline retains the ending “अभी बाकी है”; disabling VAD does not
improve it. This provides no evidence that base VAD lost the final words in this
clip. It does **not** establish that every quiet syllable was retained. Identifying
speech in discarded intervals still needs listening/annotation; `lost_final_words`
and `lost_quiet_syllables` remain null rather than invented labels. Lossless
discarded excerpts and playback WAVs are under
`benchmark-recordings/hindi-investigation/discarded-hinglish/`, with exact paths
and coordinates in each report's `vad_audit` field.

For the unverified 5 s / 15 s recordings, base VAD discards .564 s / 9.504 s.
Without references or boundary annotations these durations cannot be called
lost speech. The full table keeps those experiments separate.

## Decoding, prompts and vocabulary

All experiments use transcription mode and preserve the model output. The
controlled base is auto language, multilingual windows, beam 5, temperature 0,
no prompt, and the VAD settings above. Nineteen profiles test one change at a
time; a separate row tests the application's current temperature fallback list.
Two passes per profile measure repeatability, not independent speakers/samples.

On medium int8_float16, beams 1 / 3 / 5 / 8 gave raw WER
57.89% / 52.63% / 21.05% / 21.05%. Beam 8 added latency without better accuracy.
The current temperature fallback list matched temperature 0 on this clip.

| Prompt | Small raw WER | Medium int8_float16 raw WER | Observation |
| --- | ---: | ---: | --- |
| None | 78.95% | 21.05% | Best medium result, tied with several controls. |
| General Hindi | 68.42% | 52.63% | Hurt medium. |
| General Hindi–English mix | 68.42% | 31.58% | Hurt medium overall. |
| Keval Minda, VoiceOS, Python, Gemini, DBMS | 131.58% | 26.32% | Small repeated “प्रूजक”; medium recognized DBMS but worsened overall WER. |

WER can exceed 100% because insertions count as errors. Prompt success on a
target spelling is not success on the whole sentence. No prompt was promoted
into the default configuration, and no transcript-specific correction was added.

The configurable normalizer remains conservative: exact complete-word/phrase
aliases, no fuzzy phonetics, sentence rewriting, translation or global name
substitution. Ambiguous Keval/Minda aliases require the configured adjacent full
name and an introduction context. The confirmed Hinglish sentence contains no
personal name, so this run does not establish Keval Minda recognition accuracy.
The older samples are not substitutes for a verified name reference.

WER uses NFC, case folding, punctuation removal and whitespace tokenization;
CER uses Unicode code points with spaces omitted. This reference has 19 words
and 72 scored code points. Latin technical terms versus Devanagari spellings
count as errors. These metrics measure the specified written reference and do
not independently measure meaning or prove absence of translation. No special
scoring exceptions were introduced to improve the reported numbers.

## GPU performance and candidate configuration

RTX 3050 Laptop GPU, 4096 MiB; Faster-Whisper 1.2.1, CTranslate2 4.8.5, Python
3.11.9. Each model was loaded in a separate sequential process. No simultaneous
model residency, CUDA failure, OOM or CPU fallback was observed in these runs.
Weights were already local; load time is a new-process load with potentially
warm OS file cache, not an uncached download or guaranteed cold-disk measurement.

The separate five-pass baseline check used identical settings and the exact
confirmed waveform. Accuracy was unchanged in all five passes for each model.

| Model / compute | Load s | First decode s | Passes 2–5 median s | Passes 2–5 range s | Sampled run GPU MiB | Raw WER / CER |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| small / float16 | 3.117 | 1.469 | 1.090 | 1.058–1.101 | 785 | 78.95% / 58.33% |
| small / int8_float16 | 2.838 | 1.631 | 1.175 | 1.166–1.206 | 497 | 78.95% / 58.33% |
| medium / int8_float16 | 5.079 | 2.276 | 4.210 | 1.984–5.460 | 1233 | 21.05% / 15.28% |
| medium / float16 | 3.420 | 2.066 | 3.743 | 1.788–6.709 | 2161 | 21.05% / 15.28% |

Memory is sampled total device usage, including other processes, not an exact
per-process allocation peak. The longer 19-profile matrix peaked at 849 MiB for
small float16 and 1265 MiB for medium int8_float16. The full per-call table also
reports sampled memory or explicitly says no sample occurred during that call.
Medium timings varied considerably; these measurements do not justify a stable
latency guarantee or a universal speed ranking. CPU fallback remains available
in the application, but this investigation did not repeat CPU accuracy testing.

**Best measured candidate for this sample:** medium, auto language, multilingual
windows, beam 5, no prompt, base VAD. Both medium compute types tie on accuracy.
`int8_float16` leaves more VRAM headroom; float16 also fits in the observed runs.
The application's small default and private `.env` were not silently changed
from one verified sentence. To try the candidate through the existing reviewed
VoiceOS flow:

```powershell
$env:VOICEOS_STT_BEAM_SIZE = '5'
$env:VOICEOS_STT_VAD = 'true'
$env:VOICEOS_VAD_THRESHOLD = '0.5'
$env:VOICEOS_VAD_MIN_SILENCE_MS = '500'
$env:VOICEOS_VAD_SPEECH_PAD_MS = '300'
$env:VOICEOS_STT_TEMPERATURES = '0'
python main.py --voice --model .models/faster-whisper-medium --device cuda --compute-type int8_float16 --language auto --multilingual --prompt= --save-recordings benchmark-recordings/voice-captures
```

Those environment settings affect only that terminal process and its children.
Persistent preferences can use the documented `VOICEOS_*` configuration. Known
English or Hindi speech can explicitly select `en` or `hi`; Hinglish keeps auto
available and is never automatically forced into one language by a probability.

## Changes, integration and checks

Speech-scoped code/configuration changes in this investigation:

- `audio/capture_archive.py` (new): opt-in unique lossless audio plus metadata,
  written before inference and retained even if inference fails.
- `audio/audio_config.py`, `.env.example`: optional recording directory setting.
- `audio/speech_to_text.py`: archive integration, failure retention and clearing
  stale recording paths before the next microphone activation.
- `main.py`: `--save-recordings` and display of the saved path, including on
  failed recognition. Existing review/confirmation flow retained.
- `audio/vad_diagnostics.py` (new): exact discard intervals, signal levels and
  lossless/playback exports; unknown speech-loss labels stay unknown.
- `audio/diagnostics.py`: per-call memory sampling slices.
- `audio/benchmark.py`: controlled profiles, VAD/GPU audit fields, CUDA-required
  runs, hash-checked capture import without inferred references, and reference
  withdrawal that preserves the original waveform.
- `audio/comparison_report.py` (new): complete per-sample comparison tables.
- `tests/test_hindi_investigation.py` (new), `tests/test_voice_integration.py`:
  offline archive/VAD/reference/GPU-slice contracts and CLI integration.
- `README.md`, this report and the generated comparison tables: reproduction,
  reference checks, measured results and limitations.

Automatic silence recording and model/VAD reuse remain in place. Recognition
and benchmarks do not call Gemini. In the application, displayed raw/normalized
text still goes through `yes/no/edit` review and the existing validated command
boundary. Deletion retains its additional confirmation. Allowlisted intents,
unknown-command rejection and storage validation were not changed. No shell,
arbitrary SQL or unrestricted system-action route was introduced.

Final checks with the workspace Python 3.11 environment:

```text
python -m pytest -q       289 passed, 1 deselected
python -m ruff check .    All checks passed!
```

Pytest uses mocks/temporary data and performs no live microphone, model or API
accuracy testing. The recordings and GPU experiments above are separate manual
hardware results. Existing English, Hindi, Hinglish, silence/noise, short/long,
ambiguous language, pause/endpoint, vocabulary and action-safety contracts remain
covered offline; derived silence/noise/tiled stress signals are not a substitute
for independently verified natural recordings.

Original file SHA-256 values were unchanged after verification:

```text
test_audio.wav  8b9074714649e3829a57d441c99ba73b184662b4b692ee8ac2eac96dc0d515ab
stt-mixed.wav   3d4a8e044fb518deab3b648275299b4b46da9fdd946781e2efdb66f6e31fa7cb
voiceos.db     d59667a69e74ff737354c0eab356231592733ac0b45c4681a90de2c6871ae219
```

No notes/reminders/router/storage implementation was modified. Remaining work
for general accuracy claims is a correct-source Hindi reference, independent
English and mixed-language/name recordings, more speakers/noise/pause conditions,
and human review of discarded quiet boundaries. No current configuration
guarantees faithful native code-switching, technical spelling or no paraphrasing.
