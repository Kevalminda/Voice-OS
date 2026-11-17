# Medium candidate recheck — 16 November 2026

Subsequent validation collected five English and five Hindi recordings:
[completed Round 2 report](ROUND2_VALIDATION_REPORT.md).

**Recommendation: continue with Medium int8_float16 as the opt-in candidate;
defer making it the global default.** The verified Hinglish improvement and real
application integration reproduced. English and Hindi remain unscored under the
user's existing instruction. Repetition of one recording does not establish
general multilingual accuracy, and the candidate still makes four word errors.

The existing implementation and investigation reports were inspected. No source
code, decoding settings, vocabulary, private `.env`, recording logic, action
router or storage implementation was changed in this recheck. New dated reports
and replay archives preserve the prior evidence.

## Candidate selection and application execution

The shared `medium-candidate` preset remains Medium / CUDA / int8_float16,
automatic language with multilingual decoding windows, beam 5, no initial
prompt, temperature 0, VAD enabled with threshold .5, minimum silence 500 ms and
speech padding 300 ms. These are the previously measured settings, with no new
decoding search.

Effective speech configuration was reloaded from the application environment.
An ordinary `--voice` invocation still selects Small/auto compute. Medium is
actually selected through the existing opt-in command:

```powershell
python -X utf8 main.py --voice --stt-profile medium-candidate --save-recordings benchmark-recordings/voice-captures
```

The real application voice loop was replayed with the preserved waveform, actual
Whisper inference and normalization. Only microphone input, terminal answers and
the storage/action boundaries were substituted, as documented in
[the validation procedure](MEDIUM_CANDIDATE_VALIDATION.md).

- One actual CUDA/int8_float16 model creation, reused for two transcriptions.
- Model load 6.826 s; decode calls 3.847 s and 1.776 s.
- Sampled total-device GPU peak 1235 MiB; zero action submissions.
- Both transcripts were displayed and declined through the existing review flow.
- Both replay archives match the preserved float32 waveform hash and retain
  raw and normalized transcripts separately. They are marked saved-waveform
  replay, not fresh microphone recordings.

In addition to inspecting configuration, this run observed the options received
by the real Faster-Whisper `transcribe` method without changing them. Both calls
used `task="transcribe"`, `language=None`, `multilingual=True`, `beam_size=5`,
`initial_prompt=None`, temperature 0 and the expected VAD parameters. Those
observations are stored in `stt-medium-validation-app-20261116.json` under
`observed_decoder_options`.

## Reference-gated language and sample results

The existing repeatable manifest is
`benchmark-recordings/medium-validation/session.json`. English/Hindi empty slots,
record-once preservation, playback-before-verification procedure, hash-bound
manual references, and language/sample summaries already exist. No new reference
was invented or obtained from model output. The withdrawn background-Hindi file
remains excluded.

| Corpus language / sample | Verified recordings | Small raw WER / CER | Medium raw WER / CER |
| --- | ---: | --- | --- |
| English / no verified recording | 0 | Unavailable | Unavailable |
| Hindi / no verified recording | 0 | Unavailable | Unavailable |
| Hinglish / `hinglish_technical.npy`, 12.832 s | 1 | 78.95% / 58.33% | **21.05% / 15.28%** |

The user-verified reference remains:

> आज मैं VoiceOS project में Python और Gemini इस्तेमाल कर रहा हूँ, और DBMS का काम अभी बाकी है।

Waveform SHA-256:
`1adf58c7904830c398a02ca2145e9a4514a51968b4c0691e950f56cfc6add5c2`.

Each model was evaluated five times on that identical waveform, with model
instances loaded in separate sequential processes. Every pass reproduced its
model's WER/CER. Normalized WER/CER equaled raw WER/CER for these baseline runs.
Scores were independently recomputed from the human reference to check report
integrity. Results remain separated by sample and corpus language, independently
of the decoder's detected `hi` label.

[Complete current comparison table](MEDIUM_CANDIDATE_COMPARISON_20261116.md)
contains every raw/normalized transcript, reference, configuration, pass, latency,
VAD discard duration, GPU measurement and missing-language status. Its sources
are `stt-medium-validation-candidate-20261116.json` and
`stt-medium-validation-small-20261116.json`. Five repeats are one independent
recording, not five new accuracy samples.

## Loading, warm latency and GPU memory

| Model / compute | Fresh-process model load s | First decode s | Passes 2–5 median s | Passes 2–5 range s | Sampled total GPU MiB |
| --- | ---: | ---: | ---: | --- | ---: |
| Small / float16 | 3.760 | 1.377 | 1.050 | 1.022–1.172 | 787 |
| Medium / int8_float16 | 6.254 | 3.257 | 1.862 | 1.834–1.918 | 1235 |

Hardware: RTX 3050 Laptop GPU, 4096 MiB. No OOM, CUDA failure or CPU fallback
occurred. GPU benchmarks required CUDA explicitly. Loading measures a cold model
instance in a fresh process; it does not guarantee an empty Windows file cache.
Warm latency excludes the first pass. Sampled total GPU usage includes other
processes and can miss brief peaks. Older runs exhibited larger Medium latency
variation; the faster warm median in this run is not a new decoding improvement
or a latency guarantee.

## Recording, translation and future activation

Real streaming Silero and the unchanged endpoint state machine were replayed:
the saved waveform stopped for silence at 12.832 s with 2.016 s trailing silence.
Synthetic digital silence reached the initial timeout at 8 s without speech
starting. This validates replay behavior, not a new physical microphone capture.
Existing offline tests also cover pauses, noise, initial/max timeouts, final
samples, stream stalls and overflow.

The Medium raw output remained:

> आज मैं voiceOS प्रोजेक पे Python और Gemini इस्तमाल कर रहा हूँ और BBMS का काम अभी बाकी है।

Hindi grammar and Latin technical words were retained in all candidate calls;
no entirely English translation appeared. Normalization only canonicalized
`voiceOS` to `VoiceOS`; the DBMS and other recognition errors remain. Source
inspection and observed decoder arguments confirm transcription mode and no LLM
transcript rewriting. This one-sample result does not guarantee that Whisper
will never paraphrase or mistranscribe unseen mixed speech. Language probability
is not used as an accuracy score.

Wake-word integration remains architecturally possible: activation, capture,
`transcribe(audio)`, model reuse and reviewed command submission are separate.
A future trigger can activate recording, or a shared-stream adapter can supply
audio to the transcriber. Microphone ownership, buffering and handoff still need
design when that feature is requested. No wake-word code, continuous listener,
new action or confirmation bypass was implemented.

## Verification and readiness decision

```text
python -m pytest -q       295 passed, 1 deselected
python -m ruff check .    All checks passed!
```

Offline tests were completed before timed GPU runs. Source and original
audio/database hashes were checked before/after and remained unchanged. Replay
NPY hashes, stored raw transcripts and reference-derived metrics also matched.
The audit is retained in
`benchmark-recordings/medium-validation/audit-20261116.json`.

Keep the candidate available through `--stt-profile medium-candidate`. Default
promotion still needs independently verified English/Hindi and additional natural
Hinglish recordings, including names, technical terms, speakers, noise and pauses,
plus acceptable command accuracy and latency criteria. Current evidence supports
reviewed trials on the candidate, not general multilingual readiness.

To reproduce these fixed-setting runs, use the commands in
[the validation procedure](MEDIUM_CANDIDATE_VALIDATION.md), choosing new dated
output names to preserve earlier results. This recheck changed only reports and
replay/audit artifacts; the existing safety and confirmation flow remains intact.
