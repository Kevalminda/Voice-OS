# Verification

## Automatic recording and native defaults — 15 November 2026

See [AUTOMATIC_RECORDING_REPORT.md](AUTOMATIC_RECORDING_REPORT.md) for the latest
streaming capture tests, actual saved-audio Silero replay and small/medium GPU
comparison. Saved-audio results and mocked tests do not establish live microphone
or per-language recognition success.

## Speech fidelity benchmark — 15 November 2026

See [SPEECH_BENCHMARK.md](SPEECH_BENCHMARK.md) for the latest Windows/Python 3.11.9
verification, reference-gated metrics, actual GPU/CPU comparisons, and live recording
procedure. Fresh microphone tests were deferred by the user. Existing unverified
WAVs were not assigned invented references or WER/CER results.

## Speech subsystem — 15 November 2026

See [STT_REPORT.md](STT_REPORT.md) for the Windows/Python 3.11.9 speech audit,
offline regression results and actual GPU/CPU WAV measurements. The older results
below describe the previous text-only milestone and its separate environment.

## Previous verification — 9 November 2026

- Environment: Linux, Python 3.12.14; intended Python 3.11 not available here.
- Installed SDK inspected: google-genai 1.75.0 (pinned in requirements).
- Pydantic 2.14.3; pytest 9.2.0; Ruff 0.17.4; httpx 0.28.1.
- Baseline: 21 passed, 7 failed (fresh DB without initialization; network blocked).
- Final offline suite: 78 passed, 1 live test deselected.
- Ruff lint and formatting: passed.
- Python compilation: passed.
- CRUD, initialization, preview/approval/cancellation, changed-content race,
  strict parser and mocked SDK failures exercised by tests.
- Production-path sentinel was byte-for-byte unchanged after full test suite;
  no local project SQLite database was created by tests.
- Final archive was CRC-checked, extracted into a fresh directory and retested.
- Live Gemini response/model compatibility and Windows execution NOT verified.
  No live API quota was consumed. Python 3.11 compatibility is a target, not a
  runtime test claim; Ruff targets Python 3.11 syntax.
- Original archive SHA-256: `4bc29badc678818e2dba6e1b1a976cb2705ecd1038809acba9ba9360a2d5c348` (preserved unchanged).

SDK references consulted:
https://googleapis.github.io/python-genai/
https://github.com/googleapis/python-genai/blob/main/google/genai/_api_client.py

Actual installed HttpOptions confirms timeout units are milliseconds;
HttpRetryOptions attempts includes the first request; GenerateContentConfig
supports response_json_schema. Local tests verify the configured arguments.
