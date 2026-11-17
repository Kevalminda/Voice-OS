# Changes — reliability milestone

## Speech subsystem — 15 November 2026

- Added validated capture/configuration, reusable local inference and bounded CPU fallback.
- Added raw/normalized transcript results, conservative vocabulary correction and timing diagnostics.
- Replaced import-time microphone scripts with guarded record-once/replay diagnostics.
- Added reviewed voice CLI integration through the existing text processor and safety controls.
- Added mocked speech/voice regression tests and actual cached-WAV GPU/CPU measurements.
- See STT_REPORT.md for measurements, limitations and manual accuracy checks.

## Previous text reliability milestone

- Centralized existence check, content preview, explicit approval and checked deletion.
- Changed confirmation callback from `() -> bool` to `(ActionResult) -> bool`.
- Tightened JSON, entity, language, confidence and note-ID handling.
- Added configurable Gemini model/timeout, structured response schema and bounded retries.
- Preserved mandatory Pydantic validation after SDK output.
- Added isolated SQLite fixtures, blocked offline sockets, and opt-in live API test.
- Closed database initialization/client resources; added safe logs and graceful shutdown.
- Removed duplicate imports and false-success reminder/search handlers.
- Removed unused planned dependencies; added environment template and project configuration.
- Added regression coverage and setup/audit/verification documentation.
