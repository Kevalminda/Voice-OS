# Reliability audit — 9 November 2026

## Baseline and working features

All 27 supplied files were inspected. The archive used Windows backslashes;
extraction normalized separators safely. The project root contains main.py and
requirements.txt directly. The original archive was preserved.

Existing Pydantic enums/results, confidence and required-field validation, Gemini
NLU service, JSON parser, note CRUD and parameterized SQL were present. Connection
closing already existed in note actions. There were no circular imports found.

Safe baseline execution used a temporary uninitialized database and blocked
network connections: **21 passed, 7 failed**. These seven failures exposed reliance
on an existing initialized local database. This was not a claim that the historical
manual run failed. The original collection also imported a script that attempted a
live Gemini call; it was blocked during baseline execution.

## Findings and priority plan (implemented)

1. Critical: router deletion bypassed CLI confirmation; IDs/existence weren't checked
   before prompting. Centralize confirmation in the router, provide a preview and
   guard the final DELETE against a content change.
2. High: parser converted booleans, containers and arbitrary values into strings.
   Normalize only allowed names, text values and integer IDs; reject duplicate JSON
   keys, malformed shapes, invalid confidence and unsupported intent strings.
3. High: tests shared production storage and live calls could run at import.
   Use per-test initialized temporary DBs, block offline networking, isolate live test.
4. High: Gemini model was hardcoded, timeout/retry/structured output absent.
   Inspect installed SDK, require configured model, bound requests and retry only
   transient failures. Return sanitized errors; close client resources.
5. Medium: database initialization did not explicitly close connections, CLI debug
   prints duplicated user content, main's confirmation callback only existed under
   its execution guard. Close resources and introduce safe logging/shutdown.
6. Medium: reminder/search placeholders returned false success; unused audio/UI
   dependencies increased installation burden. Reject unimplemented routes and
   remove dependencies for deferred features.
7. Medium: README and environment example were missing. Add setup, architecture,
   privacy, limitations, configuration and verification documentation.

## Security and architecture boundaries

No credentials or personal DB were present in the uploaded archive. .gitignore
already excluded common secrets/databases; SQLite sidecar exclusions were added.
SQL remains parameterized. LLM JSON never supplies executable code or confirmation.
There is no implemented authenticated application session; this is documented as
an existing limitation, not silently represented as complete. No audio, reminders,
web search, paid service, external deployment or architecture replacement was added.
Trusted low-level CRUD remains separate from the validated integration router.
