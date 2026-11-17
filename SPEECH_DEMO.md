# VoiceOS speech demonstration

VoiceOS is a multilingual voice interaction middleware layer on an existing
operating system. This demonstration covers recording, local STT, conservative
normalization, transcript review and safe rejection. It does not demonstrate
notes, reminders, tasks, wake-word detection or unrestricted system control.

## Launch from the existing project directory

Use the existing Python 3.11 environment and cached models. No installation,
model download or new benchmark is needed for this workspace.

```powershell
# Optional: inspect device IDs before recording.
.\.venv\Scripts\python.exe main.py --list-devices

# Small: retain the global default, demonstrate the local voice flow.
.\.venv\Scripts\python.exe main.py --voice --speech-only --language auto --save-recordings benchmark-recordings/submission-demo-small

# Medium: explicit opt-in, the frozen multilingual candidate.
.\.venv\Scripts\python.exe main.py --voice --speech-only --stt-profile medium-candidate --save-recordings benchmark-recordings/submission-demo-medium
```

Run one instance at a time. The startup display identifies the actual device and
compute type, including CPU fallback. The model and streaming VAD initialize once.
Use `--microphone ID` if the OS default input is not the intended microphone.

Speech-only mode performs no Gemini calls or database initialization. Every
confirmed or edited transcript is rejected at the application-action boundary:
`Application actions are disabled in speech-only mode. [rejected]`. This is an
explicit capability gate, not a claim that an offline NLU classified the command.
It makes the speech demonstration independent of network access, API quota and
application actions.

## Five-minute demonstration

1. Show the startup model/device display. Press Enter only when ready. Wait for
   the terminal's **Listening** prompt, then speak. The ordinary app uses this
   visible prompt; the operator's hardware-test wrapper additionally uses a WAV
   chime. A console beep is not a reliable readiness cue on every Windows setup.
2. English: say “Restart the computer in twenty seven minutes.” Stop speaking and
   leave two seconds of silence. Show the raw text, any normalized spelling and
   number review hint. Choose `yes`; show the explicit action rejection above.
3. Hindi: say “मेरा नाम केवल मिंडा है।” Show Hindi output and any name review hint.
   If spelling is wrong, choose `edit` and type the words actually intended. The
   corrected text still meets the speech-only action rejection boundary.
4. Hinglish: say “आज मैं VoiceOS में Python और Gemini इस्तेमाल कर रहा हूँ।” Show
   the mixed-language raw/normalized output. Choose `no` to demonstrate that a
   declined transcript is not submitted.
5. Press Enter and stay quiet to demonstrate the initial silence timeout
   (default eight seconds). Empty speech is not offered as an application command.
6. Type `exit` at the activation prompt to close. Ctrl+C and EOF also stop cleanly.
   Each completed recording is preserved as a unique lossless NPY plus JSON in
   the selected directory, including raw/normalized text and review hints.

Do not promise an exact transcript for any suggested sentence. Demonstrate
correction/rejection when needed. Whisper uses transcription mode, but errors
in names, numbers, Hindi spelling and natural code-switching remain possible.
Review hints are conservative and non-exhaustive; they are not confidence scores.
Digits and common English/Hindi number words are highlighted without conversion.
Ordinary ambiguous words can also be flagged. Absence of a flag never skips review.

## Recording behavior and recovery

- Normal completion follows two seconds of continuous VAD-confirmed silence.
  Shorter pauses inside a sentence should retain capture.
- The default 60-second maximum is a safety cap, not the normal timer. A capture
  hitting it is displayed but withheld from command submission; retry with a
  complete sentence and a quiet ending.
- Background voices can extend recording; select the correct input and stop
  other speech/playback during the demonstration. Noise is not reliably rejected
  by signal level alone.
- A microphone error produces a user-facing message and returns to activation.
  Retry or exit; an error never submits the previous recording.
- CPU float32 fallback remains available if CUDA is unavailable, with higher
  latency expected. The live GPU result does not constitute a physical GPU-failure
  test; fallback and error paths are covered by offline regression tests.

Optional capture settings remain explicit per-launch overrides:

```powershell
.\.venv\Scripts\python.exe main.py --voice --speech-only --stt-profile medium-candidate --silence-seconds 2 --initial-timeout 8 --max-duration 20 --save-recordings benchmark-recordings/submission-demo-short
```

This short-demo safety limit does not change the global configuration. For
actual longer dictation use the default 60-second limit or a deliberate larger
limit supported by configuration, and leave a final quiet interval.

## Existing application integration

To use the existing Gemini-backed command processor, omit `--speech-only`:

```powershell
.\.venv\Scripts\python.exe main.py --voice --stt-profile medium-candidate --save-recordings benchmark-recordings/voice-captures
```

This path requires the existing private Gemini configuration and network/quota.
Reviewed text reaches the existing validated, allowlisted router; deletion
retains its extra confirmation. No action module was added by this speech
delivery. Do not advertise reminders, task management or arbitrary OS control
as supported features. Use the speech-only launch for a repeatable submission
demonstration when external service availability is uncertain.

Existing accuracy evidence remains in [Round 2](ROUND2_VALIDATION_REPORT.md).
No model comparisons are needed to run this demonstration.
