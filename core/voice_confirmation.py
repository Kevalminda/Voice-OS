"""Application-layer voice reply collection; never submits a command or invokes NLU."""

import copy
import math
import secrets

_CANCEL = {"cancel", "/cancel", "रद्द करो", "cancel karo", "exit"}
_REJECT = {"no", "n", "नहीं", "ना", "nahi", "nahin"}
_REVIEW_ACCEPT = {"yes", "y", "हाँ", "हां", "haan", "han"}
_KEYBOARD = {"keyboard", "type"}


class VoiceConfirmationReader:
    """Reuse one STT object, with human review and fail-closed diagnostic checks.

    Capture uses its existing automatic endpoint and a configured safety limit
    of at most 60 seconds. It can finish after confirmation expiry, but a late
    capture is never transcribed or authorized. No STT configuration is changed.
    """

    def __init__(self, stt, *, read_input=None, write=None):
        self.stt = stt
        self._input = read_input or input
        self._write = write or print
        self.last_transcript = None
        self.last_capture_files = None

    def _current(self, confirmation, pending, token):
        invalid = confirmation.check_pending(pending)
        if invalid is not None:
            self._write(f"VoiceOS: Confirmation {invalid.outcome.value}; no reply accepted.")
            return False
        # A replaced request must not receive an earlier in-flight recording,
        # even if both deletions have identical note metadata.
        if not secrets.compare_digest(confirmation.token, token):
            confirmation.cancel()
            self._write("VoiceOS: Confirmation changed; the previous reply was discarded.")
            return False
        return True

    def __call__(self, confirmation, pending, token):
        if not self._current(confirmation, pending, token):
            return None
        choice = self._input(
            f"Press Enter to speak confirmation for token {token}; or keyboard/cancel/edit: "
        )
        if not self._current(confirmation, pending, token):
            return None
        choice = " ".join(choice.casefold().split())
        if choice in _KEYBOARD:
            return self._keyboard(confirmation, pending, token)
        if choice in _CANCEL:
            return "cancel"
        if choice in _REJECT or choice == "edit":
            return choice
        if choice:
            self._write("Choose keyboard to type a reply, or press Enter to record.")
            return ""
        config = self.stt.config
        limit = config.max_recording_duration
        if (
            config.recording_mode != "automatic"
            or isinstance(limit, bool)
            or not isinstance(limit, (int, float))
            or not math.isfinite(limit)
            or not 0 < limit <= 60
        ):
            self._write(
                "Voice confirmation needs automatic recording limited to 60s; use keyboard."
            )
            return ""
        from audio.speech_to_text import SpeechToTextError

        try:
            result = self._capture(confirmation, pending, token)
        except (SpeechToTextError, OSError, ValueError, TypeError):
            self._write("Confirmation capture failed. Retry or choose keyboard at the next prompt.")
            return ""
        if result is None:
            return None
        self._write(f"Confirmation heard: {result.get('raw_text', '')}")
        warnings = result.get("warnings")
        for warning in warnings if isinstance(warnings, list) else []:
            self._write(f"VoiceOS: {warning}")
        flags = result.get("review_flags")
        for flag in flags if isinstance(flags, list) else []:
            if isinstance(flag, dict):
                self._write(
                    f"Review {flag.get('kind', 'speech')}: {flag.get('message', 'Verify speech.')}"
                )
            else:
                self._write("Confirmation needs review; use keyboard or retry.")
        if not self._usable(result):
            self._write("Unusable or uncertain confirmation. Retry or choose keyboard.")
            return ""
        review = self._input(
            "Is this your intended spoken response, exactly? (yes/no/keyboard/cancel/edit): "
        )
        if not self._current(confirmation, pending, token):
            return None
        review = " ".join(review.casefold().split())
        if review in _KEYBOARD:
            return self._keyboard(confirmation, pending, token)
        if review in _CANCEL:
            return "cancel"
        if review in _REJECT:
            return "no"
        if review == "edit":
            return "edit"
        # Review approves only the transcription. The exact raw reply must still
        # pass ConfirmationSession's token parser and final deadline checks.
        return result["raw_text"] if review in _REVIEW_ACCEPT else ""

    def _keyboard(self, confirmation, pending, token):
        if not self._current(confirmation, pending, token):
            return None
        reply = self._input(f"Approve with yes {token}; or no/cancel/edit: ")
        return reply if self._current(confirmation, pending, token) else None

    def _capture(self, confirmation, pending, token):
        from audio.capture_archive import finish_capture, preserve_capture

        if not self._current(confirmation, pending, token):
            return None
        # These are diagnostic snapshots, never cached authorization replies.
        # Keep the last recording across cancellation/keyboard selection, but
        # discard it before any new capture so a failed capture cannot reuse it.
        self.last_transcript = None
        self.last_capture_files = None
        self.stt.last_capture_files = None
        audio = self.stt.record_audio(
            on_started=lambda: self._write("Listening for confirmation. Speak, then leave silence.")
        )
        current = self._current(confirmation, pending, token)
        info = copy.deepcopy(self.stt.last_recording_info or {})
        files = None
        if self.stt.config.recording_directory:
            files = preserve_capture(
                audio, self.stt.config.recording_directory, info, self.stt.configuration()
            )
            self.last_capture_files = files
            self.stt.last_capture_files = files
            self._write(f"Saved confirmation recording: {files['metadata']}")
        if not current or not self._current(confirmation, pending, token):
            if files:
                finish_capture(files, None)
            return None
        if not self._safe_capture_info(info):
            if files:
                finish_capture(files, None)
            raise ValueError("The confirmation recording did not complete safely.")
        try:
            result = self.stt.transcribe(audio, recording_info=info)
            if not isinstance(result, dict):
                raise ValueError("STT did not return transcript metadata.")
        except BaseException:
            if files:
                try:
                    finish_capture(files, None)
                except OSError:
                    self._write("Confirmation audio saved; failure metadata could not be updated.")
            raise
        self.last_transcript = copy.deepcopy(result)
        current = self._current(confirmation, pending, token)
        if files:
            finish_capture(files, result)
        return result if current and self._current(confirmation, pending, token) else None

    @staticmethod
    def _safe_capture_info(info):
        return (
            isinstance(info, dict)
            and info.get("mode") == "automatic"
            and info.get("speech_started") is True
            and info.get("stop_reason") == "silence"
        )

    @staticmethod
    def _usable(result):
        raw = result.get("raw_text")
        info = result.get("recording_info")
        diagnostics = result.get("audio_diagnostics")
        if (
            result.get("status") != "ok"
            or result.get("empty_transcription") is not False
            or result.get("has_lexical_content") is not True
            or not isinstance(raw, str)
            or not raw.strip()
            or not VoiceConfirmationReader._safe_capture_info(info)
            or not isinstance(diagnostics, dict)
            or diagnostics.get("possible_truncation") is not False
            or diagnostics.get("near_silent") is not False
            or not isinstance(result.get("warnings"), list)
            or result["warnings"]
            or not isinstance(result.get("review_flags", []), list)
            or result.get("review_flags")
        ):
            return False
        clipped = diagnostics.get("clipped_fraction")
        # Probability/segment scores are not calibrated recognition guarantees.
        return (
            not isinstance(clipped, bool)
            and isinstance(clipped, (int, float))
            and math.isfinite(clipped)
            and clipped == 0
        )
