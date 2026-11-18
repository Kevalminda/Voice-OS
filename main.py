import argparse
import json
import logging
import os
import sys
import time

from dotenv import load_dotenv

from core.command_processor import process_text_command as process_text_command
from core.confirmation import ConfirmationOutcome, ConfirmationSession
from core.dialogue import DialogueSession
from core.schemas import ActionResult
from core.timing import InteractionTiming, current_timing
from core.voice_confirmation import VoiceConfirmationReader
from storage.database import initialize_database


class _UnicodeSafeWriter:
    """Escape unsupported characters when a redirected stream cannot be reconfigured."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        try:
            self._stream.write(text)
        except UnicodeEncodeError:
            encoding = self._stream.encoding
            escaped = text.encode(encoding, errors="backslashreplace").decode(encoding)
            self._stream.write(escaped)
        return len(text)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _configure_voice_output():
    """Support multilingual redirected output in both text and voice modes."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream is None or isinstance(stream, _UnicodeSafeWriter):
            continue
        isatty = getattr(stream, "isatty", None)
        if callable(isatty) and isatty():
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
                continue
            except (OSError, ValueError, TypeError):
                pass
        if getattr(stream, "encoding", None):
            setattr(sys, name, _UnicodeSafeWriter(stream))


def confirm_deletion(pending: ActionResult) -> bool:
    return _confirm_deletion(pending)


def _confirm_deletion(pending: ActionResult, response_reader=None, *, timing=None) -> bool:
    timing = timing if timing is not None else current_timing.get()
    if timing is None:
        clock = (
            response_reader.measurement_clock
            if isinstance(response_reader, VoiceConfirmationReader) else time.perf_counter
        )
        timing = InteractionTiming(clock=clock)
    if isinstance(response_reader, VoiceConfirmationReader):
        response_reader.timing = timing
    with timing.measure("confirmation_total_duration"):
        return _read_deletion_confirmation(pending, response_reader, timing)


def _read_deletion_confirmation(pending, response_reader, timing):
    confirmation = ConfirmationSession()
    timing.record["confirmation_outcome"] = "cancelled"
    try:
        token = confirmation.begin(pending)
        print(f"VoiceOS: {pending.message}")
        while confirmation.has_pending:
            invalid = confirmation.check_pending(pending)
            if invalid is not None:
                timing.record["confirmation_outcome"] = invalid.outcome.value
                return False
            timing.record["confirmation_attempt_count"] += 1
            if response_reader is None:
                with timing.measure("keyboard_fallback_duration"):
                    reply = input(f"Approve with yes {token}; or no/cancel/edit: ")
            else:
                reply = response_reader(confirmation, pending, token)
            if reply is None:
                return False
            result = confirmation.respond(reply, pending)
            timing.record["confirmation_outcome"] = result.outcome.value
            if result.outcome is ConfirmationOutcome.APPROVED:
                return True
            if result.outcome is ConfirmationOutcome.INVALID and confirmation.has_pending:
                print(
                    "VoiceOS: Confirmation invalid. Use an approval word and the exact token; "
                    f"{confirmation.remaining_retries} attempt(s) remain."
                )
                continue
            if result.edit_requested:
                print(
                    "VoiceOS: Deletion cancelled. Submit a corrected command separately; "
                    "it will require a fresh confirmation."
                )
            else:
                print(f"VoiceOS: Confirmation {result.outcome.value}.")
            return False
        return False
    except (KeyboardInterrupt, EOFError):
        timing.record["confirmation_outcome"] = "interrupted"
        raise
    except BaseException:
        timing.record["confirmation_outcome"] = "error"
        raise
    finally:
        confirmation.cancel()


def _voice_text(stt, *, timing=None):
    """Review recognized text before submitting it to the same text-command boundary."""
    from audio.speech_to_text import SpeechToTextError

    trigger = input("Press Enter to record, or type exit: ").strip()
    if trigger.lower() == "exit":
        return "exit"
    timing = timing if timing is not None else InteractionTiming()
    timing.resume()

    def ready():
        if stt.config.recording_mode == "automatic":
            print(f"Listening. Speak now; stops after {stt.config.silence_duration:g}s of silence.")
        else:
            print(f"Speak now for up to {stt.config.duration:g} seconds; leave trailing silence.")

    print("Opening microphone...")
    try:
        transcript = stt.listen_and_transcribe(on_recording_started=ready)
    except (SpeechToTextError, ValueError) as error:
        for field in (
            "command_capture_duration", "transcription_duration", "normalization_duration",
        ):
            timing.add(field, None)
        print(f"VoiceOS: {error}")
        if getattr(stt, "last_capture_files", None):
            print(f"Saved recording: {stt.last_capture_files['metadata']}")
        return ""
    timing.add("command_capture_duration", transcript.get("recording_time"))
    timing.add("transcription_duration", transcript.get("transcription_duration"))
    timing.add("normalization_duration", transcript.get("normalization_duration"))
    if transcript.get("recording_files"):
        print(f"Saved recording: {transcript['recording_files']['metadata']}")
    if transcript["empty_transcription"] or not transcript.get("has_lexical_content", True):
        print("VoiceOS: No usable speech. Check microphone/gain or retry with a longer recording.")
        return ""
    print(f"Heard: {transcript['raw_text']}")
    if transcript["raw_text"] != transcript["normalized_text"]:
        print(f"Normalized: {transcript['normalized_text']}")
    for warning in transcript["warnings"]:
        print(f"VoiceOS: {warning}")
    for flag in transcript.get("review_flags", []):
        print(f"Review {flag['kind']}: {flag['text']} — {flag['message']}")
    stop_reason = transcript.get("recording_info", {}).get("stop_reason")
    if stop_reason == "max_duration" or (
        transcript["audio_diagnostics"]["possible_truncation"] and stop_reason != "silence"
    ):
        print("VoiceOS: Please re-record with more time and a silent pause at the end.")
        return ""
    with timing.measure("transcript_review_duration"):
        choice = input("Use this command? (yes/no/edit): ").strip().lower()
        if choice == "edit":
            return input("Corrected command: ").strip()
    return transcript["normalized_text"] if choice in {"yes", "y"} else ""


def _publish_timing(timing, records, *, finish=True):
    if not timing.started or timing.published:
        return
    if finish:
        timing.finish()
    record = timing.snapshot()
    timing.published = True
    try:
        if records is not None:
            records.append(record)
        logging.getLogger(__name__).debug("Interaction timing: %s", json.dumps(record))
    except Exception:
        pass


def main(argv=(), *, measurement_clock=time.perf_counter, timing_records=None):
    _configure_voice_output()
    parser = argparse.ArgumentParser(description="VoiceOS text and reviewed voice commands")
    parser.add_argument("--voice", action="store_true")
    parser.add_argument(
        "--speech-only",
        action="store_true",
        help="Local voice demo: reject all actions; no Gemini or database use.",
    )
    parser.add_argument(
        "--stt-profile",
        choices=("medium-candidate",),
        help="Opt in to the measured Medium candidate; explicit CLI options override it.",
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument(
        "--max-duration",
        "--duration",
        type=float,
        dest="max_duration",
        help="Automatic recording safety limit, not a fixed capture timer.",
    )
    parser.add_argument("--silence-seconds", type=float)
    parser.add_argument("--initial-timeout", type=float)
    parser.add_argument(
        "--language",
        "--preferred-language",
        choices=("auto", "en", "hi"),
        help="Preferred decoder language; auto explicitly enables detection, overriding .env.",
    )
    parser.add_argument("--microphone")
    parser.add_argument(
        "--model", help="Multilingual model name or converted local model directory."
    )
    parser.add_argument(
        "--compute-type", help="Explicit compute type; auto selects a compatible type."
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--save-recordings",
        metavar="DIRECTORY",
        help="Preserve lossless audio and metadata before each transcription.",
    )
    parser.add_argument(
        "--multilingual",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow per-segment language detection in auto mode; evaluate on your own speech.",
    )
    parser.add_argument("--prompt", help="Optional vocabulary hint; --prompt= disables it.")
    args = parser.parse_args(argv)
    if args.speech_only and not args.voice:
        parser.error("--speech-only requires --voice")
    load_dotenv()
    level = os.getenv("VOICEOS_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO), format="%(levelname)s %(name)s: %(message)s"
    )
    # SDK/transport debug logs can contain request payloads; keep them quiet.
    for name in ("google", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    dialogue = None
    active_timing = None
    confirmation_callback = confirm_deletion
    try:
        stt = None
        if args.voice or args.list_devices:
            from dataclasses import replace

            from audio import AudioConfig, SpeechToText, SpeechToTextError, list_input_devices
            from audio.audio_config import language_code, microphone_device
            from audio.stt_profiles import apply_stt_profile

            if args.list_devices:
                for device in list_input_devices():
                    print(f"{device['id']}: {device['name']}")
                return
            overrides = {
                key: value
                for key, value in {
                    "max_recording_duration": args.max_duration,
                    "silence_duration": args.silence_seconds,
                    "initial_silence_timeout": args.initial_timeout,
                    "microphone_id": microphone_device(args.microphone),
                    "multilingual": args.multilingual,
                    "initial_prompt": args.prompt,
                    "model_size": args.model,
                    "compute_type": args.compute_type,
                    "device": args.device,
                    "recording_directory": args.save_recordings,
                }.items()
                if value is not None
            }
            if args.language is not None:
                overrides["language"] = language_code(args.language)
            config = replace(
                apply_stt_profile(AudioConfig.from_env(), args.stt_profile), **overrides
            )
            stt = SpeechToText(config=config)
            print("Loading speech model...")
            try:
                loading_time = stt.load_model()
                stt.load_vad()
            except SpeechToTextError as error:
                print(f"VoiceOS: {error}")
                return
            print(f"Speech model ready ({loading_time:.2f}s loading).")
            print(f"Speech language mode: {config.language or 'auto'}.")
            print(
                f"Speech model: {config.model_size}; execution: "
                f"{stt.execution_device}/{stt.compute_type}; beam: {config.beam_size}; "
                f"VAD: {'on' if config.vad_filter else 'off'}; "
                f"prompt: {'configured' if config.initial_prompt else 'none'}."
            )
        if args.speech_only:
            print("Speech-only mode: all application actions are disabled; no Gemini calls.")
        else:
            initialize_database()
            dialogue = DialogueSession(measurement_clock=measurement_clock)

            if stt is not None:
                response_reader = VoiceConfirmationReader(stt, measurement_clock=measurement_clock)

                def confirmation_callback(pending):
                    return _confirm_deletion(pending, response_reader, timing=dialogue.timing)
        print("Welcome to VoiceOS. Type 'exit' to close.")
        while True:
            active_timing = (
                dialogue.prepare_turn() if dialogue is not None
                else InteractionTiming(clock=measurement_clock)
            )
            text = (
                _voice_text(stt, timing=active_timing) if stt is not None
                else input("You: ").strip()
            )
            if text.lower() == "exit":
                break
            if not text:
                if stt is None:
                    if dialogue is not None and dialogue.has_pending:
                        result = dialogue.process_text_command(
                            "", confirm_action=confirmation_callback
                        )
                        print(f"VoiceOS: {result.message} [{result.status}]")
                    else:
                        print("Please enter a command.")
                if dialogue is not None and dialogue.has_pending:
                    active_timing.wait_for_reply()
                else:
                    _publish_timing(active_timing, timing_records)
                continue
            result = (
                ActionResult(
                    status="rejected",
                    message="Application actions are disabled in speech-only mode.",
                )
                if args.speech_only
                else dialogue.process_text_command(text, confirm_action=confirmation_callback)
            )
            print(f"VoiceOS: {result.message} [{result.status}]")
            if dialogue is not None and active_timing is not dialogue.timing:
                _publish_timing(active_timing, timing_records, finish=False)
                active_timing = dialogue.timing
            if dialogue is None or not dialogue.has_pending:
                _publish_timing(active_timing, timing_records)
            else:
                active_timing.wait_for_reply(displayed=True)
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:
        logging.getLogger(__name__).error(
            "Application stopped. Check local configuration and storage."
        )
    finally:
        if dialogue is not None:
            if active_timing is not None and active_timing is not dialogue.timing:
                _publish_timing(active_timing, timing_records, finish=False)
                active_timing = dialogue.timing
            dialogue.cancel()
        print("VoiceOS shutting down.")
        if active_timing is not None:
            _publish_timing(active_timing, timing_records)


if __name__ == "__main__":
    main(sys.argv[1:])
