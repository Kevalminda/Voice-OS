import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from core.command_processor import process_text_command as process_text_command
from core.dialogue import DialogueSession
from core.schemas import ActionResult
from storage.database import initialize_database


def _configure_voice_output():
    """Hindi output must also work when Windows redirects the terminal streams."""
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def confirm_deletion(pending: ActionResult) -> bool:
    print(f"VoiceOS: {pending.message}")
    return input("Confirm deletion (yes/no): ").strip().lower() in {"yes", "y"}


def _voice_text(stt):
    """Review recognized text before submitting it to the same text-command boundary."""
    from audio.speech_to_text import SpeechToTextError

    trigger = input("Press Enter to record, or type exit: ").strip()
    if trigger.lower() == "exit":
        return "exit"

    def ready():
        if stt.config.recording_mode == "automatic":
            print(f"Listening. Speak now; stops after {stt.config.silence_duration:g}s of silence.")
        else:
            print(f"Speak now for up to {stt.config.duration:g} seconds; leave trailing silence.")

    print("Opening microphone...")
    try:
        transcript = stt.listen_and_transcribe(on_recording_started=ready)
    except (SpeechToTextError, ValueError) as error:
        print(f"VoiceOS: {error}")
        if getattr(stt, "last_capture_files", None):
            print(f"Saved recording: {stt.last_capture_files['metadata']}")
        return ""
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
    choice = input("Use this command? (yes/no/edit): ").strip().lower()
    if choice == "edit":
        return input("Corrected command: ").strip()
    return transcript["normalized_text"] if choice in {"yes", "y"} else ""


def main(argv=()):
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
    if args.voice:
        _configure_voice_output()
    load_dotenv()
    level = os.getenv("VOICEOS_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO), format="%(levelname)s %(name)s: %(message)s"
    )
    # SDK/transport debug logs can contain request payloads; keep them quiet.
    for name in ("google", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    dialogue = None
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
            dialogue = DialogueSession()
        print("Welcome to VoiceOS. Type 'exit' to close.")
        while True:
            text = _voice_text(stt) if stt is not None else input("You: ").strip()
            if text.lower() == "exit":
                break
            if not text:
                if stt is None:
                    if dialogue is not None and dialogue.has_pending:
                        result = dialogue.process_text_command("", confirm_action=confirm_deletion)
                        print(f"VoiceOS: {result.message} [{result.status}]")
                    else:
                        print("Please enter a command.")
                continue
            result = (
                ActionResult(
                    status="rejected",
                    message="Application actions are disabled in speech-only mode.",
                )
                if args.speech_only
                else dialogue.process_text_command(text, confirm_action=confirm_deletion)
            )
            print(f"VoiceOS: {result.message} [{result.status}]")
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:
        logging.getLogger(__name__).error(
            "Application stopped. Check local configuration and storage."
        )
    finally:
        if dialogue is not None:
            dialogue.cancel()
        print("VoiceOS shutting down.")


if __name__ == "__main__":
    main(sys.argv[1:])
