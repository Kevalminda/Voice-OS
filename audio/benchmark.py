"""Saved-audio accuracy benchmark and human-verified capture; never invokes NLU/actions."""

import argparse
import hashlib
import json
import re
import statistics
import time
import unicodedata
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from audio import AudioConfig, SpeechToText, SpeechToTextError
from audio.diagnostics import GpuMemorySampler, load_wav, save_wav
from audio.speech_to_text import audio_diagnostics
from audio.stt_profiles import STT_PROFILES, apply_stt_profile
from audio.vad_diagnostics import export_discarded, inspect_vad

PROFILE_OVERRIDES = {
    "baseline": {},
    "legacy": {
        "language": None,
        "multilingual": False,
        "initial_prompt": "VoiceOS, Python, Gemini, project, application.",
    },
    "native": {"language": None, "multilingual": True, "initial_prompt": ""},
    "auto_multilingual": {"language": None, "multilingual": True},
    "auto_no_prompt": {"language": None, "multilingual": True, "initial_prompt": ""},
    "auto_no_vad": {
        "language": None,
        "multilingual": True,
        "initial_prompt": "",
        "vad_filter": False,
    },
    "auto_gentle_vad": {
        "language": None,
        "multilingual": True,
        "initial_prompt": "",
        "vad_threshold": 0.35,
        "vad_speech_pad_ms": 400,
        "vad_min_silence_ms": 700,
    },
    "auto_deterministic": {
        "language": None,
        "multilingual": True,
        "initial_prompt": "",
        "temperatures": (0.0,),
    },
    "english": {"language": "en", "multilingual": False},
    "hindi": {"language": "hi", "multilingual": False},
}
INVESTIGATION_BASE = {
    "language": None,
    "multilingual": True,
    "initial_prompt": "",
    "beam_size": 5,
    "vad_filter": True,
    "vad_threshold": 0.5,
    "vad_min_silence_ms": 500,
    "vad_speech_pad_ms": 300,
    "temperatures": (0.0,),
}
INVESTIGATION_CHANGES = {
    "base": {},
    "current_fallback": {"temperatures": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)},
    "vad_off": {"vad_filter": False},
    "silence_250": {"vad_min_silence_ms": 250},
    "silence_1000": {"vad_min_silence_ms": 1000},
    "silence_2000": {"vad_min_silence_ms": 2000},
    "pad_0": {"vad_speech_pad_ms": 0},
    "pad_150": {"vad_speech_pad_ms": 150},
    "pad_600": {"vad_speech_pad_ms": 600},
    "threshold_035": {"vad_threshold": 0.35},
    "threshold_065": {"vad_threshold": 0.65},
    "beam_1": {"beam_size": 1},
    "beam_3": {"beam_size": 3},
    "beam_8": {"beam_size": 8},
    "prompt_hindi": {"initial_prompt": "यह हिंदी में सामान्य बातचीत है।"},
    "prompt_mixed": {"initial_prompt": "यह Hindi और English में मिली-जुली बातचीत है।"},
    "prompt_terms": {"initial_prompt": "Keval Minda, VoiceOS, Python, Gemini, DBMS."},
    "forced_hi": {"language": "hi", "multilingual": False},
    "forced_en": {"language": "en", "multilingual": False},
}
INVESTIGATION_PROFILES = tuple(f"invest_{name}" for name in INVESTIGATION_CHANGES)
PROFILE_OVERRIDES.update(
    {
        f"invest_{name}": {**INVESTIGATION_BASE, **changes}
        for name, changes in INVESTIGATION_CHANGES.items()
    }
)
REQUIRED_CATEGORIES = (
    "english",
    "hindi",
    "hinglish_technical",
    "hinglish_names",
    "short",
    "long",
    "trailing_silence",
    "noisy",
)
METRIC_POLICY = (
    "NFC, casefold, Unicode punctuation to spaces, collapsed whitespace; CER omits spaces."
)


def metric_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join(
        "".join(
            " " if unicodedata.category(char).startswith("P") else char for char in text
        ).split()
    )


def edit_distance(reference, hypothesis) -> int:
    """Levenshtein distance with linear memory; works for words or Unicode characters."""
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def error_rates(reference: str, hypothesis: str) -> dict:
    reference, hypothesis = metric_text(reference), metric_text(hypothesis)
    words, chars = reference.split(), reference.replace(" ", "")
    word_errors = edit_distance(words, hypothesis.split())
    char_errors = edit_distance(chars, hypothesis.replace(" ", ""))
    return {
        "word_errors": word_errors,
        "reference_words": len(words),
        "wer": word_errors / len(words) if words else None,
        "character_errors": char_errors,
        "reference_characters": len(chars),
        "cer": char_errors / len(chars) if chars else None,
    }


def term_recall(reference: str, hypothesis: str, terms: list[str]) -> dict:
    reference, hypothesis = f" {metric_text(reference)} ", f" {metric_text(hypothesis)} "
    present = [term for term in terms if f" {metric_text(term)} " in reference]
    matched = [term for term in present if f" {metric_text(term)} " in hypothesis]
    return {
        "reference_terms": present,
        "matched": matched,
        "missed": [term for term in present if term not in matched],
        "recall": len(matched) / len(present) if present else None,
    }


def score_transcript(case: dict, hypothesis: str) -> dict:
    verified = case.get("reference_verified") is True and case.get("reference_source") == "human"
    reference = case.get("reference_text")
    if not verified or not isinstance(reference, str):
        return {"reference_status": "unavailable", "errors": None, "terms": None, "names": None}
    return {
        "reference_status": "verified",
        "errors": error_rates(reference, hypothesis),
        "terms": term_recall(reference, hypothesis, case.get("terms", [])),
        "names": term_recall(reference, hypothesis, case.get("names", [])),
    }


def script_counts(text: str) -> dict:
    """Descriptive script counts are not language-preservation or translation scores."""
    counts = {"latin_letters": 0, "devanagari_letters": 0, "other_letters": 0}
    for char in text:
        if not unicodedata.category(char).startswith("L"):
            continue
        if "LATIN" in unicodedata.name(char, ""):
            counts["latin_letters"] += 1
        elif "\u0900" <= char <= "\u097f":
            counts["devanagari_letters"] += 1
        else:
            counts["other_letters"] += 1
    return counts


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError("Benchmark manifest must contain a cases list.")
    seen = set()
    for case in data["cases"]:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ValueError("Each benchmark case needs an ID.")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", case["id"]) or case["id"] in seen:
            raise ValueError("Case IDs must be unique safe filename components.")
        seen.add(case["id"])
        if type(case.get("reference_verified", False)) is not bool:
            raise ValueError("reference_verified must be a boolean.")
        for field in ("terms", "names"):
            terms = case.get(field, [])
            if not isinstance(terms, list) or not all(
                isinstance(term, str) and metric_text(term) for term in terms
            ):
                raise ValueError(f"{field} must be a list of nonempty terms.")
        if case.get("path") is not None and not isinstance(case["path"], str):
            raise ValueError("Recording path must be a string or null.")
    return data


def write_json(path: Path, data: dict, *, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2)
        output.write("\n")


def create_manifest(path: Path):
    root = Path(__file__).resolve().parents[1]
    cases = [
        {
            "id": name,
            "category": "unverified_saved",
            "path": str(root / filename),
            "reference_text": None,
            "reference_verified": False,
            "reference_source": None,
        }
        for name, filename in (
            ("saved_original", "test_audio.wav"),
            ("saved_mixed", "stt-mixed.wav"),
        )
    ]
    cases.extend(
        {
            "id": name,
            "category": name,
            "path": None,
            "reference_text": None,
            "reference_verified": False,
            "reference_source": None,
        }
        for name in REQUIRED_CATEGORIES
    )
    write_json(path, {"cases": cases}, exclusive=True)


def load_recording(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        audio = np.load(path, allow_pickle=False)
        if audio.dtype != np.float32 or audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Raw NPY recording must be finite mono float32 at 16000 Hz.")
        return audio
    return load_wav(str(path))


def _beep(frequency=880):
    try:
        import winsound

        winsound.Beep(frequency, 150)
    except (ImportError, RuntimeError):
        pass


def record_case(
    manifest: Path,
    case_id: str,
    duration: float,
    *,
    wait=True,
    delay=3,
    automatic=False,
    max_duration=None,
):
    if not manifest.exists():
        create_manifest(manifest)
    data = load_manifest(manifest)
    case = next((item for item in data["cases"] if item["id"] == case_id), None)
    if case is None or case.get("path") is not None:
        raise ValueError("Choose an existing empty case slot; previous recordings are preserved.")
    settings = {"duration": duration, "recording_mode": "automatic" if automatic else "fixed"}
    if max_duration is not None:
        settings["max_recording_duration"] = max_duration
    config = replace(AudioConfig.from_env(), **settings)
    engine = SpeechToText(config=config)
    if automatic:
        engine.load_vad()
    raw_path = manifest.parent / f"{case_id}.npy"
    wav_path = manifest.parent / f"{case_id}.wav"
    if raw_path.exists() or wav_path.exists():
        raise FileExistsError(
            "A previous raw/WAV recording exists; choose a new session directory."
        )
    if wait:
        input("Press Enter when ready; wait for 'Listening' before speaking: ")
    if not 0 <= delay <= 10:
        raise ValueError("Start delay must be from 0 to 10 seconds.")
    if automatic:
        print(
            f"Opening microphone in {delay:g}s; wait for 'Listening' before speaking.", flush=True
        )
    else:
        print(
            f"Recording starts in {delay:g}s, for {duration:g}s. Leave trailing silence.",
            flush=True,
        )
    time.sleep(delay)

    def ready():
        if not automatic:
            _beep()
        print("Listening. Speak now.", flush=True)

    audio = engine.record_audio(on_started=ready)
    if not automatic:
        _beep(440)
    with raw_path.open("xb") as output:
        np.save(output, audio, allow_pickle=False)
    save_wav(str(wav_path), audio)
    case.update(
        {
            "path": raw_path.name,
            "wav_path": wav_path.name,
            "sample_rate": 16000,
            "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
            "audio_diagnostics": audio_diagnostics(audio, config),
            "recording_info": getattr(engine, "last_recording_info", None),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "reference_text": None,
            "reference_verified": False,
            "reference_source": None,
        }
    )
    write_json(manifest, data)
    print(json.dumps(case["audio_diagnostics"], indent=2))
    print(
        f"Saved {raw_path}. Confirm the exact words spoken before scoring. No decode/actions ran."
    )


def verify_reference(manifest: Path, case_id: str, reference_path: Path):
    data = load_manifest(manifest)
    case = next((item for item in data["cases"] if item["id"] == case_id), None)
    if case is None or case.get("path") is None:
        raise ValueError("Record this case before confirming its reference.")
    audio = load_recording(manifest.parent / case["path"])
    digest = hashlib.sha256(audio.tobytes()).hexdigest()
    if case.get("audio_sha256") and case["audio_sha256"] != digest:
        raise ValueError("Recording changed before reference verification.")
    case.update(
        {
            "reference_text": reference_path.read_text(encoding="utf-8").strip(),
            "reference_verified": True,
            "reference_source": "human",
            "reference_verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "audio_sha256": digest,
        }
    )
    write_json(manifest, data)


def invalidate_reference(manifest: Path, case_id: str, reason: str):
    data = load_manifest(manifest)
    case = next((item for item in data["cases"] if item["id"] == case_id), None)
    if case is None or not reason.strip():
        raise ValueError("Choose an existing case and provide the reference-withdrawal reason.")
    if case.get("reference_text") is not None:
        case["withdrawn_reference_text"] = case["reference_text"]
    case.update(
        {
            "reference_text": None,
            "reference_verified": False,
            "reference_source": None,
            "reference_withdrawal_reason": reason,
        }
    )
    write_json(manifest, data)


def import_capture(manifest: Path, metadata_path: Path, case_id: str):
    """Attach a preserved capture without treating its model output as a reference."""
    data = load_manifest(manifest)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    path = Path(metadata["path"])
    if not path.is_absolute():
        path = metadata_path.parent / path
    audio = load_recording(path)
    digest = hashlib.sha256(audio.tobytes()).hexdigest()
    if digest != metadata.get("audio_sha256"):
        raise ValueError("Capture waveform does not match its saved hash.")
    case = next((item for item in data["cases"] if item["id"] == case_id), None)
    if case is None or case.get("path") is not None:
        raise ValueError("Choose an empty existing manifest slot for import.")
    case.update(
        {
            "path": str(path.resolve()),
            "audio_sha256": digest,
            "sample_rate": 16000,
            "reference_text": None,
            "reference_verified": False,
            "reference_source": None,
            "capture_metadata": str(metadata_path.resolve()),
        }
    )
    write_json(manifest, data)


def aggregate_results(results: list[dict]) -> dict:
    summaries = {}
    for profile in sorted({row["profile"] for row in results}):
        rows = [row for row in results if row["profile"] == profile and "transcript" in row]
        scored = [
            row["raw_metrics"]["errors"]
            for row in rows
            if row["raw_metrics"]["reference_status"] == "verified"
        ]
        words = sum(item["reference_words"] for item in scored)
        chars = sum(item["reference_characters"] for item in scored)
        normalized = [
            row["normalized_metrics"]["errors"]
            for row in rows
            if row["normalized_metrics"]["reference_status"] == "verified"
        ]
        summaries[profile] = {
            "calls": len(rows),
            "scored_calls": len(scored),
            "raw_wer": sum(item["word_errors"] for item in scored) / words if words else None,
            "raw_cer": sum(item["character_errors"] for item in scored) / chars if chars else None,
            "normalized_wer": (
                sum(item["word_errors"] for item in normalized) / words if words else None
            ),
            "normalized_cer": (
                sum(item["character_errors"] for item in normalized) / chars if chars else None
            ),
            "median_decode_seconds": statistics.median(
                row["transcript"]["transcription_duration"] for row in rows
            )
            if rows
            else None,
            "warm_calls": sum(row.get("repeat", 0) > 1 for row in rows),
            "median_warm_decode_seconds": statistics.median(
                row["transcript"]["transcription_duration"]
                for row in rows
                if row.get("repeat", 0) > 1
            )
            if any(row.get("repeat", 0) > 1 for row in rows)
            else None,
        }
    return summaries


def language_group(case: dict) -> str:
    """Corpus label supplied by a human, independent of the detected decoder language."""
    category = case.get("category") or ""
    if category in {"english", "hindi"}:
        return category
    return "hinglish" if category.startswith("hinglish") else "unclassified"


def grouped_summaries(results: list[dict], field: str) -> dict:
    rows = [row for row in results if "transcript" in row]
    return {
        value: aggregate_results([row for row in rows if row[field] == value])
        for value in sorted({row[field] for row in rows})
    }


def language_coverage(cases: list[dict], results: list[dict]) -> dict:
    available = {row["case_id"] for row in results if "transcript" in row}
    verified = {
        row["case_id"]
        for row in results
        if row.get("raw_metrics", {}).get("reference_status") == "verified"
    }
    coverage = {}
    for language in ("english", "hindi", "hinglish"):
        expected = {case["id"] for case in cases if language_group(case) == language}
        coverage[language] = {
            "available_sample_ids": sorted(expected & available),
            "verified_sample_ids": sorted(expected & verified),
            "missing_sample_ids": sorted(expected - available),
            "unverified_sample_ids": sorted((expected & available) - verified),
        }
    return coverage


def run_benchmark(
    manifest: Path,
    config: AudioConfig,
    profiles,
    repeats=2,
    case_ids=None,
    engine_factory=SpeechToText,
    gpu_sampler=None,
    vad_audit=False,
    export_vad_directory=None,
    require_cuda=False,
) -> dict:
    if not 1 <= repeats <= 10 or not profiles or any(p not in PROFILE_OVERRIDES for p in profiles):
        raise ValueError("Choose known profiles and 1–10 repeats.")
    data = load_manifest(manifest)
    cases = [case for case in data["cases"] if case_ids is None or case["id"] in case_ids]
    if case_ids and set(case_ids) - {case["id"] for case in cases}:
        raise ValueError("Unknown benchmark case selection.")
    engine = engine_factory(config=config)
    loading_time = engine.load_model()
    if require_cuda and engine.execution_device != "cuda":
        raise SpeechToTextError(
            "CUDA was required for this benchmark; CPU fallback was not timed as GPU."
        )
    results = []
    audits = {}
    for case in cases:
        path = manifest.parent / case["path"] if case.get("path") else None
        if path is None or not path.exists():
            results.append({"case_id": case["id"], "profile": None, "status": "missing_recording"})
            continue
        audio = load_recording(path)
        if case.get("sample_rate", 16000) != 16000:
            raise ValueError("Benchmark samples must be recorded at 16000 Hz.")
        audio.setflags(write=False)
        digest = hashlib.sha256(audio.tobytes()).hexdigest()
        if case.get("audio_sha256") and case["audio_sha256"] != digest:
            raise ValueError(f"Recording changed since reference capture: {case['id']}.")
        for repeat in range(1, repeats + 1):
            for profile in profiles:
                engine.config = replace(config, **PROFILE_OVERRIDES[profile])
                audit = None
                if vad_audit:
                    key = (
                        digest,
                        engine.config.vad_filter,
                        engine.config.vad_threshold,
                        engine.config.vad_min_silence_ms,
                        engine.config.vad_speech_pad_ms,
                    )
                    if key not in audits:
                        audit = inspect_vad(audio, engine.config)
                        if export_vad_directory is not None:
                            audit["exported_files"] = export_discarded(
                                audio,
                                audit,
                                Path(export_vad_directory),
                                f"{case['id']}-{len(audits)}",
                            )
                        audits[key] = audit
                    audit = audits[key]
                print(f"Evaluating {case['id']} / {profile} / pass {repeat}", flush=True)
                memory_index = len(gpu_sampler.samples) if gpu_sampler is not None else 0
                transcript = engine.transcribe(audio)
                if require_cuda and transcript["execution_device"] != "cuda":
                    raise SpeechToTextError("CUDA failed during inference; GPU benchmark stopped.")
                review = case.get("reviews", {}).get(profile, {})
                results.append(
                    {
                        "case_id": case["id"],
                        "category": case.get("category"),
                        "language_group": language_group(case),
                        "profile": profile,
                        "repeat": repeat,
                        "audio_sha256": digest,
                        "audio_path": str(path.resolve()),
                        "reference_text": case.get("reference_text")
                        if (
                            case.get("reference_verified") is True
                            and case.get("reference_source") == "human"
                        )
                        else None,
                        "configuration": engine.configuration(),
                        "vad_audit": audit,
                        "gpu_memory": gpu_sampler.report(memory_index)
                        if gpu_sampler is not None
                        else None,
                        "transcript": transcript,
                        "raw_metrics": score_transcript(case, transcript["raw_text"]),
                        "normalized_metrics": score_transcript(case, transcript["normalized_text"]),
                        "script_counts": script_counts(transcript["raw_text"]),
                        "manual_review": {
                            "language_preserved": review.get("language_preserved"),
                            "translation_error": review.get("translation_error"),
                            "notes": review.get("notes"),
                        },
                    }
                )
    engine.config = config
    return {
        "metric_policy": METRIC_POLICY,
        "timing_policy": (
            "Model load measured before decoding in this process; OS file cache may be warm. "
            "Warm summaries exclude pass 1 for each sample/profile. No model reload per call."
        ),
        "model_load_duration": loading_time,
        "configuration": {**engine.configuration(), "profiles": list(profiles)},
        "missing_categories": sorted(
            set(REQUIRED_CATEGORIES)
            - {row.get("category") for row in results if "transcript" in row}
        ),
        "summary": aggregate_results([row for row in results if row["profile"] is not None]),
        "summary_by_language": grouped_summaries(results, "language_group"),
        "summary_by_sample": grouped_summaries(results, "case_id"),
        "coverage_by_language": language_coverage(cases, results),
        "results": results,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create unverified saved cases and empty recording slots.")
    record = sub.add_parser("record", help="Record raw float32 and WAV; no automatic reference.")
    verify = sub.add_parser("verify", help="Confirm a reference file you manually checked.")
    attach = sub.add_parser("import", help="Attach a losslessly preserved voice capture.")
    run = sub.add_parser("run", help="Evaluate identical saved samples across decoder profiles.")
    for command in (init, record, verify, run, attach):
        command.add_argument("--manifest", type=Path, required=True)
    record.add_argument("--case", required=True)
    record.add_argument("--duration", type=float, default=12)
    record.add_argument(
        "--automatic",
        action="store_true",
        help="Use the same silence endpointing as the VoiceOS voice loop.",
    )
    record.add_argument("--max-duration", type=float)
    record.add_argument("--no-wait", action="store_true")
    record.add_argument("--delay", type=float, default=3)
    verify.add_argument("--case", required=True)
    verify.add_argument("--reference-file", type=Path, required=True)
    attach.add_argument("--case", required=True)
    attach.add_argument("--capture", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--profiles",
        nargs="+",
        choices=PROFILE_OVERRIDES,
        default=None,
    )
    run.add_argument("--repeat", type=int, default=2)
    run.add_argument("--cases", nargs="+")
    run.add_argument("--model")
    run.add_argument("--device", choices=("auto", "cpu", "cuda"))
    run.add_argument("--compute-type")
    run.add_argument("--stt-profile", choices=STT_PROFILES)
    run.add_argument("--allow-download", action="store_true")
    run.add_argument("--gpu-memory", action="store_true")
    run.add_argument(
        "--investigation", action="store_true", help="Run one-variable Hindi/Hinglish comparisons."
    )
    run.add_argument("--vad-audit", action="store_true")
    run.add_argument("--export-vad", type=Path, help="New directory for discarded audio excerpts.")
    run.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        if args.command == "init":
            create_manifest(args.manifest)
        elif args.command == "record":
            record_case(
                args.manifest,
                args.case,
                args.duration,
                wait=not args.no_wait,
                delay=args.delay,
                automatic=args.automatic,
                max_duration=args.max_duration,
            )
        elif args.command == "verify":
            verify_reference(args.manifest, args.case, args.reference_file)
        elif args.command == "import":
            import_capture(args.manifest, args.capture, args.case)
        else:
            overrides = {
                key: value
                for key, value in {
                    "model_size": args.model,
                    "device": args.device,
                    "compute_type": args.compute_type,
                }.items()
                if value is not None
            }
            config = replace(
                apply_stt_profile(AudioConfig.from_env(), args.stt_profile),
                local_files_only=not args.allow_download,
                **overrides,
            )
            with GpuMemorySampler() if args.gpu_memory else nullcontext() as sampler:
                profiles = args.profiles or (
                    INVESTIGATION_PROFILES
                    if args.investigation
                    else ["legacy", "native", "english", "hindi"]
                )
                report = run_benchmark(
                    args.manifest,
                    config,
                    profiles,
                    args.repeat,
                    args.cases,
                    gpu_sampler=sampler,
                    vad_audit=args.vad_audit or args.export_vad is not None,
                    export_vad_directory=args.export_vad,
                    require_cuda=args.require_cuda,
                )
            if args.gpu_memory:
                report["gpu_memory"] = sampler.report()
            write_json(args.output, report)
            print(json.dumps(report["summary"], indent=2))
        return 0
    except (ValueError, OSError, SpeechToTextError) as error:
        print(f"Benchmark failed: {error}")
        return 1
    except (KeyboardInterrupt, EOFError):
        print("Benchmark cancelled.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
