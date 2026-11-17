"""Render measured benchmark rows; never infer a reference or an accuracy winner."""

import argparse
import html
import json
from pathlib import Path


def cell(value):
    return html.escape(str(value)).replace("|", "\\|").replace("\n", "<br>")


def render_comparison(reports: list[dict]) -> str:
    lines = [
        "# Hindi/Hinglish configuration comparison",
        "",
        "Each experiment reuses its sample's exact decoded waveform. "
        "GPU figures are sampled total device memory, including other processes.",
        "",
    ]
    verified = any(
        row.get("raw_metrics", {}).get("reference_status") == "verified"
        for report in reports
        for row in report["results"]
    )
    if not verified:
        lines += [
            "**No accuracy winner can be selected: no evaluated sample has a verified reference.**",
            "Unverified text cannot be ranked by fluency, probability or speed.",
            "",
        ]
    else:
        lines += [
            "Verified-reference metrics are shown where available. "
            "Language preservation and translation errors still require human review.",
            "",
        ]
    lines += [
        "| Source report | Model | Compute | Load seconds | Model-run sampled GPU MiB |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for report in reports:
        config = report["configuration"]
        peak = (report.get("gpu_memory") or {}).get("sampled_peak_total_mib")
        lines.append(
            f"| {cell(report.get('source_report', 'unspecified'))} | "
            f"{cell(config['model_size'])} | {cell(config['compute_type'])} | "
            f"{report['model_load_duration']:.3f} | {peak if peak is not None else 'N/A'} |"
        )
    lines.append("")
    if any(report.get("coverage_by_language") for report in reports):
        lines += [
            "## Corpus coverage and warm latency",
            "",
            "Counts are unique recordings, not repeated passes. Missing references stay unscored. "
            "Warm latency excludes pass 1 for each sample/profile; OS file cache may be warm.",
            "",
            "| Source report | Corpus language | Available / verified samples | Profile | "
            "Raw WER / CER | Normalized WER / CER | Warm median s |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for report in reports:
            for language, coverage in report.get("coverage_by_language", {}).items():
                summaries = report.get("summary_by_language", {}).get(language, {})
                for profile, summary in (summaries or {"(none)": {}}).items():
                    scores = []
                    for prefix in ("raw", "normalized"):
                        wer, cer = summary.get(f"{prefix}_wer"), summary.get(f"{prefix}_cer")
                        scores.append(
                            f"{wer:.4f} / {cer:.4f}"
                            if wer is not None and cer is not None
                            else "Unavailable"
                        )
                    warm = summary.get("median_warm_decode_seconds")
                    values = [
                        report.get("source_report", "unspecified"),
                        language,
                        f"{len(coverage['available_sample_ids'])} / "
                        f"{len(coverage['verified_sample_ids'])}",
                        profile,
                        *scores,
                        f"{warm:.3f}" if warm is not None else "N/A",
                    ]
                    lines.append("| " + " | ".join(cell(value) for value in values) + " |")
        lines.append("")
    for report in reports:
        if report.get("reference_validation_note"):
            lines += [
                f"Reference warning ({cell(report.get('source_report', 'unspecified'))}): "
                f"{cell(report['reference_validation_note'])}",
                "",
            ]
    samples = {}
    for report in reports:
        for row in report["results"]:
            if "transcript" in row:
                samples.setdefault((row["case_id"], row["audio_sha256"]), []).append(
                    {**row, "source_report": report.get("source_report", "unspecified")}
                )
    for (case_id, digest), rows in samples.items():
        lines += [
            f"## Sample {cell(case_id)}",
            "",
            f"Path: {cell(rows[0].get('audio_path', 'unrecorded'))}",
            "",
            f"Decoded float32 SHA-256: `{digest}`",
            "",
            "| Source report | Model / compute | Profile / pass | "
            "VAD (threshold; silence ms; pad ms) | "
            "Language | Beam | Prompt | Raw transcript | Normalized transcript | "
            "Human reference | Raw WER / CER | Normalized WER / CER | Decode s | "
            "Discarded s | Sampled GPU MiB |",
            "| " + " | ".join(["---"] * 15) + " |",
        ]
        for row in rows:
            config, transcript = row["configuration"], row["transcript"]
            vad = (
                (
                    f"on ({config['vad_threshold']}; {config['vad_min_silence_ms']}; "
                    f"{config['vad_speech_pad_ms']})"
                )
                if config["vad_filter"]
                else "off"
            )
            scores = []
            for key in ("raw_metrics", "normalized_metrics"):
                errors = row[key]["errors"]
                scores.append(
                    f"{errors['wer']:.4f} / {errors['cer']:.4f}"
                    if (errors and errors["wer"] is not None and errors["cer"] is not None)
                    else "Unavailable"
                )
            audit = row.get("vad_audit") or {}
            discarded = audit.get("discarded_duration")
            gpu = (row.get("gpu_memory") or {}).get("sampled_peak_total_mib")
            values = [
                row["source_report"],
                f"{config['model_size']} / {config['compute_type']}",
                f"{row['profile']} / {row['repeat']}",
                vad,
                config["language"] or "auto",
                config["beam_size"],
                config["initial_prompt"] or "(none)",
                transcript["raw_text"],
                transcript["normalized_text"],
                row.get("reference_text")
                if row.get("reference_text") is not None
                else "Unavailable",
                *scores,
                f"{transcript['transcription_duration']:.3f}",
                f"{discarded:.3f}" if discarded is not None else "N/A",
                gpu if gpu is not None else "No sample in call interval",
            ]
            lines.append("| " + " | ".join(cell(value) for value in values) + " |")
        lines += [
            "",
            "VAD discard duration is signal duration, not proof of removed speech. "
            "Use the JSON interval coordinates and exported excerpts for human review.",
            "",
        ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    reports = [
        {**json.loads(path.read_text(encoding="utf-8")), "source_report": str(path)}
        for path in args.reports
    ]
    args.output.write_text(render_comparison(reports), encoding="utf-8")


if __name__ == "__main__":
    main()
