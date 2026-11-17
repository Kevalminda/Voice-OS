"""Opt-in lossless capture persistence before inference, independent of NLU/actions."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np


def preserve_capture(audio: np.ndarray, directory: str, recording_info: dict, configuration: dict):
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:8]
    raw_path = folder / f"{identifier}.npy"
    metadata_path = folder / f"{identifier}.json"
    with raw_path.open("xb") as output:
        np.save(output, audio, allow_pickle=False)
    metadata = {
        "id": identifier,
        "category": "unverified_saved",
        "path": str(raw_path.resolve()),
        "sample_rate": 16000,
        "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
        "recording_info": recording_info,
        "configuration": configuration,
        "reference_text": None,
        "reference_verified": False,
        "reference_source": None,
        "status": "captured",
        "transcript": None,
    }
    with metadata_path.open("x", encoding="utf-8") as output:
        json.dump(metadata, output, ensure_ascii=False, indent=2)
    return {"audio": str(raw_path.resolve()), "metadata": str(metadata_path.resolve())}


def finish_capture(files: dict, transcript: dict | None):
    path = Path(files["metadata"])
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["status"] = "transcribed" if transcript is not None else "transcription_failed"
    metadata["transcript"] = transcript
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(metadata, output, ensure_ascii=False, indent=2)
    temporary.replace(path)
