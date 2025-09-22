# src/services/diarization_service.py

import json
import logging
import os
import tempfile

# We will use a global lock for diarization to avoid concurrency issues (since file-based output is used)
from threading import Lock

from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from omegaconf import OmegaConf
from pydub import AudioSegment

from utils import cleanup_dir, create_diarization_config

_diarization_lock = Lock()


def diarize_audio(audio_path: str, domain: str = "general") -> list:
    """
    Perform speaker diarization on the given audio file.
    Returns a list of segments: {"speaker": "Speaker X", "start": float_seconds, "end": float_seconds}.
    """
    # Create a temporary directory for this diarization session
    tmp_dir = tempfile.mkdtemp(prefix="diarization_")
    rttm_path = None
    segments = []
    try:
        # Convert audio to mono wav at 16k if needed
        wav_path = os.path.join(tmp_dir, "audio_mono.wav")
        sound = AudioSegment.from_file(audio_path)
        sound = sound.set_frame_rate(16000).set_channels(1)
        sound.export(wav_path, format="wav")

        # Prepare NeMo config and manifest
        config_path = create_diarization_config(tmp_dir, domain_type=domain)
        config = OmegaConf.load(config_path)
        # Update config paths
        manifest_path = os.path.join(tmp_dir, "manifest.json")
        # Write manifest file expected by NeMo (pointing to our wav)
        with open(manifest_path, "w") as m:
            manifest_entry = {
                "audio_filepath": wav_path,
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
            }
            m.write(json.dumps(manifest_entry) + "\n")
        config.diarizer.manifest_filepath = manifest_path
        config.diarizer.out_dir = tmp_dir  # diarizer will output RTTM here
        config.diarizer.speaker_embeddings.model_path = "titanet_large"
        config.diarizer.vad.model_path = "vad_multilingual_marblenet"
        config.diarizer.msdd_model.model_path = "diar_msdd_" + (
            "telephonic" if domain == "telephonic" else "general"
        )
        config.diarizer.clustering.parameters.oracle_num_speakers = False
        config.num_workers = 0  # to avoid multiprocessing issues

        # Run diarization within a thread-safe block
        with _diarization_lock:
            diarizer_model = NeuralDiarizer(cfg=config)
            diarizer_model.diarize()
        # After diarization, the output RTTM should be in tmp_dir/pred_rttms
        rttm_dir = os.path.join(tmp_dir, "pred_rttms")
        # There should be a single RTTM file named after the audio (here "audio_mono.rttm")
        # Find the RTTM file
        for file in os.listdir(rttm_dir):
            if file.endswith(".rttm"):
                rttm_path = os.path.join(rttm_dir, file)
                break
        if not rttm_path or not os.path.exists(rttm_path):
            raise RuntimeError("Diarization RTTM output not found")

        # Parse RTTM to extract speaker segments
        segments = _parse_rttm(rttm_path)
    finally:
        # Clean up the temporary directory and files
        cleanup_dir(tmp_dir)
    return segments


def _parse_rttm(rttm_path: str) -> list:
    """
    Internal helper to parse an RTTM file into a list of speaker segments.
    """
    segments = []
    try:
        with open(rttm_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                # RTTM format: Type, File ID, Chan, Start, Dur, [other fields], Speaker label
                if parts[0] != "SPEAKER":
                    continue
                start_time = float(parts[3])
                duration = float(parts[4])
                speaker_label = parts[7]  # e.g., "speaker_1"
                speaker_num = speaker_label.split("_")[-1]
                try:
                    speaker_idx = int(speaker_num)
                except ValueError:
                    speaker_idx = speaker_num  # if not numeric for some reason
                segments.append(
                    {
                        "speaker": f"Speaker {speaker_idx}",
                        "start": start_time,
                        "end": start_time + duration,
                    }
                )
    except FileNotFoundError:
        logging.error(f"RTTM file not found: {rttm_path}")
    return segments
