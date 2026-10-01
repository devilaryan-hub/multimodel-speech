"""
tests/test_features_forced_align.py
===================================
Tests for forced alignment (WhisperX, Torchaudio MMS_FA, proportional fallback).
Includes cache verification, interpolation for unaligned words, and skipped-if-no-data
tests for real audio speeches.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.config import ALIGNMENT_CACHE_DIR, RAW_DIR, SAMPLE_RATE
from src.features.forced_align import (
    align_transcript,
    compute_alignment_hash,
    interpolate_unaligned_words,
)


def _make_dummy_wav(duration_s: float = 3.0) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    sf.write(tmp.name, sig, SAMPLE_RATE)
    return Path(tmp.name)


def test_compute_alignment_hash_deterministic():
    wav = _make_dummy_wav(1.0)
    text = "hello world"
    h1 = compute_alignment_hash(wav, text)
    h2 = compute_alignment_hash(wav, text)
    assert h1 == h2
    assert len(h1) == 24


def test_interpolate_unaligned_words():
    words = ["the", "quick", "brown", "fox", "jumps"]
    # Only "the" (idx 0) and "fox" (idx 3) were aligned by the model
    raw = [
        {"word": "the", "start": 0.1, "end": 0.4},
        {"word": "fox", "start": 1.5, "end": 1.8},
    ]
    interpolated = interpolate_unaligned_words(words, raw, duration=2.5)

    assert len(interpolated) == len(words)
    # Check aligned flags
    assert interpolated[0]["aligned"] is True
    assert interpolated[0]["word"] == "the"
    assert interpolated[1]["aligned"] is False
    assert interpolated[2]["aligned"] is False
    assert interpolated[3]["aligned"] is True
    assert interpolated[3]["word"] == "fox"
    assert interpolated[4]["aligned"] is False

    # Check monotonicity
    for i in range(len(interpolated) - 1):
        assert interpolated[i]["end"] <= interpolated[i + 1]["start"] + 1e-4
        assert interpolated[i]["end"] > interpolated[i]["start"]


def test_proportional_fallback_and_caching():
    wav = _make_dummy_wav(2.0)
    transcript = "testing forced alignment caching and fallback"
    # Force proportional backend for unit testing without GPU/model download
    res1 = align_transcript(wav, transcript, backend="proportional", use_cache=True)
    assert len(res1) == len(transcript.split())
    for item in res1:
        assert item["end"] > item["start"]
        assert item["aligned"] is False

    # Verify cache file exists
    f_hash = compute_alignment_hash(wav, transcript)
    cache_file = ALIGNMENT_CACHE_DIR / f"{f_hash}.json"
    assert cache_file.exists()

    # Second call reads from cache
    res2 = align_transcript(wav, transcript, backend="proportional", use_cache=True)
    assert res1 == res2


def test_real_speech_alignment_skipped_if_no_data():
    """Real forced alignment on true human speech can only be verified when real speech files exist.

    If no real speech WAVs exist in data/raw/, this test is cleanly skipped.
    """
    real_wavs = list(RAW_DIR.glob("*.wav"))
    if not real_wavs:
        pytest.skip(
            "Unverified: Real speech forced alignment is unverified because no real speech files exist in data/raw/. "
            "Verification will run automatically once speeches are added."
        )

    # If real speech is present, run alignment
    sample_wav = real_wavs[0]
    txt_file = sample_wav.with_suffix(".txt")
    if not txt_file.exists():
        pytest.skip(f"No corresponding transcript file found for {sample_wav}")

    transcript = txt_file.read_text(encoding="utf-8")
    aligned = align_transcript(sample_wav, transcript, backend="whisperx")
    assert len(aligned) > 0
    assert any(w["aligned"] is True for w in aligned)
