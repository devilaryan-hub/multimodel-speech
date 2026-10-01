"""
src/pipeline.py
===============
End-to-end speech evaluation pipeline.

Given a candidate audio file + transcript and a baseline audio file,
produces an EvaluationResult JSON-serialisable object.

Usage:
    from src.pipeline import evaluate
    result = evaluate(
        candidate_path="data/flawed/speaker1.wav",
        baseline_path="data/raw/baseline.wav",
        transcript="Hello world, this is a test speech.",
        audio_id="speaker1_run1",
    )
    print(result.model_dump_json(indent=2))
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from src.config import SEED
from src.analysis.detector import detect_all
from src.analysis.scoring import (
    compute_composite,
    score_energy_consistency,
    score_pace,
    score_pause_pattern,
    score_pitch_variation,
)
from src.explain.templates import explain
from src.features.alignment import align_features
from src.features.audio import AudioFeatures, extract_all
from src.features.forced_align import align_transcript
from src.features.transcript import Word, estimate_word_times, tokenize, words_to_pace_wpm
from src.schema import EvaluationResult, FlawRegion, RubricScore
from src.analysis.match import compare_words, detect_from_matches, match_words


def _seed_all() -> None:
    """Seed Python's random and NumPy for reproducibility."""
    random.seed(SEED)
    np.random.seed(SEED)


def _build_summary(
    composite: float,
    rubric_scores: list[RubricScore],
    flaw_regions: list[FlawRegion],
) -> str:
    """Build a one-paragraph human-readable summary of the evaluation.

    Args:
        composite:     Overall composite score.
        rubric_scores: Per-dimension scores.
        flaw_regions:  Detected flaw regions.

    Returns:
        Summary string.
    """
    grade = (
        "excellent" if composite >= 0.85
        else "good" if composite >= 0.70
        else "fair" if composite >= 0.55
        else "needs improvement"
    )
    dims = ", ".join(
        f"{s.dimension.value} ({s.score:.2f})" for s in rubric_scores
    )
    n_flaws = len(flaw_regions)
    flaw_str = (
        f"No flaw regions were detected."
        if n_flaws == 0
        else f"{n_flaws} flaw region(s) were detected across the recording."
    )
    return (
        f"Overall performance is {grade} with a composite score of {composite:.2f}. "
        f"Rubric breakdown — {dims}. {flaw_str}"
    )


def evaluate(
    candidate_path: str | Path,
    baseline_path: str | Path,
    transcript: str,
    audio_id: str,
) -> EvaluationResult:
    """Run the full speech evaluation pipeline.

    Steps:
    1. Seed random state for reproducibility.
    2. Extract audio features from candidate and baseline.
    3. DTW-align candidate feature streams to baseline.
    4. Estimate word timings from transcript + candidate duration.
    5. Score each rubric dimension.
    6. Detect time-stamped flaw regions.
    7. Build summary and assemble EvaluationResult.

    Args:
        candidate_path: Path to the candidate speaker's audio file.
        baseline_path:  Path to the baseline (ideal) audio file.
        transcript:     The spoken text (same for both speakers).
        audio_id:       Unique identifier string for this evaluation.

    Returns:
        EvaluationResult with rubric scores, flaw regions, and summary.
    """
    _seed_all()

    # ── 1. Feature extraction ─────────────────────────────────────────────
    cand: AudioFeatures = extract_all(candidate_path)
    base: AudioFeatures = extract_all(baseline_path)

    # ── 2. Align candidate to baseline frame grid ─────────────────────────
    _b_pitch, _b_energy, w_pitch, w_energy = align_features(
        baseline_pitch=base.pitch_st,
        baseline_energy=base.energy_z,
        candidate_pitch=cand.pitch_st,
        candidate_energy=cand.energy_z,
    )

    # ── 3. Forced Alignment → word timings → pace ────────────────────────
    aligned_res = align_transcript(candidate_path, transcript, duration=cand.duration)
    words = [Word(text=aw["word"], start=aw["start"], end=aw["end"]) for aw in aligned_res]
    tokens = [w.text for w in words]
    wpm = words_to_pace_wpm(words)

    # Baseline word timings for contrastive detection
    base_aligned = align_transcript(baseline_path, transcript, duration=base.duration)
    base_words = [Word(text=aw["word"], start=aw["start"], end=aw["end"]) for aw in base_aligned]


    # ── 4. Rubric scoring ─────────────────────────────────────────────────
    rubric: list[RubricScore] = [
        score_pace(wpm),
        score_pitch_variation(w_pitch),
        score_energy_consistency(w_energy),
        score_pause_pattern(cand.pauses, cand.duration, len(tokens)),
    ]
    composite = compute_composite(rubric)

    # ── 5. Flaw detection ─────────────────────────────────────────────────
    # 5a. Single-signal detectors (pitch, energy, pause, filler)
    flaw_regions = detect_all(
        pitch_st=w_pitch,
        energy_z=w_energy,
        pauses=cand.pauses,
        words=words,
        duration=cand.duration,
    )
    # 5b. Contrastive word-matching detectors (pace, pause delta, energy delta)
    matched = match_words(base_words, words)
    comparisons = compare_words(
        matched,
        baseline_pitch=base.pitch_st,
        candidate_pitch=cand.pitch_st,
        baseline_energy=base.energy_z,
        candidate_energy=cand.energy_z,
    )
    match_flaws = detect_from_matches(comparisons)
    # Merge and deduplicate by type+time (keep contrastive results which are more precise)
    flaw_regions = sorted(
        flaw_regions + match_flaws, key=lambda r: r.start
    )

    # ── 6. Summary ────────────────────────────────────────────────────────
    summary = _build_summary(composite, rubric, flaw_regions)

    return EvaluationResult(
        audio_id=audio_id,
        transcript=transcript,
        duration=cand.duration,
        rubric_scores=rubric,
        composite_score=composite,
        flaw_regions=flaw_regions,
        summary=summary,
    )
