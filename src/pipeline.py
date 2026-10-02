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
    score_energy_consistency_relative,
    score_pace_relative,
    score_pause_pattern_relative,
    score_pitch_variation_relative,
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

    Rules:
    - Grade is derived from the composite score, but overridden to at most
      'fair' when any dimension score is below 0.70 or any flaw regions exist.
    - Cites the lowest-scoring dimension and its score.
    - Cites the flaw region count when non-zero.
    - Uses only plain ASCII punctuation to avoid encoding artefacts.

    Args:
        composite:     Overall composite score.
        rubric_scores: Per-dimension scores.
        flaw_regions:  Detected flaw regions.

    Returns:
        Summary string (plain ASCII, no Unicode special chars).
    """
    # Determine grade from composite
    grade = (
        "excellent" if composite >= 0.85
        else "good" if composite >= 0.70
        else "fair" if composite >= 0.55
        else "needs improvement"
    )

    # Override grade if any dimension is low or flaws detected
    n_flaws = len(flaw_regions)
    min_score = min((s.score for s in rubric_scores), default=1.0)
    if min_score < 0.70 or n_flaws > 0:
        if grade in ("excellent", "good"):
            grade = "fair" if composite >= 0.55 else "needs improvement"

    # Find worst dimension
    worst = min(rubric_scores, key=lambda s: s.score) if rubric_scores else None
    dims = ", ".join(
        f"{s.dimension.value} ({s.score:.2f})" for s in rubric_scores
    )

    # Flaw string
    if n_flaws == 0:
        flaw_str = "No flaw regions were detected."
    else:
        types = sorted({r.flaw_type.value for r in flaw_regions})
        flaw_str = (
            f"{n_flaws} flaw region(s) detected "
            f"({', '.join(types)})."
        )

    # Worst dimension citation
    worst_str = ""
    if worst and worst.score < 0.90:
        worst_str = (
            f" Lowest dimension: {worst.dimension.value} ({worst.score:.2f})."
        )

    return (
        f"Overall performance is {grade} with a composite score of {composite:.2f}. "
        f"Rubric breakdown - {dims}.{worst_str} {flaw_str}"
    )


def evaluate(
    candidate_path: str | Path,
    baseline_path: str | Path | None = None,
    transcript: str = "",
    audio_id: str = "",
) -> EvaluationResult:
    """Run the full speech evaluation pipeline.

    Steps:
    1. Seed random state for reproducibility.
    2. Extract audio features from candidate and baseline (if provided).
    3. DTW-align candidate feature streams to baseline.
    4. Estimate word timings from transcript + candidate duration.
    5. Score each rubric dimension.
    6. Detect time-stamped flaw regions (contrastive if baseline given, else standalone).
    7. Build summary and assemble EvaluationResult.

    Args:
        candidate_path: Path to the candidate speaker's audio file.
        baseline_path:  Path to the baseline (ideal) audio file, or None.
        transcript:     The spoken text (or path to transcript file).
        audio_id:       Unique identifier string for this evaluation.

    Returns:
        EvaluationResult with rubric scores, flaw regions, and summary.
    """
    _seed_all()

    # Support passing a file path to the transcript
    try:
        tr_path = Path(transcript)
        if tr_path.is_file():
            transcript = tr_path.read_text(encoding="utf-8-sig").strip()
    except (OSError, ValueError):
        pass

    # ── 1. Feature extraction ─────────────────────────────────────────────
    cand: AudioFeatures = extract_all(candidate_path)
    has_baseline = baseline_path is not None and str(baseline_path).strip() != ""

    # ── 2. Align candidate to baseline frame grid ─────────────────────────
    if has_baseline:
        base: AudioFeatures = extract_all(baseline_path)
        _b_pitch, _b_energy, w_pitch, w_energy = align_features(
            baseline_pitch=base.pitch_st,
            baseline_energy=base.energy_z,
            candidate_pitch=cand.pitch_st,
            candidate_energy=cand.energy_z,
        )
    else:
        w_pitch, w_energy = cand.pitch_st, cand.energy_z

    # ── 3. Forced Alignment → word timings → pace ────────────────────────
    aligned_res = align_transcript(candidate_path, transcript, duration=cand.duration)
    words = [Word(text=aw["word"], start=aw["start"], end=aw["end"]) for aw in aligned_res]
    tokens = [w.text for w in words]
    wpm = words_to_pace_wpm(words)

    if has_baseline:
        # Baseline word timings for contrastive scoring and detection
        base_aligned = align_transcript(baseline_path, transcript, duration=base.duration)
        base_words = [Word(text=aw["word"], start=aw["start"], end=aw["end"]) for aw in base_aligned]
        # Compute word comparisons once — shared by scoring and detection steps
        matched = match_words(base_words, words)
        comparisons = compare_words(
            matched,
            baseline_pitch=base.pitch_st,
            candidate_pitch=cand.pitch_st,
            baseline_energy=base.energy_z,
            candidate_energy=cand.energy_z,
        )

    # ── 4. Rubric scoring ─────────────────────────────────────────────────
    if has_baseline:
        # Baseline-relative scoring: every dimension measures deviation from
        # the reference read. Identical inputs give deviation=0, score=1.0.
        # Standalone absolute-range scorers are NOT used in this mode.
        rubric: list[RubricScore] = [
            score_pace_relative(comparisons),
            score_pitch_variation_relative(comparisons, base.pitch_st, cand.pitch_st),
            score_energy_consistency_relative(comparisons),
            score_pause_pattern_relative(
                comparisons, cand.pauses, base.pauses, cand.duration
            ),
        ]
    else:
        # No-baseline mode: absolute heuristic ranges (original behaviour).
        rubric = [
            score_pace(wpm),
            score_pitch_variation(w_pitch),
            score_energy_consistency(w_energy),
            score_pause_pattern(cand.pauses, cand.duration, len(tokens)),
        ]
    composite = compute_composite(rubric)

    # ── 5. Flaw detection ─────────────────────────────────────────────────
    if has_baseline:
        # When a baseline is provided, only baseline-relative (match-based)
        # detection may emit regions. Standalone absolute-threshold detectors run
        # only when no baseline is given, because an ideal reference speech can
        # itself contain natural stylistic variations (e.g., steady pitch on a clause,
        # natural rhetorical pauses) that absolute heuristic thresholds would falsely flag.
        flaw_regions = detect_from_matches(
            comparisons,
            cand_pauses=cand.pauses,
            base_pauses=base.pauses,
        )
    else:
        # Standalone mode: when no reference baseline is given, evaluate against
        # absolute heuristic thresholds across single-signal feature dimensions.
        flaw_regions = detect_all(
            pitch_st=w_pitch,
            energy_z=w_energy,
            pauses=cand.pauses,
            words=words,
            duration=cand.duration,
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
