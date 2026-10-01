"""
src/analysis/scoring.py
=======================
Rubric scoring functions.

Each scorer receives extracted features and returns a RubricScore.
All computations are purely numerical – no I/O.

Scorers:
- score_pace               – penalise WPM outside ideal range
- score_pitch_variation    – penalise std-dev of pitch outside ideal range
- score_energy_consistency – penalise high variance of energy z-scores
- score_pause_pattern      – penalise missing, excessive, or misplaced pauses
- compute_composite        – weighted mean of a list of RubricScores
"""
from __future__ import annotations

import math
import numpy as np

from src.config import (
    ENERGY_Z_CLIP,
    PACE_IDEAL_WPM_MAX,
    PACE_IDEAL_WPM_MIN,
    PAUSE_IDEAL_MAX,
    PITCH_VAR_IDEAL_MAX,
    PITCH_VAR_IDEAL_MIN,
    RUBRIC_WEIGHTS,
    SCORE_K_ENERGY,
    SCORE_K_PACE,
    SCORE_K_PAUSE,
    SCORE_K_PITCH,
)
from src.schema import RubricDimension, RubricScore


def _linear_penalty(value: float, lo: float, hi: float) -> float:
    """Return a 0-1 penalty score: 1.0 when value is inside [lo, hi], scaling
    linearly to 0.0 as it moves further outside the range.

    The maximum allowed deviation before score hits 0 is equal to the width
    of the ideal range (hi - lo).
    """
    if lo <= value <= hi:
        return 1.0
    span = hi - lo if hi > lo else 1.0
    if value < lo:
        return float(max(0.0, 1.0 - (lo - value) / span))
    else:
        return float(max(0.0, 1.0 - (value - hi) / span))


def exponential_score(deviation: float, k: float) -> float:
    """Compute normalized score in [0, 1] using exponential decay.

    Formula:
        score = exp(-k * deviation)
    (Display score = 100 * score).

    Args:
        deviation: Non-negative deviation from ideal value/boundary.
        k:         Decay constant (> 0). Larger k → faster drop in score.

    Returns:
        Score in [0.0, 1.0].
    """
    dev = max(0.0, float(deviation))
    return float(np.clip(math.exp(-k * dev), 0.0, 1.0))


def score_pace(wpm: float) -> RubricScore:
    """Score speaking pace relative to the ideal WPM range.

    Args:
        wpm: Measured speaking rate in words-per-minute.

    Returns:
        RubricScore for PACE dimension.
    """
    score = _linear_penalty(wpm, PACE_IDEAL_WPM_MIN, PACE_IDEAL_WPM_MAX)
    if wpm < PACE_IDEAL_WPM_MIN:
        details = f"Pace {wpm:.0f} WPM is below ideal minimum {PACE_IDEAL_WPM_MIN:.0f} WPM."
    elif wpm > PACE_IDEAL_WPM_MAX:
        details = f"Pace {wpm:.0f} WPM exceeds ideal maximum {PACE_IDEAL_WPM_MAX:.0f} WPM."
    else:
        details = f"Pace {wpm:.0f} WPM is within ideal range."
    return RubricScore(
        dimension=RubricDimension.PACE,
        score=round(score, 4),
        weight=RUBRIC_WEIGHTS.get("pace", 1.0),
        details=details,
    )


def score_pitch_variation(pitch_st: np.ndarray) -> RubricScore:
    """Score pitch expressiveness based on std-dev of voiced-frame pitch values.

    Args:
        pitch_st: Per-frame pitch in semitones relative to speaker median F0.
                  Unvoiced frames carry value NaN.

    Returns:
        RubricScore for PITCH_VARIATION dimension.
    """
    voiced = pitch_st[np.isfinite(pitch_st)]
    std = float(np.std(voiced)) if voiced.size > 1 else 0.0
    score = _linear_penalty(std, PITCH_VAR_IDEAL_MIN, PITCH_VAR_IDEAL_MAX)

    if std < PITCH_VAR_IDEAL_MIN:
        details = (
            f"Pitch std-dev {std:.2f} st is below ideal minimum "
            f"{PITCH_VAR_IDEAL_MIN:.1f} st – speech sounds monotone."
        )
    elif std > PITCH_VAR_IDEAL_MAX:
        details = (
            f"Pitch std-dev {std:.2f} st exceeds ideal maximum "
            f"{PITCH_VAR_IDEAL_MAX:.1f} st – pitch changes are erratic."
        )
    else:
        details = f"Pitch variation std-dev {std:.2f} st is within ideal range."
    return RubricScore(
        dimension=RubricDimension.PITCH_VARIATION,
        score=round(score, 4),
        weight=RUBRIC_WEIGHTS.get("pitch_variation", 1.0),
        details=details,
    )


def score_energy_consistency(energy_z: np.ndarray) -> RubricScore:
    """Score loudness consistency based on the std-dev of energy z-scores.

    A speaker whose energy z-scores fluctuate wildly (high std) is penalised.
    After clipping outliers at ±ENERGY_Z_CLIP, ideal std-dev is near 1.0;
    values much higher indicate inconsistency.

    Args:
        energy_z: Per-frame energy z-scores.

    Returns:
        RubricScore for ENERGY_CONSISTENCY dimension.
    """
    clipped = np.clip(energy_z, -ENERGY_Z_CLIP, ENERGY_Z_CLIP)
    std = float(np.std(clipped)) if clipped.size > 1 else 0.0

    # Ideal: std close to 1.0.  Penalise if std > 2.0 or < 0.3.
    score = _linear_penalty(std, 0.3, 2.0)
    details = f"Energy z-score std-dev {std:.2f} (ideal 0.3–2.0)."
    return RubricScore(
        dimension=RubricDimension.ENERGY_CONSISTENCY,
        score=round(score, 4),
        weight=RUBRIC_WEIGHTS.get("energy_consistency", 1.0),
        details=details,
    )


def score_pause_pattern(
    pauses: list[tuple[float, float]],
    duration: float,
    word_count: int,
) -> RubricScore:
    """Score pause usage: penalise missing pauses in long speeches and
    excessive pauses.

    Rules:
    - If duration > 60 s and no pauses detected → penalty (missing pauses).
    - Each pause longer than PAUSE_IDEAL_MAX adds to the penalty.
    - Score is 1.0 when pauses are absent (short speeches) or appropriate.

    Args:
        pauses:     List of (start_s, end_s) pause intervals.
        duration:   Total audio duration in seconds.
        word_count: Number of words in the transcript.

    Returns:
        RubricScore for PAUSE_PATTERN dimension.
    """
    if duration <= 0:
        return RubricScore(
            dimension=RubricDimension.PAUSE_PATTERN,
            score=1.0,
            weight=RUBRIC_WEIGHTS.get("pause_pattern", 1.0),
            details="No audio to evaluate.",
        )

    penalty = 0.0

    # Missing pause penalty: for speeches > 60 s expect at least one pause
    if duration > 60.0 and len(pauses) == 0:
        penalty += 0.4

    # Excessive pause penalty
    long_pauses = [p for p in pauses if (p[1] - p[0]) > PAUSE_IDEAL_MAX]
    if pauses:
        excess_ratio = len(long_pauses) / len(pauses)
        penalty += 0.6 * excess_ratio
    elif long_pauses:
        penalty += 0.6

    score = max(0.0, 1.0 - penalty)

    n_long = len(long_pauses)
    if n_long == 0 and len(pauses) == 0:
        details = "No significant pauses detected."
    elif n_long == 0:
        details = f"{len(pauses)} pause(s) detected, all within ideal duration."
    else:
        details = (
            f"{n_long}/{len(pauses)} pause(s) exceed ideal maximum "
            f"{PAUSE_IDEAL_MAX:.1f} s."
        )

    return RubricScore(
        dimension=RubricDimension.PAUSE_PATTERN,
        score=round(score, 4),
        weight=RUBRIC_WEIGHTS.get("pause_pattern", 1.0),
        details=details,
    )


def compute_composite(scores: list[RubricScore]) -> float:
    """Compute the weighted mean composite score from rubric scores.

    Args:
        scores: List of RubricScore objects.

    Returns:
        Weighted mean score in [0, 1].
    """
    if not scores:
        return 0.0
    total_weight = sum(s.weight for s in scores)
    return round(sum(s.score * s.weight for s in scores) / total_weight, 4)
