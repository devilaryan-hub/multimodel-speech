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
    SCORE_K_PACE_REL,
    SCORE_K_PITCH_REL,
    SCORE_K_ENERGY_REL,
    SCORE_K_PAUSE_REL,
    SCORE_LOCAL_WEIGHT,
    SCORE_WINDOW_WORDS,
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


# ---------------------------------------------------------------------------
# Baseline-relative helpers
# ---------------------------------------------------------------------------

def _blend_global_local(global_vals: list[float], window: int = SCORE_WINDOW_WORDS, local_weight: float = SCORE_LOCAL_WEIGHT) -> float:
    """Blend file-wide mean deviation with worst sliding-window deviation.

    Combining both terms ensures a 3-6 word injected flaw still registers even
    when the rest of the file is ideal. The blend formula is:
        deviation = (1 - local_weight) * global_mean + local_weight * worst_window_mean

    Args:
        global_vals: Per-word deviation values (non-NaN).
        window:      Sliding window size in words.
        local_weight: Weight of the localized worst-window term (0-1).

    Returns:
        Blended scalar deviation >= 0.
    """
    if not global_vals:
        return 0.0
    global_mean = float(np.mean(global_vals))
    if len(global_vals) < window:
        return global_mean
    worst = max(
        float(np.mean(global_vals[i: i + window]))
        for i in range(len(global_vals) - window + 1)
    )
    return (1.0 - local_weight) * global_mean + local_weight * worst


# ---------------------------------------------------------------------------
# Baseline-relative rubric scorers
# ---------------------------------------------------------------------------

def score_pace_relative(comparisons: list) -> RubricScore:
    """Baseline-relative pace score.

    Deviation per word = |log(candidate_duration / baseline_duration)|.
    Identical timing gives deviation 0 and score 1.0.
    Blends file-wide mean with worst sliding window.

    Args:
        comparisons: List of WordComparison from match.compare_words().

    Returns:
        RubricScore for PACE dimension.
    """
    devs = [
        abs(math.log(c.duration_ratio))
        for c in comparisons
        if not math.isnan(c.duration_ratio) and c.duration_ratio > 0
    ]
    deviation = _blend_global_local(devs)
    score = exponential_score(deviation, SCORE_K_PACE_REL)
    details = (
        f"Pace deviation {deviation:.3f} nats (mean |log dur-ratio| blended with "
        f"worst {SCORE_WINDOW_WORDS}-word window)."
    )
    return RubricScore(
        dimension=RubricDimension.PACE,
        score=round(score, 4),
        weight=RUBRIC_WEIGHTS.get("pace", 1.0),
        details=details,
    )


def score_pause_pattern_relative(
    comparisons: list,
    cand_pauses: list[tuple[float, float]],
    base_pauses: list[tuple[float, float]],
    cand_duration: float,
) -> RubricScore:
    """Baseline-relative pause score.

    Global deviation = extra candidate pause time vs baseline, as a fraction
    of candidate audio duration (extra_pause_frac).
    Local deviation = worst sliding window of positive pause_delta_s / window span.
    Blends both terms.

    Args:
        comparisons:   List of WordComparison.
        cand_pauses:   Pause intervals from the candidate audio.
        base_pauses:   Pause intervals from the baseline audio.
        cand_duration: Candidate audio duration in seconds.

    Returns:
        RubricScore for PAUSE_PATTERN dimension.
    """
    # Global term: extra silence as fraction of total duration
    cand_total_pause = sum(p[1] - p[0] for p in cand_pauses)
    base_total_pause = sum(p[1] - p[0] for p in base_pauses)
    extra_pause_frac = max(0.0, cand_total_pause - base_total_pause) / max(cand_duration, 1.0)

    # Local term: per-word positive pause_delta contributions
    # Use the absolute pause_delta_s values per word for windowing
    pause_devs = [
        max(0.0, c.pause_delta_s)
        for c in comparisons
        if not math.isnan(c.pause_delta_s)
    ]
    # Normalise per-word local window by window duration span
    W = SCORE_WINDOW_WORDS
    local_pause = 0.0
    if len(pause_devs) >= W:
        # Use fraction of candidate duration for each window
        for i in range(len(comparisons) - W + 1):
            win = [c for c in comparisons[i: i + W] if not math.isnan(c.pause_delta_s)]
            if not win:
                continue
            win_extra = sum(max(0.0, c.pause_delta_s) for c in win)
            win_span = max(win[-1].candidate.end - win[0].candidate.start, 1.0)
            local_pause = max(local_pause, win_extra / win_span)

    deviation = (1.0 - SCORE_LOCAL_WEIGHT) * extra_pause_frac + SCORE_LOCAL_WEIGHT * local_pause
    score = exponential_score(deviation, SCORE_K_PAUSE_REL)
    details = (
        f"Pause deviation {deviation:.4f} "
        f"(extra pause fraction {extra_pause_frac:.3f} blended with "
        f"worst {SCORE_WINDOW_WORDS}-word window {local_pause:.3f})."
    )
    return RubricScore(
        dimension=RubricDimension.PAUSE_PATTERN,
        score=round(score, 4),
        weight=RUBRIC_WEIGHTS.get("pause_pattern", 1.0),
        details=details,
    )


def score_pitch_variation_relative(
    comparisons: list,
    base_pitch: np.ndarray,
    cand_pitch: np.ndarray,
) -> RubricScore:
    """Baseline-relative pitch variation score.

    Deviation = |log(candidate_voiced_std / baseline_voiced_std)| computed per
    word (NaN-aware: words where either side has <2 voiced frames are skipped).
    Blends file-wide mean with worst sliding window.

    Args:
        comparisons: List of WordComparison.
        base_pitch:  Baseline pitch array (semitones, NaN=unvoiced).
        cand_pitch:  Candidate pitch array.

    Returns:
        RubricScore for PITCH_VARIATION dimension.
    """
    from src.config import HOP_LENGTH, SAMPLE_RATE

    def _word_pitch_std(word, pitch: np.ndarray) -> float:
        f_start = int(round(word.start * SAMPLE_RATE / HOP_LENGTH))
        f_end = max(f_start + 1, int(round(word.end * SAMPLE_RATE / HOP_LENGTH)))
        f_start = max(0, min(f_start, len(pitch)))
        f_end = max(f_start, min(f_end, len(pitch)))
        voiced = pitch[f_start:f_end]
        voiced = voiced[np.isfinite(voiced)]
        return float(np.std(voiced)) if voiced.size >= 2 else float("nan")

    devs = []
    for c in comparisons:
        if c.candidate is None:
            continue
        cw = c.candidate
        c_std = _word_pitch_std(cw, cand_pitch)
        if math.isnan(c_std) or c_std < 1e-6:
            continue
        # Use pitch_range_ratio as a proxy: ratio = cand_pitch_range / base_pitch_range
        # Skip NaN (unvoiced on either side) and zero/negative values
        if not math.isnan(c.pitch_range_ratio) and c.pitch_range_ratio > 0:
            devs.append(abs(math.log(c.pitch_range_ratio)))


    # Fallback: compare whole-file voiced std
    if not devs:
        base_voiced = base_pitch[np.isfinite(base_pitch)]
        cand_voiced = cand_pitch[np.isfinite(cand_pitch)]
        b_std = float(np.std(base_voiced)) if base_voiced.size >= 2 else 1.0
        c_std_global = float(np.std(cand_voiced)) if cand_voiced.size >= 2 else 1.0
        deviation = abs(math.log(max(c_std_global, 1e-6) / max(b_std, 1e-6)))
    else:
        deviation = _blend_global_local(devs)

    score = exponential_score(deviation, SCORE_K_PITCH_REL)
    details = (
        f"Pitch deviation {deviation:.3f} nats (mean |log pitch-range-ratio| "
        f"blended with worst {SCORE_WINDOW_WORDS}-word window)."
    )
    return RubricScore(
        dimension=RubricDimension.PITCH_VARIATION,
        score=round(score, 4),
        weight=RUBRIC_WEIGHTS.get("pitch_variation", 1.0),
        details=details,
    )


def score_energy_consistency_relative(comparisons: list) -> RubricScore:
    """Baseline-relative energy consistency score.

    Deviation per word = |candidate_energy_z - baseline_energy_z|.
    Identical energy gives deviation 0 and score 1.0.
    Blends file-wide mean with worst sliding window.

    Args:
        comparisons: List of WordComparison.

    Returns:
        RubricScore for ENERGY_CONSISTENCY dimension.
    """
    devs = [
        abs(c.energy_delta)
        for c in comparisons
        if not math.isnan(c.energy_delta)
    ]
    deviation = _blend_global_local(devs)
    score = exponential_score(deviation, SCORE_K_ENERGY_REL)
    details = (
        f"Energy deviation {deviation:.3f} z-units (mean |energy-delta| "
        f"blended with worst {SCORE_WINDOW_WORDS}-word window)."
    )
    return RubricScore(
        dimension=RubricDimension.ENERGY_CONSISTENCY,
        score=round(score, 4),
        weight=RUBRIC_WEIGHTS.get("energy_consistency", 1.0),
        details=details,
    )

