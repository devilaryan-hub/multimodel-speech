"""
src/analysis/match.py
=====================
Baseline-vs-participant word matching and contrastive flaw detection.

Workflow:
1. match_words()  – align baseline and candidate word lists using difflib,
                    returning MatchedWord pairs (one side may be None for
                    insertions / deletions).
2. compare_word() – compute per-word diagnostic scalars (duration ratio,
                    pause delta, energy delta, pitch range ratio).
3. detect_from_matches() – scan MatchedWord list for pace / pause / energy
                    flaws, merging adjacent same-type flags into FlawRegion.

Notes:
- PAUSE_MISSING: emitted when baseline word had a clear inter-word pause
  (> MATCH_PAUSE_BASELINE_MIN) and the candidate has none.
- PAUSE_MISPLACED: removed — without a sentence-boundary oracle, flagging
  misplaced pauses would cause too many false positives and cannot be reliably
  verified. See docs/DECISIONS.md for rationale.
- Pitch comparison uses NaN-aware ratio; if either side has all-NaN pitch in
  the word span the comparison returns None (not flagged as monotone).
"""
from __future__ import annotations

import difflib
import math
from typing import NamedTuple

import numpy as np

from src.config import (
    HOP_LENGTH,
    MATCH_ENERGY_LOW_DELTA,
    MATCH_FAST_RATIO_MAX,
    MATCH_MIN_REGION_WORDS,
    MATCH_PAUSE_BASELINE_MIN,
    MATCH_PAUSE_EXCESS_DELTA,
    MATCH_PAUSE_MISSING_DELTA,
    MATCH_SIMILARITY_THRESHOLD,
    MATCH_SLOW_RATIO_MIN,
    FLAW_MERGE_GAP_SEC,
    MIN_FLAW_DURATION_SEC,
    SAMPLE_RATE,
)
from src.features.transcript import Word
from src.schema import FlawRegion, FlawType


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class MatchedWord(NamedTuple):
    """A pair of matched baseline and candidate words.

    Either side may be None:
    - baseline=None : word was inserted in the candidate.
    - candidate=None: word was deleted (skipped) by the candidate.
    """

    baseline: Word | None
    candidate: Word | None


class WordComparison(NamedTuple):
    """Per-matched-word scalar diagnostics."""

    candidate: Word
    duration_ratio: float       # candidate_dur / baseline_dur (nan if baseline missing)
    pause_delta_s: float        # candidate_pause_before - baseline_pause_before (nan if no ref)
    energy_delta: float         # candidate_energy_z - baseline_energy_z (nan if no ref)
    pitch_range_ratio: float    # candidate_range / baseline_range (nan if either is nan)


# ---------------------------------------------------------------------------
# Step 1: Word matching
# ---------------------------------------------------------------------------

def match_words(
    baseline_words: list[Word],
    candidate_words: list[Word],
    threshold: float = MATCH_SIMILARITY_THRESHOLD,
) -> list[MatchedWord]:
    """Align baseline and candidate word lists using difflib opcodes.

    Uses difflib.SequenceMatcher on the lowercased word text. Operations:
      - equal / replace → paired MatchedWord(baseline, candidate)
      - insert          → MatchedWord(None, candidate) — extra candidate word
      - delete          → MatchedWord(baseline, None)  — missing candidate word

    For 'replace' blocks the words are zipped; any remainder from the longer
    side is emitted as insertions or deletions.

    Args:
        baseline_words:  Words from the ideal (reference) recording.
        candidate_words: Words from the participant recording.
        threshold:       Minimum SequenceMatcher ratio to treat a replace as a
                         valid pair (else treated as delete + insert).

    Returns:
        Ordered list of MatchedWord pairs.
    """
    b_texts = [w.text.lower() for w in baseline_words]
    c_texts = [w.text.lower() for w in candidate_words]

    matcher = difflib.SequenceMatcher(None, b_texts, c_texts, autojunk=False)
    result: list[MatchedWord] = []

    for tag, b1, b2, c1, c2 in matcher.get_opcodes():
        b_chunk = baseline_words[b1:b2]
        c_chunk = candidate_words[c1:c2]

        if tag == "equal":
            for bw, cw in zip(b_chunk, c_chunk):
                result.append(MatchedWord(bw, cw))

        elif tag == "replace":
            # Pair up as many as possible
            for bw, cw in zip(b_chunk, c_chunk):
                ratio = difflib.SequenceMatcher(
                    None, bw.text.lower(), cw.text.lower()
                ).ratio()
                if ratio >= threshold:
                    result.append(MatchedWord(bw, cw))
                else:
                    result.append(MatchedWord(bw, None))
                    result.append(MatchedWord(None, cw))
            # Leftovers
            for bw in b_chunk[len(c_chunk) :]:
                result.append(MatchedWord(bw, None))
            for cw in c_chunk[len(b_chunk) :]:
                result.append(MatchedWord(None, cw))

        elif tag == "delete":
            for bw in b_chunk:
                result.append(MatchedWord(bw, None))

        elif tag == "insert":
            for cw in c_chunk:
                result.append(MatchedWord(None, cw))

    return result


# ---------------------------------------------------------------------------
# Helpers: word-level feature extraction
# ---------------------------------------------------------------------------

def _word_energy_mean(word: Word, energy_z: np.ndarray) -> float:
    """Mean energy z-score over the word's frame span."""
    f_start = int(round(word.start * SAMPLE_RATE / HOP_LENGTH))
    f_end = max(f_start + 1, int(round(word.end * SAMPLE_RATE / HOP_LENGTH)))
    f_start = max(0, min(f_start, len(energy_z)))
    f_end = max(f_start, min(f_end, len(energy_z)))
    if f_end <= f_start:
        return 0.0
    return float(np.mean(energy_z[f_start:f_end]))


def _word_pitch_range(word: Word, pitch_st: np.ndarray) -> float:
    """Voiced pitch range (max - min) in semitones, NaN if < 2 voiced frames."""
    f_start = int(round(word.start * SAMPLE_RATE / HOP_LENGTH))
    f_end = max(f_start + 1, int(round(word.end * SAMPLE_RATE / HOP_LENGTH)))
    f_start = max(0, min(f_start, len(pitch_st)))
    f_end = max(f_start, min(f_end, len(pitch_st)))
    if f_end <= f_start:
        return float("nan")
    voiced = pitch_st[f_start:f_end]
    voiced = voiced[np.isfinite(voiced)]
    if voiced.size < 2:
        return float("nan")
    return float(np.max(voiced) - np.min(voiced))


def _pause_before(word_idx: int, words: list[Word]) -> float:
    """Gap between the previous word's end and this word's start (clamped to 0)."""
    if word_idx == 0:
        return 0.0
    return max(0.0, words[word_idx].start - words[word_idx - 1].end)


# ---------------------------------------------------------------------------
# Step 2: Per-word comparison
# ---------------------------------------------------------------------------

def compare_words(
    matched: list[MatchedWord],
    baseline_pitch: np.ndarray,
    candidate_pitch: np.ndarray,
    baseline_energy: np.ndarray,
    candidate_energy: np.ndarray,
) -> list[WordComparison]:
    """Compute diagnostic scalars for each matched word pair.

    Args:
        matched:          Output of match_words().
        baseline_pitch:   Pitch array (semitones) for the baseline.
        candidate_pitch:  Pitch array for the candidate.
        baseline_energy:  Energy z-score array for the baseline.
        candidate_energy: Energy z-score array for the candidate.

    Returns:
        One WordComparison per MatchedWord that has a non-None candidate word.
    """
    results: list[WordComparison] = []

    # Build index-aware lists
    b_words = [m.baseline for m in matched if m.baseline is not None]
    c_words = [m.candidate for m in matched if m.candidate is not None]

    # We iterate matched pairs; for pause_delta we need position in each list
    b_idx = 0
    c_idx = 0

    for m in matched:
        if m.candidate is None:
            if m.baseline is not None:
                b_idx += 1
            continue

        cw = m.candidate
        bw = m.baseline

        # Duration ratio
        c_dur = cw.end - cw.start
        if bw is not None:
            if cw.start == bw.start and cw.end == bw.end:
                dur_ratio = 1.0
            else:
                b_dur = max(bw.end - bw.start, 1e-6)
                dur_ratio = c_dur / b_dur
        else:
            dur_ratio = float("nan")

        # Pause delta (candidate pause_before - baseline pause_before)
        c_pause = _pause_before(c_idx, c_words)
        if bw is not None:
            if c_idx == b_idx and cw.start == bw.start and (c_idx == 0 or c_words[c_idx - 1].end == b_words[b_idx - 1].end):
                pause_delta = 0.0
            else:
                b_pause = _pause_before(b_idx, b_words)
                pause_delta = c_pause - b_pause
        else:
            pause_delta = float("nan")

        # Energy delta
        c_energy = _word_energy_mean(cw, candidate_energy)
        if bw is not None:
            b_energy = _word_energy_mean(bw, baseline_energy)
            if cw.start == bw.start and cw.end == bw.end and np.isclose(c_energy, b_energy, atol=1e-5):
                energy_delta = 0.0
            else:
                energy_delta = c_energy - b_energy
        else:
            energy_delta = float("nan")

        # Pitch range ratio (NaN-aware)
        c_pitch_range = _word_pitch_range(cw, candidate_pitch)
        if bw is not None:
            b_pitch_range = _word_pitch_range(bw, baseline_pitch)
            if math.isnan(c_pitch_range) or math.isnan(b_pitch_range) or b_pitch_range < 1e-6:
                pitch_ratio = float("nan")
            elif cw.start == bw.start and cw.end == bw.end and np.isclose(c_pitch_range, b_pitch_range, atol=1e-5):
                pitch_ratio = 1.0
            else:
                pitch_ratio = c_pitch_range / b_pitch_range
        else:
            pitch_ratio = float("nan")

        results.append(
            WordComparison(
                candidate=cw,
                duration_ratio=dur_ratio,
                pause_delta_s=pause_delta,
                energy_delta=energy_delta,
                pitch_range_ratio=pitch_ratio,
            )
        )

        c_idx += 1
        if bw is not None:
            b_idx += 1

    return results


# ---------------------------------------------------------------------------
# Step 3: Merge adjacent flags into FlawRegions
# ---------------------------------------------------------------------------

def _merge_runs(
    comparisons: list[WordComparison],
    flag_fn,
    flaw_type: FlawType,
    severity_fn,
    metadata_fn,
) -> list[FlawRegion]:
    """Merge adjacent flagged words into FlawRegion objects.

    Args:
        comparisons:  List of WordComparison.
        flag_fn:      (WordComparison) → bool — True if word should be flagged.
        flaw_type:    The FlawType to assign.
        severity_fn:  (list[WordComparison]) → float — compute severity for a run.
        metadata_fn:  (list[WordComparison]) → dict — build metadata for a run.

    Returns:
        List of FlawRegion, one per contiguous run of flagged words.
    """
    from src.explain.templates import explain

    flaws: list[FlawRegion] = []
    run: list[WordComparison] = []

    def _emit(run: list[WordComparison]) -> None:
        if len(run) < MATCH_MIN_REGION_WORDS:
            return
        start_s = run[0].candidate.start
        end_s = run[-1].candidate.end
        if end_s <= start_s:
            return
        sev = severity_fn(run)
        meta = metadata_fn(run)
        flaws.append(
            FlawRegion(
                start=round(start_s, 3),
                end=round(end_s, 3),
                flaw_type=flaw_type,
                severity=round(sev, 4),
                explanation=explain(flaw_type, sev, meta),
                metadata=meta,
            )
        )

    for comp in comparisons:
        if flag_fn(comp):
            # Adjacent flagged words can have an unflagged word between them;
            # time, rather than list position, defines whether they are one run.
            if run and comp.candidate.start - run[-1].candidate.end > FLAW_MERGE_GAP_SEC:
                _emit(run)
                run = []
            run.append(comp)

    if run:
        _emit(run)

    return [
        flaw for flaw in flaws
        if flaw.end - flaw.start >= MIN_FLAW_DURATION_SEC
    ]


# ---------------------------------------------------------------------------
# Public: detect flaws from word comparisons
# ---------------------------------------------------------------------------

def detect_from_matches(
    comparisons: list[WordComparison],
) -> list[FlawRegion]:
    """Detect pace, pause, and energy flaws from word-level comparisons.

    Returns:
        Time-sorted list of FlawRegion.
    """
    from src.analysis.detector import _severity

    flaws: list[FlawRegion] = []

    # --- PACE_TOO_SLOW ---
    flaws.extend(
        _merge_runs(
            comparisons,
            flag_fn=lambda c: not math.isnan(c.duration_ratio)
                              and c.duration_ratio > MATCH_SLOW_RATIO_MIN,
            flaw_type=FlawType.PACE_TOO_SLOW,
            severity_fn=lambda run: _severity(
                float(np.mean([c.duration_ratio for c in run])) - MATCH_SLOW_RATIO_MIN,
                scale=2.0,
            ),
            metadata_fn=lambda run: {
                "mean_duration_ratio": round(
                    float(np.mean([c.duration_ratio for c in run])), 3
                )
            },
        )
    )

    # --- PACE_TOO_FAST ---
    flaws.extend(
        _merge_runs(
            comparisons,
            flag_fn=lambda c: not math.isnan(c.duration_ratio)
                              and c.duration_ratio < MATCH_FAST_RATIO_MAX,
            flaw_type=FlawType.PACE_TOO_FAST,
            severity_fn=lambda run: _severity(
                MATCH_FAST_RATIO_MAX - float(np.mean([c.duration_ratio for c in run])),
                scale=2.0,
            ),
            metadata_fn=lambda run: {
                "mean_duration_ratio": round(
                    float(np.mean([c.duration_ratio for c in run])), 3
                )
            },
        )
    )

    # --- PAUSE_EXCESSIVE ---
    flaws.extend(
        _merge_runs(
            comparisons,
            flag_fn=lambda c: not math.isnan(c.pause_delta_s)
                              and c.pause_delta_s > MATCH_PAUSE_EXCESS_DELTA,
            flaw_type=FlawType.PAUSE_EXCESSIVE,
            severity_fn=lambda run: _severity(
                float(np.mean([c.pause_delta_s for c in run])) - MATCH_PAUSE_EXCESS_DELTA,
                scale=1.5,
            ),
            metadata_fn=lambda run: {
                "mean_pause_delta_s": round(
                    float(np.mean([c.pause_delta_s for c in run])), 3
                )
            },
        )
    )

    # --- PAUSE_MISSING ---
    # Only flag when baseline had a substantial pause that the candidate omitted.
    flaws.extend(
        _merge_runs(
            comparisons,
            flag_fn=lambda c: not math.isnan(c.pause_delta_s)
                              and c.pause_delta_s < MATCH_PAUSE_MISSING_DELTA,
            flaw_type=FlawType.PAUSE_MISSING,
            severity_fn=lambda run: _severity(
                abs(float(np.mean([c.pause_delta_s for c in run])))
                - abs(MATCH_PAUSE_MISSING_DELTA),
                scale=1.5,
            ),
            metadata_fn=lambda run: {
                "mean_pause_delta_s": round(
                    float(np.mean([c.pause_delta_s for c in run])), 3
                )
            },
        )
    )

    # --- ENERGY_LOW ---
    flaws.extend(
        _merge_runs(
            comparisons,
            flag_fn=lambda c: not math.isnan(c.energy_delta)
                              and c.energy_delta < MATCH_ENERGY_LOW_DELTA,
            flaw_type=FlawType.ENERGY_LOW,
            severity_fn=lambda run: _severity(
                abs(float(np.mean([c.energy_delta for c in run])))
                - abs(MATCH_ENERGY_LOW_DELTA),
                scale=0.8,
            ),
            metadata_fn=lambda run: {
                "mean_energy_delta": round(
                    float(np.mean([c.energy_delta for c in run])), 3
                )
            },
        )
    )

    flaws.sort(key=lambda r: r.start)
    return flaws
