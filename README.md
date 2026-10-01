# Speech Evaluation System (Track C: Contrastive Speech Analytics)

A contrastive speech evaluation system that takes a participant's audio and transcript, compares it to an "ideal" baseline recording of the same text, and outputs rubric scores, time-stamped flaw regions, and human-readable causal explanations as JSON.

---

## Quickstart

### 1. Prerequisites & Environment Setup
- Python 3.11 (e.g. Python 3.11.9)

```bash
# Activate existing virtual environment (Windows PowerShell)
.\.venv\Scripts\activate

# Or create fresh
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Full Test Suite
```bash
pytest tests/ -v
```

### 3. One-Command End-to-End Pipeline
```bash
python scripts/run_all.py
```

---

## Adding Real Speeches

1. Place source `.wav` audio files into `data/raw/` (e.g. `speech_01.wav`).
2. Add a matching transcript `.txt` file with the exact same base name (e.g. `speech_01.txt`).
3. Run the one-command pipeline:
   ```bash
   python scripts/run_all.py
   ```
   This will automatically:
   - Generate flawed speech variations across 7 flaw types and 5 severity levels into `data/flawed/`.
   - Write ground-truth flaw intervals to `data/labels/injections.csv`.
   - Validate labels with `scripts/validate_labels.py`.
   - Evaluate each candidate file contrastively against its baseline.
   - Save output JSON results to `outputs/<filename>_eval.json`.
   - Print detection Precision, Recall, and F1 metrics.

---

## System Architecture

```
speech_eval/
  src/
    config.py                 All constants, paths, thresholds, and seeds (seed=42)
    schema.py                 Pydantic v2 output schemas (EvaluationResult, FlawRegion, RubricScore)
    pipeline.py               End-to-end evaluation orchestrator
    features/
      audio.py                pYIN pitch tracking (NaN unvoiced), RMS energy z-score, pause detection
      forced_align.py         WhisperX phone-level forced alignment with MMS_FA and proportional fallbacks
      transcript.py           Tokenization and character-proportional timing fallback
      spectral.py             13 MFCCs, spectral centroid, spectral flatness, percentile clarity SNR
      word_features.py        Per-word acoustic metrics (duration, pitch mean/range, energy z, pauses)
      alignment.py            Sakoe-Chiba DTW feature alignment
    analysis/
      scoring.py              Exponential decay rubric scoring: score = exp(-k * dev)
      detector.py             Single-stream sliding window flaw detectors (pace, pitch, energy, pause, filler)
      match.py                difflib word matching and contrastive flaw detection
    dataset/
      generator.py            Region-based flaw injection (slow, fast, pause, monotone, low energy, filler, unclear)
      generate.py             Dataset generation CLI (python -m src.dataset.generate)
    explain/
      templates.py            Deterministic template-based causal explanations
  scripts/
    run_all.py                One-command end-to-end runner
    validate_labels.py        Validates injections.csv integrity
    evaluate_detections.py    Event-level 1-D temporal IoU detection accuracy benchmark
  data/
    raw/                      Source recordings (*.wav + *.txt)
    flawed/                   Injected flaw recordings
    labels/
      injections.csv          Ground-truth flaw annotations
      alignments/             Cached forced-alignment JSONs
  docs/
    DECISIONS.md              Design decisions log with technical justifications
    features.md               Mathematical definitions of acoustic and spectral features
```

---

## Scoring Model & Output Contract

Scores are normalized to `[0.0, 1.0]` (1.0 = ideal, matching the baseline).

- **Formula:** `score = exp(-k * deviation)`
- **Rubric Dimensions:**
  - `pace`: Speaking rate in WPM relative to ideal range (120–160 WPM).
  - `pitch_variation`: Expressiveness via pitch std-dev (voiced frames only).
  - `energy_consistency`: Volume stability via RMS z-score variability.
  - `pause_pattern`: Penalizes omitted pauses in long speeches and excessive pauses.
- **Flaw Regions:** Time-stamped intervals `[start, end]` with normalized severity `[0, 1]`, taxonomic `FlawType`, and human-readable causal explanation.
