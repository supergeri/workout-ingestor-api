"""Shared utilities for workout data sanitization.

This module provides a centralized function for fixing common LLM structural
mistakes in workout data, used by both YouTube and Instagram ingest paths.
"""

import re
from typing import Dict

_REPS_RANGE_RE = re.compile(r"(\d+)\s*[-–]\s*(\d+)")
_REPS_INT_RE = re.compile(r"^\d+$")


def _sanitize_exercise_reps(exercise: dict) -> None:
    """Coerce string reps to int or reps_range before Pydantic validation.

    LLMs sometimes return reps as a string (e.g. "6-8 each leg" or "10").
    This normalizes:
      - "10"          → reps=10
      - "6-8"         → reps=None, reps_range="6-8"
      - "6-8 each leg"→ reps=None, reps_range="6-8", notes+=" (per side)"
      - other strings → reps=None
    """
    reps = exercise.get("reps")
    if reps is None or isinstance(reps, int):
        return
    reps_str = str(reps).strip()
    if _REPS_INT_RE.match(reps_str):
        exercise["reps"] = int(reps_str)
        return
    m = _REPS_RANGE_RE.search(reps_str)
    if m:
        exercise.setdefault("reps_range", f"{m.group(1)}-{m.group(2)}")
        exercise["reps"] = None
        remainder = reps_str[m.end():].strip().lower()
        if any(word in remainder for word in ("leg", "side", "arm", "each")):
            notes = (exercise.get("notes") or "").strip()
            if "(per side)" not in notes:
                exercise["notes"] = (notes + " (per side)").strip()
    else:
        exercise["reps"] = None


def sanitize_workout_data(workout_data: Dict) -> Dict:
    """Sanitize LLM output to fix common structural mistakes.

    Fixes:
    1. Preserve circuit/rounds/amrap/emom/for-time blocks (exercises stay in exercises[])
    2. If supersets is non-empty and NOT a circuit: set structure to "superset", clear exercises[]
    3. If structure == "superset" but supersets is empty: reset structure to null
    4. Validates blocks is a list and superset entries have exercises

    Args:
        workout_data: Raw workout dict from LLM parsing

    Returns:
        Sanitized workout dict with fixed structure
    """
    circuit_structures = {"circuit", "rounds", "amrap", "emom", "for-time"}

    blocks = workout_data.get("blocks")
    if not isinstance(blocks, list):
        return workout_data

    for block in blocks:
        structure = block.get("structure")

        # Preserve circuit-type blocks — exercises belong in exercises[], not supersets
        # But only if exercises[] is non-empty; if LLM put everything in supersets
        # with a circuit label, fall through to the superset path to avoid data loss
        if structure in circuit_structures and block.get("exercises"):
            block["supersets"] = []
        else:
            supersets = block.get("supersets", [])
            # Filter out malformed superset entries (must have exercises list)
            valid_supersets = [
                s for s in supersets
                if isinstance(s, dict) and isinstance(s.get("exercises"), list) and len(s["exercises"]) > 0
            ]
            block["supersets"] = valid_supersets

            if valid_supersets:
                block["structure"] = "superset"
                block["exercises"] = []
            elif structure == "superset":
                # Structure says superset but no valid supersets — reset
                block["structure"] = None

        # Normalize rest field aliases → rest_between_rounds_sec
        if "rest_sec" in block and "rest_between_rounds_sec" not in block:
            block["rest_between_rounds_sec"] = block.pop("rest_sec")
        elif "rest_sec" in block:
            block.pop("rest_sec")  # already has rest_between_rounds_sec, discard alias
        if "rest_between_sec" in block and "rest_between_rounds_sec" not in block:
            block["rest_between_rounds_sec"] = block.pop("rest_between_sec")
        elif "rest_between_sec" in block:
            block.pop("rest_between_sec")

        # Default missing exercise type to "strength"; coerce string reps (all blocks)
        for exercise in block.get("exercises", []):
            if not exercise.get("type"):
                exercise["type"] = "strength"
            _sanitize_exercise_reps(exercise)
        for superset in block.get("supersets", []):
            for exercise in superset.get("exercises", []):
                if not exercise.get("type"):
                    exercise["type"] = "strength"
                _sanitize_exercise_reps(exercise)
    return workout_data
