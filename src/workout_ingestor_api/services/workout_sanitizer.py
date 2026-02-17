"""Shared utilities for workout data sanitization.

This module provides a centralized function for fixing common LLM structural
mistakes in workout data, used by both YouTube and Instagram ingest paths.
"""

from typing import Dict


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
            continue

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
    return workout_data
