"""
Parse endpoints for structured text parsing

Provides POST /parse/text for Instagram caption and general workout text parsing.
Returns structured exercise data with sets, reps, superset_group, etc.
"""

import re
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from workout_ingestor_api.parsers.models import (
    ParseResult,
    ParsedWorkout,
    ParsedExercise,
)
from workout_ingestor_api.parsers.text_parser import TextParser
from workout_ingestor_api.services.parser_service import ParserService

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class ParseTextRequest(BaseModel):
    """Request model for POST /parse/text"""
    text: str = Field(..., description="Text to parse (e.g., Instagram caption)")
    source: Optional[str] = Field(default=None, description="Optional source hint (e.g., 'instagram_caption')")


class ParsedExerciseResponse(BaseModel):
    """Single parsed exercise response"""
    raw_name: str = Field(..., description="Original exercise name")
    sets: Optional[int] = Field(default=None, description="Number of sets")
    reps: Optional[str] = Field(default=None, description="Reps (may include ranges like '8-12')")
    distance: Optional[str] = Field(default=None, description="Distance if applicable (e.g., '10m')")
    superset_group: Optional[str] = Field(default=None, description="Group ID for supersets (e.g., 'A', 'B')")
    order: int = Field(..., description="Exercise order (0-indexed)")
    weight: Optional[str] = Field(default=None)
    weight_unit: Optional[str] = Field(default=None)
    rpe: Optional[float] = Field(default=None)
    notes: Optional[str] = Field(default=None)
    rest_seconds: Optional[int] = Field(default=None)


class ParseTextResponse(BaseModel):
    """Response model for POST /parse/text"""
    success: bool
    exercises: List[ParsedExerciseResponse]
    detected_format: Optional[str] = None
    confidence: float = Field(default=0, ge=0, le=100)
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Instagram-specific patterns
# ---------------------------------------------------------------------------

# Pattern to detect set/rep notation in text
SETS_REPS_PATTERN = re.compile(
    r'(?P<name>.+?)\s+(?P<sets>\d+)\s*[x×]\s*(?P<reps>\d+(?:[-–]\d+)?)(?:\s*(?P<unit>m|s|sec|seconds))?',
    re.IGNORECASE
)

# Pattern for distance notation (e.g., "5 x 10m")
DISTANCE_PATTERN = re.compile(
    r'(?P<name>.+?)\s+(?P<sets>\d+)\s*[x×]\s*(?P<distance>\d+)\s*(?P<unit>m|meters|yards|yd)',
    re.IGNORECASE
)

# Pattern for RPE notation
RPE_PATTERN = re.compile(
    r'@\s*RPE?\s*(?P<rpe>\d+(?:\.\d+)?)',
    re.IGNORECASE
)

# Pattern to split supersets (only when both sides have set/rep notation)
SUPERSET_SPLIT_PATTERN = re.compile(
    r'\s*\+\s*'
)

# Patterns to skip (hashtags, CTAs, headers)
SKIP_PATTERNS = [
    re.compile(r'^#\w+'),  # Hashtags
    re.compile(r'^follow\s+me', re.IGNORECASE),  # CTAs
    re.compile(r'^subscribe', re.IGNORECASE),
    re.compile(r'^check\s+out', re.IGNORECASE),
    re.compile(r'^link\s+in\s+bio', re.IGNORECASE),
    re.compile(r'^save\s+this', re.IGNORECASE),
    re.compile(r'^upper\s+body:', re.IGNORECASE),  # Section headers
    re.compile(r'^lower\s+body:', re.IGNORECASE),
    re.compile(r'^warmup?:', re.IGNORECASE),
    re.compile(r'^cool\s*down?:', re.IGNORECASE),
    re.compile(r'^round\s+\d+:', re.IGNORECASE),
    re.compile(r'^day\s+\d+:', re.IGNORECASE),
    re.compile(r'^week\s+\d+:', re.IGNORECASE),
]

# Numbered/bullet patterns
NUMBERED_PATTERN = re.compile(r'^\s*(?:\d+[.):]|\d+\s*[.):])\s*(.+)')
BULLET_PATTERN = re.compile(r'^[\s•\-→>]+(.+)')


def has_sets_reps_notation(text: str) -> bool:
    """Check if text contains set/rep notation like '4x8' or '3 x 10'"""
    return bool(SETS_REPS_PATTERN.search(text))


def should_skip_line(text: str) -> bool:
    """Check if line should be skipped (hashtag, CTA, header)"""
    trimmed = text.strip()
    if not trimmed:
        return True
    
    for pattern in SKIP_PATTERNS:
        if pattern.match(trimmed):
            return True
    
    # Skip standalone "Workout:" with nothing after
    if re.match(r'^workout[:\s]*$', trimmed, re.IGNORECASE):
        return True
    
    return False


def strip_workout_prefix(text: str) -> str:
    """Strip 'Workout:' prefix if present"""
    if text.lower().startswith('workout:'):
        return text[8:].strip()
    return text


def split_superset_intelligently(text: str) -> List[str]:
    """
    Split text on '+' only if both sides have set/rep notation.
    
    Examples:
    - "Pull-ups 4x8 + Z Press 4x8" -> ["Pull-ups 4x8", "Z Press 4x8"]
    - "Chin-up + Negative Hold" -> ["Chin-up + Negative Hold"] (kept together)
    """
    parts = SUPERSET_SPLIT_PATTERN.split(text)
    
    if len(parts) <= 1:
        return [text]
    
    # Only split if ALL parts have set/rep notation
    all_have_sets_reps = all(has_sets_reps_notation(part) for part in parts)
    
    if all_have_sets_reps:
        return [p.strip() for p in parts]
    
    return [text]


def parse_exercise_line(line: str, order: int, superset_group: Optional[str] = None) -> Optional[ParsedExerciseResponse]:
    """
    Parse a single exercise line and return structured data.
    
    Handles:
    - "Pull-ups 4x8"
    - "SA cable row 4x12"
    - "Seated sled pull 5 x 10m"
    - "Squats 4x8-12"
    - "Plank 3x30s"
    """
    line = line.strip()
    if not line or len(line) <= 2:
        return None
    
    # Try distance pattern first (e.g., "5 x 10m")
    distance_match = DISTANCE_PATTERN.match(line)
    if distance_match:
        name = distance_match.group('name').strip()
        sets = int(distance_match.group('sets'))
        distance_val = distance_match.group('distance')
        unit = distance_match.group('unit')
        
        # Extract RPE if present
        rpe_match = RPE_PATTERN.search(line)
        rpe = float(rpe_match.group('rpe')) if rpe_match else None
        
        return ParsedExerciseResponse(
            raw_name=name,
            sets=sets,
            reps=None,
            distance=f"{distance_val}{unit}",
            superset_group=superset_group,
            order=order,
            rpe=rpe,
        )
    
    # Try sets/reps pattern
    sets_reps_match = SETS_REPS_PATTERN.match(line)
    if sets_reps_match:
        name = sets_reps_match.group('name').strip()
        sets = int(sets_reps_match.group('sets'))
        reps = sets_reps_match.group('reps')
        unit = sets_reps_match.group('unit')
        
        # Extract RPE if present
        rpe_match = RPE_PATTERN.search(line)
        rpe = float(rpe_match.group('rpe')) if rpe_match else None
        
        # Handle time-based exercises (e.g., "30s", "60sec")
        if unit and unit.lower() in ('s', 'sec', 'seconds'):
            return ParsedExerciseResponse(
                raw_name=name,
                sets=sets,
                reps=f"{reps}{unit}",
                distance=None,
                superset_group=superset_group,
                order=order,
                rpe=rpe,
            )
        
        return ParsedExerciseResponse(
            raw_name=name,
            sets=sets,
            reps=reps,
            distance=None,
            superset_group=superset_group,
            order=order,
            rpe=rpe,
        )
    
    # No set/rep notation - just an exercise name
    return ParsedExerciseResponse(
        raw_name=line,
        sets=None,
        reps=None,
        distance=None,
        superset_group=superset_group,
        order=order,
    )


def clean_exercise_name(name: str) -> str:
    """Clean up exercise name by removing annotations"""
    return (
        name
        .replace('→', '')
        .replace('→', '')  # Different arrow types
        .replace('➜', '')
        .replace('➡', '')
        .replace('=>', '')
        .strip()
    )


# ---------------------------------------------------------------------------
# Main parsing logic
# ---------------------------------------------------------------------------

def parse_instagram_caption(text: str) -> ParseTextResponse:
    """
    Parse Instagram-style caption text into structured exercises.
    
    Handles:
    - Superset notation with "+"
    - Distance notation (e.g., "5 x 10m")
    - Rep ranges (e.g., "4x8-12")
    - Time-based exercises (e.g., "3x30s")
    - RPE notation (e.g., "@RPE8")
    - Hashtag/CTA filtering
    - Section header filtering
    """
    lines = text.strip().split('\n')
    exercises: List[ParsedExerciseResponse] = []
    current_superset_group: Optional[str] = None
    superset_counter = 0
    
    for line in lines:
        trimmed = line.strip()
        
        # Skip filtered content
        if should_skip_line(trimmed):
            continue
        
        # Strip "Workout:" prefix
        trimmed = strip_workout_prefix(trimmed)
        if not trimmed:
            continue
        
        # Try numbered/bullet patterns to extract content
        numbered_match = NUMBERED_PATTERN.match(trimmed)
        if numbered_match:
            trimmed = numbered_match.group(1).strip()
        else:
            bullet_match = BULLET_PATTERN.match(trimmed)
            if bullet_match:
                trimmed = bullet_match.group(1).strip()
        
        # Handle supersets
        parts = split_superset_intelligently(trimmed)
        
        if len(parts) > 1:
            # This is a superset - assign group letter
            superset_counter += 1
            group_letter = chr(64 + superset_counter)  # A, B, C, ...
            
            for i, part in enumerate(parts):
                exercise = parse_exercise_line(part, len(exercises), group_letter)
                if exercise:
                    exercise.raw_name = clean_exercise_name(exercise.raw_name)
                    exercises.append(exercise)
        else:
            # Single exercise
            exercise = parse_exercise_line(trimmed, len(exercises))
            if exercise:
                exercise.raw_name = clean_exercise_name(exercise.raw_name)
                exercises.append(exercise)
    
    # Calculate confidence based on how many exercises had structured data
    if not exercises:
        confidence = 0
    else:
        structured_count = sum(1 for e in exercises if e.sets is not None)
        confidence = min(90, int((structured_count / len(exercises)) * 100))
    
    return ParseTextResponse(
        success=len(exercises) > 0,
        exercises=exercises,
        detected_format="instagram_caption" if confidence > 50 else "text_unstructured",
        confidence=confidence,
    )


async def parse_with_llm_fallback(text: str, source: Optional[str]) -> ParseTextResponse:
    """
    Use LLM parsing as fallback when structured parsing doesn't find exercises.
    """
    try:
        workout = ParserService.parse_free_text_to_workout(text, source or "text")
        
        exercises = []
        order = 0
        
        for block in workout.blocks:
            block_label = block.label
            # Check if block label indicates superset
            is_superset = block.structure == "superset" or "superset" in block_label.lower()
            
            superset_group = None
            if is_superset and block.exercises:
                superset_group = chr(65 + order // 10)  # Simple grouping
            
            for ex in block.exercises:
                exercises.append(ParsedExerciseResponse(
                    raw_name=ex.name,
                    sets=ex.sets,
                    reps=str(ex.reps) if ex.reps else None,
                    distance=f"{ex.distance_m}m" if ex.distance_m else None,
                    superset_group=superset_group,
                    order=order,
                    weight=str(ex.weight_kg) if ex.weight_kg else None,
                    notes=ex.notes,
                ))
                order += 1
        
        return ParseTextResponse(
            success=len(exercises) > 0,
            exercises=exercises,
            detected_format="text_llm",
            confidence=70,
            metadata={"model": "gpt-4o-mini"}
        )
    
    except Exception as e:
        logger.warning(f"LLM fallback failed: {e}")
        return ParseTextResponse(
            success=False,
            exercises=[],
            detected_format="text_unstructured",
            confidence=0,
            metadata={"error": str(e)}
        )


# ---------------------------------------------------------------------------
# API Endpoint
# ---------------------------------------------------------------------------

@router.post("/parse/text")
async def parse_text(request: ParseTextRequest) -> JSONResponse:
    """
    Parse workout text (e.g., Instagram caption) into structured exercise data.
    
    ## Request Body
    - **text**: The text to parse (e.g., "Pull-ups 4x8 + Z Press 4x8")
    - **source**: Optional source hint (e.g., "instagram_caption")
    
    ## Response
    Returns structured exercise data with:
    - raw_name: Exercise name
    - sets: Number of sets
    - reps: Reps (may include ranges like "8-12")
    - distance: Distance if applicable (e.g., "10m")
    - superset_group: Group ID for supersets (e.g., "A", "B")
    - order: Exercise order (0-indexed)
    
    ## Example
    ```json
    {
      "text": "Workout: Pull-ups 4x8 + Z Press 4x8\\nSA cable row 4x12\\nSeated sled pull 5 x 10m",
      "source": "instagram_caption"
    }
    ```
    
    Returns:
    ```json
    {
      "success": true,
      "exercises": [
        {"raw_name": "Pull-ups", "sets": 4, "reps": "8", "superset_group": "A", "order": 0},
        {"raw_name": "Z Press", "sets": 4, "reps": "8", "superset_group": "A", "order": 1},
        {"raw_name": "SA cable row", "sets": 4, "reps": "12", "order": 2},
        {"raw_name": "Seated sled pull", "sets": 5, "distance": "10m", "order": 3}
      ],
      "confidence": 90
    }
    ```
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    
    text = request.text.strip()
    source = request.source
    
    # Try structured parsing first
    result = parse_instagram_caption(text)
    result.source = source
    
    # If structured parsing found exercises, return them
    if result.success and result.exercises:
        return JSONResponse(result.model_dump())
    
    # Fall back to LLM parsing
    llm_result = await parse_with_llm_fallback(text, source)
    llm_result.source = source
    
    return JSONResponse(llm_result.model_dump())
