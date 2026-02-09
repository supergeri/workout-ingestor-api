"""
Parse endpoints for structured text parsing

Provides POST /parse/text for Instagram caption and general workout text parsing.
Returns structured exercise data with sets, reps, superset_group, etc.

Uses TextParser._try_structured_parse() as per AMA-555 requirements.
"""

import asyncio
import re
import logging
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from workout_ingestor_api.parsers.models import (
    ParseResult,
    ParsedWorkout,
    ParsedExercise,
    FileInfo,
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
# Patterns for preprocessing and custom parsing
# ---------------------------------------------------------------------------

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
BULLET_PATTERN = re.compile(r'^\s*[•\-→>]\s*(.+)')

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


# ---------------------------------------------------------------------------
# Helper functions (preserved from original)
# ---------------------------------------------------------------------------

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


def has_sets_reps_notation(text: str) -> bool:
    """Check if text contains set/rep notation like '4x8' or '3 x 10'"""
    pattern = re.compile(r'\d+\s*[x×]\s*\d+(?:[-–]\d+)?', re.IGNORECASE)
    return bool(pattern.search(text))


def split_superset_intelligently(text: str) -> List[str]:
    """
    Split text on '+' only if both sides have set/rep notation.
    
    Examples:
    - "Pull-ups 4x8 + Z Press 4x8" -> ["Pull-ups 4x8", "Z Press 4x8"]
    - "Chin-up + Negative Hold" -> ["Chin-up + Negative Hold"] (kept together)
    """
    SUPERSET_SPLIT_PATTERN = re.compile(r'\s*\+\s*')
    parts = SUPERSET_SPLIT_PATTERN.split(text)
    
    if len(parts) <= 1:
        return [text]
    
    # Only split if ALL parts have set/rep notation
    all_have_sets_reps = all(has_sets_reps_notation(part) for part in parts)
    
    if all_have_sets_reps:
        return [p.strip() for p in parts]
    
    return [text]


def clean_exercise_name(name: str) -> str:
    """Clean up exercise name by removing annotations"""
    return (
        name
        .replace('→', '')    # U+2192
        .replace('➜', '')    # U+279C
        .replace('➡', '')    # U+27A1
        .replace('=>', '')
        .strip()
    )


# ---------------------------------------------------------------------------
# Preprocessing and manual parsing helpers
# ---------------------------------------------------------------------------

def preprocess_and_split_lines(text: str) -> Tuple[List[str], Dict[int, str]]:
    """
    Preprocess text into lines with superset detection.
    Returns (processed_lines, superset_mapping).
    
    The superset_mapping tracks which exercises belong to which superset group.
    """
    lines = text.strip().split('\n')
    processed_lines = []
    superset_mapping = {}
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
        
        if not trimmed:
            continue
        
        # Handle supersets - split intelligently
        parts = split_superset_intelligently(trimmed)
        
        if len(parts) > 1:
            # This is a superset - assign group letter
            superset_counter += 1
            group_letter = chr(64 + superset_counter)  # A, B, C, ...
            
            for part in parts:
                superset_mapping[len(processed_lines)] = group_letter
                processed_lines.append(part)
        else:
            processed_lines.append(parts[0])
    
    return processed_lines, superset_mapping


def parse_line_with_original_patterns(
    line: str, 
    order: int, 
    superset_group: Optional[str] = None
) -> Optional[ParsedExerciseResponse]:
    """
    Parse a single line using the original patterns from parse_instagram_caption.
    This is used as a fallback when TextParser doesn't find exercises.
    
    Only returns exercises that have set/rep or distance notation.
    Returns None for bare exercise names without structured data.
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
            raw_name=clean_exercise_name(name),
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
                raw_name=clean_exercise_name(name),
                sets=sets,
                reps=f"{reps}{unit}",
                distance=None,
                superset_group=superset_group,
                order=order,
                rpe=rpe,
            )
        
        return ParsedExerciseResponse(
            raw_name=clean_exercise_name(name),
            sets=sets,
            reps=reps,
            distance=None,
            superset_group=superset_group,
            order=order,
            rpe=rpe,
        )
    
    # No set/rep or distance notation - return as exercise with no sets/reps
    # TextParser uses lenient parsing for compound names
    return ParsedExerciseResponse(
        raw_name=clean_exercise_name(line),
        sets=None,
        reps=None,
        distance=None,
        superset_group=superset_group,
        order=order,
    )


# ---------------------------------------------------------------------------
# TextParser integration
# ---------------------------------------------------------------------------

def parse_result_to_response(
    parse_result: ParseResult,
    source: Optional[str] = None,
    superset_mapping: Optional[Dict[int, str]] = None,
    original_lines: Optional[List[str]] = None
) -> ParseTextResponse:
    """
    Convert TextParser's ParseResult to ParseTextResponse format (AMA-555).
    
    Maps ParsedExercise to ParsedExerciseResponse, preserving all fields.
    If TextParser finds exercises, use them. Otherwise, fallback to manual parsing
    of original_lines with the superset_mapping.
    
    Also handles extracting distance information from exercises that have it.
    """
    exercises = []
    order = 0
    
    # First, try to use TextParser results
    if parse_result.workouts:
        for workout in parse_result.workouts:
            for ex in workout.exercises:
                # Get superset group from mapping (if available)
                superset_group = ex.superset_group
                if superset_mapping and order in superset_mapping:
                    superset_group = superset_mapping[order]
                
                # Check if we have distance information in the original line
                distance = None
                reps = ex.reps
                if original_lines and order < len(original_lines):
                    # Try to extract distance from original line
                    original_line = original_lines[order]
                    distance_match = DISTANCE_PATTERN.match(original_line)
                    if distance_match:
                        distance = f"{distance_match.group('distance')}{distance_match.group('unit')}"
                        reps = None
                
                exercise_response = ParsedExerciseResponse(
                    raw_name=clean_exercise_name(ex.raw_name),
                    sets=ex.sets if ex.sets and ex.sets > 1 else None,
                    reps=reps if reps and reps != "1" else None,
                    distance=distance,
                    superset_group=superset_group,
                    order=order,
                    weight=ex.weight,
                    weight_unit=ex.weight_unit,
                    rpe=ex.rpe,
                    notes=ex.notes,
                    rest_seconds=ex.rest_seconds,
                )
                exercises.append(exercise_response)
                order += 1
    
    # If TextParser found nothing but we have original lines, fall back to manual parsing
    if not exercises and original_lines:
        for i, line in enumerate(original_lines):
            superset_group = superset_mapping.get(i) if superset_mapping else None
            exercise = parse_line_with_original_patterns(line, i, superset_group)
            if exercise:
                exercises.append(exercise)
    
    # Calculate confidence based on structured data
    if not exercises:
        confidence = 0
    else:
        structured_count = sum(1 for e in exercises if e.sets is not None)
        confidence = min(90, int((structured_count / len(exercises)) * 100))
    
    # Determine detected format
    detected_format = parse_result.detected_format
    if detected_format == "text_structured":
        detected_format = "instagram_caption"
    
    return ParseTextResponse(
        success=len(exercises) > 0,
        exercises=exercises,
        detected_format=detected_format or "instagram_caption",
        confidence=confidence,
        source=source,
        metadata=parse_result.metadata,
    )


async def parse_with_text_parser(text: str, source: Optional[str]) -> ParseTextResponse:
    """
    Parse text using TextParser._try_structured_parse() as per AMA-555.
    """
    # Preprocess lines and detect supersets
    lines, superset_mapping = preprocess_and_split_lines(text)
    
    if not lines:
        return ParseTextResponse(
            success=False,
            exercises=[],
            detected_format="text_unstructured",
            confidence=0,
            source=source,
        )
    
    # Create TextParser instance
    parser = TextParser()
    
    # Call _try_structured_parse with the preprocessed text
    processed_text = '\n'.join(lines)
    try:
        result = await parser._try_structured_parse(processed_text)
    except Exception as e:
        logger.warning(f"TextParser._try_structured_parse failed: {e}")
        result = ParseResult(success=False)
    
    # Convert to response format, passing original lines for distance extraction
    response = parse_result_to_response(result, source, superset_mapping, lines)
    
    return response


async def parse_with_llm_fallback(text: str, source: Optional[str]) -> ParseTextResponse:
    """
    Use LLM parsing as fallback when structured parsing doesn't find exercises.
    """
    try:
        workout = await asyncio.to_thread(
            ParserService.parse_free_text_to_workout, text, source or "text"
        )
        
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
            source=source,
            metadata={"parser": "llm_fallback"}
        )
    
    except Exception as e:
        logger.warning(f"LLM fallback failed: {e}")
        return ParseTextResponse(
            success=False,
            exercises=[],
            detected_format="text_unstructured",
            confidence=0,
            source=source,
            metadata={"error": str(e)}
        )


# ---------------------------------------------------------------------------
# API Endpoint
# ---------------------------------------------------------------------------

@router.post("/parse/text")
async def parse_text(request: ParseTextRequest) -> JSONResponse:
    """
    Parse workout text (e.g., Instagram caption) into structured exercise data.
    
    Uses TextParser._try_structured_parse() per AMA-555 requirements.
    
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
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    text = request.text.strip()
    if not text:
        # Return successful response with no exercises for whitespace-only input
        return JSONResponse(ParseTextResponse(
            success=False,
            exercises=[],
            detected_format="text_unstructured",
            confidence=0,
            source=request.source,
        ).model_dump())
    
    source = request.source or "instagram_caption"
    
    # Try structured parsing first using TextParser
    result = await parse_with_text_parser(text, source)
    
    # If structured parsing found exercises, return them
    if result.success and result.exercises:
        return JSONResponse(result.model_dump())
    
    # Fall back to LLM parsing if no exercises found
    llm_result = await parse_with_llm_fallback(text, source)
    
    return JSONResponse(llm_result.model_dump())
