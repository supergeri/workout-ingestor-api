"""
Parse endpoints for structured text parsing

Provides POST /parse/text for Instagram caption and general workout text parsing.
Returns structured exercise data with sets, reps, superset_group, etc.

Uses TextParser._try_structured_parse() as per AMA-555 requirements.
"""

import asyncio
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
# Preprocessing Patterns
# ---------------------------------------------------------------------------

NUMBERED_PATTERN = re.compile(r'^\s*(?:\d+[.):]|\d+\s*[.):])\s*(.+)')
BULLET_PATTERN = re.compile(r'^\s*[•\-→>]\s*(.+)')


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Preprocess text to extract exercise lines from various formats.
    Removes numbered/bullet markers for TextParser to handle.
    """
    lines = text.strip().split('\n')
    processed_lines = []
    
    for line in lines:
        trimmed = line.strip()
        
        if not trimmed:
            continue
        
        # Try to extract content from numbered list format
        numbered_match = NUMBERED_PATTERN.match(trimmed)
        if numbered_match:
            trimmed = numbered_match.group(1).strip()
        else:
            # Try to extract content from bullet format
            bullet_match = BULLET_PATTERN.match(trimmed)
            if bullet_match:
                trimmed = bullet_match.group(1).strip()
        
        if trimmed:
            processed_lines.append(trimmed)
    
    return '\n'.join(processed_lines)


def clean_exercise_name(name: str) -> str:
    """Clean up exercise name by removing special annotations"""
    return (
        name
        .replace('→', '')    # U+2192
        .replace('➜', '')    # U+279C
        .replace('➡', '')    # U+27A1
        .replace('=>', '')
        .strip()
    )


# ---------------------------------------------------------------------------
# Main Parsing (uses TextParser)
# ---------------------------------------------------------------------------

def parse_result_to_response(parse_result: ParseResult, source: Optional[str]) -> ParseTextResponse:
    """
    Convert TextParser ParseResult to HTTP response format (AMA-555).
    
    Transforms ParsedWorkout/ParsedExercise models to ParsedExerciseResponse objects
    for the HTTP API response.
    """
    exercises: List[ParsedExerciseResponse] = []
    order = 0
    
    for workout in parse_result.workouts:
        for exercise in workout.exercises:
            # Clean exercise name by removing special characters
            clean_name = clean_exercise_name(exercise.raw_name)
            
            # Determine if part of superset (from exercise metadata)
            superset_group = exercise.superset_group
            
            # Parse distance from reps if it contains units like 'm'
            distance = None
            reps = exercise.reps
            
            # Check if reps contains distance marker (e.g., "10m")
            if reps and 'm' in reps.lower():
                # Try to extract distance - only if it's just a number + unit (no dash range)
                distance_match = re.match(r'^(\d+(?:\.\d+)?)(m|meters?)$', str(reps), re.IGNORECASE)
                if distance_match:
                    distance = reps
                    reps = None
            
            exercises.append(ParsedExerciseResponse(
                raw_name=clean_name,
                sets=exercise.sets if exercise.sets > 1 else None,
                reps=reps if reps != "1" else None,
                distance=distance,
                superset_group=superset_group,
                order=order,
                weight=exercise.weight,
                weight_unit=exercise.weight_unit,
                rpe=exercise.rpe,
                notes=exercise.notes,
                rest_seconds=exercise.rest_seconds,
            ))
            order += 1
    
    # Map TextParser format names to API format names
    detected_format = parse_result.detected_format
    if detected_format == "text_structured":
        detected_format = "instagram_caption"
    
    return ParseTextResponse(
        success=parse_result.success,
        exercises=exercises,
        detected_format=detected_format,
        confidence=parse_result.confidence,
        source=source,
        metadata=parse_result.metadata,
    )


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
            metadata={"parser": "llm_fallback"}
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
    
    # Preprocess text to extract exercise lines (handle numbered/bullet formats)
    preprocessed_text = preprocess_text(text)
    
    # Use TextParser._try_structured_parse() as per AMA-555 requirement
    parser = TextParser()
    parse_result = await parser._try_structured_parse(preprocessed_text)
    
    # Convert TextParser result to HTTP response format
    response = parse_result_to_response(parse_result, source)
    
    # If structured parsing found exercises, return them
    if response.success and response.exercises:
        return JSONResponse(response.model_dump())
    
    # Fall back to LLM parsing if no exercises found
    llm_result = await parse_with_llm_fallback(text, source)
    
    return JSONResponse(llm_result.model_dump())
