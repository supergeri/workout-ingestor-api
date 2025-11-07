"""LLM service for structuring workout text into canonical JSON format."""
import os
import json
import re
from typing import Dict, Optional
from fastapi import HTTPException

# Optional dependencies
try:
    import openai  # type: ignore
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    from anthropic import Anthropic  # type: ignore
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None


class LLMService:
    """Service for using LLMs to structure workout data."""
    
    WORKOUT_STRUCTURE_PROMPT = """You are a fitness workout parser. Convert the following workout text into structured JSON format.

The workout text may contain:
- Exercise names
- Sets and reps (e.g., "3 sets of 10 reps", "5x5")
- Weights/loads (e.g., "32kg", "100lb")
- Time intervals (e.g., "30 seconds", "2 minutes")
- Rest periods
- Equipment notes
- Exercise cues/instructions

Extract and structure this into a JSON format matching:
{
  "title": "workout title",
  "blocks": [
    {
      "label": "block name (e.g., 'Warm-up', 'Strength', 'Conditioning')",
      "structure": "structure description (e.g., '3 sets', 'for time')",
      "exercises": [
        {
          "name": "exercise name",
          "sets": number or null,
          "reps": number or null,
          "reps_range": "range like '8-10' or null",
          "weight_kg": number or null,
          "duration_sec": number or null,
          "rest_sec": number or null,
          "distance_m": number or null,
          "notes": "any additional cues or instructions"
        }
      ],
      "supersets": [
        {
          "exercises": [/* same format as above */],
          "rest_between_sec": number or null
        }
      ]
    }
  ]
}

Focus on accuracy:
- Prefer OCR-extracted numbers for reps/weights/times (they're more accurate)
- Use transcript/ASR text for exercise names and instructions
- If numbers conflict, prefer OCR values
- Group exercises into supersets if they're performed together
- Preserve time caps, intervals, and structure information

Return ONLY valid JSON, no additional text."""

    @staticmethod
    def structure_with_openai(
        text: str,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None
    ) -> Dict:
        """
        Structure workout text using OpenAI API.
        
        Args:
            text: Fused workout text
            model: OpenAI model to use
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            
        Returns:
            Structured workout JSON
        """
        if not OPENAI_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="OpenAI library not installed. Run: pip install openai"
            )
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not provided. Set OPENAI_API_KEY environment variable."
            )
        
        client = openai.OpenAI(api_key=api_key)
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": LLMService.WORKOUT_STRUCTURE_PROMPT},
                    {"role": "user", "content": text}
                ],
                temperature=0.1,  # Low temperature for structured output
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            return json.loads(result_text)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI API call failed: {e}"
            )
    
    @staticmethod
    def structure_with_anthropic(
        text: str,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None
    ) -> Dict:
        """
        Structure workout text using Anthropic Claude API.
        
        Args:
            text: Fused workout text
            model: Claude model to use
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
            
        Returns:
            Structured workout JSON
        """
        if not ANTHROPIC_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="Anthropic library not installed. Run: pip install anthropic"
            )
        
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable."
            )
        
        client = Anthropic(api_key=api_key)
        
        try:
            message = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": f"{LLMService.WORKOUT_STRUCTURE_PROMPT}\n\nWorkout text:\n{text}"}
                ],
                temperature=0.1,
            )
            
            result_text = message.content[0].text
            # Extract JSON from response (Claude may add markdown formatting)
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return json.loads(result_text)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Anthropic API call failed: {e}"
            )
    
    @staticmethod
    def structure_workout(
        text: str,
        provider: str = "openai",
        **kwargs
    ) -> Dict:
        """
        Structure workout text using specified LLM provider.
        
        Args:
            text: Fused workout text
            provider: LLM provider ("openai" or "anthropic")
            **kwargs: Additional arguments for provider
            
        Returns:
            Structured workout JSON
        """
        if provider.lower() == "openai":
            return LLMService.structure_with_openai(text, **kwargs)
        elif provider.lower() == "anthropic":
            return LLMService.structure_with_anthropic(text, **kwargs)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown LLM provider: {provider}. Use 'openai' or 'anthropic'."
            )

