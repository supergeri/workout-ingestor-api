"""
Tests for POST /parse/text endpoint (AMA-555)
"""

import pytest
from fastapi.testclient import TestClient
from workout_ingestor_api.main import app


client = TestClient(app)


class TestParseTextEndpoint:
    """Test the POST /parse/text endpoint"""
    
    def test_parse_standard_instagram_notation(self):
        """Test standard Instagram fitness notation parsing"""
        text = """Workout:
Pull-ups 4x8 + Z Press 4x8
SA cable row 4x12 + SA DB press 4x8
Seated sled pull 5 x 10m"""
        
        response = client.post("/parse/text", json={
            "text": text,
            "source": "instagram_caption"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["exercises"]) == 5
        
        # Check first superset
        assert data["exercises"][0]["raw_name"] == "Pull-ups"
        assert data["exercises"][0]["sets"] == 4
        assert data["exercises"][0]["reps"] == "8"
        assert data["exercises"][0]["superset_group"] == "A"
        
        assert data["exercises"][1]["raw_name"] == "Z Press"
        assert data["exercises"][1]["sets"] == 4
        assert data["exercises"][1]["reps"] == "8"
        assert data["exercises"][1]["superset_group"] == "A"
        
        # Check distance exercise
        assert data["exercises"][4]["raw_name"] == "Seated sled pull"
        assert data["exercises"][4]["sets"] == 5
        assert data["exercises"][4]["distance"] == "10m"
    
    def test_parse_superset_notation(self):
        """Test superset notation with + delimiter"""
        text = "Bench Press 4x8 + Rows 3x10"
        
        response = client.post("/parse/text", json={"text": text})
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["exercises"]) == 2
        
        # Both should be in superset group A
        assert data["exercises"][0]["superset_group"] == "A"
        assert data["exercises"][1]["superset_group"] == "A"
    
    def test_mixed_formats_numbered_and_fitness(self):
        """Test mixed numbered and fitness notation"""
        text = """1. Squats 4x8
2. Bench Press 3x10
3. Deadlifts 5x5"""
        
        response = client.post("/parse/text", json={"text": text})
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["exercises"]) == 3
        
        assert data["exercises"][0]["raw_name"] == "Squats"
        assert data["exercises"][0]["sets"] == 4
        assert data["exercises"][0]["reps"] == "8"
    
    def test_skip_hashtags(self):
        """Test that hashtags are filtered out"""
        text = """Squats 4x8
#fitness #legday #workout
Bench Press 3x10"""
        
        response = client.post("/parse/text", json={"text": text})
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["exercises"]) == 2
        assert data["exercises"][0]["raw_name"] == "Squats"
        assert data["exercises"][1]["raw_name"] == "Bench Press"
    
    def test_skip_ctas(self):
        """Test that CTAs like 'Follow me!' are filtered out"""
        text = """Squats 4x8
Follow me for more workouts!
Bench Press 3x10"""
        
        response = client.post("/parse/text", json={"text": text})
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["exercises"]) == 2
        exercise_names = [e["raw_name"] for e in data["exercises"]]
        assert "Follow me for more workouts!" not in exercise_names
    
    def test_skip_section_headers(self):
        """Test that section headers like 'Upper Body:' are filtered out"""
        text = """Upper Body:
Bench Press 4x8
Lower Body:
Squats 3x10"""
        
        response = client.post("/parse/text", json={"text": text})
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["exercises"]) == 2
        exercise_names = [e["raw_name"] for e in data["exercises"]]
        assert "Upper Body:" not in exercise_names
        assert "Lower Body:" not in exercise_names
    
    def test_no_split_compound_names_without_sets_reps(self):
        """Test that compound exercise names without sets/reps are parsed as single exercise"""
        # Even without set/rep notation, exercise names are accepted
        text = "Chin-up + Negative Hold"
        
        response = client.post("/parse/text", json={"text": text})
        
        assert response.status_code == 200
        data = response.json()
        
        # Parsed as 1 exercise (not split on +) with no sets/reps data
        assert len(data["exercises"]) == 1
        assert data["exercises"][0]["raw_name"] == "Chin-up + Negative Hold"
        assert data["exercises"][0]["sets"] is None
        assert data["exercises"][0]["reps"] is None
    
    def test_rep_ranges(self):
        """Test rep ranges like '4x8-12'"""
        text = "Squats 4x8-12"
        
        response = client.post("/parse/text", json={"text": text})
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["exercises"]) == 1
        assert data["exercises"][0]["reps"] == "8-12"
    
    def test_time_based_exercises(self):
        """Test time-based exercises like '3x30s'"""
        text = "Plank 3x30s"
        
        response = client.post("/parse/text", json={"text": text})
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["exercises"]) == 1
        assert data["exercises"][0]["sets"] == 3
        assert data["exercises"][0]["reps"] == "30s"
    
    def test_rpe_notation(self):
        """Test RPE notation like '@RPE8'"""
        text = "Squats 4x8 @RPE8"
        
        response = client.post("/parse/text", json={"text": text})
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["exercises"]) == 1
        assert data["exercises"][0]["rpe"] == 8.0
    
    def test_empty_input(self):
        """Test empty input handling"""
        response = client.post("/parse/text", json={"text": ""})
        
        assert response.status_code == 400
        assert "Text is required" in response.json()["detail"]
    
    def test_whitespace_only_input(self):
        """Test whitespace-only input handling"""
        response = client.post("/parse/text", json={"text": "   \n\n   "})
        
        # Should fallback to LLM or return empty success
        assert response.status_code == 200
        data = response.json()
        # LLM fallback might still try to parse
        assert "exercises" in data
    
    def test_confidence_score(self):
        """Test that confidence score is calculated"""
        text = """Squats 4x8
Bench Press 3x10
Random text without structure"""
        
        response = client.post("/parse/text", json={"text": text})
        
        assert response.status_code == 200
        data = response.json()
        
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 100
    
    def test_detected_format(self):
        """Test that detected_format is returned"""
        text = "Squats 4x8"
        
        response = client.post("/parse/text", json={"text": text})
        
        assert response.status_code == 200
        data = response.json()
        
        assert "detected_format" in data
        assert data["detected_format"] in ["instagram_caption", "text_unstructured", "text_llm"]
    
    def test_exercise_order(self):
        """Test that exercises have correct order"""
        text = """Squats 4x8
Bench Press 3x10
Deadlifts 5x5"""
        
        response = client.post("/parse/text", json={"text": text})
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["exercises"]) == 3
        assert data["exercises"][0]["order"] == 0
        assert data["exercises"][1]["order"] == 1
        assert data["exercises"][2]["order"] == 2
    
    def test_bullet_format(self):
        """Test bullet point format parsing"""
        text = """• Squats 4x8
- Bench Press 3x10
→ Deadlifts 5x5"""
        
        response = client.post("/parse/text", json={"text": text})
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["exercises"]) == 3
        assert data["exercises"][0]["raw_name"] == "Squats"
        assert data["exercises"][1]["raw_name"] == "Bench Press"
        assert data["exercises"][2]["raw_name"] == "Deadlifts"
    
    def test_structured_parsing_extracts_embedded_notation(self):
        """Test that structured parser extracts exercises from conversational text"""
        # Text with structured notation (4x8, 4x5) embedded in conversational text
        text = "Today I did some squats 4x8 and bench press 4x5. It was a great workout..."
        
        response = client.post("/parse/text", json={"text": text})
        
        assert response.status_code == 200
        data = response.json()
        
        # Structured parser successfully extracts notation from conversational text
        assert "exercises" in data
        assert data["detected_format"] == "instagram_caption"  # Structured notation found
        assert data["success"] is True
        assert "confidence" in data
        assert len(data["exercises"]) > 0
