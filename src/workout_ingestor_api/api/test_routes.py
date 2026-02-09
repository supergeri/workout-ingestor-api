"""Test-only endpoints for automation framework."""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from workout_ingestor_api.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/test", tags=["test"])

# In-memory test data (reset on each /test/reset call)
test_data = {
    "users": [],
    "workouts": [],
    "sessions": []
}


@router.get("/health")
def test_health():
    """Health check for test endpoints."""
    return {"test_endpoints": "ok", "test_data_count": len(test_data["users"])}


@router.post("/reset")
def test_reset():
    """Reset all test data and state."""
    global test_data
    test_data = {
        "users": [],
        "workouts": [],
        "sessions": []
    }
    logger.info("Test data reset completed")
    return {"status": "reset_complete", "message": "All test data cleared"}


@router.post("/seedUser")
def seed_test_user(user_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Seed a test user with optional custom data."""
    if user_data is None:
        user_data = {}
    
    default_user = {
        "id": "test-user-001",
        "email": "test@amakaflow.com",
        "name": "Test User",
        "username": "testuser",
        "created_at": "2024-01-01T00:00:00Z",
        "test_data": True
    }
    
    # Merge with provided data
    seeded_user = {**default_user, **user_data}
    
    # Store in test data
    test_data["users"].append(seeded_user)
    
    logger.info(f"Test user seeded: {seeded_user['id']}")
    return {"status": "user_seeded", "user": seeded_user}


@router.post("/seedWorkout")
def seed_test_workout(workout_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Seed a test workout with optional custom data."""
    if workout_data is None:
        workout_data = {}
    
    default_workout = {
        "id": "test-workout-001",
        "user_id": "test-user-001",
        "title": "Test Workout",
        "description": "Automated test workout",
        "type": "strength",
        "duration_minutes": 45,
        "exercises": [
            {
                "name": "Push-ups",
                "sets": 3,
                "reps": 15,
                "rest_seconds": 60
            },
            {
                "name": "Squats", 
                "sets": 3,
                "reps": 20,
                "rest_seconds": 60
            }
        ],
        "created_at": "2024-01-01T10:00:00Z",
        "test_data": True
    }
    
    # Merge with provided data
    seeded_workout = {**default_workout, **workout_data}
    
    # Store in test data
    test_data["workouts"].append(seeded_workout)
    
    logger.info(f"Test workout seeded: {seeded_workout['id']}")
    return {"status": "workout_seeded", "workout": seeded_workout}


@router.get("/users")
def list_test_users() -> Dict[str, Any]:
    """List all test users."""
    return {"users": test_data["users"], "count": len(test_data["users"])}


@router.get("/workouts")
def list_test_workouts() -> Dict[str, Any]:
    """List all test workouts."""
    return {"workouts": test_data["workouts"], "count": len(test_data["workouts"])}


@router.delete("/users/{user_id}")
def delete_test_user(user_id: str) -> Dict[str, Any]:
    """Delete a specific test user."""
    global test_data
    original_count = len(test_data["users"])
    test_data["users"] = [user for user in test_data["users"] if user["id"] != user_id]
    
    if len(test_data["users"]) < original_count:
        logger.info(f"Test user deleted: {user_id}")
        return {"status": "user_deleted", "user_id": user_id}
    else:
        raise HTTPException(status_code=404, detail=f"Test user {user_id} not found")


@router.delete("/workouts/{workout_id}")
def delete_test_workout(workout_id: str) -> Dict[str, Any]:
    """Delete a specific test workout."""
    global test_data
    original_count = len(test_data["workouts"])
    test_data["workouts"] = [workout for workout in test_data["workouts"] if workout["id"] != workout_id]
    
    if len(test_data["workouts"]) < original_count:
        logger.info(f"Test workout deleted: {workout_id}")
        return {"status": "workout_deleted", "workout_id": workout_id}
    else:
        raise HTTPException(status_code=404, detail=f"Test workout {workout_id} not found")