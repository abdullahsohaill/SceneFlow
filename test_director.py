import os
import sys

# Add the scneneflow folder to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sceneflow'))

from sceneflow.config import settings
from sceneflow.schemas import DraftRequest
from sceneflow.llm_engine import generate_director_plan

def test_director_agent():
    print("Testing the Director Planner Agent (Gemini 3 Flash)...")
    if not settings.GEMINI_API_KEY:
        print("GEMINI_API_KEY missing - skipping active API hit.")
        return
        
    req = DraftRequest(
        topic="How Neural Networks Work",
        audience="high school students",
        num_scenes=3
    )
    
    try:
        draft = generate_director_plan(req)
        print(f"\nSUCCESS! Multi-Agent Planner generated {len(draft.scenes)} scenes.")
        for i, scene in enumerate(draft.scenes, 1):
            print(f"\n--- Scene {i} [{scene.estimated_duration}s] ---")
            print(f"Narration: {scene.narration_text[:100]}...")
            print(f"Visuals:   {scene.visual_description[:100]}...")
    except Exception as e:
        print(f"\nFAILED: {e}")

if __name__ == '__main__':
    test_director_agent()
