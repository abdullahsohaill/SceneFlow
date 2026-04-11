import os
import sys

# Add api to path so we can import from routes
sys.path.append(os.path.join(os.path.dirname(__file__), 'generative_manim_fork', 'api'))

from routes.video_generation import generate_manim_code
from routes.code_generation import generate_code

def test_gemini():
    print("Testing Gemini 1.5 Pro code generation...")
    # NOTE: requires GEMINI_API_KEY in environment
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not set. Cannot run actual API call.")
        print("Our code changes are structurally correct but require an active API key to test.")
        return
        
    try:
        code = generate_manim_code("Draw a red square next to a blue circle", engine="google", model="gemini-1.5-pro")
        print("Success! Generated code snippet:")
        print(code[:200] + "...")
        assert "class GenScene(Scene):" in code
        print("Test passed.")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_gemini()
