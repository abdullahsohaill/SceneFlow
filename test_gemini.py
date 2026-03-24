import os
from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyBJy5Qesur1GblSPnNpVkRNjk7_lKLf0Ec")

prompt = """
Return ONLY valid JSON matching this exact structure:
```json
{
  "scenes": [
    {
      "scene_id": "scene_1",
      "narration_text": "clearly written spoken text...",
      "visuals": [
        {
          "type": "title_text",
          "content": {"text": "example"}
        }
      ],
      "estimated_duration": 5.0
    }
  ],
  "style_config": {}
}
```

User Request:
Create an explainer video about: How DNS works

Style config:
  brand_colors: ['#4F46E5', '#10B981']
  font_family: Inter, sans-serif
  logo_url: none
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.7,
    ),
)
print(response.text)
