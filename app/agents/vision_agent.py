from openai import OpenAI
import base64
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


def analyze_image(image_path):

    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
Analyze this image and return ONLY valid JSON.

{
  "objects": [],
  "number_of_people": 0,
  "scene": "",
  "activity": "",
  "emotion": "",
  "dominant_colors": [],
  "image_type": "",
  "contains_text": false,
  "ocr_text": "",
  "language": ""
}

Rules:
1. Return JSON only.
2. Do not use markdown.
3. Do not add explanations.
4. Fill every field.
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0
    )

    content = response.choices[0].message.content
    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()

    return json.loads(content)

