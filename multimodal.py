#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip --version


# In[2]:


pip show autogen


# In[2]:


import autogen

config_list = [
    {
        "model": "qwen2.5:7b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    }
]

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": config_list,
    }
)


# In[3]:


from pydantic import BaseModel
from typing import Literal

class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]

reply = assistant.generate_reply(
    messages=[
        {
            "role": "user",
            "content": """
            Return JSON only.
            Analyze sentiment of:
            I love AutoGen
            """
        }
    ]
)

print(reply)


# In[4]:


assistant.generate_reply(
    messages=[
        {
            "role": "user",
            "content": "Describe this image"
        }
    ]
)


# In[6]:


pip show autogen


# In[7]:


import autogen
print(autogen.__version__)


# In[8]:


import autogen

config_list = [
    {
        "model": "qwen2.5:7b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    }
]

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list}
)

response = assistant.generate_reply(
    messages=[
        {
            "role": "user",
            "content": "What is Python?"
        }
    ]
)

print(response)


# In[9]:


import autogen

config_list = [
    {
        "model": "qwen2.5:7b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    }
]

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": config_list,
        "temperature": 0
    }
)

user_proxy = autogen.UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1
)

user_proxy.initiate_chat(
    assistant,
    message="Explain what a Large Language Model is in 3 lines."
)


# In[10]:


import autogen

config_list = [
    {
        "model": "qwen2.5:7b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    }
]

describer = autogen.AssistantAgent(
    name="description_agent",
    system_message="You are good at describing images",
    llm_config={"config_list": config_list}
)


# In[11]:


prompt = """
Return JSON only.

{
    "scene":"",
    "message":"",
    "style":"",
    "orientation":""
}
"""


# In[12]:


import json

response = describer.generate_reply(
    messages=[{"role":"user","content":prompt}]
)

data = json.loads(response)
print(data["scene"])


# In[16]:


from dotenv import load_dotenv
load_dotenv(override=True)


# In[18]:


import base64
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

image_path = "kyuli.jpg"   

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
                    Identify:
                    1. Main objects
                    2. Scene description
                    3. Classification category
                    4. Any text present
                    Return JSON only.
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

print(response.choices[0].message.content)


# In[19]:


{
  "objects": [],
  "scene": "",
  "category": "",
  "confidence": 0.0,
  "text_present": [],
  "colors": [],
  "sentiment": "",
  "image_type": ""
}


# In[20]:


{
  "ocr_text": "...",
  "language": "English"
}


# In[21]:


pip install pydantic


# In[22]:


pip install --upgrade pip


# In[42]:


from pydantic import BaseModel
import base64
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

class ImageDescription(BaseModel):
    objects: list[str]
    scene: str
    category: str
    text_present: list[str]

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

image_path = "kyuli.jpg"

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
Return ONLY JSON.

{
    "objects": [],
    "scene": "",
    "category": "",
    "text_present": []
}
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

print(response.choices[0].message.content)


# In[51]:


from openai import OpenAI
import base64
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Groq Client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Convert image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Image path
image_path = "kyuli.jpg"

# Encode image
base64_image = encode_image(image_path)

# Vision Request
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

# Print result
print(response.choices[0].message.content)


# In[52]:


from pydantic import BaseModel
from typing import List

class ImageDescription(BaseModel):
    objects: List[str]
    number_of_people: int
    scene: str
    activity: str
    emotion: str
    dominant_colors: List[str]
    image_type: str
    contains_text: bool
    ocr_text: str
    language: str


# In[60]:


import json

def save_analysis(result):
    with open("image_analysis.json", "w") as f:
        json.dump(result, f, indent=4)

    return "Analysis saved successfully."


# In[62]:


response = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "user",
            "content": "Analyze the image and save the results."
        }
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "save_analysis",
                "description": "Save image analysis to a JSON file",
                "parameters": {
                    "type": "object",
                    "properties": {
                       "result": {
    "type": "string"
}
                    },
                    "required": ["result"]
                }
            }
        }
    ],
    tool_choice="auto"
)


# In[63]:


def save_analysis(result):
    print(result)
    return "Saved"


# In[64]:


print(response.choices[0].message)


# In[65]:


print(response.choices[0].message.tool_calls)


# In[66]:


import json

tool_calls = response.choices[0].message.tool_calls

for tool_call in tool_calls:

    args = json.loads(tool_call.function.arguments)

    print("Function:", tool_call.function.name)
    print("Arguments:", args)

    result = save_analysis(args["result"])

    print(result)


# In[67]:


def vision_agent(image_result):
    return image_result


# In[68]:


{
  "objects": ["glasses", "hoodie"],
  "scene": "indoor",
  "emotion": "happy"
}


# In[69]:


def classification_agent(vision_result):

    people = vision_result["number_of_people"]

    if people > 1:
        category = "Group Photo"
    else:
        category = "Single Person"

    vision_result["category"] = category

    return vision_result


# In[70]:


def report_agent(data):

    report = f"""
IMAGE REPORT

Scene: {data['scene']}
People: {data['number_of_people']}
Activity: {data['activity']}
Emotion: {data['emotion']}
Category: {data['category']}
Objects: {', '.join(data['objects'])}
"""

    return report


# In[71]:


vision_result = {
    "objects": ["glasses","earbuds","hoodie"],
    "number_of_people": 4,
    "scene": "indoor",
    "activity": "posing for a photo",
    "emotion": "happy"
}


# In[72]:


step1 = vision_agent(vision_result)

step2 = classification_agent(step1)

final_report = report_agent(step2)

print(final_report)


# In[ ]:




