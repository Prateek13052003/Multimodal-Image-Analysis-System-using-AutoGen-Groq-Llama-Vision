## What This Project Does

This notebook explores the full stack of **multimodal AI engineering** — from running LLMs locally to sending images to cloud vision APIs and building a chained multi-agent pipeline that produces structured reports. It is a hands-on lab covering five distint areas:

| Area | Description |
|---|---|
| AutoGen Agents | Build and configure LLM-backed agents using Microsoft AutoGen |
| Vision Analysis | Analyze images using Llama 4 Scout via Groq's OpenAI-compatible API |
| Structured Outputs | Force the LLM to return strict JSON using Pydantic schemas |
| Function Calling | Register Python functions as tools the LLM can invoke autonomously |
| Multi-Agent Pipeline | Chain vision → classification → report agents in a sequential pipeline |

---

## Project Structure

```
vision-agent-pipeline/
│
├── multimodal.ipynb        # Main research notebook
├── kyu.jpg               # Sample image used for vision analysis
├── image_analysis.json     # Output: saved analysis results (auto-generated)
├── .env                    # API keys (GROQ_API_KEY)
└── README.md
```

---

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/yourname/vision-agent-pipeline.git
cd vision-agent-pipeline
pip install pyautogen openai pydantic python-dotenv
```

### 2. Set Up Environment Variables

Create a `.env` file in the root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Install & Run Ollama (for local LLM)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the Qwen 2.5 7B model
ollama pull qwen2.5:7b

# Ollama runs at http://localhost:11434 by default
```

---

## Core Concepts

### 1. AutoGen — Multi-Agent Conversations 

[AutoGen](https://github.com/microsoft/autogen) is a Microsoft framework for building **conversational AI agents** that can talk to each other, use tools, and produce structured outputs.

This notebook uses two key AutoGen primitives:

**`AssistantAgent`** — An LLM-powered agent that responds to messages:
```python
import autogen

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list}
)
```

**`UserProxyAgent`** — A proxy agent that represents the human and initiates conversations:
```python
user_proxy = autogen.UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1
)

user_proxy.initiate_chat(
    assistant,
    message="Explain what a Large Language Model is in 3 lines."
)
```

The `NEVER` mode means the agent never prompts for human input — it runs fully autonomously. `max_consecutive_auto_reply=1` caps the back-and-forth to prevent infinite loops.

---

### 2. Connecting AutoGen to Local LLMs via Ollama

By default AutoGen works with OpenAI. But it supports any **OpenAI-compatible endpoint**, including Ollama running locally:

```python
config_list = [
    {
        "model": "qwen2.5:7b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",   # Ollama doesn't need a real key
    }
]
```

This means you can run the entire pipeline **100% offline**, with no API costs.

---

### 3. Vision Analysis — Base64 Image Encoding

To send an image to a multimodal LLM, you first convert it to a **base64 string** and embed it directly into the API message:

```python
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image("kyuli.jpg")
```

The encoded image is then passed in the `content` array of the message as an `image_url` block using the `data:image/jpeg;base64,...` URI format. This is the standard approach for all OpenAI-compatible vision APIs.

---

### 4. Groq + Llama 4 Scout — Cloud Vision Model

The notebook uses [Groq](https://groq.com) as the inference provider and `meta-llama/llama-4-scout-17b-16e-instruct` as the vision model. Groq's API is fully OpenAI-compatible, so the standard `openai` Python client works without changes:

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

response = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image and return ONLY valid JSON."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ],
    temperature=0
)
```

Using `temperature=0` makes the output deterministic — essential when you need reliable JSON structure.

---

### 5. Structured Outputs with Pydantic

Rather than parsing free-form text, the notebook forces the LLM to return JSON that maps directly to a **Pydantic schema**. This creates a contract between the model's output and your Python code:

```python
from pydantic import BaseModel
from typing import list 

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
```

The schema is passed to the model as a JSON template in the prompt. The model fills it in and returns it. You then parse and validate it:

```python
import json
data = json.loads(response.choices[0].message.content)
desc = ImageDescription(**data)
```

**Why this matters:** Pydantic validates field types at parse time. If the model returns a string where an `int` is expected, it raises an error immediately — making your pipeline robust and debuggable.

---

### 6. Function Calling (Tool Use)

Function calling lets the LLM **decide to invoke a Python function** based on the conversation context. You register a function as a tool by describing it in a JSON schema:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "save_analysis",
            "description": "Save image analysis to a JSON file",
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {"type": "string"}
                },
                "required": ["result"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[...],
    tools=tools,
    tool_choice="auto"
)
```

When the model decides to call the function, you extract and execute the call:

```python
for tool_call in response.choices[0].message.tool_calls:
    args = json.loads(tool_call.function.arguments)
    result = save_analysis(args["result"])
```

`tool_choice="auto"` lets the model decide whether to call a tool or just reply with text.

---

### 7. Multi-Agent Pipeline

The notebook builds a **sequential three-agent pipeline** where the output of each agent feeds into the next:

```
[Image Input]
      │
      ▼
vision_agent()          → extracts objects, scene, emotion, people count
      │
      ▼
classification_agent()  → adds a "category" field (Group Photo / Single Person)
      │
      ▼
report_agent()          → formats everything into a human-readable IMAGE REPORT
      │
      ▼
[Printed Report]
```

Each agent is a simple Python function that receives and enriches a shared data dictionary:

```python
step1 = vision_agent(vision_result)
step2 = classification_agent(step1)
final_report = report_agent(step2)
print(final_report)
```

**Output:**
```
IMAGE REPORT

Scene: indoor
People: 4
Activity: posing for a photo
Emotion: happy
Category: Group Photo
Objects: glasses, earbuds, hoodie
```

This pattern — passing state through a chain of specialized agents — is the foundation of more advanced agentic architectures like LangGraph and AutoGen's GroupChat.

---

## How Everything Connects

```
┌─────────────────────────────────────────────────────────────┐
│                    vision-agent-pipeline                    │
│                                                             │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────────┐ │
│  │  Ollama  │    │  Groq Cloud   │    │   Python Agents  │ │
│  │ (Local)  │    │ (Llama 4      │    │ (vision →        │ │
│  │ qwen2.5  │    │  Scout Vision)│    │  classify →      │ │
│  └────┬─────┘    └──────┬────────┘    │  report)         │ │
│       │                 │             └──────────────────┘ │
│       ▼                 ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              AutoGen Agent Framework                │   │
│  │  AssistantAgent + UserProxyAgent + generate_reply   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Structured Outputs & Tool Calling           │   │
│  │     Pydantic schemas  +  Function/Tool dispatch     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `pyautogen` | 0.2.40 | Multi-agent framework |
| `openai` | latest | OpenAI-compatible client for Groq |
| `pydantic` | 2.x | Structured output validation |
| `python-dotenv` | latest | API key management |
| Ollama | latest | Local LLM server |
| Groq API | — | Cloud inference (Llama 4 Scout) |

---

## Key Takeaways

- **Local + Cloud hybrid**: Use Ollama for cost-free local inference and Groq for high-speed cloud vision when you need multimodal capability.
- **Structured outputs > free text**: Prompting for JSON and validating with Pydantic eliminates post-processing headaches.
- **Function calling is the bridge**: It lets the LLM drive real Python execution, not just generate text.
- **Agent pipelines are composable**: Each agent does one thing well. Chaining them is more maintainable than one giant prompt.

---

