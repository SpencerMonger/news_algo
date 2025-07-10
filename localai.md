# LocalAI API Migration Guide

## Overview
This guide shows how to migrate from LM Studio to your LocalAI setup running the Claude 3.7 Sonnet Reasoning Gemma3-12B model.

## Server Information
- **Base URL**: `http://localhost:8080`
- **API**: OpenAI-compatible REST API
- **Model Name**: `claude-3.7-sonnet-reasoning-gemma3-12B.Q8_0.gguf`

## Quick Migration Changes

### From LM Studio to LocalAI
```diff
- BASE_URL = "http://localhost:1234"  # LM Studio
+ BASE_URL = "http://localhost:8080"  # LocalAI

- MODEL_NAME = "google/gemma-3-12b"  # LM Studio model ID
+ MODEL_NAME = "claude-3.7-sonnet-reasoning-gemma3-12B.Q8_0.gguf"  # LocalAI model
```

## API Endpoints

### 1. List Available Models
```bash
curl http://localhost:8080/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {"id": "claude-3.7-sonnet-reasoning-gemma3-12B.Q8_0.gguf", "object": "model"},
    {"id": "gpt-4", "object": "model"},
    {"id": "gpt-4o", "object": "model"},
    // ... other models
  ]
}
```

### 2. Chat Completions (Recommended)
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3.7-sonnet-reasoning-gemma3-12B.Q8_0.gguf",
    "messages": [
      {"role": "user", "content": "Hello! How are you?"}
    ],
    "max_tokens": 150,
    "temperature": 0.7,
    "stream": false
  }'
```

### 3. Text Completions (Legacy)
```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3.7-sonnet-reasoning-gemma3-12B.Q8_0.gguf",
    "prompt": "Hello! Please introduce yourself.",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### 4. Streaming Responses
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3.7-sonnet-reasoning-gemma3-12B.Q8_0.gguf",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "max_tokens": 200,
    "stream": true
  }'
```

## Code Examples

### Python with OpenAI Library
```python
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # LocalAI doesn't require API key
)

# Chat completion
response = client.chat.completions.create(
    model="claude-3.7-sonnet-reasoning-gemma3-12B.Q8_0.gguf",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=150,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### Python with Requests
```python
import requests
import json

def chat_with_localai(message, temperature=0.7, max_tokens=150):
    url = "http://localhost:8080/v1/chat/completions"
    
    payload = {
        "model": "claude-3.7-sonnet-reasoning-gemma3-12B.Q8_0.gguf",
        "messages": [{"role": "user", "content": message}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code}"

# Usage
result = chat_with_localai("Hello! How are you?")
print(result)
```

### JavaScript/Node.js
```javascript
const axios = require('axios');

async function chatWithLocalAI(message, temperature = 0.7, maxTokens = 150) {
    const url = 'http://localhost:8080/v1/chat/completions';
    
    const payload = {
        model: 'claude-3.7-sonnet-reasoning-gemma3-12B.Q8_0.gguf',
        messages: [{ role: 'user', content: message }],
        max_tokens: maxTokens,
        temperature: temperature
    };
    
    try {
        const response = await axios.post(url, payload);
        return response.data.choices[0].message.content;
    } catch (error) {
        return `Error: ${error.response?.status || error.message}`;
    }
}

// Usage
chatWithLocalAI("What is machine learning?").then(console.log);
```

### cURL Examples for Testing
```bash
# Simple chat
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3.7-sonnet-reasoning-gemma3-12B.Q8_0.gguf",
    "messages": [{"role": "user", "content": "Explain quantum computing in simple terms"}],
    "max_tokens": 200,
    "temperature": 0.7
  }'

# Multi-turn conversation
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3.7-sonnet-reasoning-gemma3-12B.Q8_0.gguf",
    "messages": [
      {"role": "user", "content": "What is Python?"},
      {"role": "assistant", "content": "Python is a programming language..."},
      {"role": "user", "content": "Show me a simple example"}
    ],
    "max_tokens": 150
  }'
```

## Model-Specific Notes

### Claude 3.7 Sonnet Reasoning Model
This model has some special characteristics:
- **Reasoning tokens**: The model shows `<think>...</think>` blocks in responses - these represent the model's reasoning process
- **High quality**: Based on Gemma3-12B but fine-tuned for reasoning tasks
- **Context size**: 4096 tokens (configured in your setup)

### Response Format
```json
{
  "created": 1752174653,
  "object": "chat.completion",
  "id": "unique-id",
  "model": "claude-3.7-sonnet-reasoning-gemma3-12B.Q8_0.gguf",
  "choices": [{
    "index": 0,
    "finish_reason": "stop",
    "message": {
      "role": "assistant",
      "content": "<think>reasoning process</think>\nActual response"
    }
  }],
  "usage": {
    "prompt_tokens": 21,
    "completion_tokens": 100,
    "total_tokens": 121
  }
}
```

## Migration Checklist

- [ ] Update base URL from `localhost:1234` to `localhost:8080`
- [ ] Change model name to `claude-3.7-sonnet-reasoning-gemma3-12B.Q8_0.gguf`
- [ ] Remove API key requirements (LocalAI doesn't need them)
- [ ] Test with a simple request
- [ ] Update error handling for LocalAI-specific responses
- [ ] Consider the reasoning tokens in response parsing

## Docker Management Commands

```bash
# Check if LocalAI is running
docker ps

# Start LocalAI
docker start localai

# Stop LocalAI
docker stop localai

# Restart LocalAI
docker restart localai

# View logs
docker logs localai

# Follow logs in real-time
docker logs -f localai
```

## Performance Notes

- **Response time**: ~3-5 seconds per request
- **Concurrent requests**: Currently sequential processing
- **Memory usage**: ~12GB model loaded in VRAM
- **Context window**: 4096 tokens

## Troubleshooting

### Common Issues:
1. **Connection refused**: Make sure Docker container is running
2. **Model not found**: Use exact model name from `/v1/models` endpoint
3. **Slow responses**: Normal for first request (model loading)
4. **GPU memory**: Monitor with `rocm-smi`

### Health Check:
```bash
# Test if server is responding
curl http://localhost:8080/v1/models

# Test a simple completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3.7-sonnet-reasoning-gemma3-12B.Q8_0.gguf", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10}'
```