# QV Ollama SDK

A simple, Domain-Driven Design SDK for interacting with the Ollama API.

## Features

- Clean, DDD-inspired architecture
- Simple conversation management
- Support for synchronous API calls
- Streaming response support
- Customizable model parameters
- User-friendly client interface

## Installation

```bash
pip install qv-ollama-sdk
```

## Quick Start

```python
from qv_ollama_sdk import OllamaChatClient

# Create a client with a system message
client = OllamaChatClient(
    model_name="gemma2:2b",
    system_message="You are a helpful assistant."
)

# Simple chat
response = client.chat("What is the capital of France?")
print(response)

# Continue the conversation
response = client.chat("And what is its population?")
print(response)

# Update parameters
client.set_parameters(temperature=1.0)

# Get conversation history
history = client.get_history()
```

## Streaming Responses

```python
from qv_ollama_sdk import OllamaChatClient

client = OllamaChatClient(model_name="gemma2:2b")

# Stream the response
for chunk in client.stream_chat("Explain quantum computing."):
    print(chunk, end="", flush=True)
```

## Advanced Usage

For more control, you can use the lower-level API:

```python
from qv_ollama_sdk import Conversation, OllamaConversationService, ModelParameters

# Create a conversation
conversation = Conversation(model_name="gemma2:2b")
conversation.add_system_message("You are a helpful assistant.")
conversation.add_user_message("What is the capital of France?")

# Generate a response
service = OllamaConversationService()
parameters = ModelParameters(temperature=0.7)
response = service.generate_response(conversation, parameters)

print(response.content)
```

## Requirements

- Python 3.12+
- Ollama 0.4.7+

## License

MIT