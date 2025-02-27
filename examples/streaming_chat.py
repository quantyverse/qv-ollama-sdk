"""Example of using the QV Ollama SDK with streaming responses."""

import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qv_ollama_sdk.domain.models import (
    Conversation,
    ModelParameters
)
from src.qv_ollama_sdk.services.ollama_conversation_service import OllamaConversationService


def streaming_example():
    """Example of using the streaming API."""
    # Create a new conversation
    conversation = Conversation(model_name="gemma2:2b")
    
    # Add a system message to set the context
    conversation.add_system_message(
        "You are a helpful AI assistant. Answer questions concisely and accurately."
    )
    
    # Add a user message
    conversation.add_user_message("Explain quantum computing in simple terms.")
    
    # Create the Ollama conversation service
    service = OllamaConversationService()
    
    # Set custom parameters
    parameters = ModelParameters(
        temperature=0.7,
        max_tokens=500
    )
    
    # Generate a streaming response
    print("Generating streaming response...\n")
    
    # Collect the full response to add to conversation history
    full_response = ""
    
    # Stream the response
    for chunk in service.stream_response(conversation, parameters):
        full_response += chunk
        print(chunk, end="", flush=True)
    
    print("\n\nStreaming complete!")
    
    # Add the response to the conversation
    conversation.add_assistant_message(full_response)


if __name__ == "__main__":
    streaming_example()
    