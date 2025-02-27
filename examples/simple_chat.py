"""Simple example of using the QV Ollama SDK."""

import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qv_ollama_sdk.domain.models import (
    Conversation,
    ModelParameters
)
from src.qv_ollama_sdk.services.ollama_conversation_service import OllamaConversationService


def main():
    # Create a new conversation
    conversation = Conversation(model_name="gemma2:2b")
    
    # Add a system message to set the context
    conversation.add_system_message(
        "You are a helpful AI assistant. Answer questions concisely and accurately."
    )
    
    # Add a user message
    conversation.add_user_message("What is the capital of France?")
    
    # Create the Ollama conversation service
    service = OllamaConversationService()
    
    # Set custom parameters
    parameters = ModelParameters(
        temperature=0.7,
        max_tokens=500
    )
    
    # Generate a response
    print("Generating response...")
    response = service.generate_response(conversation, parameters)
    
    # Print the response
    print(f"\nResponse: {response.content}")
    
    # Add the response to the conversation
    conversation.add_assistant_message(response.content)
    
    # Continue the conversation
    conversation.add_user_message("And what is the population of Paris?")
    
    # Generate another response
    print("\nGenerating response...")
    response = service.generate_response(conversation, parameters)
    
    # Print the response
    print(f"\nResponse: {response.content}")


if __name__ == "__main__":
    main() 