"""Example of using the simplified OllamaChatClient."""

import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qv_ollama_sdk.client import OllamaChatClient


def main():
    # Create a client with a system message
    client = OllamaChatClient(
        model_name="gemma2:2b",
        system_message="You are a helpful AI assistant. Answer questions concisely and accurately."
    )
    
    # Simple chat
    print("User: What is the capital of France?")
    response = client.chat("What is the capital of France?")
    print(f"Assistant: {response}")
    
    # Continue the conversation
    print("\nUser: And what is its population?")
    response = client.chat("And what is its population?")
    print(f"Assistant: {response}")
    
    # Update parameters
    print("\nChanging temperature to 1.0 for more creative responses...")
    client.set_parameters(temperature=1.0)
    
    # Ask another question
    print("\nUser: Give me a fun fact about Paris")
    response = client.chat("Give me a fun fact about Paris")
    print(f"Assistant: {response}")
    
    # Print the conversation history
    print("\nConversation History:")
    for i, message in enumerate(client.get_history()):
        role = message["role"].capitalize()
        content = message["content"]
        print(f"{i+1}. {role}: {content}")


if __name__ == "__main__":
    main() 