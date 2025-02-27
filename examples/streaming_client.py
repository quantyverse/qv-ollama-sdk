"""Example of using the OllamaChatClient with streaming responses."""

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
    
    # Set parameters for a shorter response
    client.set_parameters(max_tokens=500)
    
    # Stream a response
    print("User: Explain quantum computing in simple terms.")
    print("Assistant: ", end="", flush=True)
    
    for chunk in client.stream_chat("Explain quantum computing in simple terms."):
        print(chunk, end="", flush=True)
    
    print("\n\nStreaming complete!")
    
    # Continue the conversation with streaming
    print("\nUser: Can you give an example of a quantum algorithm?")
    print("Assistant: ", end="", flush=True)
    
    for chunk in client.stream_chat("Can you give an example of a quantum algorithm?"):
        print(chunk, end="", flush=True)
    
    print("\n\nStreaming complete!")


if __name__ == "__main__":
    main() 