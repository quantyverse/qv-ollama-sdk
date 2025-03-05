"""Example of using the OllamaChatClient with streaming responses and flexible parameters."""

import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qv_ollama_sdk.client import OllamaChatClient
from src.qv_ollama_sdk.domain.models import ModelParameters


def main():
    # Create a client with custom parameters
    client = OllamaChatClient(
        model_name="gemma2:2b",
        system_message="You are a helpful AI assistant. Answer questions concisely and accurately.",
        parameters=ModelParameters(
            temperature=0.7,
            max_tokens=300,  # Shorter responses
            top_p=0.95,      # Slightly more diverse token selection
            num_ctx=2048     # Reduced context window
        )
    )
    
    # Stream a response
    print("User: Explain quantum computing in simple terms.")
    print("Assistant: ", end="", flush=True)
    
    for chunk in client.stream_chat("Explain quantum computing in simple terms."):
        print(chunk, end="", flush=True)
    
    print("\n\nStreaming complete!")
    
    # Continue the conversation with streaming
    # Set max_tokens directly using the property
    client.max_tokens = 400
    
    print("\nUser: Can you give an example of a quantum algorithm?")
    print("Assistant: ", end="", flush=True)
    
    for chunk in client.stream_chat("Can you give an example of a quantum algorithm?"):
        print(chunk, end="", flush=True)
    
    print("\n\nStreaming complete!")
    
    # Try with some model-specific parameters (if supported by your model)
    print("\nUpdating with some advanced parameters...")
    client.set_parameters(
        temperature=0.9,
        repeat_penalty=1.3,
        # These are examples of model-specific parameters
        seed=42,              # For reproducibility 
        typical_p=0.7,        # Alternative diversity control
        mirostat_mode=1       # Advanced sampling technique
    )
    
    print("\nUser: What are the ethical implications of quantum computing?")
    print("Assistant: ", end="", flush=True)
    
    for chunk in client.stream_chat("What are the ethical implications of quantum computing?"):
        print(chunk, end="", flush=True)
    
    print("\n\nStreaming complete!")


if __name__ == "__main__":
    main() 