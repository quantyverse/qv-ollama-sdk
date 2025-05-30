"""Example demonstrating tool calling functionality with the QV Ollama SDK."""

import sys
import os
import json

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qv_ollama_sdk import OllamaChatClient, ModelParameters


def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers together.

    Args:
        a: The first number as an int
        b: The second number as an int

    Returns:
        int: The sum of the two numbers
    """
    result = a + b
    print(f"🧮 Tool called: add_two_numbers({a}, {b}) = {result}")
    return result


def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Get current weather information for a location.

    Args:
        location: The location to get weather for
        unit: Temperature unit (celsius or fahrenheit)

    Returns:
        str: Weather information
    """
    # Simulate weather data
    weather_data = {
        "stuttgart": {"temp": 22, "condition": "Sunny"},
        "london": {"temp": 18, "condition": "Cloudy"},
        "tokyo": {"temp": 28, "condition": "Rainy"},
        "new york": {"temp": 25, "condition": "Partly cloudy"}
    }
    
    location_lower = location.lower()
    if location_lower in weather_data:
        weather = weather_data[location_lower]
        temp = weather["temp"]
        if unit.lower() == "fahrenheit":
            temp = (temp * 9/5) + 32
        
        result = f"Weather in {location}: {weather['condition']}, {temp}°{unit[0].upper()}"
        print(f"🌤️ Tool called: get_weather('{location}', '{unit}') = {result}")
        return result
    else:
        result = f"Weather data not available for {location}"
        print(f"🌤️ Tool called: get_weather('{location}', '{unit}') = {result}")
        return result


def main():
    print("=" * 60)
    print("QV Ollama SDK - Tool Calling Example")
    print("=" * 60)
    
    # Initialize the client
    client = OllamaChatClient(
        model_name="qwen3:8b",  # Tool calling supported model
        system_message="You are a helpful assistant that can perform calculations and get weather information."
    )
    
    # Define available tools
    tools = [add_two_numbers, get_weather]
    
    print("\n1. REGULAR CHAT (without tools)")
    print("-" * 40)
    
    # Regular chat without tools (backward compatibility)
    response = client.chat("Hello! How are you?")
    print(f"Assistant: {response}")
    
    print("\n2. CHAT WITH TOOLS (math calculation)")
    print("-" * 40)
    
    # Chat with tools - mathematical calculation
    response = client.chat("What is 15 plus 27?", tools=tools)
    print(f"Assistant: {response}")
    
    print("\n3. CHAT WITH TOOLS (weather query)")
    print("-" * 40)
    
    # Chat with tools - weather query
    response = client.chat("What's the weather like in Stuttgart?", tools=tools)
    print(f"Assistant: {response}")
    
    print("\n4. FULL TOOL RESPONSE (with tool call details)")
    print("-" * 40)
    
    # Get full response including tool call details
    full_response = client.chat_with_tools("What's 42 times 3? Also tell me about Tokyo weather.", tools=tools)
    print(f"Assistant: {full_response.content}")
    
    if full_response.tool_calls:
        print("\nTool calls made:")
        for i, tool_call in enumerate(full_response.tool_calls, 1):
            print(f"  {i}. Function: {tool_call.function.name}")
            print(f"     Arguments: {json.dumps(tool_call.function.arguments, indent=6)}")
    
    print("\n5. STREAMING CHAT (with tools)")
    print("-" * 40)
    
    print("Assistant: ", end="", flush=True)
    for chunk in client.stream_chat("Calculate 100 minus 73, and tell me about London weather.", tools=tools):
        print(chunk, end="", flush=True)
    print()  # New line after streaming
    
    print("\n6. STREAMING WITH TOOL DETAILS")
    print("-" * 40)
    
    print("Streaming response with tool call details...")
    content_parts = []
    all_tool_calls = []
    
    for chunk in client.stream_chat_with_tools("What is 88 divided by 4?", tools=tools):
        if chunk.content:
            content_parts.append(chunk.content)
            print(chunk.content, end="", flush=True)
        
        if chunk.tool_calls:
            all_tool_calls.extend(chunk.tool_calls)
    
    print()  # New line
    
    if all_tool_calls:
        print("\nTool calls during streaming:")
        for i, tool_call in enumerate(all_tool_calls, 1):
            print(f"  {i}. Function: {tool_call.function.name}")
            print(f"     Arguments: {json.dumps(tool_call.function.arguments, indent=6)}")
    
    print("\n7. CONVERSATION HISTORY")
    print("-" * 40)
    
    # Show conversation history
    history = client.get_history()
    print(f"Total messages in conversation: {len(history)}")
    
    for i, msg in enumerate(history[-3:], 1):  # Show last 3 messages
        role = msg["role"].title()
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"  {i}. {role}: {content}")
    
    print("\n" + "=" * 60)
    print("Tool calling example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main() 