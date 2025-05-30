import ollama
import json

# Test with the official thinking mode model
print("Testing with deepseek-r1 (official thinking model)...")

try:
    response = ollama.chat(
        model='qwen3:8b',
        messages=[{'role': 'user', 'content': 'How many Rs are in strawberry?'}],
        think=True
    )

    print("Raw response:")
    print(json.dumps(response, indent=2, default=str))

    print("\nMessage content:")
    print(repr(response.get("message", {}).get("content", "")))

    print("\nMessage thinking:")
    print(repr(response.get("message", {}).get("thinking", "")))
    
    print("\nThinking present:", response.get("message", {}).get("thinking") is not None)
    print("Content has <think> tags:", "<think>" in response.get("message", {}).get("content", ""))

except Exception as e:
    print(f"Error with deepseek-r1: {e}")
    print("\nTrying with qwen3:8b...")
    
    response = ollama.chat(
        model='qwen3:8b',
        messages=[{'role': 'user', 'content': 'How many Rs are in strawberry?'}],
        think=True
    )
    
    print("Raw response with qwen3:")
    print(json.dumps(response, indent=2, default=str))
    
    print("\nThinking present:", response.get("message", {}).get("thinking") is not None)
    print("Content has <think> tags:", "<think>" in response.get("message", {}).get("content", "")) 