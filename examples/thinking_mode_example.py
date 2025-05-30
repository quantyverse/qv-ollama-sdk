"""Example demonstrating Ollama's Thinking Mode integration."""

from qv_ollama_sdk import OllamaChatClient, ModelParameters

def add_numbers(a: str, b: str) -> str:
    """Add two numbers."""
    return str(int(a) + int(b))

def count_letters(text: str, letter: str) -> str:
    """Count occurrences of a letter in text."""
    return str(text.lower().count(letter.lower()))

def main():
    print("🧠 Thinking Mode Example - QV Ollama SDK")
    print("=" * 50)
    
    # Test with models that support thinking
    thinking_models = ["qwen3:8b"]
    
    for model in thinking_models:
        print(f"\n🤖 Testing with {model}")
        print("-" * 40)
        
        try:
            # Initialize client with thinking enabled
            client = OllamaChatClient(
                model_name=model,
                parameters=ModelParameters(think=True),
                system_message="You are a helpful assistant."
            )
            
            # Test 1: Basic thinking mode
            print("\n1. Basic Thinking Mode")
            print("-" * 25)
            
            result = client.chat_with_thinking("How many Rs are in strawberry?")
            print(f"Question: How many Rs are in strawberry?")
            print(f"Thinking: {result['thinking'][:200]}{'...' if len(result['thinking']) > 200 else ''}")
            print(f"Answer: {result['content']}")
            
            # Test 2: Thinking with tools
            print("\n2. Thinking Mode with Tools")
            print("-" * 25)
            
            tools = [add_numbers, count_letters]
            result = client.chat_with_thinking("What is 15 + 27? Also count how many 'r' letters are in 'strawberry'", tools)
            
            print(f"Question: What is 15 + 27? Also count 'r' letters in 'strawberry'")
            if result['has_thinking']:
                print(f"Thinking: {result['thinking'][:300]}{'...' if len(result['thinking']) > 300 else ''}")
            print(f"Answer: {result['content']}")
            
            # Test 3: Controlled thinking on/off
            print("\n3. Thinking Control")
            print("-" * 25)
            
            # Without thinking
            client.disable_thinking()
            response_no_think = client.chat("Solve 8 * 7")
            print(f"Without thinking: {response_no_think}")
            
            # With thinking
            client.enable_thinking()
            result_with_think = client.chat_with_thinking("Solve 8 * 7")
            print(f"With thinking - Answer: {result_with_think['content']}")
            if result_with_think['has_thinking']:
                print(f"With thinking - Process: {result_with_think['thinking'][:150]}...")
            
            # Test 4: Streaming with thinking
            print("\n4. Streaming with Thinking")
            print("-" * 25)
            
            print("Question: Why is the sky blue?")
            print("Streaming response:")
            
            full_thinking = ""
            full_content = ""
            
            for chunk in client.stream_chat_with_tools("Why is the sky blue?"):
                if chunk.thinking:
                    full_thinking += chunk.thinking
                if chunk.content:
                    full_content += chunk.content
                    print(chunk.content, end="", flush=True)
            
            print(f"\nThinking captured: {len(full_thinking)} characters")
            
            # Test 5: Thinking mode comparison
            print("\n5. Thinking Mode Comparison")
            print("-" * 25)
            
            question = "What's the capital of France and why?"
            
            # Quick response (no thinking)
            client.disable_thinking()
            quick_response = client.chat(question)
            
            # Thoughtful response (with thinking)
            client.enable_thinking()
            thoughtful_result = client.chat_with_thinking(question)
            
            print(f"Question: {question}")
            print(f"\nQuick mode: {quick_response}")
            print(f"\nThoughtful mode: {thoughtful_result['content']}")
            if thoughtful_result['has_thinking']:
                print(f"Thinking process: {thoughtful_result['thinking'][:200]}...")
            
        except Exception as e:
            print(f"❌ Error with {model}: {e}")
            print("Make sure the model is installed: ollama pull " + model)
            continue
    
    # Demo different thinking approaches
    print("\n" + "=" * 50)
    print("💡 Thinking Mode Use Cases")
    print("=" * 50)
    
    print("""
🧠 WHEN TO USE THINKING MODE:

✅ Complex reasoning tasks
  - Math word problems  
  - Logic puzzles
  - Analysis questions

✅ Debugging and explanation
  - "Explain step by step..."
  - "Show your work..."
  - "How did you arrive at this?"

✅ Quality control
  - Compare thinking vs. content
  - Validate reasoning process
  - Understand model behavior

❌ WHEN NOT TO USE:

❌ Simple factual questions
❌ Real-time chat (adds latency)  
❌ When you only need the answer

🎯 INTEGRATION PATTERNS:

# Auto-enable for complex queries
if is_complex_question(message):
    client.enable_thinking()
else:
    client.disable_thinking()

# Separate thinking UI
result = client.chat_with_thinking(message)
show_answer(result['content'])
if user_wants_explanation:
    show_thinking(result['thinking'])

# Quality assurance
result = client.chat_with_thinking(critical_question)
if validate_thinking(result['thinking']):
    return result['content']
else:
    retry_with_different_approach()
""")

if __name__ == "__main__":
    main() 