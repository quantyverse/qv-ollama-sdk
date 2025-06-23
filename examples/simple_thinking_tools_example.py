"""Simple Example: Clean API with Thinking + Tool Calling"""

from qv_ollama_sdk import OllamaChatClient

# Simple tool functions
def add_numbers(a: str, b: str) -> str:
    """Add two numbers."""
    return str(int(a) + int(b))

def get_weather(city: str) -> str:
    """Get weather for a city (simulated)."""
    return f"The weather in {city} is sunny and 23°C"

def main():
    print("🚀 Clean API Example - Thinking + Tools")
    print("=" * 45)
    
    # Setup
    client = OllamaChatClient(model_name="qwen3:8b")
    tools = [add_numbers, get_weather]
    
    # ===== EXAMPLE 1: Basic Chat (No Tools) =====
    print("\n1️⃣ Basic Chat (No Tools)")
    print("-" * 25)
    client.disable_thinking()
    
    response = client.chat("What is the capital of France?")
    print(f"Question: What is the capital of France?")
    print("Thinking: ", response.thinking)
    print(f"Answer: {response.content}")
    print(f"Thinking available: {response.thinking is not None}")
    
    # ===== EXAMPLE 2: Chat with Tools =====
    print("\n\n2️⃣ Chat with Tools (Auto-Execute)")
    print("-" * 35)
    
    response = client.chat("What is 15 + 27?", tools=tools)
    print(f"Question: What is 15 + 27?")
    print(f"Answer: {response.content}")
    print(f"Tool calls made: {len(response.tool_calls) if response.tool_calls else 0} {response.tool_calls}")
    print(f"Tool results: {len(response.tool_results) if response.tool_results else 0} {response.tool_results}")
    print("💡 Tools automatically executed!")
    
    # ===== EXAMPLE 3: Enable Thinking Mode =====
    print("\n\n3️⃣ Enable Thinking Mode")
    print("-" * 28)
    
    client.enable_thinking()
    print("✅ Thinking mode enabled globally")
    
    response = client.chat("What is 25 + 18?", tools=tools)
    print(f"Question: What is 25 + 18?")
    print(f"🧠 Thinking: {response.thinking if response.thinking else 'None'}...")
    print(f"💬 Answer: {response.content}")
    print("💡 Now shows AI's reasoning process!")
    
    # ===== EXAMPLE 4: Streaming with Thinking =====
    print("\n\n4️⃣ Streaming with Thinking + Tools")
    print("-" * 35)
    
    print("Question: Add 12 + 8, then tell me weather in Stuttgart")
    print("Streaming response:")
    print()
    
    thinking_displayed = False
    content_started = False
    content_chunks = 0
    tool_calls_made = []
    
    for chunk in client.stream_chat("Add 12 + 8, then tell me weather in Stuttgart", tools=tools):
        # Handle thinking - clean one-time display
        if chunk.thinking and not thinking_displayed:
            print("🧠 Thinking: ", end="", flush=True)
            thinking_displayed = True
        
        if chunk.thinking:
            print(chunk.thinking, end="", flush=True)
        
        # Handle tool calls
        if chunk.tool_calls:
            tool_calls_made.extend(chunk.tool_calls)
            if thinking_displayed and not content_started:
                print()  # Newline after thinking
            print(f"🛠️ Tool called: {chunk.tool_calls[0].function.name}({chunk.tool_calls[0].function.arguments})")
        
        # Handle content - clean display
        if chunk.content:
            if not content_started:
                if thinking_displayed:
                    print()  # Newline after thinking
                print("💬 Answer: ", end="", flush=True)
                content_started = True
            
            content_chunks += 1
            print(chunk.content, end="", flush=True)
    
    print()  # Final newline
    print(f"\n📊 Summary:")
    print(f"   Content chunks: {content_chunks}")
    print(f"   Tool calls: {len(tool_calls_made)}")
    print("   💡 Thinking streamed in readable blocks!")
    
    # ===== EXAMPLE 5: Raw Tool Calls (No Auto-Execute) =====
    print("\n\n5️⃣ Raw Tool Calls (No Auto-Execute)")
    print("-" * 37)
    
    response = client.chat("Calculate 9 * 7", tools=tools, auto_execute=False)
    print(f"Question: Calculate 9 * 7")
    print(f"Answer: {response.content}")
    print(f"Tool calls made: {len(response.tool_calls) if response.tool_calls else 0}")
    print(f"Tool results: {len(response.tool_results) if response.tool_results else 0}")
    print("💡 Tools NOT executed - raw tool calls only!")
    
    # ===== EXAMPLE 6: Thinking Control =====
    print("\n\n6️⃣ Thinking Control")
    print("-" * 20)
    
    # Disable thinking
    client.disable_thinking()
    print("❌ Thinking disabled")
    
    response = client.chat("What is 6 + 4?", tools=tools)
    print(f"Quick response: {response.content}")
    print(f"Thinking: {response.thinking or 'None'}")
    print("💡 Back to fast mode!")
    
    # ===== SUMMARY =====
    print("\n\n🎯 Clean API Summary")
    print("=" * 25)
    print("✅ Just TWO main methods:")
    print("  • chat() - Get full response")
    print("  • stream_chat() - Stream response chunks")
    
    print("\n🎛️ Parameters:")
    print("  • tools=None - Add tool functions")
    print("  • auto_execute=True - Auto-run tools (default)")
    print("  • auto_execute=False - Raw tool calls only")
    
    print("\n🧠 Thinking Control:")
    print("  • enable_thinking() - Global thinking on")
    print("  • disable_thinking() - Global thinking off")
    
    print("\n📄 Consistent Response:")
    print("  • response.content - The answer")
    print("  • response.thinking - AI's thought process")
    print("  • response.tool_calls - Tools that were called")
    print("  • response.tool_results - Tool execution results")
    
    print("\n💡 Use Cases:")
    print("  • Simple questions: chat('Question?')")
    print("  • With tools: chat('Calculate...', tools=tools)")
    print("  • Thinking mode: enable_thinking() → chat()")
    print("  • Streaming: stream_chat() for real-time")
    print("  • Raw tools: chat(..., auto_execute=False)")

if __name__ == "__main__":
    main() 