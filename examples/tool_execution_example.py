"""Example demonstrating automatic tool execution with Thinking Mode integration."""

from qv_ollama_sdk import OllamaChatClient, ModelParameters

def add_two_numbers(a: str, b: str) -> str:
    """Add two numbers and return the result."""
    return str(int(a) + int(b))

def multiply_numbers(a: str, b: str) -> str:
    """Multiply two numbers and return the result."""
    return str(int(a) * int(b))

def get_weather(location: str, unit: str = "celsius") -> str:
    """Get weather information for a location (simulated)."""
    return f"The weather in {location} is 22°{unit[0].upper()} and sunny."

def calculate_area(shape: str, width: str, height: str = None) -> str:
    """Calculate area of a shape."""
    w = float(width)
    if shape.lower() == "square":
        return str(w * w)
    elif shape.lower() == "rectangle" and height:
        h = float(height)
        return str(w * h)
    elif shape.lower() == "circle":
        # width is radius
        import math
        return str(math.pi * w * w)
    else:
        return "Unsupported shape or missing parameters"

def main():
    print("🧠🛠️ Tool Execution + Thinking Mode Example - QV Ollama SDK")
    print("=" * 65)
    
    # Define tools
    tools = [add_two_numbers, multiply_numbers, get_weather, calculate_area]
    
    # Initialize the client
    client = OllamaChatClient(
        model_name="qwen3:8b",
        system_message="You are a helpful assistant that can perform calculations and provide weather information. Think step by step when using tools."
    )
    
    # ===== SECTION 1: Classic Tool Execution (No Thinking) =====
    print("\n🔧 SECTION 1: Classic Tool Execution (No Thinking)")
    print("=" * 55)
    
    print("\n1.1 Simple Auto Tool Execution")
    print("-" * 30)
    response = client.chat_with_auto_tools("What is 15 + 27?", tools=tools)
    print(f"Response: {response}")
    
    print("\n1.2 Complex Multi-Tool Scenario")
    print("-" * 30)
    response = client.chat_with_auto_tools("Calculate 8 * 7, then add 10 to that result", tools=tools)
    print(f"Response: {response}")
    
    # ===== SECTION 2: Tool Execution WITH Thinking =====
    print("\n\n🧠 SECTION 2: Auto-Tools WITH Thinking Mode")
    print("=" * 55)
    print("💡 KEY FEATURE: auto-tools methods automatically use thinking_mode!")
    
    print("\n2.1 Enable Thinking → Auto-Tools Get Thinking Automatically")
    print("-" * 55)
    client.enable_thinking()
    print("✅ Thinking mode enabled globally")
    
    # Now ALL auto-tools methods will automatically use thinking=True
    response = client.chat_with_auto_tools("Calculate the area of a rectangle 5x8", tools=tools)
    print(f"🛠️ Auto-tools response: {response}")
    print("   ↳ This used thinking=True automatically!")
    
    print("\n2.2 Auto-Tools Full Response WITH Automatic Thinking")
    print("-" * 50)
    full_response = client.chat_with_auto_tools_full("What is 12 * 15?", tools=tools)
    print(f"💬 Content: {full_response.content}")
    print(f"🧠 Thinking available: {full_response.thinking is not None}")
    if full_response.thinking:
        print(f"🧠 Thinking preview: {full_response.thinking[:200]}...")
    print("   ↳ Auto-tools automatically included thinking!")
    
    print("\n2.3 Streaming Auto-Tools WITH Automatic Thinking")
    print("-" * 45)
    print("Question: Get weather in Munich and multiply 7 * 6")
    
    thinking_chunks = 0
    content_chunks = 0
    
    for chunk in client.stream_chat_with_auto_tools_full("Get weather in Munich and multiply 7 * 6", tools=tools):
        if chunk.thinking:
            thinking_chunks += 1
            print(f"🧠 Thinking: {chunk.thinking[:80]}{'...' if len(chunk.thinking) > 80 else ''}")
        if chunk.content:
            content_chunks += 1
            print(f"💬 Content: {chunk.content}", end="", flush=True)
        if chunk.tool_calls:
            print(f"\n🛠️ Tool: {chunk.tool_calls[0].function.name}({chunk.tool_calls[0].function.arguments})")
    
    print(f"\n📊 Captured: {thinking_chunks} thinking chunks, {content_chunks} content chunks")
    print("   ↳ Streaming auto-tools automatically included thinking!")
    
    print("\n2.4 Disable Thinking → Auto-Tools Stop Using Thinking")
    print("-" * 50)
    client.disable_thinking()
    print("❌ Thinking mode disabled globally")
    
    response = client.chat_with_auto_tools("Calculate 25 + 17", tools=tools)
    print(f"🛠️ Auto-tools response: {response}")
    print("   ↳ This used thinking=False automatically!")
    
    full_response = client.chat_with_auto_tools_full("What is 9 * 4?", tools=tools)
    print(f"🧠 Thinking available: {full_response.thinking is not None}")
    print("   ↳ No thinking because it's disabled globally!")
    
    # Re-enable for remaining sections
    client.enable_thinking()
    
    # ===== SECTION 3: Streaming + Thinking + Tools =====
    print("\n\n📡 SECTION 3: Manual Thinking Methods")
    print("=" * 50)
    print("💡 Using explicit chat_with_thinking() for fine control")
    
    print("\n3.1 Manual Thinking with Tool Selection")
    print("-" * 40)
    result = client.chat_with_thinking("I need to calculate the area of a rectangle that is 5 units wide and 8 units tall", tools=tools)
    print(f"Question: Calculate rectangle area (5x8)")
    print(f"🧠 Thinking: {result['thinking'][:300]}{'...' if len(result['thinking']) > 300 else ''}")
    print(f"💬 Answer: {result['content']}")
    print("   ↳ chat_with_thinking() forces thinking regardless of global setting!")
    
    print("\n3.2 Thinking Through Multi-Step Problems")
    print("-" * 40)
    result = client.chat_with_thinking("What is 12 * 15, and then what's the square root of that result?", tools=tools)
    print(f"Question: Multi-step calculation (12*15, then sqrt)")
    if result['has_thinking']:
        print(f"🧠 Thinking: {result['thinking'][:400]}{'...' if len(result['thinking']) > 400 else ''}")
    print(f"💬 Answer: {result['content']}")
    
    print("\n3.3 Streaming with Manual Tool Calls")
    print("-" * 35)
    print("Question: Calculate the area of a circle with radius 4")
    print("Streaming response:")
    
    full_thinking = ""
    full_content = ""
    tool_calls_seen = []
    
    for chunk in client.stream_chat_with_tools("Calculate the area of a circle with radius 4", tools=tools):
        if chunk.thinking:
            full_thinking += chunk.thinking
            print(f"🧠 Thinking chunk: {chunk.thinking[:100]}{'...' if len(chunk.thinking) > 100 else ''}")
        if chunk.content:
            full_content += chunk.content
            print(f"💬 Content: '{chunk.content}'", end="", flush=True)
        if chunk.tool_calls:
            tool_calls_seen.extend(chunk.tool_calls)
            print(f"\n🛠️ Tool call: {chunk.tool_calls[0].function.name}({chunk.tool_calls[0].function.arguments})")
    
    print(f"\n📊 Summary:")
    print(f"   Thinking captured: {len(full_thinking)} characters")
    print(f"   Content captured: {len(full_content)} characters") 
    print(f"   Tool calls made: {len(tool_calls_seen)}")
    print("   ↳ stream_chat_with_tools() uses current thinking_mode setting!")
    
    # ===== SECTION 4: Thinking Mode Comparison =====
    print("\n\n⚖️ SECTION 4: Thinking vs Non-Thinking Comparison")
    print("=" * 55)
    
    question = "Calculate 123 * 456 and explain why this calculation might be useful"
    
    print(f"Question: {question}")
    print("\n4.1 WITHOUT Thinking:")
    print("-" * 25)
    client.disable_thinking()
    quick_response = client.chat_with_auto_tools(question, tools=tools)
    print(f"Quick Response: {quick_response}")
    
    print("\n4.2 WITH Thinking:")
    print("-" * 20)
    client.enable_thinking()
    thoughtful_result = client.chat_with_thinking(question, tools=tools)
    print(f"💬 Thoughtful Response: {thoughtful_result['content']}")
    if thoughtful_result['has_thinking']:
        print(f"🧠 Reasoning Process: {thoughtful_result['thinking'][:300]}{'...' if len(thoughtful_result['thinking']) > 300 else ''}")
    
    # ===== SECTION 5: Advanced Tool + Thinking Scenarios =====
    print("\n\n🎯 SECTION 5: Advanced Tool + Thinking Scenarios")
    print("=" * 55)
    
    print("\n5.1 Error Handling with Thinking")
    print("-" * 35)
    result = client.chat_with_thinking("Calculate the area of a triangle with width 5", tools=tools)
    print(f"Question: Invalid tool usage (missing parameters)")
    if result['has_thinking']:
        print(f"🧠 Thinking: {result['thinking'][:250]}{'...' if len(result['thinking']) > 250 else ''}")
    print(f"💬 Answer: {result['content']}")
    
    print("\n5.2 Tool Chain Reasoning")
    print("-" * 25)
    result = client.chat_with_thinking("If a square has an area of 25, what's the weather like in that many cities?", tools=tools)
    print(f"Question: Complex tool chaining")
    if result['has_thinking']:
        print(f"🧠 Thinking: {result['thinking'][:350]}{'...' if len(result['thinking']) > 350 else ''}")
    print(f"💬 Answer: {result['content']}")
    
    # ===== SECTION 6: Full Response Analysis =====
    print("\n\n🔍 SECTION 6: Full Response Analysis")
    print("=" * 45)
    
    print("\n6.1 Complete Tool Execution Details")
    print("-" * 35)
    client.enable_thinking()
    full_response = client.chat_with_auto_tools_full("Calculate 42 * 37 and tell me about the weather in Paris", tools=tools)
    
    print(f"Content: {full_response.content}")
    print(f"Thinking Available: {full_response.thinking is not None}")
    if full_response.thinking:
        print(f"Thinking Length: {len(full_response.thinking)} characters")
        print(f"Thinking Preview: {full_response.thinking[:200]}...")
    
    if full_response.tool_calls:
        print(f"Tool calls made: {len(full_response.tool_calls)}")
        for i, tool_call in enumerate(full_response.tool_calls, 1):
            print(f"  {i}. {tool_call.function.name}({tool_call.function.arguments})")
    
    if full_response.tool_results:
        print(f"Tool results: {len(full_response.tool_results)}")
        for i, result in enumerate(full_response.tool_results, 1):
            status = "✅ Success" if result.error is None else "❌ Error"
            output = result.result if result.error is None else result.error
            print(f"  {i}. {status}: {output}")
    
    # ===== SECTION 7: Conversation History =====
    print("\n\n📚 SECTION 7: Conversation History Analysis")
    print("=" * 50)
    
    history = client.get_history()
    print(f"Total messages in conversation: {len(history)}")
    
    thinking_messages = 0
    tool_messages = 0
    regular_messages = 0
    
    for i, msg in enumerate(history, 1):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        if role == 'tool':
            tool_messages += 1
        elif role == 'assistant' and 'thinking' in str(msg):
            thinking_messages += 1
        else:
            regular_messages += 1
        
        if i <= 5:  # Show first 5 messages
            print(f"  {i}. {role}: {content[:60]}{'...' if len(content) > 60 else ''}")
    
    print(f"\nMessage Distribution:")
    print(f"  Regular messages: {regular_messages}")
    print(f"  Tool result messages: {tool_messages}")
    print(f"  Messages with thinking: {thinking_messages}")
    
    # ===== FINAL SUMMARY =====
    print("\n\n🎉 FINAL SUMMARY: Tool + Thinking Integration")
    print("=" * 55)
    
    print("✅ Successfully Demonstrated:")
    print("  🧠 Thinking Mode + Tool Selection Reasoning")
    print("  🛠️ Automatic Tool Execution with Thought Process")
    print("  📡 Streaming with Real-time Thinking + Tool Calls")
    print("  ⚖️ Quality Comparison: Thinking vs Non-Thinking")
    print("  🎯 Advanced Scenarios: Error Handling, Tool Chains")
    print("  🔍 Complete Response Analysis with Full Details")
    print("  📚 Conversation History with Tool + Thinking Messages")
    
    print("\n🚀 QV Ollama SDK Features Showcased:")
    print("  - chat_with_thinking() + tools: One-shot thinking with tools")
    print("  - chat_with_auto_tools(): Classic automatic tool execution")
    print("  - chat_with_auto_tools_full(): Full response + tool details")
    print("  - stream_chat_with_tools(): Streaming with thinking capture")
    print("  - enable_thinking() / disable_thinking(): Dynamic control")
    print("  - Complete tool calling pipeline with thinking transparency")
    
    print("\n🎯 Perfect for:")
    print("  - Debugging tool selection logic")
    print("  - Understanding AI reasoning process")  
    print("  - Quality assurance for critical calculations")
    print("  - Educational AI applications")
    print("  - Transparent AI systems")

if __name__ == "__main__":
    main() 