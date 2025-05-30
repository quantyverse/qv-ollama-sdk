"""Comparison of stream_chat_with_auto_tools vs stream_chat_with_auto_tools_full"""

from qv_ollama_sdk import OllamaChatClient

def add_numbers(a: str, b: str) -> str:
    """Add two numbers."""
    return str(int(a) + int(b))

def multiply_numbers(a: str, b: str) -> str:
    """Multiply two numbers."""
    return str(int(a) * int(b))

def main():
    client = OllamaChatClient(
        model_name="qwen3:8b",
        system_message="Du bist ein Assistent für Berechnungen."
    )
    
    tools = [add_numbers, multiply_numbers]
    question = "Was ist 15 + 27?"
    
    print("🔄 Streaming Methods Comparison")
    print("=" * 50)
    
    # Method 1: Simple streaming (strings only)
    print("\n1. stream_chat_with_auto_tools() - Simple")
    print("-" * 40)
    print("Output: ", end="", flush=True)
    
    for chunk in client.stream_chat_with_auto_tools(question, tools):
        if chunk:  # chunk ist ein STRING
            print(chunk, end="", flush=True)
    
    print("\n")
    
    # Clear conversation for clean test
    client.clear_history()
    
    # Method 2: Full streaming (GenerationResponse objects)
    print("\n2. stream_chat_with_auto_tools_full() - Full Details")
    print("-" * 40)
    
    chunk_count = 0
    content_chunks = []
    tool_calls_seen = 0
    tool_results_seen = 0
    
    for chunk in client.stream_chat_with_auto_tools_full(question, tools):
        chunk_count += 1
        
        # Sammle Content für finale Ausgabe
        if chunk.content:
            content_chunks.append(chunk.content)
        
        # Zeige Tool Activity
        if chunk.tool_calls:
            tool_calls_seen += len(chunk.tool_calls)
            for tc in chunk.tool_calls:
                print(f"🔧 Tool Call: {tc.function.name}({tc.function.arguments})")
        
        if chunk.tool_results:
            tool_results_seen += len(chunk.tool_results)
            for tr in chunk.tool_results:
                status = "✅ Success" if tr.error is None else "❌ Error"
                result = tr.result if tr.error is None else tr.error
                print(f"{status}: {result}")
    
    print(f"\nFinal Content: {''.join(content_chunks)}")
    print(f"Chunks processed: {chunk_count}")
    print(f"Tool calls detected: {tool_calls_seen}")
    print(f"Tool results processed: {tool_results_seen}")
    
    # Comparison Table
    print("\n" + "=" * 50)
    print("📊 METHOD COMPARISON")
    print("-" * 50)
    print(f"{'Feature':<25} {'Simple':<10} {'Full':<10}")
    print("-" * 50)
    print(f"{'Return Type':<25} {'str':<10} {'GenResp':<10}")
    print(f"{'Content Access':<25} {'direct':<10} {'obj.content':<10}")
    print(f"{'Tool Call Info':<25} {'❌':<10} {'✅':<10}")
    print(f"{'Tool Results':<25} {'❌':<10} {'✅':<10}")
    print(f"{'Debug Info':<25} {'❌':<10} {'✅':<10}")
    print(f"{'UI Complexity':<25} {'Low':<10} {'High':<10}")
    print(f"{'Best For':<25} {'Simple UI':<10} {'Advanced':<10}")

def demo_ui_implementations():
    """Show how to use each method in different UI scenarios"""
    
    print("\n" + "=" * 50)
    print("💡 UI IMPLEMENTATION EXAMPLES")
    print("=" * 50)
    
    print("""
🔹 SIMPLE UI (Basic Chat Interface):
```python
def simple_chat_ui(message):
    response_text = ""
    for chunk in client.stream_chat_with_auto_tools(message, tools):
        response_text += chunk
        update_ui(response_text)  # Live update
    return response_text
```

🔹 ADVANCED UI (With Tool Indicators):
```python
def advanced_chat_ui(message):
    response_text = ""
    tool_activity = []
    
    for chunk in client.stream_chat_with_auto_tools_full(message, tools):
        if chunk.content:
            response_text += chunk.content
            update_chat_content(response_text)
        
        if chunk.tool_calls:
            for tc in chunk.tool_calls:
                show_tool_indicator(f"🔧 Using {tc.function.name}")
                tool_activity.append(tc)
        
        if chunk.tool_results:
            for tr in chunk.tool_results:
                show_tool_result(f"✅ Result: {tr.result}")
    
    return response_text, tool_activity
```

🎯 RECOMMENDATION FOR YOUR CHAT APP:
- Start with: stream_chat_with_auto_tools() (simple)
- Upgrade to: stream_chat_with_auto_tools_full() (when you want tool indicators)
""")

if __name__ == "__main__":
    main()
    demo_ui_implementations() 