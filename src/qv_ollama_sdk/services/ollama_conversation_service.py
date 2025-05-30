"""Service for interacting with the Ollama API for conversation generation."""

import ollama
from typing import Dict, Any, Optional, Iterator, List, Callable
import re

from ..domain.models import (
    Conversation, 
    Message, 
    ModelParameters, 
    GenerationResponse,
    MessageRole,
    ToolCall,
    Function,
    ToolResult,
    ToolRegistry
)


class OllamaConversationService:
    """Service for generating responses in conversations using the Ollama API."""
    
    def __init__(self):
        """Initialize the Ollama conversation service."""
        pass

    def generate_response(
        self, 
        conversation: Conversation, 
        parameters: Optional[ModelParameters] = None,
        tools: Optional[List[Callable]] = None
    ) -> GenerationResponse:
        """Generate a response for the given conversation.
        
        Args:
            conversation: The conversation to generate a response for
            parameters: Optional model parameters to use for generation
            tools: Optional list of Python functions that can be called by the model
            
        Returns:
            A GenerationResponse containing the generated content and any tool calls
        """
        params = parameters.to_dict() if parameters else ModelParameters().to_dict()
        
        # Build chat request
        chat_kwargs = {
            "model": conversation.model_name,
            "messages": conversation.get_message_history(),
            "options": params
        }
        
        # Add tools if provided
        if tools:
            chat_kwargs["tools"] = tools
        
        # Handle thinking mode separately (not in options)
        if parameters and hasattr(parameters, 'think'):
            chat_kwargs["think"] = parameters.think
        
        response = ollama.chat(**chat_kwargs)
        
        # Extract the assistant's response
        content = response.get("message", {}).get("content", "")
        
        # Extract thinking if present (both from separate field and from <think> tags)
        thinking = None
        if "message" in response and "thinking" in response["message"]:
            thinking = response["message"]["thinking"]
        
        # Also check for <think> tags in content and extract them
        if "<think>" in content and "</think>" in content:
            # Extract thinking from <think> tags
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            if think_match:
                if not thinking:  # Only use <think> content if no separate thinking field
                    thinking = think_match.group(1).strip()
                # Remove <think> tags from content
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        # Extract tool calls if present
        tool_calls = None
        if "message" in response and "tool_calls" in response["message"]:
            tool_calls = [
                ToolCall(
                    function=Function(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"]
                    ),
                    id=tc.get("id")  # Extract tool call ID if available
                )
                for tc in response["message"]["tool_calls"]
            ]
        
        # Create and return the generation response
        return GenerationResponse(
            model_name=conversation.model_name,
            content=content,
            raw_response=response,
            finish_reason=response.get("done"),
            usage=response.get("prompt_eval_count", {}),
            tool_calls=tool_calls,
            thinking=thinking
        )
    
    def generate_response_with_tool_execution(
        self,
        conversation: Conversation,
        parameters: Optional[ModelParameters] = None,
        tools: Optional[List[Callable]] = None,
        auto_execute: bool = True
    ) -> GenerationResponse:
        """Generate a response with automatic tool execution.
        
        Args:
            conversation: The conversation to generate a response for
            parameters: Optional model parameters to use for generation
            tools: Optional list of Python functions that can be called by the model
            auto_execute: Whether to automatically execute tool calls and get final response
            
        Returns:
            A GenerationResponse with executed tool results and final response
        """
        # First, get the initial response with tool calls
        initial_response = self.generate_response(conversation, parameters, tools)
        
        # If no tool calls or auto_execute is False, return initial response
        if not initial_response.tool_calls or not auto_execute or not tools:
            return initial_response
        
        # Create tool registry from provided tools
        tool_registry = ToolRegistry()
        for tool in tools:
            tool_registry.register(tool)
        
        # Execute tool calls
        tool_results = []
        for tool_call in initial_response.tool_calls:
            result = tool_registry.execute_tool_call(tool_call)
            tool_results.append(result)
        
        # Add assistant message with tool calls to conversation (temporarily)
        temp_conversation = Conversation(
            model_name=conversation.model_name,
            messages=conversation.messages.copy()
        )
        
        # Add assistant message with tool calls
        assistant_msg = temp_conversation.add_assistant_message(initial_response.content)
        assistant_msg.tool_calls = initial_response.tool_calls
        
        # Add tool result messages  
        temp_conversation.add_tool_results(tool_results)
        
        # Get final response from model with tool results
        final_response = self.generate_response(temp_conversation, parameters)
        
        # Combine responses
        return GenerationResponse(
            model_name=conversation.model_name,
            content=final_response.content,
            raw_response=final_response.raw_response,
            finish_reason=final_response.finish_reason,
            usage=final_response.usage,
            tool_calls=initial_response.tool_calls,
            tool_results=tool_results
        )

    def stream_response(
        self, 
        conversation: Conversation, 
        parameters: Optional[ModelParameters] = None,
        tools: Optional[List[Callable]] = None
    ) -> Iterator[GenerationResponse]:
        """Generate a streaming response for the given conversation.
        
        Args:
            conversation: The conversation to generate a response for
            parameters: Optional model parameters to use for generation
            tools: Optional list of Python functions that can be called by the model
            
        Yields:
            GenerationResponse chunks as they become available, including content and tool calls
        """
        params = parameters.to_dict() if parameters else ModelParameters().to_dict()
        
        # Build chat request
        chat_kwargs = {
            "model": conversation.model_name,
            "messages": conversation.get_message_history(),
            "options": params,
            "stream": True
        }
        
        # Add tools if provided
        if tools:
            chat_kwargs["tools"] = tools
        
        # Handle thinking mode separately (not in options)
        if parameters and hasattr(parameters, 'think'):
            chat_kwargs["think"] = parameters.think
        
        stream = ollama.chat(**chat_kwargs)
        
        for chunk in stream:
            content = ""
            tool_calls = None
            thinking = None
            
            # Extract content
            if "message" in chunk and "content" in chunk["message"]:
                content = chunk["message"]["content"]
            
            # Extract thinking if present (both from separate field and from <think> tags)
            if "message" in chunk and "thinking" in chunk["message"]:
                thinking = chunk["message"]["thinking"]
            
            # Also check for <think> tags in content and extract them
            if "<think>" in content and "</think>" in content:
                # Extract thinking from <think> tags
                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                if think_match:
                    if not thinking:  # Only use <think> content if no separate thinking field
                        thinking = think_match.group(1).strip()
                    # Remove <think> tags from content
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
            # Extract tool calls if present
            if "message" in chunk and "tool_calls" in chunk["message"]:
                tool_calls = [
                    ToolCall(
                        function=Function(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"]
                        ),
                        id=tc.get("id")  # Extract tool call ID if available
                    )
                    for tc in chunk["message"]["tool_calls"]
                ]
            
            # Yield response chunk
            yield GenerationResponse(
                model_name=conversation.model_name,
                content=content,
                raw_response=chunk,
                finish_reason=chunk.get("done"),
                usage=chunk.get("prompt_eval_count", {}),
                tool_calls=tool_calls,
                thinking=thinking
            )
    
    def stream_response_with_tool_execution(
        self,
        conversation: Conversation,
        parameters: Optional[ModelParameters] = None,
        tools: Optional[List[Callable]] = None,
        auto_execute: bool = True
    ) -> Iterator[GenerationResponse]:
        """Generate a streaming response with automatic tool execution.
        
        Args:
            conversation: The conversation to generate a response for
            parameters: Optional model parameters to use for generation
            tools: Optional list of Python functions that can be called by the model
            auto_execute: Whether to automatically execute tool calls and continue streaming
            
        Yields:
            GenerationResponse chunks, tool execution results, and final response chunks
        """
        # Collect initial response and tool calls
        initial_content = ""
        collected_tool_calls = []
        
        # Stream initial response
        for chunk in self.stream_response(conversation, parameters, tools):
            if chunk.content:
                initial_content += chunk.content
            
            if chunk.tool_calls:
                collected_tool_calls.extend(chunk.tool_calls)
            
            yield chunk
        
        # If no tool calls or auto_execute is False, we're done
        if not collected_tool_calls or not auto_execute or not tools:
            return
        
        # Create tool registry and execute tool calls
        tool_registry = ToolRegistry()
        for tool in tools:
            tool_registry.register(tool)
        
        tool_results = []
        for tool_call in collected_tool_calls:
            result = tool_registry.execute_tool_call(tool_call)
            tool_results.append(result)
            
            # Yield tool execution result
            yield GenerationResponse(
                model_name=conversation.model_name,
                content="",
                tool_results=[result]
            )
        
        # Create temporary conversation with tool results
        temp_conversation = Conversation(
            model_name=conversation.model_name,
            messages=conversation.messages.copy()
        )
        
        # Add assistant message with tool calls
        assistant_msg = temp_conversation.add_assistant_message(initial_content)
        assistant_msg.tool_calls = collected_tool_calls
        
        # Add tool result messages
        temp_conversation.add_tool_results(tool_results)
        
        # Stream final response
        for chunk in self.stream_response(temp_conversation, parameters):
            yield chunk
    