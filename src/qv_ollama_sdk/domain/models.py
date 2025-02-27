from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4


class MessageRole(str, Enum):
    """Enumeration of possible message roles in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: MessageRole
    content: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to a dictionary format for the Ollama API."""
        return {
            "role": self.role.value,
            "content": self.content
        }


@dataclass
class Conversation:
    """Represents a complete conversation consisting of multiple messages."""
    model_name: str = "llama3"
    id: UUID = field(default_factory=uuid4)
    title: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_system_message(self, content: str) -> Message:
        """Add a system message to the conversation."""
        message = Message(role=MessageRole.SYSTEM, content=content)
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message
    
    def add_user_message(self, content: str) -> Message:
        """Add a user message to the conversation."""
        message = Message(role=MessageRole.USER, content=content)
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message
    
    def add_assistant_message(self, content: str) -> Message:
        """Add an assistant message to the conversation."""
        message = Message(role=MessageRole.ASSISTANT, content=content)
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message
    
    def get_message_history(self) -> List[Dict[str, str]]:
        """Get the message history in a format suitable for the Ollama API."""
        return [message.to_dict() for message in self.messages]
    
    def clear(self) -> None:
        """Clear all messages from the conversation."""
        self.messages.clear()
        self.updated_at = datetime.now()


@dataclass
class ModelParameters:
    """Parameters for model inference."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary format for the Ollama API."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop": self.stop_sequences if self.stop_sequences else None
        }


@dataclass
class GenerationResponse:
    """Represents a response from the model generation."""
    model_name: str
    content: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    raw_response: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None 