from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    title: str = Field(default="新对话", max_length=200)


class ChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=20000)


class MemoryUpdate(BaseModel):
    content: str = Field(min_length=1, max_length=10000)

