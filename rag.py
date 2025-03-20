from typing import List
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.chat_engine.types import StreamingAgentChatResponse

def aggregate_answer(chat_engine: BaseChatEngine, query: str, chat_history: List[ChatMessage]) -> StreamingAgentChatResponse:
    response = chat_engine.stream_chat(message=query, chat_history=chat_history)
    return response