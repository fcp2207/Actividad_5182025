from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """Implementación en memoria del historial de mensajes."""
    messages: list[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Añadir un conjunto de mensajes al historial."""
        self.messages.extend(messages)

    def clear(self) -> None:
        """Limpiar el historial."""
        self.messages = []

store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    """Recupera el historial por ID de sesión."""
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]
