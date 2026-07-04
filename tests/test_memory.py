from app.ports.llm import ChatMessage
from app.services.memory import MemoryService
from app.services.tokens import TokenCounter


def test_token_counter_counts_messages():
    counter = TokenCounter()
    messages = [ChatMessage(role="system", content="hello"), ChatMessage(role="user", content="world")]

    assert counter.count_messages(messages) >= 8


async def test_context_messages_accepts_persisted_string_roles():
    class Repo:
        async def recent_messages(self, conversation_id: str, limit: int):
            class Message:
                role = "user"
                content = "hello from storage"

            return [Message()]

    class Conversation:
        id = "conversation-id"
        summary = None

    service = MemoryService(
        Repo(),
        llm=None,
        token_counter=TokenCounter(),
        context_budget=1000,
        summary_trigger_tokens=1000,
        window_turn_count=12,
        model="test-model",
        timeout_seconds=1,
    )

    messages = await service.context_messages(Conversation(), "system prompt")

    assert messages[-1] == ChatMessage(role="user", content="hello from storage")
