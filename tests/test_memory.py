from app.ports.llm import ChatMessage
from app.services.tokens import TokenCounter


def test_token_counter_counts_messages():
    counter = TokenCounter()
    messages = [ChatMessage(role="system", content="hello"), ChatMessage(role="user", content="world")]

    assert counter.count_messages(messages) >= 8

