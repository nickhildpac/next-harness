from app.ports.llm import ChatMessage


class TokenCounter:
    def count(self, text: str) -> int:
        # Approximation keeps the service provider-neutral; replace with model-specific encoders as needed.
        return max(1, len(text) // 4) if text else 0

    def count_messages(self, messages: list[ChatMessage]) -> int:
        return sum(self.count(message.content) + 4 for message in messages)

