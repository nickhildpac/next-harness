from app.db.models import Conversation
from app.ports.llm import ChatMessage, GenerationParams, LLMClient
from app.repositories.conversations import ConversationRepository
from app.services.tokens import TokenCounter


class MemoryService:
    def __init__(
        self,
        repo: ConversationRepository,
        llm: LLMClient,
        token_counter: TokenCounter,
        context_budget: int,
        summary_trigger_tokens: int,
        window_turn_count: int,
        model: str,
        timeout_seconds: float,
    ):
        self.repo = repo
        self.llm = llm
        self.token_counter = token_counter
        self.context_budget = context_budget
        self.summary_trigger_tokens = summary_trigger_tokens
        self.window_turn_count = window_turn_count
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def context_messages(self, conversation: Conversation, system_prompt: str) -> list[ChatMessage]:
        messages = [ChatMessage(role="system", content=system_prompt)]
        if conversation.summary:
            messages.append(
                ChatMessage(
                    role="system",
                    content=f"Conversation summary so far: {conversation.summary.content}",
                )
            )
        recent = await self.repo.recent_messages(conversation.id, self.window_turn_count * 2)
        messages.extend(ChatMessage(role=message.role.value, content=message.content) for message in recent)
        if self.token_counter.count_messages(messages) > self.context_budget:
            return self._trim_to_budget(messages)
        return messages

    async def summarize_if_needed(self, conversation: Conversation) -> None:
        unsummarized = await self.repo.unsummarized_messages(conversation)
        token_total = sum(message.token_count for message in unsummarized)
        if token_total < self.summary_trigger_tokens or len(unsummarized) < 4:
            return

        existing = conversation.summary.content if conversation.summary else ""
        transcript = "\n".join(f"{message.role}: {message.content}" for message in unsummarized[:-2])
        prompt = [
            ChatMessage(
                role="system",
                content=(
                    "Summarize the conversation for future context. Preserve facts, decisions, "
                    "preferences, open tasks, and important constraints. Do not invent details."
                ),
            ),
            ChatMessage(
                role="user",
                content=f"Existing summary:\n{existing}\n\nNew transcript:\n{transcript}",
            ),
        ]
        result = await self.llm.chat(
            prompt,
            GenerationParams(
                model=self.model,
                temperature=0.2,
                top_p=0.9,
                timeout_seconds=self.timeout_seconds,
            ),
        )
        covered_until = unsummarized[-3].created_at
        await self.repo.upsert_summary(
            conversation,
            content=result.content,
            covered_until=covered_until,
            token_count=self.token_counter.count(result.content),
        )

    def _trim_to_budget(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        required = messages[:1]
        optional = messages[1:]
        kept: list[ChatMessage] = []
        for message in reversed(optional):
            candidate = required + [message] + kept
            if self.token_counter.count_messages(candidate) <= self.context_budget:
                kept.insert(0, message)
        return required + kept
