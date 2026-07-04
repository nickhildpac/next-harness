from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.ports.llm import ChatMessage, GenerationParams, LLMClient, LLMResult


class ChatState(TypedDict):
    messages: list[ChatMessage]
    params: GenerationParams
    result: LLMResult | None


class ChatGraph:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        graph = StateGraph(ChatState)
        graph.add_node("generate", self._generate)
        graph.set_entry_point("generate")
        graph.add_edge("generate", END)
        self.graph = graph.compile()

    async def _generate(self, state: ChatState) -> ChatState:
        result = await self.llm.chat(state["messages"], state["params"])
        return {**state, "result": result}

    async def run(self, messages: list[ChatMessage], params: GenerationParams) -> LLMResult:
        state = await self.graph.ainvoke({"messages": messages, "params": params, "result": None})
        result = state["result"]
        if result is None:
            raise RuntimeError("LLM graph completed without a result")
        return result

