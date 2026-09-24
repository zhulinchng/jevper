"""The model jevper exposes to MLflow: a ``pyfunc.ChatModel`` over ``SystemOneClient``.

Logged with models-from-code (``python_model=<this file>``) because a cloudpickled instance cannot
carry the client — the OpenAI client is built in ``load_context`` from ``model_config``, which is the
pattern MLflow documents for a model that talks to a service.
"""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import ChatChoice, ChatCompletionResponse, ChatMessage

from jevper import Choice, SystemOneClient

CRITERIA = {"billing": None, "technical": None, "sales": None}


class JevperChatModel(ChatModel):
    def load_context(self, context: Any) -> None:
        from openai import OpenAI

        config = context.model_config or {}
        self.client = SystemOneClient(
            OpenAI(base_url=config["base_url"], api_key="test", max_retries=0, timeout=30),
            model=config["model"],
            api="chat_completions",
            method=config.get("method", "structured"),
        )

    def predict(self, context: Any, messages: list[ChatMessage], params: Any) -> ChatCompletionResponse:
        state = "\n".join(str(getattr(message, "content", "")) for message in messages)
        response = self.client.system_one(state=state, questions={"intent": Choice(criteria=CRITERIA)})
        answer = response.answers["intent"]
        return ChatCompletionResponse(
            model="jevper",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=answer.choice or ""),
                    finish_reason="stop",
                )
            ],
        )


mlflow.models.set_model(JevperChatModel())
