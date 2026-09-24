"""The same model behind ``pyfunc.ResponsesAgent``, MLflow's current recommendation.

``ChatModel`` is deprecated since MLflow 3.0.0, so the Responses-shaped base class is what new code
should use; both are exercised by the integration suite. MLflow hands the agent ``Message`` objects
(not dictionaries), and its serving path can ask for a stream, so both are handled here.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponseOutputItemDoneEvent, ResponsesAgentResponse

from jevper import Choice, SystemOneClient

CRITERIA = {"billing": None, "technical": None, "sales": None}


def message_text(item: Any) -> str:
    """The text of one input item, whether MLflow sent a ``Message`` or a plain mapping.

    ``Message.content`` is a string or a list of content parts; both shapes arrive in practice.
    """
    content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for part in content:
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    return "" if content is None else str(content)


class JevperResponsesAgent(ResponsesAgent):
    def load_context(self, context: Any) -> None:
        from openai import OpenAI

        config = context.model_config or {}
        self.client = SystemOneClient(
            OpenAI(base_url=config["base_url"], api_key="test", max_retries=0, timeout=30),
            model=config["model"],
            api="chat_completions",
            method=config.get("method", "structured"),
        )

    def _answer(self, request: Any) -> str:
        state = "\n".join(message_text(item) for item in request.input)
        response = self.client.system_one(state=state, questions={"intent": Choice(criteria=CRITERIA)})
        return response.answers["intent"].choice or ""

    def _output(self, text: str) -> list[Any]:
        return [
            {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
                "status": "completed",
            }
        ]

    def predict(self, request: Any) -> ResponsesAgentResponse:
        return ResponsesAgentResponse(output=self._output(self._answer(request)))

    def predict_stream(self, request: Any) -> Iterator[Any]:
        """One jevper call, emitted as a single completed output item.

        The base class raises ``NotImplementedError``, and MLflow's serving path asks for a stream when
        the request carries ``stream=True``.
        """
        response = self.predict(request)
        for index, item in enumerate(response.output):
            yield ResponseOutputItemDoneEvent(item=item, output_index=index)


mlflow.models.set_model(JevperResponsesAgent())
