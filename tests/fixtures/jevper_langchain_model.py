"""A LangChain chat model backed by jevper, for ``mlflow.langchain.log_model``.

Only configuration lives on the instance — the OpenAI client is built lazily in a module-level cache —
so the model stays picklable and reloadable, which is what the LangChain flavour requires.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlflow
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage
from mlflow.models import ModelConfig

from jevper import Choice, SystemOneClient

CRITERIA = {"billing": None, "technical": None, "sales": None}

_CLIENTS: dict[tuple[str, str], SystemOneClient] = {}


def client_for(base_url: str, model: str) -> SystemOneClient:
    key = (base_url, model)
    if key not in _CLIENTS:
        from openai import OpenAI

        _CLIENTS[key] = SystemOneClient(
            OpenAI(base_url=base_url, api_key="test", max_retries=0, timeout=30),
            model=model,
            api="chat_completions",
            method="structured",
        )
    return _CLIENTS[key]


class JevperLangChainChat(SimpleChatModel):
    """One jevper question per turn: the state is the conversation, the answer is the label.

    ``SimpleChatModel`` (not ``BaseChatModel``) is the base the MLflow LangChain flavour accepts, and it
    turns ``_call`` into a chat result for us.
    """

    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "stub"

    @property
    def _llm_type(self) -> str:
        return "jevper"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        state = "\n".join(str(message.content) for message in messages)
        response = client_for(self.base_url, self.model).system_one(
            state=state, questions={"intent": Choice(criteria=CRITERIA)}
        )
        return response.answers["intent"].choice or ""


def messages_for(text: str) -> Sequence[BaseMessage]:
    from langchain_core.messages import HumanMessage

    return [HumanMessage(content=text)]


# Models-from-code: with no argument, ModelConfig reads the config MLflow logged beside the code, and
# raises FileNotFoundError when there is none — which is also the state a first run is in, since
# logging is what writes the config. MLflow refuses that read on purpose, for a model whose settings
# are required; the two settings here are optional, so the documented defaults stand in and
# `log_model(model_config=...)` overrides them whenever a deployment has something else to say.
try:
    _config = ModelConfig().to_dict()
except FileNotFoundError:
    _config = {}
mlflow.models.set_model(
    JevperLangChainChat(
        base_url=_config.get("base_url", "http://127.0.0.1:8000/v1"),
        model=_config.get("model", "stub"),
    )
)
