from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI
from typing import List

from agent_platform.core.secrets import Secrets
from agent_platform.config import config_holder


class BaseLLM(ABC):
    def __init__(self):
        config = config_holder.get_config()
        self.config = config
        self.secrets = Secrets(self.config.secrets.path)
        self.client = self._create_client()

    def _create_client(self):
        return self._get_client_class()(
            base_url=self.config.urls.llm,
            api_key=self.secrets.get_value("LITELLM_TOKEN"),
            model=self.config.model.name,
            extra_body={"ttl": self.config.model.timeout},
        )

    @abstractmethod
    def _get_client_class(self):
        pass

    def bind_tools(self, tools: List):
        self.client = self.client.bind_tools(tools)


class LLM(BaseLLM):
    def _get_client_class(self):
        return ChatOpenAI
