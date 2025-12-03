from langchain_openai import ChatOpenAI
from typing import List

from agent_platform.core.secrets import Secrets


class LLM:
    def __init__(self, config):
        self.config = config
        self.secrets = Secrets(self.config.secrets)
        self.client = ChatOpenAI(
            base_url=self.config.urls.llm + self.config.model.llm,
            api_key=self.secrets.get_value("LITELLM_TOKEN"),  # Can be any string
            model=self.config.model.name,
            extra_body={"ttl": self.config.model.timeout},
        )

    def bind_tools(self, tools: List):
        self.client = self.client.bind_tools(tools)
