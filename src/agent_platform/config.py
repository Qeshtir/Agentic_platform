import os

import yaml
from loguru import logger
from pydantic import BaseModel


class Model(BaseModel):
    max_tokens: int
    name: str
    timeout: int


class Server(BaseModel):
    host: str
    port: int
    threads: int


class Url(BaseModel):
    llm: str
    tokens: str


class SecretsConfig(BaseModel):
    path: str


class Config(BaseModel):
    model: Model
    server: Server
    urls: Url
    secrets: SecretsConfig


class ConfigHolder:
    def __init__(self):
        self.config = None

    def load_config(self, path):
        logger.info("Loading config...")
        logger.info("Current working directory {}", os.getcwd())
        config_path = os.path.normpath(os.path.join(os.getcwd(), path))
        with open(config_path) as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        self.config = Config.model_validate(cfg)

    def get_config(self) -> Config:
        return self.config


config_holder = ConfigHolder()
