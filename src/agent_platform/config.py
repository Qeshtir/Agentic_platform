from pathlib import Path
from typing import Optional
from loguru import logger
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


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


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False, env_file=None, extra="ignore"
    )

    model: Model
    server: Server
    urls: Url
    secrets: SecretsConfig


class ConfigHolder:
    def __init__(self):
        self.config = None

    def load_config(self, path: Optional[Path] = None):
        logger.info("Loading config...")
        logger.info("Current working directory {}", Path.cwd())

        if path:
            config_path = Path(path)
            if not config_path.is_absolute():
                config_path = Path.cwd() / config_path

            with open(config_path, "r", encoding="utf-8") as f:
                config_data = f.read()

            import yaml

            cfg = yaml.safe_load(config_data)
            self.config = Config.model_validate(cfg)
        else:
            self.config = Config()

    def get_config(self) -> Config:
        return self.config


config_holder = ConfigHolder()
