from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # Model settings
    default_model: str = "microsoft/DialoGPT-medium"
    models_dir: str = "./models"
    max_new_tokens: int = 256
    device: Literal["cuda", "mps", "cpu", "auto"] = "auto"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
