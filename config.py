from pathlib import Path
from typing import Optional
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import sys

class Settings(BaseSettings):
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent)

    # 2. Paths
    prompts_path: Path = Field(default=Path("prompts"))
    logs_path: Path = Field(default=Path("second_run"))

    # 3. Security (API Keys)
    openrouter_api_key: SecretStr
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra variables in .env so it doesn't crash
        case_sensitive=False
    )

    def model_post_init(self, __context):
        """
        Runs automatically after the model is initialized.
        We use this to resolve paths and ensure directories exist.
        """
        self.logs_path.mkdir(parents=True, exist_ok=True)

try:
    settings = Settings()
    
except Exception as e:
    sys.exit(1)

