from pathlib import Path
from typing import Optional
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import sys

class Settings(BaseSettings):
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent)

    # 2. Paths
    prompts_path: Path = Field(default=Path("prompts"))
    logs_path: Path = Field(default=Path("logs"))

    # 3. Security (API Keys)
    open_router_api_key: SecretStr
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
        if not self.prompts_path.is_absolute():
            self.prompts_path = self.base_dir / self.prompts_path

        if not self.logs_path.is_absolute():
            self.logs_path = self.base_dir / self.logs_path

        self.prompts_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)



try:
    settings = Settings()
    
except Exception as e:
    sys.exit(1)

