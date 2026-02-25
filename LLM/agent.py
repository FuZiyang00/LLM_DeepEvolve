from __future__ import annotations
from typing import Generic, TypeVar, Tuple, Any, Dict
import httpx
from httpx import HTTPStatusError, RequestError
from config import settings
from pipelines.utils import setup_logger

logger = setup_logger(__name__)

T = TypeVar("T")

class BaseLLMAgent(Generic[T]):
    def __init__(self,
                 model: str = "anthropic/claude-sonnet-4.5",
                 ):
        
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        self._client = httpx.AsyncClient(timeout=60.0,
                                         http2=True,
                                         limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
                                         )
    
    async def aclose(self) -> None:
        """Close the underlying HTTP client (call on app shutdown)."""
        await self._client.aclose()

    async def _post(self, payload: Dict[str, Any]) -> httpx.Response:
        """Internal helper to POST and raise for status with good error logging."""
        try:
            r = await self._client.post("https://openrouter.ai/api/v1/chat/completions",
                                        headers=self.headers,
                                        json=payload,)
            r.raise_for_status()
            return r

        except HTTPStatusError as e:
            status_code = e.response.status_code if e.response is not None else "Unknown"
            body = e.response.text if e.response is not None else ""
            logger.error(f"HTTP error {status_code}: {e}")
            if body:
                logger.error(f"Server Response Body: {body}")
            raise

        except RequestError as e:
            logger.error(f"Network/client error occurred: {e}")
            raise

        except Exception as e:
            logger.error(f"General error: {e}")
            raise

    async def _run(self, 
                   messages: list[dict[str, str]], 
                   temperature: float) -> Tuple[str, Dict[str, Any]]:
        
        payload = {
            "model": self.model,
            "messages": messages,
            "reasoning": {"enabled": False},
            "temperature": temperature,
        }

        r = await self._post(payload)

        response_json = r.json()
        raw_content = response_json["choices"][0]["message"]["content"]


        return raw_content


    