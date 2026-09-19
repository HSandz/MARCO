import os
from typing import Any, Dict, Optional

from loguru import logger

from marco.llms.basellm import BaseLLM
from marco.utils import read_json

DEFAULT_MODEL = 'gemini-2.5-flash-lite'
DEFAULT_LOCATION = 'global'
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64
DEFAULT_MAX_RETRIES = 100
DEFAULT_RETRY_DELAY_BASE = 1
DEFAULT_MAX_RETRY_DELAY_SECONDS = 60
DEFAULT_REQUEST_TIMEOUT_SECONDS = 90
DEFAULT_AGENT_CONTEXT = 'Unknown'
VERTEX_SCOPES = ['https://www.googleapis.com/auth/cloud-platform']


class VertexAILLM(BaseLLM):

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        json_mode: bool = False,
        agent_context: Optional[str] = None,
        config_file: str = 'config/api-config.json',
        api_key: str = '',
        project_id: str = '',
        location: str = '',
        credentials_path: str = '',
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        self.model = model or DEFAULT_MODEL
        self.json_mode = json_mode
        self.agent_context = agent_context or DEFAULT_AGENT_CONTEXT
        self.config_file = config_file

        self.max_tokens: int = kwargs.get('max_tokens') or DEFAULT_MAX_TOKENS
        self.temperature: float = (
            kwargs.get('temperature') if kwargs.get('temperature') is not None else DEFAULT_TEMPERATURE
        )
        self.top_p: float = kwargs.get('top_p') or DEFAULT_TOP_P
        self.top_k: int = kwargs.get('top_k') or DEFAULT_TOP_K
        self.max_retries = kwargs.get('max_retries') or DEFAULT_MAX_RETRIES
        self.retry_delay_base = kwargs.get('retry_delay_base') or DEFAULT_RETRY_DELAY_BASE
        self.max_retry_delay_seconds = (
            kwargs.get('max_retry_delay_seconds') or DEFAULT_MAX_RETRY_DELAY_SECONDS
        )
        self.enable_prompt_logging: bool = kwargs.get('enable_prompt_logging', True)
        self.request_timeout_seconds: int = (
            kwargs.get('timeout_seconds')
            or kwargs.get('request_timeout_seconds')
            or DEFAULT_REQUEST_TIMEOUT_SECONDS
        )

        provider_cfg = self._load_provider_cfg()
        self.api_key = self._normalize_api_key(
            api_key or provider_cfg.get('api_key') or provider_cfg.get('vertexai_api_key')
        )
        self.location = location or provider_cfg.get('location') or DEFAULT_LOCATION
        self.credentials_path = ''
        self.project_id = ''

        if not self.api_key:
            config_project_id = project_id or provider_cfg.get('project_id') or provider_cfg.get('project')
            self.credentials_path = credentials_path or provider_cfg.get('credentials_path') or ''
            if not self.credentials_path:
                default_sa_path = 'config/vertex-service-account.json'
                if os.path.exists(default_sa_path):
                    self.credentials_path = default_sa_path
                    logger.debug(f"Auto-detected Vertex AI service account at {default_sa_path}")

            project_id_from_credentials = self._read_project_id_from_credentials(self.credentials_path)
            if project_id_from_credentials and config_project_id:
                logger.debug(
                    'Both project_id and credentials_path were provided; using project_id from credentials_path'
                )
            self.project_id = project_id_from_credentials or config_project_id

            if not self.project_id:
                raise ValueError(
                    f"Vertex AI requires api_key, project_id in credentials file, or provider config 'providers.vertexai' in {self.config_file}"
                )

        try:
            from google import genai
            from google.genai import errors as genai_errors
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "google-genai package is required for Vertex AI. Install it with: pip install google-genai"
            ) from exc

        self._genai_errors = genai_errors
        self._types = types

        credentials = None
        if self.api_key:
            self.credentials = None
            logger.info("Using Vertex AI API key authentication")
        elif self.credentials_path:
            if os.path.exists(self.credentials_path):
                from google.oauth2 import service_account

                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=VERTEX_SCOPES,
                )
                self.credentials = credentials
                logger.info(f"Using Vertex AI service account: {self.credentials_path}")
            else:
                logger.warning(
                    f"Credentials file not found at {self.credentials_path}, falling back to Application Default Credentials"
                )
                self.credentials = None
        else:
            self.credentials = None

        client_kwargs: Dict[str, Any] = {
            'vertexai': True,
            'http_options': self._types.HttpOptions(timeout=self.request_timeout_seconds * 1000),
        }
        if self.api_key:
            client_kwargs['api_key'] = self.api_key
        else:
            client_kwargs['location'] = self.location
            client_kwargs['project'] = self.project_id
            client_kwargs['credentials'] = credentials

        self.client = genai.Client(**client_kwargs)

        self.max_context_length = self._get_context_length(self.model)
        self.generation_config = self._build_generation_config()

        logger.info(
            f"Initialized Vertex AI LLM with model '{self.model}', auth='{self._auth_mode}', location='{self.location}'"
        )

    def _load_provider_cfg(self) -> Dict[str, Any]:
        try:
            api_config = read_json(self.config_file)
            return api_config.get('providers', {}).get('vertexai', {})
        except Exception as e:
            logger.warning(f"Failed to load Vertex AI provider config from {self.config_file}: {e}")
            return {}

    @staticmethod
    def _normalize_api_key(api_key: Any) -> str:
        if isinstance(api_key, list):
            api_key = next((key for key in api_key if key), '')
        return str(api_key or '').strip()

    @property
    def _auth_mode(self) -> str:
        if self.api_key:
            return 'api_key'
        if self.credentials_path:
            return 'service_account'
        return f'project:{self.project_id}'

    @staticmethod
    def _read_project_id_from_credentials(credentials_path: str) -> str:
        if not credentials_path:
            return ''
        if not os.path.exists(credentials_path):
            return ''

        try:
            credentials_json = read_json(credentials_path)
            return credentials_json.get('project_id') or credentials_json.get('project') or ''
        except Exception as e:
            logger.warning(f"Failed to read project_id from credentials file {credentials_path}: {e}")
            return ''

    def _build_generation_config(self):
        config_kwargs: Dict[str, Any] = {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'max_output_tokens': self.max_tokens,
        }
        if self.top_k > 0:
            config_kwargs['top_k'] = self.top_k
        if self.json_mode:
            config_kwargs['response_mime_type'] = 'application/json'

        return self._types.GenerateContentConfig(**config_kwargs)

    @staticmethod
    def _get_context_length(model: str) -> int:
        return 1048576 if 'gemini' in model else 1048576

    @property
    def tokens_limit(self) -> int:
        return self.max_context_length

    def _get_retriable_errors(self) -> tuple:
        base_errors = super()._get_retriable_errors()

        try:
            import httpx

            return base_errors + (
                httpx.ProtocolError,
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.RemoteProtocolError,
                httpx.LocalProtocolError,
                httpx.ReadError,
                httpx.WriteError,
                self._genai_errors.APIError,
            )
        except Exception:
            return base_errors + (self._genai_errors.APIError,)

    def _is_retriable_exception(self, exception: Exception) -> bool:
        error_str = str(exception).lower()

        if isinstance(exception, self._genai_errors.APIError):
            code = getattr(exception, 'code', None)
            if isinstance(code, int) and (code == 429 or code == 499 or code >= 500):
                return True

            status = str(getattr(exception, 'status', '') or '').upper()
            if status in {'RESOURCE_EXHAUSTED', 'CANCELLED', 'CANCELED'}:
                return True

            error_text = str(exception).upper()
            return 'RESOURCE_EXHAUSTED' in error_text or 'CANCELLED' in error_text or 'CANCELED' in error_text

        try:
            import httpx

            if isinstance(
                exception,
                (
                    httpx.TimeoutException,
                    httpx.ConnectError,
                    httpx.RemoteProtocolError,
                    httpx.ProtocolError,
                    httpx.LocalProtocolError,
                    httpx.ReadError,
                    httpx.WriteError,
                ),
            ):
                return True
        except Exception:
            pass

        return super()._is_retriable_exception(exception)

    def _make_api_request(self, prompt_text: str):
        if not isinstance(prompt_text, str):
            prompt_text = str(prompt_text)

        try:
            return self.client.models.generate_content(
                model=self.model,
                contents=prompt_text,
                config=self.generation_config,
            )
        except Exception as e:
            logger.error(f"Vertex AI API request failed: {type(e).__name__}: {str(e)[:200]}")
            raise

    @staticmethod
    def _extract_text(response: Any) -> str:
        if hasattr(response, 'text') and response.text:
            try:
                return response.text.strip()
            except Exception:
                pass

        candidates = getattr(response, 'candidates', None)
        if candidates:
            for candidate in candidates:
                content = getattr(candidate, 'content', None)
                parts = getattr(content, 'parts', None)
                if not parts:
                    continue

                text_parts = []
                for part in parts:
                    text = getattr(part, 'text', '')
                    if text:
                        text_parts.append(text)

                if text_parts:
                    return ' '.join(text_parts).strip()

        return ''

    def __call__(self, prompt: str, *args, **kwargs) -> str:
        try:
            if self.enable_prompt_logging:
                logger.debug(f"LLM Prompt ({self.agent_context} -> {self.model}):\n{prompt}")

            actual_prompt = prompt
            if self.json_mode:
                actual_prompt = f"{prompt}\n\nPlease respond with valid JSON only."

            response = self.execute_with_retry(self._make_api_request, prompt_text=actual_prompt)
            content = self._extract_text(response)
            usage_metadata = getattr(response, 'usage_metadata', None)
            input_tokens = getattr(usage_metadata, 'prompt_token_count', None) if usage_metadata else None
            output_tokens = getattr(usage_metadata, 'candidates_token_count', None) if usage_metadata else None
            total_tokens = getattr(usage_metadata, 'total_token_count', None) if usage_metadata else None

            if usage_metadata:
                if input_tokens is not None and output_tokens is None and total_tokens is not None:
                    output_tokens = max(total_tokens - input_tokens, 0)
                elif output_tokens is not None and input_tokens is None and total_tokens is not None:
                    input_tokens = max(total_tokens - output_tokens, 0)

            self.track_usage(
                prompt,
                content or '',
                input_tokens if usage_metadata else None,
                output_tokens if usage_metadata else None,
                api_usage=(
                    {
                        'prompt_tokens': input_tokens,
                        'completion_tokens': output_tokens,
                        'total_tokens': total_tokens,
                    }
                    if usage_metadata
                    else {}
                ),
            )

            if not content:
                logger.warning("Empty response from Vertex AI")
                return ''

            logger.debug(f"LLM Response ({self.agent_context} -> {self.model}):\n{content}")
            return content

        except Exception as e:
            return self.handle_api_error(e)
