import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from loguru import logger

from marco.llms.basellm import BaseLLM
from marco.utils import read_json

DEFAULT_MODEL = 'gpt-4o-mini'
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_BASE = 1
DEFAULT_MAX_TOKENS_LIMIT = 4096
DEFAULT_CONTEXT_LENGTH = 128000
DEFAULT_AGENT_CONTEXT = "Unknown"

class OpenAILLM(BaseLLM):

    def __init__(
        self,
        model: str = '',
        api_key: str = '',
        json_mode: bool = False,
        agent_context: Optional[str] = None,
        config_file: str = 'config/api-config.json',
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        self.config_file = config_file
        api_config = self._load_api_config()
        provider_cfg = api_config.get('providers', {}).get('openai', {}) if api_config else {}
        
        self.api_keys = []
        self.current_key_index = 0
        
        if not model:
            model = provider_cfg.get('model', '') or DEFAULT_MODEL
        
        self.original_model = model or DEFAULT_MODEL
        
        self._provider_cfg = provider_cfg
        
        self.json_mode = json_mode
        self.agent_context = agent_context or DEFAULT_AGENT_CONTEXT
        self.enable_prompt_logging: bool = kwargs.get('enable_prompt_logging', True)
        
        self.max_tokens: int = kwargs.get('max_tokens') or provider_cfg.get('max_tokens') or DEFAULT_MAX_TOKENS
        self.temperature: float = kwargs.get('temperature') or provider_cfg.get('temperature') or DEFAULT_TEMPERATURE
        self.top_p: float = kwargs.get('top_p') or provider_cfg.get('top_p') or DEFAULT_TOP_P
        self.max_retries = kwargs.get('max_retries') or provider_cfg.get('max_retries') or DEFAULT_MAX_RETRIES
        self.retry_delay_base = kwargs.get('retry_delay_base') or provider_cfg.get('retry_delay_base') or DEFAULT_RETRY_DELAY_BASE

        self.api_keys = self._resolve_api_keys(api_key)
        self.api_key = self.api_keys[0] if self.api_keys else ''

        try:
            if not provider_cfg or not provider_cfg.get('base_url'):
                raise ValueError(f"Missing 'base_url' for OpenAI provider in {self.config_file}!")
            
            raw_base_url = provider_cfg['base_url'].strip()
            self.base_url = raw_base_url.rstrip('/')
            
            self.is_official_openai = self._is_official_openai_url(self.base_url)
            
            if self.is_official_openai:
                self.model = self._normalize_model_for_official(self.original_model)
                logger.info("Detected official OpenAI API endpoint")
            else:
                self.model = self._normalize_model_for_proxy(self.original_model, provider_cfg)
                logger.info(f"Detected proxy provider: {self.base_url}")
                logger.info(f"Using model: {self.model} (proxy supports multi-provider models)")
            
            self.max_context_length = self._get_context_length(self.model)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenAI base_url from {self.config_file}: {e}")

        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAI LLMs. Install it with: pip install openai>=1.0.0"
            ) from exc

        self.client = OpenAI(api_key=self.api_key or None, base_url=self.base_url)

        provider_type = "Official OpenAI" if self.is_official_openai else "Proxy Provider"
        logger.info(
            f"Initialized OpenAI LLM ({provider_type}) with model '{self.model}' "
            f"(display: '{self.original_model}'), agent='{self.agent_context}', "
            f"max_retries={self.max_retries}, base_url={self.base_url}"
        )

    def _load_api_config(self) -> Dict[str, Any]:
        try:
            return read_json(self.config_file)
        except Exception as e:
            logger.warning(f"Failed to load API config from {self.config_file}: {e}")
            return {}

    @staticmethod
    def _normalize_model_for_official(model: str) -> str:
        cleaned = (model or '').strip()
        if '/' in cleaned:
            prefix, suffix = cleaned.split('/', 1)
            if prefix.lower() == 'openai':
                return suffix
        return cleaned or DEFAULT_MODEL

    @staticmethod
    def _normalize_model_for_proxy(model: str, provider_cfg: Dict[str, Any]) -> str:
        cleaned = (model or '').strip()
        
        if provider_cfg and 'model_mappings' in provider_cfg:
            model_mappings = provider_cfg['model_mappings']
            if cleaned in model_mappings:
                mapped_model = model_mappings[cleaned]
                logger.info(f"Model alias '{cleaned}' mapped to '{mapped_model}'")
                return mapped_model
        
        return cleaned or DEFAULT_MODEL

    @staticmethod
    def _is_official_openai_url(url: str) -> bool:
        if not url:
            return False
        
        url_lower = url.lower().strip()
        
        official_domains = [
            'api.openai.com',
            'openai.com',
            'https://api.openai.com',
            'http://api.openai.com',
        ]
        
        try:
            parsed = urlparse(url_lower)
            hostname = parsed.netloc or parsed.path.split('/')[0] if not parsed.netloc else parsed.netloc
            
            if ':' in hostname:
                hostname = hostname.split(':')[0]
            
            for domain in official_domains:
                domain_clean = domain.replace('https://', '').replace('http://', '')
                if hostname == domain_clean or hostname.endswith('.' + domain_clean):
                    return True
        except Exception:
            for domain in ['api.openai.com', 'openai.com']:
                if domain in url_lower:
                    return True
        
        return False

    @staticmethod
    def _get_context_length(model: str) -> int:
        context_map = {
            'gpt-4o': DEFAULT_CONTEXT_LENGTH,
            'gpt-4o-mini': DEFAULT_CONTEXT_LENGTH,
            'gpt-4.1-mini': DEFAULT_CONTEXT_LENGTH,
            'gpt-4.1': DEFAULT_CONTEXT_LENGTH,
            'o4-mini': 200000,
            'o4-mini-high': 200000,
            'gpt-4-turbo': DEFAULT_CONTEXT_LENGTH,
            'gpt-4': 8192,
            'gpt-3.5-turbo': 16384,
        }
        return context_map.get(model, DEFAULT_CONTEXT_LENGTH)

    def _resolve_api_keys(self, explicit_key: str) -> list:
        if explicit_key:
            return [explicit_key]
        
        env_key = os.getenv('OPENAI_API_KEY')
        if env_key:
            return [env_key]
        
        try:
            api_config = self._load_api_config()
            provider_cfg = api_config.get('providers', {}).get('openai', {})
            
            keys = provider_cfg.get('api_keys', [])
            if keys and isinstance(keys, list):
                return keys
            
            key = provider_cfg.get('api_key', '')
            if key:
                return [key]
            
            provider = api_config.get('provider', '').lower()
            if provider == 'openai':
                legacy_key = api_config.get('api_key', '') or api_config.get('openai_api_key', '')
                if legacy_key:
                    return [legacy_key]
            
            if 'openai_api_key' in api_config:
                return [api_config['openai_api_key']]
        except Exception:
            pass
        
        logger.warning(f"OpenAI API key not found. Set OPENAI_API_KEY env var or update {self.config_file}.")
        return []
    
    def _switch_to_next_api_key(self) -> bool:
        if len(self.api_keys) <= 1:
            return False
        
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.api_key = self.api_keys[self.current_key_index]
        
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key or None, base_url=self.base_url)
        
        logger.info(f"Switched to API key #{self.current_key_index + 1} (total: {len(self.api_keys)})")
        return True

    def _get_retriable_errors(self) -> tuple:
        base_errors = super()._get_retriable_errors()
        try:
            from openai import APIConnectionError, APITimeoutError, RateLimitError  # type: ignore
            return base_errors + (APIConnectionError, APITimeoutError, RateLimitError)
        except ImportError:
            return base_errors

    def _is_retriable_exception(self, exception: Exception) -> bool:
        try:
            from openai import APIError, APIConnectionError, APITimeoutError, RateLimitError  # type: ignore

            if isinstance(exception, (APIConnectionError, APITimeoutError, RateLimitError)):
                return True
            if isinstance(exception, APIError):
                status = getattr(exception, 'status', None)
                return status == 429 or (isinstance(status, int) and status >= 500)
        except ImportError:
            pass

        return super()._is_retriable_exception(exception)
    
    def _is_rate_limit_error(self, exception: Exception) -> bool:
        try:
            from openai import RateLimitError, APIError  # type: ignore
            
            if isinstance(exception, RateLimitError):
                return True
            
            if isinstance(exception, APIError):
                status = getattr(exception, 'status', None)
                if status == 429:
                    return True
        except ImportError:
            pass
        
        return False

    def _make_api_request(
        self,
        messages: list[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
    ) -> Any:
        logger.info(f"[API CALL] Provider: openai | Model: {self.model}")
        request_payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": min(self.max_tokens, DEFAULT_MAX_TOKENS_LIMIT),
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        if response_format:
            request_payload["response_format"] = response_format

        return self.client.chat.completions.create(**request_payload)

    @staticmethod
    def _extract_message_content(response: Any) -> str:
        choices = getattr(response, 'choices', None)
        if not choices:
            return ""

        message = getattr(choices[0], 'message', None)
        if not message:
            return ""

        content = getattr(message, 'content', None)
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get('type') == 'text':
                        parts.append(part.get('text', ''))
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts).strip()

        return str(content or "").strip()

    def __call__(self, prompt: str, *args, **kwargs) -> str:
        try:
            if self.enable_prompt_logging:
                logger.debug(f"LLM Prompt ({self.agent_context} | {self.model}):\n{prompt}")
            estimated_prompt_tokens = self.estimate_tokens(prompt)
            if self.enable_prompt_logging:
                logger.debug(
                    f"Token usage estimate ({self.agent_context}): ~{estimated_prompt_tokens} prompt tokens"
                )

            messages = [{"role": "user", "content": prompt}]
            response_format = {"type": "json_object"} if self.json_mode else None
            if self.json_mode:
                api_config = self._load_api_config()
                provider_cfg = api_config.get('providers', {}).get('openai', {}) if api_config else {}
                json_instruction = provider_cfg.get('json_instruction', 'Return valid JSON only.')
                messages[0]["content"] = f"{prompt}\n\n{json_instruction}"

            response = None
            last_exception = None
            max_rotations = 3
            max_attempts = len(self.api_keys) * max_rotations
            attempts = 0
            
            while attempts < max_attempts:
                try:
                    response = self.execute_with_retry(
                        self._make_api_request,
                        messages=messages,
                        response_format=response_format,
                    )
                    break
                except Exception as e:
                    last_exception = e
                    attempts += 1
                    
                    is_rate_limit = self._is_rate_limit_error(e)
                    
                    if is_rate_limit and attempts < max_attempts:
                        rotation = (attempts // len(self.api_keys)) + 1
                        logger.warning(f"Rate limit hit on API key #{self.current_key_index + 1} (rotation {rotation}/{max_rotations}). Switching to next key...")
                        self._switch_to_next_api_key()
                        continue
                    
                    raise
            
            if response is None and last_exception:
                raise last_exception

            content = self._extract_message_content(response)
            usage_info = getattr(response, 'usage', None)
            input_tokens = getattr(usage_info, 'prompt_tokens', None) if usage_info else None
            output_tokens = getattr(usage_info, 'completion_tokens', None) if usage_info else None

            usage_dict: Dict[str, Any] = {}
            if usage_info:
                usage_dict = {
                    "prompt_tokens": getattr(usage_info, 'prompt_tokens', None),
                    "completion_tokens": getattr(usage_info, 'completion_tokens', None),
                    "total_tokens": getattr(usage_info, 'total_tokens', None),
                }

            self.track_usage(
                prompt,
                content,
                input_tokens,
                output_tokens,
                api_usage=usage_dict,
            )

            logger.debug(f"LLM Response ({self.agent_context} | {self.model}):\n{content}")

            if usage_info:
                prompt_tokens = usage_dict.get('prompt_tokens') or 0
                completion_tokens = usage_dict.get('completion_tokens') or 0
                total_tokens = usage_dict.get('total_tokens') or (prompt_tokens + completion_tokens)
                logger.debug(
                    f"Token usage ({self.agent_context}): "
                    f"{prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total"
                )

            return content.strip()

        except Exception as exc:
            return self.handle_api_error(exc)
