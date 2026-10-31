import requests
import json
import time
from loguru import logger
from typing import Any, Dict

from marco.llms.basellm import BaseLLM

class OpenRouterLLM(BaseLLM):
    def __init__(self, model: str = 'google/gemini-2.0-flash-001', api_key: str = '', json_mode: bool = False, agent_context: str = None, config_file: str = 'config/api-config.json', *args, **kwargs):
        super().__init__()

        self.model = model
        self.json_mode = json_mode
        self.agent_context = agent_context or "Unknown"
        self.config_file = config_file
        self.max_tokens: int = kwargs.get('max_tokens', 1024)
        self.temperature: float = kwargs.get('temperature', 0.7)
        self.top_p: float = kwargs.get('top_p', 0.95)
        self.enable_prompt_logging: bool = kwargs.get('enable_prompt_logging', True)
        
        self.api_keys = []
        self.current_key_index = 0
        
        if api_key:
            self.api_keys = [api_key]
        else:
            import os
            env_key = os.getenv('OPENROUTER_API_KEY', '')
            if env_key:
                self.api_keys = [env_key]
            else:
                try:
                    from marco.utils import read_json
                    api_config = read_json(self.config_file)
                    
                    if 'providers' in api_config and 'openrouter' in api_config['providers']:
                        provider_cfg = api_config['providers']['openrouter']
                        keys = provider_cfg.get('api_keys', [])
                        if keys and isinstance(keys, list):
                            self.api_keys = keys
                        else:
                            single_key = provider_cfg.get('api_key', '')
                            if single_key:
                                self.api_keys = [single_key]
                    
                    if not self.api_keys:
                        if api_config.get('provider') == 'openrouter':
                            legacy_key = api_config.get('api_key', '')
                            if legacy_key:
                                self.api_keys = [legacy_key]
                        if not self.api_keys:
                            legacy_key = api_config.get('openrouter_api_key', '')
                            if legacy_key:
                                self.api_keys = [legacy_key]
                except:
                    pass
        
        self.api_key = self.api_keys[0] if self.api_keys else ''
        
        if not self.api_key:
            logger.warning("OpenRouter API key not found. Please set it in config/api-config.json or as OPENROUTER_API_KEY environment variable.")
        else:
            logger.info(f"OpenRouter API key loaded: {self.api_key[:6]}...{self.api_key[-4:]} (total keys: {len(self.api_keys)})")
        
        model_context_lengths = {
            'google/gemini-2.5-flash-001': 1048576,
            'google/gemini-2.0-flash-001': 1048576,
            'google/gemini-2.0-flash-lite-001': 1048576,
        }
        
        self.max_context_length = model_context_lengths.get(model, 1048576)
        
        self._update_headers()
        
        try:
            from marco.utils import read_json
            api_config = read_json(self.config_file)
            if 'providers' in api_config and 'openrouter' in api_config['providers']:
                provider_cfg = api_config['providers']['openrouter']
                if provider_cfg.get('base_url'):
                    self.base_url = provider_cfg['base_url']
                else:
                    raise ValueError(f"Missing 'base_url' for OpenRouter provider in {self.config_file}!")
            else:
                raise ValueError(f"Missing OpenRouter provider config in {self.config_file}!")
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenRouter base_url from config: {e}")
        
        self.max_retries = kwargs.get('max_retries', 3)
        self.retry_delay_base = kwargs.get('retry_delay_base', 1)
        
        logger.info(f"Initialized OpenRouter LLM with model: {model}, max_retries: {self.max_retries}")

    def _update_headers(self):
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _switch_to_next_api_key(self) -> bool:
        if len(self.api_keys) <= 1:
            return False
        
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.api_key = self.api_keys[self.current_key_index]
        self._update_headers()
        
        logger.info(f"Switched to API key #{self.current_key_index + 1} (total: {len(self.api_keys)})")
        return True
    
    def _is_rate_limit_error(self, exception: Exception) -> bool:
        if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
            if exception.response.status_code == 429:
                return True
        
        if hasattr(exception, 'status_code') and exception.status_code == 429:
            return True
        
        return False

    @property
    def tokens_limit(self) -> int:
        return self.max_context_length
    
    def _make_api_request(
        self,
        payload: Dict[str, Any],
        timeout: int = 60
    ) -> requests.Response:
        return requests.post(
            self.base_url,
            headers=self.headers,
            json=payload,
            timeout=timeout,
            stream=False
        )

    def __call__(self, prompt: str, *args, **kwargs) -> str:
        try:
            if self.enable_prompt_logging:
                logger.debug(f"LLM Prompt ({self.agent_context} → {self.model}):\n{prompt}")
            
            estimated_prompt_tokens = self.estimate_tokens(prompt)
            if self.enable_prompt_logging:
                logger.debug(f"Token Usage ({self.agent_context}): ~{estimated_prompt_tokens} prompt tokens estimated")
            messages = [{"role": "user", "content": prompt}]
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": min(self.max_tokens, 4096),
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            
            if self.json_mode:
                payload["response_format"] = {"type": "json_object"}
                messages[0]["content"] = f"{prompt}\n\nPlease respond with valid JSON only."
            
            response = None
            last_exception = None
            max_rotations = 3
            max_attempts = len(self.api_keys) * max_rotations
            attempts = 0
            
            while attempts < max_attempts:
                try:
                    response = self.execute_with_retry(
                        self._make_api_request,
                        payload=payload,
                        timeout=60
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
            
            logger.debug(f"OpenRouter API response: status={response.status_code}, length={len(response.text)} chars")
            
            if len(response.text) > 500000:
                logger.warning(f"Very large response from OpenRouter: {len(response.text)} chars")
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if 'application/json' not in content_type:
                    logger.warning(f"Unexpected content-type from OpenRouter: {content_type}")
                
                try:
                    result = response.json()
                except json.JSONDecodeError as json_err:
                    response_text = response.text
                    logger.error(f"JSON decode error in OpenRouter response:")
                    logger.error(f"  Error: {json_err}")
                    logger.error(f"  Response length: {len(response_text)} chars")
                    logger.error(f"  Response preview (first 500 chars): {response_text[:500]}")
                    logger.error(f"  Response preview (last 500 chars): {response_text[-500:]}")
                    
                    import re
                    content_match = re.search(r'"content":\s*"([^"]*)"', response_text)
                    if content_match:
                        logger.warning("Attempting to extract content manually from malformed JSON")
                        content = content_match.group(1)
                        
                        self.track_usage(
                            prompt, 
                            content, 
                            None,
                            None,
                            api_usage={'error': 'json_parse_failed'}
                        )
                        
                        return content.strip()
                    else:
                        return f"Error: Invalid JSON response (could not extract content)"
                
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    
                    input_tokens = None
                    output_tokens = None
                    total_tokens = None
                    
                    if 'usage' in result:
                        usage = result['usage']
                        input_tokens = usage.get('prompt_tokens')
                        output_tokens = usage.get('completion_tokens')
                        total_tokens = usage.get('total_tokens')
                        
                        logger.debug(f"OpenRouter API usage: {input_tokens}+{output_tokens}={total_tokens} tokens")
                        
                        if input_tokens and output_tokens and total_tokens:
                            expected_total = input_tokens + output_tokens
                            if total_tokens != expected_total:
                                logger.warning(f"Token count mismatch: API reported total={total_tokens}, calculated={expected_total}")
                    else:
                        logger.warning("No usage information in OpenRouter API response - falling back to estimation")
                    
                    self.track_usage(
                        prompt, 
                        content, 
                        input_tokens, 
                        output_tokens,
                        api_usage=result.get('usage', {})
                    )
                    
                    logger.debug(f"LLM Response ({self.agent_context} → {self.model}):\n{content.strip()}")
                    
                    if input_tokens and output_tokens and total_tokens:
                        logger.debug(f"Token Usage ({self.agent_context}): {input_tokens} prompt + {output_tokens} completion = {total_tokens} total tokens")
                    else:
                        estimated_input = self.estimate_tokens(prompt)
                        estimated_output = self.estimate_tokens(content.strip())
                        estimated_total = estimated_input + estimated_output
                        logger.debug(f"Token Usage ({self.agent_context}): ~{estimated_input} prompt + ~{estimated_output} completion = ~{estimated_total} total tokens (estimated)")
                    
                    logger.debug(f"Cumulative Usage ({self.agent_context}): {self.total_input_tokens + (input_tokens or 0)} total tokens across {self.api_calls + 1} calls")
                    
                    return content.strip()
                else:
                    logger.warning("No choices found in OpenRouter API response")
                    return ""
            else:
                logger.error(f"OpenRouter API request failed with status {response.status_code}: {response.text}")
                return f"Error: HTTP {response.status_code}"
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return f"Error: INVALID_JSON - Failed to parse API response"
        
        except Exception as e:
            return self.handle_api_error(e)
