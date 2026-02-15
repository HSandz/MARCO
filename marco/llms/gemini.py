from loguru import logger
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

import google.generativeai as genai
from typing import Any

from marco.llms.basellm import BaseLLM

class GeminiLLM(BaseLLM):
    def __init__(self, model: str = 'gemini-2.0-flash-001', json_mode: bool = False, agent_context: str = None, config_file: str = 'config/api-config.json', *args, **kwargs):
        super().__init__()

        self.model = model
        self.json_mode = json_mode
        self.agent_context = agent_context or "Unknown"
        self.config_file = config_file
        self.max_tokens: int = kwargs.get('max_tokens', 2048)
        self.temperature: float = kwargs.get('temperature', 0)
        self.top_p: float = kwargs.get('top_p', 0.95)
        self.top_k: int = kwargs.get('top_k', 64)
        self.max_retries: int = 100
        self.retry_delay_base: int = 1
        self.enable_prompt_logging: bool = kwargs.get('enable_prompt_logging', True)

        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        if self.api_keys:
            self._configure_api_key(self.api_keys[0])
        
        if 'gemini-2.5-flash' in model:
            self.max_context_length = 1048576
        elif 'gemini-2.0-flash-lite' in model:
            self.max_context_length = 1048576
        elif 'gemini-2.0-flash' in model:
            self.max_context_length = 1048576
        else:
            self.max_context_length = 1048576
        
        generation_config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_tokens,
        }
        
        if json_mode:
            generation_config["response_mime_type"] = "application/json"
            logger.info("Using JSON mode for Gemini API.")
        
        self.client = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )

    def _load_api_keys(self) -> list:
        try:
            from marco.utils import read_json
            api_config = read_json(self.config_file)

            if 'providers' in api_config and 'gemini' in api_config['providers']:
                provider_cfg = api_config['providers']['gemini']

                keys = provider_cfg.get('api_keys', [])
                if keys and isinstance(keys, list):
                    return keys

                key = provider_cfg.get('api_key', '')
                if key:
                    return [key]
        except Exception as e:
            logger.warning(f"Failed to load Gemini API keys from {self.config_file}: {e}")

        return []
    
    def _configure_api_key(self, api_key: str):
        if api_key:
            genai.configure(api_key=api_key)
            logger.info(f"Configured Gemini API with key: {api_key[:6]}...{api_key[-4:]}")
    
    def _switch_to_next_api_key(self) -> bool:
        if len(self.api_keys) <= 1:
            return False
        
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        api_key = self.api_keys[self.current_key_index]
        self._configure_api_key(api_key)
        
        logger.info(f"Switched to API key #{self.current_key_index + 1} (total: {len(self.api_keys)})")
        return True
    
    def _is_rate_limit_error(self, exception: Exception) -> bool:
        try:
            from google.api_core import exceptions as google_exceptions
            if isinstance(exception, (google_exceptions.TooManyRequests, google_exceptions.ResourceExhausted)):
                return True
        except ImportError:
            pass
        
        error_str = str(exception).lower()
        if 'rate limit' in error_str or 'quota' in error_str or '429' in error_str:
            return True
        
        return False

    @property
    def tokens_limit(self) -> int:
        return self.max_context_length
    
    def _make_api_request(self, prompt_text: str):
        if not isinstance(prompt_text, str):
            prompt_text = str(prompt_text)
        
        try:
            response = self.client.generate_content(prompt_text)
            return response
        except Exception as e:
            logger.error(f"Gemini API error details:")
            logger.error(f"  - Error type: {type(e).__name__}")
            logger.error(f"  - Error message: {str(e)}")
            logger.error(f"  - Model: {self.model}")
            logger.error(f"  - Prompt length: {len(prompt_text) if isinstance(prompt_text, str) else 'N/A'}")
            raise
    
    def _get_retriable_errors(self) -> tuple:
        base_errors = super()._get_retriable_errors()
        
        try:
            from google.api_core import exceptions as google_exceptions
            return base_errors + (
                google_exceptions.ServiceUnavailable,
                google_exceptions.InternalServerError,
                google_exceptions.TooManyRequests,
                google_exceptions.DeadlineExceeded,
                google_exceptions.ResourceExhausted,
            )
        except ImportError:
            return base_errors
    
    def _should_retry_response(self, response: Any) -> bool:
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                finish_reason = getattr(candidate, 'finish_reason', None)
                if finish_reason == 5:
                    return True
        
        return False

    def __call__(self, prompt: str, *args, **kwargs) -> str:
        try:
            if self.enable_prompt_logging:
                logger.debug(f"LLM Prompt ({self.agent_context} → {self.model}):\n{prompt}")
            
            estimated_prompt_tokens = self.estimate_tokens(prompt)
            if self.enable_prompt_logging:
                logger.debug(f"Token Usage ({self.agent_context}): ~{estimated_prompt_tokens} prompt tokens estimated")
            
            messages = [{"role": "user", "content": prompt}]
            
            if self.json_mode:
                messages[0]["content"] = f"{prompt}\n\nPlease respond with valid JSON only."
                actual_prompt = messages[0]["content"]
            else:
                actual_prompt = prompt

            response = None
            last_exception = None
            max_rotations = 5
            max_attempts = len(self.api_keys) * max_rotations
            attempts = 0
            
            while attempts < max_attempts:
                try:
                    response = self.execute_with_retry(
                        self._make_api_request,
                        prompt_text=actual_prompt
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
            
            content = ""
            finish_reason_num = None
            
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    finish_reason_num = getattr(candidate, 'finish_reason', None)
                    if finish_reason_num == 2:
                        logger.warning(f"Response truncated: finish_reason=2 (MAX_TOKENS). Current limit: {self.max_tokens} tokens.")
            
            try:
                if hasattr(response, 'text') and response.text:
                    try:
                        content = response.text.strip()
                        logger.debug(f"Successfully extracted {len(content)} chars using response.text")
                    except (ValueError, RuntimeError, AttributeError):
                        pass
                
                if not content and hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            parts_text = []
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    parts_text.append(part.text)
                            if parts_text:
                                content = ' '.join(parts_text).strip()
                                if content:
                                    logger.debug(f"Extracted {len(content)} chars from candidate.content.parts")
                                    break
                
                if not content and hasattr(response, 'parts') and response.parts:
                    parts_text = []
                    for part in response.parts:
                        if hasattr(part, 'text') and part.text:
                            parts_text.append(part.text)
                    if parts_text:
                        content = ' '.join(parts_text).strip()
                        logger.debug(f"Extracted {len(content)} chars from response.parts")
                
                if not content and hasattr(response, 'candidates') and response.candidates:
                    try:
                        for candidate in response.candidates:
                            if hasattr(candidate, 'text'):
                                candidate_text = candidate.text.strip()
                                if candidate_text:
                                    content = candidate_text
                                    logger.debug(f"Extracted {len(content)} chars from candidate.text")
                                    break
                    except (ValueError, RuntimeError, AttributeError):
                        pass
            
            except Exception as e:
                logger.debug(f"Error during text extraction methods: {type(e).__name__}: {str(e)[:100]}")
            
            if not content:
                logger.warning(f"Could not extract text from Gemini response.")
                if hasattr(response, 'prompt_feedback'):
                    logger.warning(f"Prompt feedback: {response.prompt_feedback}")
                if hasattr(response, 'candidates') and response.candidates:
                    for i, candidate in enumerate(response.candidates):
                        reason = getattr(candidate, 'finish_reason', None)
                        logger.warning(f"Candidate {i}: finish_reason={reason}")
                        if reason == 2:
                            logger.error(f"Response truncated due to MAX_TOKENS limit ({self.max_tokens}). Try increasing max_tokens or reducing prompt size.")
            
            if content:
                
                input_tokens = None
                output_tokens = None
                
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    input_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
                    output_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)
                
                self.track_usage(
                    prompt,
                    content, 
                    input_tokens, 
                    output_tokens,
                    api_usage=({'prompt_tokens': input_tokens, 'completion_tokens': output_tokens} if input_tokens or output_tokens else {})
                )
                
                logger.debug(f"LLM Response ({self.agent_context} → {self.model}):\n{content}")
                
                if input_tokens and output_tokens:
                    total_tokens = input_tokens + output_tokens
                    logger.debug(f"Token Usage ({self.agent_context}): {input_tokens} prompt + {output_tokens} completion = {total_tokens} total tokens")
                else:
                    estimated_input = self.estimate_tokens(prompt)
                    estimated_output = self.estimate_tokens(content)
                    estimated_total = estimated_input + estimated_output
                    logger.debug(f"Token Usage ({self.agent_context}): ~{estimated_input} prompt + ~{estimated_output} completion = ~{estimated_total} total tokens (estimated)")
                
                logger.debug(f"Cumulative Usage ({self.agent_context}): {self.total_input_tokens + (input_tokens or 0)} total tokens across {self.api_calls + 1} calls")
                
                return content
            else:
                logger.warning("Empty response from Gemini API")
                return ""
        
        except Exception as e:
            return self.handle_api_error(e)
