from abc import ABC, abstractmethod
from typing import Dict, Any, Callable
from loguru import logger
from contextlib import contextmanager
import signal
import time
import requests

class BaseLLM(ABC):
    def __init__(self) -> None:
        self.model: str
        self.max_tokens: int
        self.max_context_length: int
        self.json_mode: bool
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.api_calls: int = 0
        self.call_history: list = []
        self.max_retries: int = 3
        self.retry_delay_base: int = 1
        self.request_timeout_seconds: int = 120
        self.enable_prompt_logging: bool = True

    @property
    def tokens_limit(self) -> int:
        return self.max_context_length - 2 * self.max_tokens - 50
    
    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def track_usage(self, prompt: str, response: str, input_tokens: int = None, output_tokens: int = None, **kwargs) -> None:
        api_usage = kwargs.get('api_usage', {})
        estimated_input = False
        estimated_output = False
        
        if input_tokens is None:
            input_tokens = self.estimate_tokens(prompt)
            estimated_input = True
            
        if output_tokens is None:
            output_tokens = self.estimate_tokens(response)
            estimated_output = True
            
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.api_calls += 1
        
        call_info = {
            'call_id': self.api_calls,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'model': self.model,
            'prompt_length': len(prompt),
            'response_length': len(response),
            'estimated_input': estimated_input,
            'estimated_output': estimated_output,
            'api_usage': api_usage
        }
        
        self.call_history.append(call_info)
        
        if estimated_input or estimated_output:
            logger.debug(f"Token tracking for call {self.api_calls}: estimated_input={estimated_input}, estimated_output={estimated_output}")
        else:
            logger.debug(f"Token tracking for call {self.api_calls}: using actual API token counts")

    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            'model': self.model,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'api_calls': self.api_calls,
            'avg_input_tokens': self.total_input_tokens / max(self.api_calls, 1),
            'avg_output_tokens': self.total_output_tokens / max(self.api_calls, 1),
        }
    
    def get_detailed_usage_stats(self) -> Dict[str, Any]:
        api_calls_with_actual = sum(1 for call in self.call_history 
                                  if not call.get('estimated_input', True) and not call.get('estimated_output', True))
        api_calls_with_estimates = self.api_calls - api_calls_with_actual
        
        return {
            'model': self.model,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'api_calls': self.api_calls,
            'api_calls_with_actual_counts': api_calls_with_actual,
            'api_calls_with_estimated_counts': api_calls_with_estimates,
            'avg_input_tokens': self.total_input_tokens / max(self.api_calls, 1),
            'avg_output_tokens': self.total_output_tokens / max(self.api_calls, 1),
            'accuracy_rate': api_calls_with_actual / max(self.api_calls, 1),
        }

    def reset_usage_stats(self) -> None:
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_calls = 0
        self.call_history = []

    def execute_with_retry(
        self,
        api_call_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        RETRIABLE_ERRORS = self._get_retriable_errors()
        timeout_seconds = kwargs.pop('timeout_seconds', getattr(self, 'request_timeout_seconds', 120))
        
        last_exception = None

        @contextmanager
        def _timeout_guard(seconds: int):
            if seconds <= 0 or not hasattr(signal, 'SIGALRM'):
                yield
                return

            def _handle_timeout(signum, frame):
                raise TimeoutError(f"Request exceeded time limit of {seconds} seconds")

            previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
            previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)

            try:
                yield
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)
                if previous_timer[0] > 0:
                    signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
        
        for attempt in range(self.max_retries):
            try:
                with _timeout_guard(timeout_seconds):
                    result = api_call_func(*args, **kwargs)
                
                if self._should_retry_response(result):
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay_base
                        logger.warning(
                            f"Server error in response (attempt {attempt + 1}/{self.max_retries}). "
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue
                
                return result
                
            except tuple(RETRIABLE_ERRORS) as e:
                last_exception = e
                error_type = type(e).__name__
                
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay_base
                    
                    logger.warning(
                        f"Transient error (attempt {attempt + 1}/{self.max_retries}): "
                        f"{error_type}: {str(e)[:100]}... Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Error persisted after {self.max_retries} attempts: "
                        f"{error_type}: {str(e)[:200]}"
                    )
                    raise
            
            except Exception as e:
                if self._is_retriable_exception(e):
                    last_exception = e
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay_base
                        logger.warning(
                            f"Retriable error (attempt {attempt + 1}/{self.max_retries}): "
                            f"{type(e).__name__}: {str(e)[:100]}... Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue
                
                logger.error(f"Non-retriable error: {type(e).__name__}: {str(e)[:200]}")
                raise
        
        if last_exception:
            raise last_exception
        
        return None
    
    def _get_retriable_errors(self) -> tuple:
        try:
            import httpx

            return (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                httpx.TransportError,
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.RemoteProtocolError,
                ConnectionResetError,
                ConnectionAbortedError,
                BrokenPipeError,
                TimeoutError,
                OSError,
            )
        except Exception:
            return (
                ConnectionResetError,
                ConnectionAbortedError,
                BrokenPipeError,
                TimeoutError,
                OSError,
            )
    
    def _should_retry_response(self, response: Any) -> bool:
        if hasattr(response, 'status_code'):
            return response.status_code >= 500 or response.status_code == 429
        
        return False
    
    def _is_retriable_exception(self, exception: Exception) -> bool:
        try:
            import httpx

            if isinstance(exception, (httpx.TransportError, httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
                return True
        except Exception:
            pass

        if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
            status_code = exception.response.status_code
            return status_code >= 500 or status_code == 429
        
        return False
    
    def classify_error(self, exception: Exception) -> tuple[str, str]:
        error_type = type(exception).__name__
        error_str = str(exception)
        
        if isinstance(exception, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return ("CONNECTION_ERROR", f"Network connection lost. {error_type}: {error_str[:100]}")
        
        if isinstance(exception, (TimeoutError,)):
            return ("TIMEOUT", f"Request exceeded time limit. {error_type}")
        
        try:
            import httpx

            if isinstance(exception, httpx.TransportError):
                return ("CONNECTION_ERROR", f"Transport error while calling LLM. {error_type}: {error_str[:100]}")

            import requests
            
            if isinstance(exception, requests.exceptions.ConnectionError):
                return ("CONNECTION_ERROR", f"Network connection error. {error_type}: {error_str[:100]}")
            
            if isinstance(exception, requests.exceptions.Timeout):
                return ("TIMEOUT", "Request timed out after retries")
            
            if isinstance(exception, requests.exceptions.HTTPError):
                status_code = exception.response.status_code if hasattr(exception, 'response') else 'unknown'
                return ("HTTP_ERROR", f"HTTP_{status_code} - Server returned error")
            
            if isinstance(exception, requests.exceptions.RequestException):
                return ("NETWORK_ERROR", f"Request error. {error_type}: {error_str[:100]}")
                
        except ImportError:
            pass
        
        try:
            import json
            if isinstance(exception, json.JSONDecodeError):
                return ("INVALID_JSON", "Failed to parse API response")
        except ImportError:
            pass
        
        return ("UNEXPECTED", f"{error_type}: {error_str[:100]}")
    
    def handle_api_error(self, exception: Exception) -> str:
        error_code, error_message = self.classify_error(exception)
        logger.error(f"{error_code}: {error_message}")
        return f"Error: {error_code} - {error_message}"

    @abstractmethod
    def __call__(self, prompt: str, *args, **kwargs) -> str:
        raise NotImplementedError("BaseLLM.__call__() not implemented")
