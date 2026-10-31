import requests
import json
from loguru import logger
from typing import Any, Dict

from marco.llms.basellm import BaseLLM

class OllamaLLM(BaseLLM):
    def __init__(self, model: str = 'llama3.2:1b', base_url: str = 'http://localhost:11434', json_mode: bool = False, agent_context: str = None, config_file: str = 'config/api-config.json', *args, **kwargs):
        super().__init__()
        
        self.model = model
        self.json_mode = json_mode
        self.agent_context = agent_context or "Unknown"
        self.config_file = config_file
        self.max_tokens: int = kwargs.get('max_tokens', 1024)
        self.temperature: float = kwargs.get('temperature', 0.7)
        self.top_p: float = kwargs.get('top_p', 0.9)
        self.top_k: int = kwargs.get('top_k', 40)
        self.timeout: int = kwargs.get('timeout', 300)
        self.enable_prompt_logging: bool = kwargs.get('enable_prompt_logging', True)
        
        try:
            from marco.utils import read_json
            api_config = read_json(self.config_file)
            if 'providers' in api_config and 'ollama' in api_config['providers']:
                provider_cfg = api_config['providers']['ollama']
                if provider_cfg.get('base_url'):
                    self.base_url = provider_cfg['base_url'].rstrip('/')
                else:
                    raise ValueError(f"Missing 'base_url' for Ollama provider in {self.config_file}!")
            else:
                raise ValueError(f"Missing Ollama provider config in {self.config_file}!")
        except Exception as e:
            raise RuntimeError(f"Failed to load Ollama base_url from config: {e}")
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        
        model_context_lengths = {
            'llama3.2': 131072,
            'llama3.2:1b': 131072,
            'llama3.2:3b': 131072,
        }
        
        self.max_context_length = model_context_lengths.get(model, 131072)
        
        self.headers = {
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized Ollama LLM with model: {model} at {base_url}")

    @property
    def tokens_limit(self) -> int:
        return self.max_context_length
    
    def _make_api_request(
        self,
        payload: Dict[str, Any],
        timeout: int = 300
    ) -> requests.Response:
        return requests.post(
            self.generate_url,
            headers=self.headers,
            json=payload,
            timeout=timeout
        )

    def _check_ollama_server(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.model in model_names:
                return True
            
            for model_name in model_names:
                if self.model in model_name or model_name.startswith(self.model):
                    logger.info(f"Found model match: {model_name} for requested {self.model}")
                    return True
            
            logger.warning(f"Model {self.model} not found in Ollama. Available models: {model_names}")
            logger.info(f"You can pull the model by running: ollama pull {self.model}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server at {self.base_url}: {e}")
            logger.info("Make sure Ollama is installed and running. Visit https://ollama.ai for installation instructions.")
            return False

    def __call__(self, prompt: str, *args, **kwargs) -> str:
        try:
            if not self._check_ollama_server():
                error_msg = f"Ollama server not available or model {self.model} not found"
                logger.error(error_msg)
                return f"Error: {error_msg}"
            
            if self.enable_prompt_logging:
                logger.debug(f"LLM Prompt ({self.agent_context} → {self.model}):\n{prompt}")
            
            estimated_prompt_tokens = self.estimate_tokens(prompt)
            if self.enable_prompt_logging:
                logger.debug(f"Token Usage ({self.agent_context}): ~{estimated_prompt_tokens} prompt tokens estimated")
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "num_predict": self.max_tokens
                }
            }
            
            if self.json_mode:
                payload["format"] = "json"
                payload["prompt"] = f"{prompt}\n\nPlease respond with valid JSON only."
            
            response = self.execute_with_retry(
                self._make_api_request,
                payload=payload,
                timeout=self.timeout
            )
            
            logger.debug(f"Ollama API response: status={response.status_code}, length={len(response.text)} chars")
            
            if len(response.text) > 500000:
                logger.warning(f"Very large response from Ollama: {len(response.text)} chars")
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if 'application/json' not in content_type:
                    logger.warning(f"Unexpected content-type from Ollama: {content_type}")
                
                try:
                    result = response.json()
                except json.JSONDecodeError as json_err:
                    response_text = response.text
                    logger.error(f"JSON decode error in Ollama response:")
                    logger.error(f"  Error: {json_err}")
                    logger.error(f"  Response length: {len(response_text)} chars")
                    logger.error(f"  Response preview (first 500 chars): {response_text[:500]}")
                    logger.error(f"  Response preview (last 500 chars): {response_text[-500:]}")
                    
                    import re
                    content_match = re.search(r'"response":\s*"([^"]*)"', response_text)
                    if content_match:
                        logger.warning("Attempting to extract content manually from malformed JSON")
                        content = content_match.group(1)
                        
                        self.track_usage(
                            prompt, 
                            content, 
                            None,
                            None
                        )
                        
                        logger.debug(f"LLM Response ({self.agent_context} → {self.model}):\n{content}")
                        return content
                    
                    return f"Error: INVALID_JSON - Failed to parse Ollama response"
                
                if 'response' in result:
                    content = result['response'].strip()
                    
                    input_tokens = None
                    output_tokens = None
                    
                    if 'prompt_eval_count' in result:
                        input_tokens = result['prompt_eval_count']
                    if 'eval_count' in result:
                        output_tokens = result['eval_count']
                    
                    self.track_usage(
                        prompt, 
                        content, 
                        input_tokens, 
                        output_tokens,
                        api_usage=result
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
                    
                    logger.debug(f"Cumulative Usage ({self.agent_context}): {self.total_input_tokens + self.total_output_tokens} total tokens across {self.api_calls} calls")
                    
                    return content
                else:
                    logger.error(f"No 'response' field in Ollama response: {result}")
                    return "Error: Invalid response format from Ollama"
                    
            else:
                error_msg = f"Ollama API request failed with status {response.status_code}"
                try:
                    error_detail = response.json()
                    if 'error' in error_detail:
                        error_msg += f": {error_detail['error']}"
                except:
                    error_msg += f": {response.text[:200]}"
                
                logger.error(error_msg)
                return self.handle_api_error(Exception(error_msg))
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return f"Error: INVALID_JSON - Failed to parse API response"
        
        except Exception as e:
            return self.handle_api_error(e)

    def list_models(self) -> list:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            else:
                logger.error(f"Failed to list Ollama models: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error listing Ollama models: {e}")
            return []

    def pull_model(self, model: str = None) -> bool:
        if model is None:
            model = self.model
            
        try:
            payload = {"name": model}
            response = requests.post(
                f"{self.base_url}/api/pull",
                headers=self.headers,
                json=payload,
                timeout=1800
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully pulled model {model}")
                return True
            else:
                logger.error(f"Failed to pull model {model}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling model {model}: {e}")
            return False
