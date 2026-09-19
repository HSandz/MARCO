import json
from loguru import logger
from typing import Any, Dict, Optional
import torch

from marco.llms.basellm import BaseLLM

class HuggingFaceLLM(BaseLLM):
    def __init__(self, model: str = 'meta-llama/Llama-3.2-3B-Instruct', api_key: str = '', json_mode: bool = False, agent_context: str = None, config_file: str = 'config/api-config.json', *args, **kwargs):
        super().__init__()

        self.model = model
        self.json_mode = json_mode
        self.agent_context = agent_context or "Unknown"
        self.config_file = config_file
        self.max_tokens: int = kwargs.get('max_tokens', 1024)
        self.temperature: float = kwargs.get('temperature', 0.7)
        self.top_p: float = kwargs.get('top_p', 0.95)
        self.top_k: int = kwargs.get('top_k', 50)
        self.device: str = kwargs.get('device', 'auto')
        self.enable_prompt_logging: bool = kwargs.get('enable_prompt_logging', True)
        
        self.hf_token: Optional[str] = None
        if api_key:
            self.hf_token = api_key
        else:
            import os
            self.hf_token = os.getenv('HUGGINGFACE_TOKEN', '') or os.getenv('HF_TOKEN', '')
            if not self.hf_token:
                try:
                    from marco.utils import read_json
                    api_config = read_json(self.config_file)
                    
                    if 'providers' in api_config and 'huggingface' in api_config['providers']:
                        self.hf_token = api_config['providers']['huggingface'].get('api_key', '') or \
                                       api_config['providers']['huggingface'].get('token', '')
                except:
                    pass
        
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        model_context_lengths = {
            'meta-llama/Llama-3.2-3B-Instruct': 131072,
            'meta-llama/Llama-3.2-1B-Instruct': 131072,
            'meta-llama/Llama-3.1-8B-Instruct': 131072,
            'mistralai/Mistral-7B-Instruct-v0.3': 32768,
            'google/gemma-2-2b-it': 8192,
            'Qwen/Qwen2.5-7B-Instruct': 131072,
        }
        
        self.max_context_length = model_context_lengths.get(model, 8192)
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            if self.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model,
                token=self.hf_token,
                trust_remote_code=True
            )
            
            model_kwargs = {
                'token': self.hf_token,
                'trust_remote_code': True,
                'dtype': torch.float16 if self.device == 'cuda' else torch.float32,
            }
            
            if self.device == 'cuda':
                if self.num_gpus > 1:
                    max_memory_per_gpu = self._get_max_memory_per_gpu()
                    model_kwargs['max_memory'] = max_memory_per_gpu
                else:
                    model_kwargs['device_map'] = 'auto'
            
            self.model_pipeline = AutoModelForCausalLM.from_pretrained(
                model,
                **model_kwargs
            )
            
            if self.device == 'cpu':
                self.model_pipeline = self.model_pipeline.to('cpu')
            
        except ImportError as e:
            raise ImportError(
                "transformers and torch packages are required for local HuggingFace inference. "
                "Install them with: pip install transformers torch"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load model {model}: {e}")
            raise RuntimeError(f"Failed to load HuggingFace model: {e}") from e

    @property
    def tokens_limit(self) -> int:
        return self.max_context_length
    
    def _get_max_memory_per_gpu(self) -> Dict[int, str]:
        max_memory = {}
        for i in range(self.num_gpus):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            max_memory_bytes = int(total_memory * 0.9)
            max_memory_gb = max_memory_bytes / (1024 ** 3)
            max_memory[i] = f"{max_memory_gb:.1f}GB"
        return max_memory
    
    def _generate_text(self, prompt: str) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            if self.device == 'cuda':
                inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model_pipeline.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    do_sample=True if self.temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            raise

    def __call__(self, prompt: str, *args, **kwargs) -> str:
        try:
            if self.enable_prompt_logging:
                logger.debug(f"LLM Prompt ({self.agent_context} → {self.model}):\n{prompt}")
            
            estimated_prompt_tokens = self.estimate_tokens(prompt)
            if self.enable_prompt_logging:
                logger.debug(f"Token Usage ({self.agent_context}): ~{estimated_prompt_tokens} prompt tokens estimated")
            
            actual_prompt = prompt
            if self.json_mode:
                actual_prompt = f"{prompt}\n\nPlease respond with valid JSON only."
            
            content = self._generate_text(actual_prompt)
            
            if not content:
                logger.warning("Empty response from local model")
                return ""
            
            input_tokens = len(self.tokenizer.encode(actual_prompt))
            output_tokens = len(self.tokenizer.encode(content))
            
            self.track_usage(
                prompt, 
                content, 
                input_tokens, 
                output_tokens,
                api_usage={'local_inference': True}
            )
                
            logger.debug(f"LLM Response ({self.agent_context} → {self.model}):\n{content}")
            
            total_tokens = input_tokens + output_tokens
            logger.debug(f"Token Usage ({self.agent_context}): {input_tokens} prompt + {output_tokens} completion = {total_tokens} total tokens")
            
            logger.debug(f"Cumulative Usage ({self.agent_context}): {self.total_input_tokens} total tokens across {self.api_calls} calls")
            
            return content
        
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return self.handle_api_error(e)
    
    def get_model_info(self) -> Dict[str, Any]:
        try:
            import requests
            info_url = f"https://huggingface.co/api/models/{self.model}"
            response = requests.get(info_url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get model info: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}
