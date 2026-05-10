import json
from abc import ABC, abstractmethod
from loguru import logger
from typing import Any, Optional, TYPE_CHECKING
from langchain.prompts import PromptTemplate

from marco.llms import BaseLLM, GeminiLLM, VertexAILLM, OpenRouterLLM, OpenAILLM, OllamaLLM, HuggingFaceLLM
from marco.tools import TOOL_MAP, Tool
from marco.utils import run_once, format_history, read_prompts, duration_tracker

if TYPE_CHECKING:
    from marco.systems import System

class Agent(ABC):
    def __init__(self, prompts: dict = dict(), prompt_config: Optional[str] = None, system: Optional['System'] = None, dataset: Optional[str] = None, *args, **kwargs) -> None:
        self.json_mode: bool
        self.system = system
        if prompt_config is not None:
            prompts = read_prompts(prompt_config)
        self.prompts = prompts
        if self.system is not None:
            for prompt_name, prompt_template in self.prompts.items():
                if isinstance(prompt_template, PromptTemplate) and 'task_type' in prompt_template.input_variables:
                    self.prompts[prompt_name] = prompt_template.partial(task_type=self.system.task_type)
        self.dataset = dataset
        self.api_config_path = kwargs.get('api_config_path', 'config/api-config.json')

    def observation(self, message: str, log_head: str = '') -> None:
        logger.debug(f'Observation: {message}')

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Agent.forward() not implemented")

    def reset(self) -> None:
        pass
    
    def track_execution(self) -> Any:
        agent_name = self.__class__.__name__
        return duration_tracker.track_agent_call(agent_name)
    
    def get_llm_instances(self) -> dict[str, BaseLLM]:
        llms = {}
        agent_name = self.__class__.__name__.lower()
        
        common_llm_attrs = ['llm', 'analyst', 'thought_llm', 'action_llm', 
                           'planner', 'solver', 'reflector']
        
        for attr_name in common_llm_attrs:
            if hasattr(self, attr_name):
                attr = getattr(self, attr_name)
                if isinstance(attr, BaseLLM):
                    if attr_name == 'llm':
                        key = agent_name
                    else:
                        clean_attr = attr_name.replace('_llm', '')
                        key = f"{agent_name}_{clean_attr}_llm"
                    llms[key] = attr
        
        if not llms:
            for attr_name in dir(self):
                if attr_name.startswith('_') or attr_name in ['get_LLM', 'get_llm_instances']:
                    continue
                try:
                    attr = getattr(self, attr_name)
                    if isinstance(attr, BaseLLM):
                        if attr_name == 'llm':
                            key = agent_name
                        else:
                            clean_attr = attr_name.replace('_llm', '')
                            key = f"{agent_name}_{clean_attr}"
                        llms[key] = attr
                except (AttributeError, RuntimeError):
                    continue
        
        return llms

    def get_LLM(self, config_path: Optional[str] = None, config: Optional[dict] = None) -> BaseLLM:
        if config is None:
            assert config_path is not None
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        config = config.copy()
        provider = config.get('provider') or config.get('model_type')
        if not provider:
            raise ValueError("Agent config must have either 'provider' or 'model_type' key")
        if 'provider' in config:
            del config['provider']
        if 'model_type' in config:
            del config['model_type']
        
        if 'model' not in config:
            raise ValueError("Agent config must have 'model' key")
        
        if 'agent_context' not in config:
            agent_name = self.__class__.__name__
            config['agent_context'] = agent_name
        
        if 'config_file' not in config:
            config['config_file'] = self.api_config_path
        
        if provider == 'gemini':
            return GeminiLLM(**config)
        elif provider == 'vertexai':
            return VertexAILLM(**config)
        elif provider == 'openrouter':
            return OpenRouterLLM(**config)
        elif provider == 'openai':
            return OpenAILLM(**config)
        elif provider == 'ollama':
            return OllamaLLM(**config)
        elif provider == 'huggingface':
            return HuggingFaceLLM(**config)
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported providers are 'gemini', 'vertexai', 'openrouter', 'openai', 'ollama', and 'huggingface'."
            )

class ToolAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tools: dict[str, Tool] = {}
        self._history = []
        self.max_turns: int = 6

    @run_once
    def validate_tools(self) -> None:
        required_tools = self.required_tools()
        for tool, tool_type in required_tools.items():
            assert tool in self.tools, f'Tool {tool} not found.'
            assert isinstance(self.tools[tool], tool_type), f'Tool {tool} must be an instance of {tool_type}.'

    @staticmethod
    @abstractmethod
    def required_tools() -> dict[str, type]:
        raise NotImplementedError("Agent.required_tools() not implemented")

    def get_tools(self, tool_config: dict[str, dict]):
        assert isinstance(tool_config, dict), 'Tool config must be a dictionary.'
        for tool_name, tool in tool_config.items():
            assert isinstance(tool, dict), 'Config of each tool must be a dictionary.'
            assert 'type' in tool, 'Tool type not found.'
            assert 'config_path' in tool, 'Tool config path not found.'
            tool_type = tool['type']
            if tool_type not in TOOL_MAP:
                raise NotImplementedError(f'Docstore {tool_type} not implemented.')
            config_path = tool['config_path']
            if self.dataset is not None:
                config_path = config_path.format(dataset=self.dataset)
            self.tools[tool_name] = TOOL_MAP[tool_type](config_path=config_path)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.validate_tools()
        self.reset()
        return self.forward(*args, **kwargs)

    @abstractmethod
    def invoke(self, argument: Any, json_mode: bool) -> str:
        raise NotImplementedError("ToolAgent.invoke() not implemented")

    def reset(self) -> None:
        self._history = []
        self.finished = False
        self.results = None
        if hasattr(self, 'queried_users'):
            self.queried_users.clear()
        if hasattr(self, 'queried_items'):
            self.queried_items.clear()
        if hasattr(self, 'gathered_info'):
            self.gathered_info.clear()
        for tool in self.tools.values():
            tool.reset()

    @property
    def history(self) -> str:
        return format_history(self._history)

    def finish(self, results: Any) -> str:
        self.results = results
        self.finished = True
        return str(self.results)

    def is_finished(self) -> bool:
        return self.finished or len(self._history) >= self.max_turns
