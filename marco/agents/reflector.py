from enum import Enum
from loguru import logger
from langchain.prompts import PromptTemplate

from marco.agents.base import Agent
from marco.utils import format_step, format_reflections, format_last_attempt, read_json, get_rm

class ReflectionStrategy(Enum):
    NONE = 'base'
    LAST_ATTEMPT = 'last_trial'
    REFLEXION = 'reflection'
    LAST_ATTEMPT_AND_REFLEXION = 'last_trial_and_reflection'

class Reflector(Agent):
    
    def __init__(self, config_path: str = None, config: dict = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if config is not None:
            agent_config = config
        else:
            assert config_path is not None, "Either config_path or config must be provided"
            agent_config = read_json(config_path)
        
        self.keep_reflections = get_rm(agent_config, 'keep_reflections', True)
        reflection_strategy = get_rm(agent_config, 'reflection_strategy', ReflectionStrategy.REFLEXION.value)
        self.llm = self.get_LLM(config=agent_config)
        self.json_mode = self.llm.json_mode
        self.enc = None
        for strategy in ReflectionStrategy:
            if strategy.value == reflection_strategy:
                self.reflection_strategy = strategy
                break
        assert self.reflection_strategy is not None, f'Unknown reflection strategy: {reflection_strategy}'
        self.reflections: list[str] = []
        self.reflections_str: str = ''

    @property
    def reflector_prompt(self) -> PromptTemplate:
        if self.json_mode:
            return self.prompts['reflect_prompt_json']
        else:
            return self.prompts['reflect_prompt']

    @property
    def reflect_examples(self) -> str:
        prompt_name = 'reflect_examples_json' if self.json_mode else 'reflect_examples'
        if prompt_name in self.prompts:
            return self.prompts[prompt_name]
        else:
            return ''

    def _build_reflector_prompt(self, input: str, scratchpad: str) -> str:
        return self.reflector_prompt.format(
            examples=self.reflect_examples,
            input=input,
            scratchpad=scratchpad
        )

    def _prompt_reflection(self, input: str, scratchpad: str) -> str:
        reflection_prompt = self._build_reflector_prompt(input, scratchpad)
        reflection_response = self.llm(reflection_prompt)
        if self.keep_reflections:
            self.reflection_input = reflection_prompt
            self.reflection_output = reflection_response
            if self.enc is None:
                input_length = len(self.reflection_input) // 4
                output_length = len(self.reflection_output) // 4
            else:
                input_length = len(self.enc.encode(self.reflection_input))
                output_length = len(self.enc.encode(self.reflection_output))
            logger.trace(f'Reflection input length: {input_length}')
            logger.trace(f"Reflection input: {self.reflection_input}")
            logger.trace(f'Reflection output length: {output_length}')
            if self.json_mode:
                self.system.log(f"[:violet[Reflection]]:\n- `{self.reflection_output}`", agent=self, logging=False)
            else:
                self.system.log(f"[:violet[Reflection]]:\n- {self.reflection_output}", agent=self, logging=False)
            logger.debug(f"Reflection output: {self.reflection_output}")
        return format_step(reflection_response)

    def forward(self, input: str, scratchpad: str, *args, **kwargs) -> str:
        logger.trace('Running Reflecion strategy...')
        if self.reflection_strategy == ReflectionStrategy.LAST_ATTEMPT:
            self.reflections = [scratchpad]
            self.reflections_str = format_last_attempt(input, scratchpad, self.prompts['last_trial_header'])
        elif self.reflection_strategy == ReflectionStrategy.REFLEXION:
            self.reflections.append(self._prompt_reflection(input=input, scratchpad=scratchpad))
            self.reflections_str = format_reflections(self.reflections, header=self.prompts['reflection_header'])
        elif self.reflection_strategy == ReflectionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str = format_last_attempt(input, scratchpad, self.prompts['last_trial_header'])
            self.reflections = self._prompt_reflection(input=input, scratchpad=scratchpad)
            self.reflections_str += format_reflections(self.reflections, header=self.prompts['reflection_last_trial_header'])
        elif self.reflection_strategy == ReflectionStrategy.NONE:
            self.reflections = []
            self.reflections_str = ''
        else:
            raise ValueError(f'Unknown reflection strategy: {self.reflection_strategy}')
        logger.trace(self.reflections_str)
        return self.reflections_str

    def reset(self) -> None:
        self.reflections = []
        self.reflections_str = ''
        if hasattr(self, 'reflection_input'):
            self.reflection_input = ''
        if hasattr(self, 'reflection_output'):
            self.reflection_output = ''
