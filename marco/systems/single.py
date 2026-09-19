import json
from typing import Any, Optional
from loguru import logger

from marco.systems.base import System
from marco.utils import parse_answer


class SingleSystem(System):

    def __init__(self, task: str, config_path: str, leak: bool = False, dataset: Optional[str] = None, *args, **kwargs) -> None:
        self.llm = None
        self.raw_response = ''
        super().__init__(task, config_path, leak, dataset, *args, **kwargs)

    @staticmethod
    def supported_tasks() -> list[str]:
        return ['rp', 'sr']

    def init(self, *args, **kwargs) -> None:
        llm_config_path = self.config.get('llm_config_path')
        llm_config = self.config.get('llm')

        if llm_config is None:
            if not llm_config_path:
                raise ValueError("Single system config must define either 'llm' or 'llm_config_path'.")
            self.llm = self.build_llm(config_path=llm_config_path, agent_context='SingleSystem')
        else:
            if not isinstance(llm_config, dict):
                raise ValueError("Single system 'llm' config must be a dictionary.")
            self.llm = self.build_llm(config=llm_config, agent_context='SingleSystem')

        self.max_step = self.config.get('max_step', 1)
        self._current_sample_idx = -1
        self._current_user_id = -1

    def reset(self, clear: bool = False, *args, **kwargs) -> None:
        super().reset(clear, *args, **kwargs)
        self.raw_response = ''

    def _parse_response(self, response: str) -> Any:
        response = (response or '').strip()

        if not response:
            return parse_answer(self.task, response, gt_answer=self.gt_answer, n_candidate=getattr(self, 'n_candidate', None), json_mode=False)['answer']

        try:
            parsed_json = json.loads(response)
        except Exception:
            parsed_json = None

        if self.task == 'sr':
            if isinstance(parsed_json, dict):
                ranked_items = parsed_json.get('ranked_items') or parsed_json.get('items') or parsed_json.get('answer')
                if isinstance(ranked_items, list):
                    return [int(item) for item in ranked_items]
            elif isinstance(parsed_json, list):
                return [int(item) for item in parsed_json]

        elif self.task == 'rp':
            if isinstance(parsed_json, dict):
                predicted_rating = parsed_json.get('predicted_rating')
                if predicted_rating is None:
                    predicted_rating = parsed_json.get('rating') or parsed_json.get('answer')
                if predicted_rating is not None:
                    return float(predicted_rating)

        parsed = parse_answer(self.task, response, gt_answer=self.gt_answer, n_candidate=getattr(self, 'n_candidate', None), json_mode=False)
        return parsed['answer']

    def forward(self, *args, **kwargs) -> Any:
        try:
            if self.llm is None:
                raise ValueError('Single system LLM is not initialized.')

            response = self.llm(self.input)
            self.raw_response = response

            answer = self._parse_response(response)
            self.finish(answer)
            logger.debug(f'Single system raw response: {response}')
            return self.answer
        except Exception as e:
            logger.error(f'Error in SingleSystem forward: {e}')
            raise