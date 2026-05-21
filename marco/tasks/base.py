from abc import ABC, abstractmethod
from argparse import ArgumentParser
from loguru import logger
from typing import Any
import os
import datetime
import json

from marco.utils import NumpyEncoder

class Task(ABC):
    def __init__(self):
        self.log_handler_id = None
        
    @staticmethod
    @abstractmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        raise NotImplementedError

    def __getattr__(self, __name: str) -> Any:
        if __name not in self.__dict__:
            return None
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{__name}'")
    
    def setup_task_logger(self, task: str, dataset: str, system: str, num_samples: int):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"{task}_{dataset}_{system}_{num_samples}_{timestamp}.log"
        result_filename = f"{task}_{dataset}_{system}_{num_samples}_{timestamp}.json"
        log_path = os.path.join("logs", log_filename)
        result_path = os.path.join("results", result_filename)

        os.makedirs("logs", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        self.log_handler_id = logger.add(log_path, level='INFO')
        self.log_path = log_path
        self.result_path = result_path
        logger.info(f"Task-specific log file: {log_path}")
        logger.info(f"Task-specific result file: {result_path}")
        
        return log_path

    def save_result_payload(self, payload: dict[str, Any]) -> str | None:
        result_path = getattr(self, 'result_path', None)
        if not result_path:
            logger.warning('Result path is not configured; skipping JSON export')
            return None

        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'w', encoding='utf-8') as result_file:
            json.dump(payload, result_file, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        logger.info(f"Task results saved to: {result_path}")
        return result_path

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    def _should_create_default_log(self) -> bool:
        task_with_custom_logs = ['GenerationTask', 'TestTask', 'EvaluateTask']
        return self.__class__.__name__ not in task_with_custom_logs
    
    def launch(self) -> Any:
        parser = ArgumentParser()
        parser = self.parse_task_args(parser)
        args, extras = parser.parse_known_args()
        self.args = args
        logger.success(args)
        
        if self._should_create_default_log() and self.log_handler_id is None:
            task_name = self.__class__.__name__.replace('Task', '').lower()
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f"{task_name}_{timestamp}.log"
            log_path = os.path.join("logs", log_filename)
            self.log_handler_id = logger.add(log_path, level='INFO')
            logger.info(f"Log file: {log_path}")
        
        return self.run(**vars(args))
