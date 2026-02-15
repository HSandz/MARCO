from abc import ABC, abstractmethod
from argparse import ArgumentParser
from loguru import logger
from typing import Any
import os
import datetime

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
        log_path = os.path.join("logs", log_filename)
        
        self.log_handler_id = logger.add(log_path, level='INFO')
        logger.info(f"Task-specific log file: {log_path}")
        
        return log_path

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
