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
        
        file_log_level = os.getenv('MARCO_LOG_LEVEL', 'INFO').upper()
        self.log_handler_id = logger.add(log_path, level=file_log_level)
        self.log_path = log_path
        self.result_path = result_path
        logger.info(f"Task-specific log file: {log_path} (level={file_log_level})")
        logger.info(f"Task-specific result file: {result_path}")
        self.log_run_arguments()
        
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

    def get_logged_arguments(self) -> dict[str, Any]:
        arguments: dict[str, Any] = {}

        global_args = getattr(self, 'global_args', None)
        if isinstance(global_args, dict):
            arguments.update({f'global.{key}': value for key, value in global_args.items()})

        task_args = getattr(self, 'args', None)
        if task_args is not None:
            arguments.update({f'task.{key}': value for key, value in vars(task_args).items()})

        effective_args = getattr(self, 'effective_args', None)
        if isinstance(effective_args, dict):
            arguments.update({f'effective.{key}': value for key, value in effective_args.items()})

        extras = getattr(self, 'unknown_args', None)
        if extras:
            arguments['unknown_args'] = extras

        return arguments

    def log_run_arguments(self) -> None:
        arguments = self.get_logged_arguments()
        if not arguments:
            logger.success('Run arguments: none recorded')
            return

        logger.success('=== Run Arguments ===')
        for key in sorted(arguments):
            logger.success(f'{key}: {arguments[key]!r}')

    def _should_create_default_log(self) -> bool:
        task_with_custom_logs = ['GenerationTask', 'TestTask', 'EvaluateTask']
        return self.__class__.__name__ not in task_with_custom_logs
    
    def launch(self, global_args: dict[str, Any] | None = None, argv: list[str] | None = None) -> Any:
        parser = ArgumentParser()
        parser = self.parse_task_args(parser)
        args, extras = parser.parse_known_args(argv)
        self.args = args
        self.global_args = global_args or {}
        self.unknown_args = extras
        logger.success(args)
        
        if self._should_create_default_log() and self.log_handler_id is None:
            task_name = self.__class__.__name__.replace('Task', '').lower()
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f"{task_name}_{timestamp}.log"
            log_path = os.path.join("logs", log_filename)
            file_log_level = os.getenv('MARCO_LOG_LEVEL', 'INFO').upper()
            self.log_handler_id = logger.add(log_path, level=file_log_level)
            logger.info(f"Log file: {log_path} (level={file_log_level})")
            self.log_run_arguments()
        
        return self.run(**vars(args))
