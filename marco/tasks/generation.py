import os
import pandas as pd
from abc import abstractmethod
from tqdm import tqdm
from typing import Any
from loguru import logger
from argparse import ArgumentParser
import datetime

from marco.tasks.base import Task
from marco.utils import init_api, read_json, token_tracker, duration_tracker
from marco.utils.prompt_builder import PromptBuilder
from marco.systems import MARCOSystem

class GenerationTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--api_config', type=str, default='config/api-config.json', help='Api configuration file')
        parser.add_argument('--dataset', type=str, default='None', help='Dataset name')
        parser.add_argument('--data_file', type=str, required=True, help='Dataset file')
        parser.add_argument('--system', type=str, default='marco', choices=['marco'], help='System name')
        parser.add_argument('--system_config', type=str, required=True, help='System configuration file')
        parser.add_argument('--task', type=str, default='sr', choices=['rp', 'sr'], help='Task name')
        parser.add_argument('--max_his', type=int, default=10, help='Max history length')
        
        parser.add_argument('--provider', type=str, choices=['openrouter', 'openai', 'ollama', 'gemini', 'huggingface'], help='LLM provider type (e.g., openrouter, openai, ollama, gemini, huggingface)')
        parser.add_argument('--model', type=str, help='Model name/version to use (e.g., google/gemini-2.0-flash-001, gpt-4o-mini, llama3.2:1b). If not specified, uses default for the provider.')
        parser.add_argument('--disable-reflection-rerun', action='store_false', dest='enable_reflection_rerun', help='Disable automatic rerun when reflector returns correctness: false (only for MARCO system)')

        return parser

    def get_data(self, data_file: str, max_his: int) -> pd.DataFrame:
        df = pd.read_csv(data_file)
        
        data_dir = os.path.dirname(data_file)
        self.prompt_builder = PromptBuilder(data_dir, self.dataset)
        
        if self.task == 'sr' and 'candidate_item_id' in df.columns:
            import ast
            first_candidates = df['candidate_item_id'].iloc[0]
            if isinstance(first_candidates, str):
                try:
                    first_candidates = ast.literal_eval(first_candidates)
                except:
                    pass
            
            if isinstance(first_candidates, list):
                self.n_candidate = len(first_candidates)
                self.system_kwargs['n_candidate'] = self.n_candidate
                logger.info(f"Detected {self.n_candidate} candidates for SR task")
        
        return df

    def prompt_data(self, df: pd.DataFrame) -> list[tuple[str, int | float | str, pd.Series]]:
        import ast

        if self.task == 'sr' and 'candidate_item_id' in df.columns:
            logger.info(f"Checking GT items in candidates for {self.task} task...")
            no_gt_count = 0

            df = df.copy()
            df['_gt_not_in_candidates'] = False

            for i in range(len(df)):
                row = df.iloc[i]
                gt_item = row['item_id']
                candidate_ids = row['candidate_item_id']

                if isinstance(candidate_ids, str):
                    try:
                        candidate_ids = ast.literal_eval(candidate_ids)
                    except:
                        logger.warning(f"Failed to parse candidate_item_id for sample {i+1}, marking as GT-not-in-candidates")
                        df.at[df.index[i], '_gt_not_in_candidates'] = True
                        no_gt_count += 1
                        continue

                if isinstance(candidate_ids, list):
                    if gt_item not in candidate_ids:
                        logger.trace(f"Sample {i+1} (User {row['user_id']}): GT item {gt_item} not in candidates - will count as automatic failure")
                        df.at[df.index[i], '_gt_not_in_candidates'] = True
                        no_gt_count += 1

            if no_gt_count > 0:
                logger.warning(f"Found {no_gt_count}/{len(df)} samples where GT item not in candidates")
            else:
                logger.info(f"All {len(df)} samples have GT in candidates")
        
        data_prompt = self.system.prompts['data_prompt']
        prompts = []

        logger.info(f"Building prompts for {len(df)} samples...")

        for i in tqdm(range(len(df)), desc="Building prompts", leave=False):
            row = df.iloc[i]

            fields = self.prompt_builder.build_prompt_fields(row, max_his=self.max_his)
            
            if self.task == 'rp':
                prompt = data_prompt.format(
                    user_id=row['user_id'],
                    user_profile=fields['user_profile'],
                    history=fields['history'],
                    target_item_id=row['item_id'],
                    target_item_attributes=fields['target_item_attributes']
                )
                target = row['rating']
            
            elif self.task == 'sr':
                prompt = data_prompt.format(
                    user_id=row['user_id'],
                    user_profile=fields['user_profile'],
                    history=fields['history'],
                    candidate_item_attributes=fields['candidate_item_attributes']
                )
                target = row['item_id']
            
            else:
                raise NotImplementedError(f"Task {self.task} not implemented")
            
            prompts.append((prompt, target, row))
        
        logger.info(f"Built {len(prompts)} prompts")
        return prompts

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [self._json_safe(item) for item in value]
        if isinstance(value, set):
            return [self._json_safe(item) for item in sorted(value, key=lambda item: str(item))]
        if isinstance(value, pd.Series):
            return self._json_safe(value.to_dict())
        if hasattr(value, 'item') and callable(getattr(value, 'item')):
            try:
                return value.item()
            except Exception:
                pass
        return value

    def _build_agent_inventory(self) -> dict[str, Any]:
        inventory: dict[str, Any] = {}
        if not hasattr(self, 'system') or not hasattr(self.system, 'agent_coordinator'):
            return inventory

        agents = getattr(self.system.agent_coordinator, 'agents', {})
        for agent_name, agent in agents.items():
            llm_instances = {}
            if hasattr(agent, 'get_llm_instances'):
                for llm_name, llm in agent.get_llm_instances().items():
                    llm_instances[llm_name] = self._json_safe({
                        'class_name': llm.__class__.__name__,
                        'model': getattr(llm, 'model', None),
                        'usage': llm.get_usage_stats() if hasattr(llm, 'get_usage_stats') else {},
                        'detailed_usage': llm.get_detailed_usage_stats() if hasattr(llm, 'get_detailed_usage_stats') else {},
                    })

            inventory[agent_name] = self._json_safe({
                'class_name': agent.__class__.__name__,
                'llms': llm_instances,
            })

        return inventory

    def _build_sample_result(self, record: dict, gt_answer: int | float | str, data_sample: pd.Series, prompt: str) -> dict[str, Any]:
        sample_result: dict[str, Any] = {
            'sample_id': record.get('sample_id'),
            'user_id': record.get('user_id', 'unknown'),
            'ground_truth': self._json_safe(gt_answer),
            'prompt': prompt,
            'skipped_no_gt': record.get('_skipped_no_gt', False),
            'system_finished': record.get('System_Finished', False),
            'steps': {key: self._json_safe(value) for key, value in record.items() if key.startswith('Answer_')},
            'data_sample': self._json_safe(data_sample.to_dict()) if hasattr(data_sample, 'to_dict') else self._json_safe(data_sample),
        }

        if hasattr(self.system, 'solver_attempt_history'):
            sample_result['solver_attempts'] = self._json_safe(self.system.solver_attempt_history)
        if hasattr(self.system, 'reflection_all_reruns'):
            sample_result['reflection_reruns'] = self._json_safe(self.system.reflection_all_reruns)
        if hasattr(self.system, 'reflection_improvements'):
            sample_result['reflection_improvements'] = self._json_safe(self.system.reflection_improvements)
        if hasattr(self.system, 'total_reflections_triggered'):
            sample_result['total_reflections_triggered'] = self.system.total_reflections_triggered

        sample_result['final_answer'] = self._json_safe(self.system.answer)
        sample_result['final_answer_type'] = type(self.system.answer).__name__

        return sample_result

    def build_result_payload(self, final_stats: dict[str, Any], duration_stats: dict[str, Any]) -> dict[str, Any]:
        task_info = final_stats.get('task_info', {}) if isinstance(final_stats, dict) else {}
        run_args = self._json_safe({
            'api_config': getattr(self, 'args', {}).api_config if getattr(self, 'args', None) else None,
            'dataset': getattr(self, 'dataset', None),
            'data_file': getattr(self, 'data_file', None),
            'system': getattr(self, 'system', None).__class__.__name__ if hasattr(self, 'system') and self.system else None,
            'system_config': getattr(self, 'args', {}).system_config if getattr(self, 'args', None) else None,
            'task': getattr(self, 'task', None),
            'max_his': getattr(self, 'max_his', None),
            'provider': getattr(self, 'provider', None),
            'model_override': getattr(self, 'model_override', None),
            'enable_reflection_rerun': getattr(self, 'args', {}).enable_reflection_rerun if getattr(self, 'args', None) else None,
            'steps': getattr(self, 'steps', None),
            'topks': getattr(self, 'topks', None),
            'num_samples': len(getattr(self, 'sample_records', [])),
        })

        payload: dict[str, Any] = {
            'run_info': {
                'task_id': getattr(self, 'task_id', None),
                'timestamp': task_info.get('start_time'),
                'task_info': self._json_safe(task_info),
                'config': run_args,
                'log_file': os.path.abspath(getattr(self, 'log_path', '')) if getattr(self, 'log_path', None) else None,
                'result_file': os.path.abspath(getattr(self, 'result_path', '')) if getattr(self, 'result_path', None) else None,
            },
            'data': {
                'dataset': getattr(self, 'dataset', None),
                'data_file': getattr(self, 'data_file', None),
                'task': getattr(self, 'task', None),
                'system': getattr(self, 'system', None).__class__.__name__ if hasattr(self, 'system') and self.system else None,
            },
            'agents': self._build_agent_inventory(),
            'samples': self._json_safe(getattr(self, 'sample_records', [])),
            'token_stats': self._json_safe(final_stats),
            'duration_stats': self._json_safe(duration_stats),
        }

        return payload

    def get_system(self, system: str, system_config: str):
        if system == 'marco':
            self.system = MARCOSystem(config_path=system_config, **self.system_kwargs)
        else:
            raise NotImplementedError(f"Unknown system: {system}. Only 'marco' system is available.")

    @property
    @abstractmethod
    def running_steps(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def before_generate(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def after_step(self, answer: Any, gt_answer: int | float | str, step: int, record: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def after_iteration(self, answer: Any, gt_answer: int | float | str, record: dict, pbar: tqdm) -> None:
        raise NotImplementedError

    @abstractmethod
    def after_generate(self) -> None:
        raise NotImplementedError

    def generate(self, data: list[tuple[str, int | float | str, pd.Series]], steps: int = 2):
        task_id = f"{self.dataset}_{self.task}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.task_id = task_id
        task_info = {
            'dataset': self.dataset,
            'task': self.task,
            'system': self.system.__class__.__name__,
            'model_override': self.model_override,
            'samples': len(data),
            'steps': steps,
            'max_history': self.max_his
        }

        token_tracker.start_task(task_id, task_info)
        duration_tracker.start_task(task_id, task_info)

        self.sample_records = []

        token_tracker.reset_agent_stats(self.system)

        self.before_generate()
        with tqdm(total=len(data)) as pbar:
            for sample_idx, (test_data, gt_answer, data_sample) in enumerate(data):
                sample_id = sample_idx + 1
                logger.info(f"Sample: {sample_id}/{len(data)}")

                record = dict()
                record['sample_id'] = sample_id
                record['user_id'] = data_sample.get('user_id', 'unknown')

                gt_not_in_candidates = data_sample.get('_gt_not_in_candidates', False)

                if gt_not_in_candidates:
                    logger.info(f"Sample {sample_id}: GT item not in candidates - skipping system call, counting as automatic failure")
                    self.system.reset(clear=True)
                    if self.task == 'sr':
                        answer = []
                    else:
                        answer = None

                    self.system.finished = False
                    self.system.answer = answer

                    record['_skipped_no_gt'] = True
                    for i in range(steps):
                        record[f'Answer_{i}'] = answer
                else:
                    record['_skipped_no_gt'] = False
                    self.system.set_data(input=test_data, context="", gt_answer=gt_answer, data_sample=data_sample)
                    self.system._current_sample_idx = sample_id
                    self.system._current_user_id = record['user_id']
                    self.system.reset(clear=True)

                    for i in range(steps):
                        logger.debug(f'===================================Running step {i}...===================================')
                        self.after_step(answer=self.system(), gt_answer=gt_answer, step=i, record=record)

                    token_tracker.collect_system_stats(self.system)

                self.after_iteration(answer=self.system.answer, gt_answer=gt_answer, record=record, pbar=pbar)
                self.sample_records.append(self._build_sample_result(record, gt_answer, data_sample, test_data))
                pbar.update(1)
                
        final_stats = token_tracker.end_task()
        duration_stats = duration_tracker.end_task()

        self.final_stats = final_stats
        self.duration_stats = duration_stats
        
        logger.success("=== Token Usage Summary ===")
        logger.success(f"Task: {self.dataset} {self.task} ({len(data)} samples)")
        logger.success(f"Data file: {self.data_file}")
        logger.success(f"Total API calls: {final_stats.get('total_api_calls', 0)}")
        logger.success(f"Total tokens: {final_stats.get('total_tokens', 0)}")
        logger.success(f"Input tokens: {final_stats.get('total_input_tokens', 0)}")
        logger.success(f"Output tokens: {final_stats.get('total_output_tokens', 0)}")
        logger.success(f"Models used: {final_stats.get('models_used', [])}")
        logger.success(f"Duration: {final_stats.get('duration', 0):.2f}s")
        
        agents = final_stats.get('agents', {})
        agent_durations = duration_stats.get('agents', {})
        
        if agents or agent_durations:
            logger.success("=== Per-Agent Statistics===")
            all_agent_names = set(agents.keys()) | set(agent_durations.keys())
            
            for agent_name in sorted(all_agent_names):
                logger.success(f"Agent: {agent_name}")
                
                if agent_name in agents:
                    agent_stats = agents[agent_name]
                    logger.success(f"  API calls: {agent_stats.get('api_calls', 0)}")
                    logger.success(f"  Total tokens: {agent_stats.get('total_tokens', 0)}")
                    logger.success(f"  Input tokens: {agent_stats.get('total_input_tokens', 0)}")
                    logger.success(f"  Output tokens: {agent_stats.get('total_output_tokens', 0)}")
                    logger.success(f"  Model: {agent_stats.get('model', 'unknown')}")
                
                if agent_name in agent_durations:
                    duration_info = agent_durations[agent_name]
                    logger.success(f"  Total duration: {duration_info.get('total_duration', 0):.3f}s")
                    logger.success(f"  Number of calls: {duration_info.get('call_count', 0)}")
                    logger.success(f"  Average duration per call: {duration_info.get('avg_duration_per_call', 0):.3f}s")
        
        self.after_generate()

        result_payload = self.build_result_payload(final_stats, duration_stats)
        self.save_result_payload(result_payload)

    def run(self, api_config: str, dataset: str, data_file: str, system: str, system_config: str, task: str, max_his: int, provider: str = None, model: str = None, enable_reflection_rerun: bool = True):
        if dataset == 'None':
            dataset = os.path.basename(os.path.dirname(data_file))
        self.dataset = dataset
        self.task = task
        self.max_his = max_his
        self.data_file = data_file
        
        data_dir = os.path.dirname(data_file)
        
        init_api(read_json(api_config))
        
        if provider:
            provider_info = self._parse_provider_options(provider, model)
            self.model_override = provider_info['model']
            self.provider = provider_info['provider']
            self.system_kwargs = {
                'task': self.task,
                'leak': False,
                'dataset': self.dataset,
                'data_dir': data_dir,
                'model_override': self.model_override,
                'provider': self.provider,
                'enable_reflection_rerun': enable_reflection_rerun,
                'api_config_path': api_config,
            }
            logger.info(f"Using {provider_info['provider']} with model: {provider_info['model']} (will override all agents except opensource)")
        else:
            self.model_override = None
            self.provider = None
            self.system_kwargs = {
                'task': self.task,
                'leak': False,
                'dataset': self.dataset,
                'data_dir': data_dir,
                'enable_reflection_rerun': enable_reflection_rerun,
                'api_config_path': api_config,
            }
            logger.info(f"No provider/model specified - using individual agent configurations")
        
        data_df = self.get_data(data_file, max_his)
        
        self.get_system(system, system_config)
        data = self.prompt_data(data_df)
        
        self.setup_task_logger(task=task, dataset=dataset, system=system, num_samples=len(data))
        
        self.generate(data, steps=self.running_steps)
    
    def _parse_provider_options(self, provider: str, model: str = None) -> dict:
        def _get_default(provider_name: str) -> str:
            default_map = {
                'openrouter': 'google/gemini-2.0-flash-001',
                'openai': 'gpt-4o-mini',
                'ollama': 'llama3.2:1b',
                'gemini': 'google/gemini-2.0-flash-001',
                'vertexai': 'gemini-2.0-flash-001'
            }
            return default_map.get(provider_name, 'google/gemini-2.0-flash-001')

        if not provider:
            return {
                'provider': None,
                'model': None
            }
        
        chosen_model = model if model else _get_default(provider)
        
        if provider == 'openai':
            chosen_model = self._normalize_openai_model(chosen_model)
        
        return {
            'provider': provider,
            'model': chosen_model
        }

    @staticmethod
    def _normalize_openai_model(model: str) -> str:
        if not model:
            return 'gpt-4o-mini'
        cleaned = model.strip()
        if '/' in cleaned:
            prefix, suffix = cleaned.split('/', 1)
            if prefix.lower() == 'openai':
                return suffix
        return cleaned
