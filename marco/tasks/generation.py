import os
import json
import tempfile
import time
import pandas as pd
from abc import abstractmethod
from tqdm import tqdm
from typing import Any
from loguru import logger
from argparse import ArgumentParser
import datetime

from marco.tasks.base import Task
from marco.utils import init_api, read_json, token_tracker, duration_tracker, system2dir
from marco.utils.vertex_batch import (
    build_gcs_uri,
    extract_batch_response_text,
    extract_batch_usage_metadata,
    load_jsonl_from_text,
    parse_gcs_uri,
)
from marco.utils.prompt_builder import PromptBuilder
from marco.systems import MARCOSystem, SingleSystem
from marco.llms import VertexAILLM

class GenerationTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--api_config', type=str, default='config/api-config.json', help='Api configuration file')
        parser.add_argument('--dataset', type=str, default='None', help='Dataset name')
        parser.add_argument('--data_file', type=str, required=True, help='Dataset file')
        parser.add_argument('--system', type=str, default='marco', choices=['marco', 'single'], help='System name')
        parser.add_argument('--system_config', type=str, required=True, help='System configuration file')
        parser.add_argument('--task', type=str, default='sr', choices=['rp', 'sr'], help='Task name')
        parser.add_argument('--max_his', type=int, default=10, help='Max history length (<=0 means unlimited)')
        parser.add_argument('--batch', action='store_true', help='Use Vertex AI batch prediction. For single, batches full prompts; for MARCO, batches Solver and/or Reflector when those agents use Vertex AI.')
        parser.add_argument('--batch_gcs_uri_prefix', type=str, default='', help='GCS URI prefix for batch input/output files, e.g. gs://bucket/marco-batch-run')
        parser.add_argument('--batch_gcs_uri_input', type=str, default='', help='Existing gs:// JSONL batch input file to reuse instead of uploading a new one')
        parser.add_argument('--no_cache', action='store_true', help='Disable Planner/Analyst cache reads while still writing successful uncached responses to cached/')
        
        parser.add_argument('--provider', type=str, choices=['openrouter', 'openai', 'codexhub', 'ollama', 'gemini', 'huggingface', 'vertexai'], help='LLM provider type (e.g., openrouter, openai, codexhub, ollama, gemini, huggingface, vertexai)')
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
            task_kwargs = {
                'task': self.task,
                'target_item_id': '',
                'target_item_attributes': '',
                'candidate_item_attributes': '',
            }
            
            if self.task == 'rp':
                task_kwargs['target_item_id'] = row['item_id']
                task_kwargs['target_item_attributes'] = fields['target_item_attributes']
                prompt = data_prompt.format(
                    user_id=row['user_id'],
                    user_profile=fields['user_profile'],
                    history=fields['history'],
                    **task_kwargs
                )
                target = row['rating']
            
            elif self.task == 'sr':
                task_kwargs['candidate_item_attributes'] = fields['candidate_item_attributes']
                prompt = data_prompt.format(
                    user_id=row['user_id'],
                    user_profile=fields['user_profile'],
                    history=fields['history'],
                    **task_kwargs
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
        if hasattr(self, 'system') and hasattr(self.system, 'llm') and self.system.llm is not None:
            inventory[f'{system2dir(self.system.__class__.__name__)}_llm'] = self._json_safe({
                'class_name': self.system.__class__.__name__,
                'llms': {
                    'llm': self._json_safe({
                        'class_name': self.system.llm.__class__.__name__,
                        'model': getattr(self.system.llm, 'model', None),
                        'usage': self.system.llm.get_usage_stats() if hasattr(self.system.llm, 'get_usage_stats') else {},
                        'detailed_usage': self.system.llm.get_detailed_usage_stats() if hasattr(self.system.llm, 'get_detailed_usage_stats') else {},
                    })
                },
            })

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
            'no_cache': getattr(self, 'no_cache', False),
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
        elif system == 'single':
            self.system = SingleSystem(config_path=system_config, **self.system_kwargs)
        else:
            raise NotImplementedError(f"Unknown system: {system}. Only 'marco' and 'single' systems are available.")

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

    def _is_vertex_batch_enabled(self) -> bool:
        return bool(getattr(self, 'batch_mode', False))

    @staticmethod
    def _apply_adjusted_duration(final_stats: dict[str, Any], duration_stats: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(final_stats, dict) or not isinstance(duration_stats, dict):
            return final_stats

        adjusted_duration = duration_stats.get('total_duration')
        if adjusted_duration is None:
            return final_stats

        raw_duration = final_stats.get('duration')
        if raw_duration is not None and 'wall_clock_duration' not in final_stats:
            final_stats['wall_clock_duration'] = raw_duration
        final_stats['duration'] = adjusted_duration
        return final_stats

    def _log_per_agent_statistics(self, final_stats: dict[str, Any], duration_stats: dict[str, Any]) -> None:
        agents = final_stats.get('agents', {}) if isinstance(final_stats, dict) else {}
        agent_durations = duration_stats.get('agents', {}) if isinstance(duration_stats, dict) else {}

        if not agents and not agent_durations:
            return

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

    def _build_vertex_batch_request(self, prompt: str, sample_key: str, llm: VertexAILLM | None = None) -> dict[str, Any]:
        llm = llm or self.system.llm
        actual_prompt = prompt

        if getattr(llm, 'json_mode', False) and 'Please respond with valid JSON only.' not in actual_prompt:
            actual_prompt = f'{prompt}\n\nPlease respond with valid JSON only.'

        generation_config: dict[str, Any] = {
            'temperature': getattr(llm, 'temperature', None),
            'topP': getattr(llm, 'top_p', None),
            'topK': getattr(llm, 'top_k', None),
            'maxOutputTokens': getattr(llm, 'max_tokens', None),
        }

        if getattr(llm, 'json_mode', False):
            generation_config['responseMimeType'] = 'application/json'

        generation_config = {key: value for key, value in generation_config.items() if value is not None}

        return {
            'key': sample_key,
            'request': {
                'contents': [
                    {
                        'role': 'user',
                        'parts': [
                            {
                                'text': actual_prompt,
                            }
                        ],
                    }
                ],
                'generationConfig': generation_config,
            },
        }

    def _resolve_batch_output_uri(self, batch_job: Any) -> str:
        dest = getattr(batch_job, 'dest', None)
        if dest is not None:
            gcs_uri = getattr(dest, 'gcs_uri', None)
            if gcs_uri:
                if isinstance(gcs_uri, list):
                    return gcs_uri[0]
                return gcs_uri

        output_info = getattr(batch_job, 'output_info', None)
        if isinstance(output_info, dict):
            for key in ('gcs_uri', 'output_uri', 'outputUriPrefix'):
                value = output_info.get(key)
                if value:
                    if isinstance(value, list):
                        return value[0]
                    return value

        raise ValueError(f'Unable to determine batch output URI for job: {getattr(batch_job, "name", batch_job)}')

    def _write_batch_input_file(self, data: list[tuple[str, int | float | str, pd.Series]]) -> tuple[str, list[dict[str, Any]]]:
        if not getattr(self, 'batch_gcs_uri_prefix', ''):
            raise ValueError('Batch mode requires --batch_gcs_uri_prefix with a gs:// URI prefix')

        if not isinstance(self.system.llm, VertexAILLM):
            raise ValueError('Batch mode is only supported for the single system with a Vertex AI LLM')

        batch_entries: list[dict[str, Any]] = []
        for sample_idx, (prompt, _gt_answer, data_sample) in enumerate(data, start=1):
            if data_sample.get('_gt_not_in_candidates', False):
                continue
            batch_entries.append(self._build_vertex_batch_request(prompt=prompt, sample_key=f'sample-{sample_idx}'))

        local_handle = tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False, encoding='utf-8')
        try:
            for entry in batch_entries:
                local_handle.write(json.dumps(entry, ensure_ascii=False))
                local_handle.write('\n')
        finally:
            local_handle.close()

        logger.info(f'Wrote {len(batch_entries)} requests to local batch file: {local_handle.name}')
        return local_handle.name, batch_entries

    def _count_expected_batch_requests(self, data: list[tuple[str, int | float | str, pd.Series]]) -> int:
        return sum(1 for _prompt, _gt_answer, data_sample in data if not data_sample.get('_gt_not_in_candidates', False))

    def _build_batch_input_blob_name(self, sample_count: int) -> str:
        data_file_name = os.path.basename(getattr(self, 'data_file', '') or 'data.csv')
        data_file_stem, data_file_ext = os.path.splitext(data_file_name)
        data_file_label = f'{data_file_stem}{data_file_ext}' if data_file_ext else data_file_stem

        dataset_label = str(getattr(self, 'dataset', 'dataset') or 'dataset')
        task_label = str(getattr(self, 'task', 'task') or 'task')

        return f'{dataset_label}_{data_file_label}_{task_label}_{sample_count}_batch_input.jsonl'

    def _prepare_batch_input_uri(self, data: list[tuple[str, int | float | str, pd.Series]]) -> tuple[str, str | None]:
        expected_request_count = self._count_expected_batch_requests(data)
        expected_batch_input_name = self._build_batch_input_blob_name(sample_count=expected_request_count)

        batch_input_uri = getattr(self, 'batch_gcs_uri_input', '')
        if batch_input_uri:
            if not batch_input_uri.startswith('gs://'):
                raise ValueError('--batch_gcs_uri_input must be a gs:// URI')

            bucket_name, blob_path = parse_gcs_uri(batch_input_uri)

            if blob_path.endswith('.jsonl'):
                actual_input_name = os.path.basename(blob_path)
                if actual_input_name != expected_batch_input_name:
                    raise ValueError(
                        'Existing batch input file name does not match the current run: '
                        f'expected {expected_batch_input_name!r} for dataset={self.dataset!r}, '
                        f'data_file={self.data_file!r}, task={self.task!r}, requests={expected_request_count}, '
                        f'but got {actual_input_name!r} from {batch_input_uri}. '
                        'Use a matching batch input file or rebuild it with --batch_gcs_uri_prefix.'
                    )

                batch_input_text = self._download_gcs_text(batch_input_uri)
                actual_request_count = len(load_jsonl_from_text(batch_input_text))

                if actual_request_count != expected_request_count:
                    raise ValueError(
                        'Existing batch input file does not match the current run: '
                        f'expected {expected_request_count} requests from {len(data)} samples, '
                        f'but found {actual_request_count} JSONL records in {batch_input_uri}. '
                        'Use a matching batch input file or rebuild it with --batch_gcs_uri_prefix.'
                    )

                logger.info(f'Reusing existing batch input file: {batch_input_uri}')
                return batch_input_uri, None

            from google.cloud import storage

            client = storage.Client(
                project=getattr(self.system.llm, 'project_id', None),
                credentials=getattr(self.system.llm, 'credentials', None),
            )

            prefix = blob_path.rstrip('/') + '/'
            blobs = list(client.list_blobs(bucket_name, prefix=prefix))
            logger.info(f'Listed {len(blobs)} objects under prefix {batch_input_uri}')

            matches = [b for b in blobs if os.path.basename(b.name) == expected_batch_input_name]
            logger.info(f'Found {len(matches)} candidate(s) matching {expected_batch_input_name!r} under the prefix')
            if len(matches) == 1:
                selected = matches[0]
                found_uri = build_gcs_uri(bucket_name, selected.name)
                logger.info(f'Found matching batch input file under prefix: {found_uri}')
                batch_input_text = selected.download_as_text()
                actual_request_count = len(load_jsonl_from_text(batch_input_text))

                if actual_request_count != expected_request_count:
                    raise ValueError(
                        'Existing batch input file does not match the current run: '
                        f'expected {expected_request_count} requests from {len(data)} samples, '
                        f'but found {actual_request_count} JSONL records in {found_uri}. '
                        'Use a matching batch input file or rebuild it with --batch_gcs_uri_prefix.'
                    )

                logger.info(f'Reusing existing batch input file: {found_uri}')
                return found_uri, None

            if len(matches) > 1:
                found_list = [build_gcs_uri(bucket_name, b.name) for b in matches]
                raise ValueError(
                    'Multiple candidate batch input files found under the provided prefix: ' + ', '.join(found_list)
                    + '. Provide a more specific path or remove duplicates.'
                )

            raise ValueError(
                f'No matching batch input file named {expected_batch_input_name!r} found under {batch_input_uri!r}. '
                'Provide a full path to the JSONL or point to a different prefix.'
            )

        local_input_file, _batch_entries = self._write_batch_input_file(data)
        bucket_name, blob_prefix = parse_gcs_uri(self.batch_gcs_uri_prefix)
        batch_input_blob_name = expected_batch_input_name

        if blob_prefix.endswith('.jsonl'):
            batch_input_uri = build_gcs_uri(bucket_name, blob_prefix)
        else:
            batch_input_uri = build_gcs_uri(bucket_name, f'{blob_prefix.rstrip("/")}/{batch_input_blob_name}')

        self._upload_file_to_gcs(local_input_file, batch_input_uri)
        return batch_input_uri, local_input_file

    def _get_storage_llm(self, llm: VertexAILLM | None = None) -> VertexAILLM:
        if llm is not None:
            return llm
        if isinstance(getattr(self.system, 'llm', None), VertexAILLM):
            return self.system.llm
        if isinstance(getattr(getattr(self.system, 'solver', None), 'llm', None), VertexAILLM):
            return self.system.solver.llm
        raise ValueError('A Vertex AI LLM is required for batch GCS operations')

    def _upload_file_to_gcs(self, local_file_path: str, gcs_uri: str, llm: VertexAILLM | None = None) -> None:
        from google.cloud import storage

        storage_llm = self._get_storage_llm(llm)
        bucket_name, blob_path = parse_gcs_uri(gcs_uri)
        client = storage.Client(
            project=getattr(storage_llm, 'project_id', None),
            credentials=getattr(storage_llm, 'credentials', None),
        )
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        logger.info(f'Uploading local batch input file {local_file_path} to {gcs_uri}...')
        blob.upload_from_filename(local_file_path)
        logger.info(f'Upload complete: {gcs_uri}')

    def _download_gcs_text(self, gcs_uri: str, llm: VertexAILLM | None = None) -> str:
        from google.cloud import storage

        storage_llm = self._get_storage_llm(llm)
        bucket_name, blob_path = parse_gcs_uri(gcs_uri)
        client = storage.Client(
            project=getattr(storage_llm, 'project_id', None),
            credentials=getattr(storage_llm, 'credentials', None),
        )
        bucket = client.bucket(bucket_name)

        if blob_path.endswith('/'):
            raise ValueError(f'Unexpected directory URI for batch output: {gcs_uri}')

        blob = bucket.blob(blob_path)
        return blob.download_as_text()

    def _load_batch_results(self, batch_output_uri: str, llm: VertexAILLM | None = None) -> dict[str, dict[str, Any]]:
        from google.cloud import storage

        storage_llm = self._get_storage_llm(llm)
        bucket_name, blob_path = parse_gcs_uri(batch_output_uri)
        client = storage.Client(
            project=getattr(storage_llm, 'project_id', None),
            credentials=getattr(storage_llm, 'credentials', None),
        )
        bucket = client.bucket(bucket_name)

        prefix = blob_path.rstrip('/') + '/'
        blobs = list(client.list_blobs(bucket_name, prefix=prefix))
        logger.info(f'Found {len(blobs)} blob(s) under batch output prefix {batch_output_uri}')

        if not blobs:
            blob = bucket.blob(blob_path)
            if blob.exists():
                blobs = [blob]

        results: dict[str, dict[str, Any]] = {}
        total_records = 0
        for blob in blobs:
            if blob.name.endswith('/'):
                continue
            blob_text = blob.download_as_text()
            for record in load_jsonl_from_text(blob_text):
                sample_key = record.get('key')
                if not sample_key:
                    continue
                results[str(sample_key)] = record
                total_records += 1

        logger.info(f'Loaded {total_records} result record(s) from batch output')
        return results

    def _run_vertex_batch_single(self, data: list[tuple[str, int | float | str, pd.Series]], steps: int) -> None:
        if not isinstance(self.system, SingleSystem):
            raise ValueError('Batch mode is only supported for the single system')
        if not isinstance(self.system.llm, VertexAILLM):
            raise ValueError('Batch mode is only supported for a Vertex AI LLM')

        task_id = f"{self.dataset}_{self.task}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.task_id = task_id
        task_info = {
            'dataset': self.dataset,
            'task': self.task,
            'system': self.system.__class__.__name__,
            'model_override': self.model_override,
            'samples': len(data),
            'steps': steps,
            'max_history': self.max_his,
            'batch': True,
            'batch_gcs_uri_prefix': self.batch_gcs_uri_prefix,
        }

        token_tracker.start_task(task_id, task_info)
        duration_tracker.start_task(task_id, task_info)

        self.sample_records = []
        token_tracker.reset_agent_stats(self.system)

        self.before_generate()

        batch_input_uri, local_input_file = self._prepare_batch_input_uri(data)
        if local_input_file:
            try:
                logger.info(f'Removing temporary local batch input file: {local_input_file}')
                os.unlink(local_input_file)
                logger.info('Temporary local batch input file removed')
            except OSError:
                logger.debug(f'Failed to remove temporary batch input file: {local_input_file}')

        with tqdm(total=len(data), desc='Waiting Vertex batch: single round 1') as batch_pbar:
            batch_job, batch_output_uri, batch_results = self._run_vertex_batch_job(
                self.system.llm,
                batch_input_uri,
                request_count=len(data),
                agent_label='single',
                round_index=1,
                progress_bar=batch_pbar,
            )

        self.system.llm.reset_usage_stats()
        total_input_tokens = 0
        total_output_tokens = 0
        batch_call_history: list[dict[str, Any]] = []

        with tqdm(total=len(data)) as pbar:
            for sample_idx, (test_data, gt_answer, data_sample) in enumerate(data, start=1):
                record: dict[str, Any] = {
                    'sample_id': sample_idx,
                    'user_id': data_sample.get('user_id', 'unknown'),
                }

                if data_sample.get('_gt_not_in_candidates', False):
                    logger.info(f'Sample {sample_idx}: GT item not in candidates - counting as automatic failure')
                    self.system.reset(clear=True)
                    answer = [] if self.task == 'sr' else None
                    self.system.finished = False
                    self.system.answer = answer
                    record['_skipped_no_gt'] = True
                    for i in range(steps):
                        record[f'Answer_{i}'] = answer
                else:
                    record['_skipped_no_gt'] = False
                    sample_key = f'sample-{sample_idx}'
                    batch_record = batch_results.get(sample_key, {})
                    response_payload = batch_record.get('response') if isinstance(batch_record, dict) else None
                    raw_response = extract_batch_response_text(response_payload if response_payload is not None else batch_record)
                    usage_payload = response_payload if response_payload is not None else batch_record
                    usage_metadata = extract_batch_usage_metadata(usage_payload)

                    self.system.set_data(input=test_data, context='', gt_answer=gt_answer, data_sample=data_sample)
                    self.system._current_sample_idx = sample_idx
                    self.system._current_user_id = record['user_id']
                    self.system.reset(clear=True)

                    self.system.raw_response = raw_response
                    answer = self.system._parse_response(raw_response)
                    self.system.finish(answer)

                    for i in range(steps):
                        record[f'Answer_{i}'] = answer
                    record['batch_key'] = sample_key
                    record['batch_raw_response'] = raw_response
                    if isinstance(batch_record, dict):
                        record['batch_status'] = batch_record.get('status', '')
                        if batch_record.get('status') and not raw_response:
                            record['batch_error'] = batch_record.get('status')

                    if usage_metadata:
                        input_tokens = usage_metadata.get('prompt_tokens')
                        output_tokens = usage_metadata.get('completion_tokens')
                        total_tokens = usage_metadata.get('total_tokens')

                        if input_tokens is None and output_tokens is None and total_tokens is None:
                            logger.warning(
                                f'Batch usage metadata missing token counts for {sample_key}; '
                                'falling back to manual token estimation'
                            )
                            input_tokens = self.system.llm.estimate_tokens(test_data) if raw_response else 0
                            output_tokens = self.system.llm.estimate_tokens(raw_response) if raw_response else 0
                            estimated_input = True
                            estimated_output = True
                        else:
                            estimated_input = input_tokens is None
                            estimated_output = output_tokens is None
                            if input_tokens is None:
                                logger.warning(
                                    f'Batch usage metadata missing prompt token count for {sample_key}; '
                                    'falling back to manual input token estimation'
                                )
                                input_tokens = self.system.llm.estimate_tokens(test_data) if raw_response else 0
                            if output_tokens is None:
                                logger.warning(
                                    f'Batch usage metadata missing completion token count for {sample_key}; '
                                    'falling back to manual output token estimation'
                                )
                                output_tokens = self.system.llm.estimate_tokens(raw_response) if raw_response else 0
                    else:
                        logger.warning(
                            f'Batch usage metadata not found for {sample_key}; '
                            'falling back to manual token estimation'
                        )
                        input_tokens = self.system.llm.estimate_tokens(test_data) if raw_response else 0
                        output_tokens = self.system.llm.estimate_tokens(raw_response) if raw_response else 0
                        estimated_input = bool(raw_response)
                        estimated_output = bool(raw_response)

                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens

                    if raw_response or usage_metadata:
                        batch_call_history.append(
                            {
                                'call_id': len(batch_call_history) + 1,
                                'input_tokens': input_tokens,
                                'output_tokens': output_tokens,
                                'total_tokens': input_tokens + output_tokens,
                                'model': self.system.llm.model,
                                'prompt_length': len(test_data),
                                'response_length': len(raw_response),
                                'estimated_input': estimated_input,
                                'estimated_output': estimated_output,
                                'api_usage': {
                                    'batch_mode': True,
                                    'batch_job_name': getattr(batch_job, 'name', None),
                                    'batch_input_uri': batch_input_uri,
                                    'batch_output_uri': batch_output_uri,
                                    'batch_key': sample_key,
                                    'usage_metadata': usage_metadata,
                                },
                            }
                        )

                self.after_iteration(answer=self.system.answer, gt_answer=gt_answer, record=record, pbar=pbar)
                self.sample_records.append(self._build_sample_result(record, gt_answer, data_sample, test_data))
                pbar.update(1)

        self.system.llm.total_input_tokens = total_input_tokens
        self.system.llm.total_output_tokens = total_output_tokens
        self.system.llm.api_calls = len(batch_call_history)
        self.system.llm.call_history = batch_call_history

        token_tracker.collect_system_stats(self.system)

        self.final_stats = token_tracker.end_task()
        self.duration_stats = duration_tracker.end_task()
        self._apply_adjusted_duration(self.final_stats, self.duration_stats)

        logger.success('=== Token Usage Summary ===')
        logger.success(f'Task: {self.dataset} {self.task} ({len(data)} samples)')
        logger.success(f'Data file: {self.data_file}')
        logger.success(f'System config: {self.system_config}')
        logger.success(f'Total API calls: {self.final_stats.get("total_api_calls", 0)}')
        logger.success(f'Total tokens: {self.final_stats.get("total_tokens", 0)}')
        logger.success(f'Input tokens: {self.final_stats.get("total_input_tokens", 0)}')
        logger.success(f'Output tokens: {self.final_stats.get("total_output_tokens", 0)}')
        logger.success(f'Models used: {self.final_stats.get("models_used", [])}')
        logger.success(f'Duration: {self.final_stats.get("duration", 0):.2f}s')
        excluded_duration = self.duration_stats.get('total_excluded_duration', 0) if isinstance(self.duration_stats, dict) else 0
        if excluded_duration:
            logger.success(f'Excluded retry time: {excluded_duration:.2f}s')
        self.log_run_arguments()
        self._log_per_agent_statistics(self.final_stats, self.duration_stats)

        self.after_generate()

        result_payload = self.build_result_payload(self.final_stats, self.duration_stats)
        result_payload.setdefault('run_info', {}).setdefault('config', {})['batch'] = True
        result_payload['run_info']['config']['batch_gcs_uri_prefix'] = self.batch_gcs_uri_prefix
        result_payload['run_info']['config']['batch_input_uri'] = batch_input_uri
        result_payload['run_info']['config']['batch_output_uri'] = batch_output_uri
        self.save_result_payload(result_payload)

    def _write_vertex_batch_entries(self, entries: list[dict[str, Any]]) -> str:
        local_handle = tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False, encoding='utf-8')
        try:
            for entry in entries:
                local_handle.write(json.dumps(entry, ensure_ascii=False))
                local_handle.write('\n')
        finally:
            local_handle.close()

        logger.info(f'Wrote {len(entries)} requests to local batch file: {local_handle.name}')
        return local_handle.name

    def _build_marco_agent_batch_input_uri(self, agent_label: str, round_index: int, request_count: int, local_input_file: str, llm: VertexAILLM) -> str:
        if not getattr(self, 'batch_gcs_uri_prefix', ''):
            raise ValueError('MARCO batch mode requires --batch_gcs_uri_prefix with a gs:// URI prefix')

        bucket_name, blob_prefix = parse_gcs_uri(self.batch_gcs_uri_prefix)
        data_file_name = os.path.basename(getattr(self, 'data_file', '') or 'data.csv')
        dataset_label = str(getattr(self, 'dataset', 'dataset') or 'dataset')
        task_label = str(getattr(self, 'task', 'task') or 'task')
        batch_input_blob_name = (
            f'{dataset_label}_{data_file_name}_{task_label}_marco_{agent_label}_round{round_index}_'
            f'{request_count}_batch_input.jsonl'
        )

        if blob_prefix.endswith('.jsonl'):
            stem, ext = os.path.splitext(blob_prefix)
            blob_path = f'{stem}_{agent_label}_round{round_index}{ext}'
        else:
            blob_path = f'{blob_prefix.rstrip("/")}/{batch_input_blob_name}'

        batch_input_uri = build_gcs_uri(bucket_name, blob_path)
        self._upload_file_to_gcs(local_input_file, batch_input_uri, llm=llm)
        return batch_input_uri

    def _build_marco_solver_batch_input_uri(self, round_index: int, request_count: int, local_input_file: str) -> str:
        return self._build_marco_agent_batch_input_uri(
            'solver',
            round_index,
            request_count,
            local_input_file,
            self.system.solver.llm,
        )

    def _build_marco_reflector_batch_input_uri(self, round_index: int, request_count: int, local_input_file: str) -> str:
        return self._build_marco_agent_batch_input_uri(
            'reflector',
            round_index,
            request_count,
            local_input_file,
            self.system.reflector.llm,
        )

    @staticmethod
    def _get_batch_job_field(obj: Any, *names: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            for name in names:
                if name in obj:
                    return obj[name]
            return None
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return None

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        seconds = max(int(seconds), 0)
        minutes, rem = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f'{hours}h{minutes:02d}m{rem:02d}s'
        if minutes:
            return f'{minutes}m{rem:02d}s'
        return f'{rem}s'

    def _format_batch_progress(
        self,
        batch_job: Any,
        request_count: int | None = None,
        elapsed_seconds: float = 0.0,
    ) -> tuple[str, bool]:
        successful_i, failed_i, incomplete_i, processed, has_completion_stats = self._get_batch_progress_counts(batch_job)

        parts = [f'elapsed={self._format_elapsed(elapsed_seconds)}']
        if request_count is not None:
            parts.append(f'requests={request_count}')

        if has_completion_stats:
            parts.append(f'successful={successful_i if successful_i is not None else "unknown"}')
            parts.append(f'failed={failed_i if failed_i is not None else "unknown"}')
            parts.append(f'incomplete={incomplete_i if incomplete_i is not None else "unknown"}')
            if request_count:
                parts.append(f'processed={processed}/{request_count}')
        else:
            parts.append('processed=unavailable')

        resources_consumed = self._get_batch_job_field(batch_job, 'resources_consumed', 'resourcesConsumed')
        replica_hours = self._get_batch_job_field(resources_consumed, 'replica_hours', 'replicaHours')
        if replica_hours is not None:
            parts.append(f'replica_hours={replica_hours}')

        partial_failures = self._get_batch_job_field(batch_job, 'partial_failures', 'partialFailures')
        if partial_failures:
            try:
                partial_failure_count = len(partial_failures)
            except TypeError:
                partial_failure_count = 'unknown'
            parts.append(f'partial_failures={partial_failure_count}')

        return ', '.join(parts), has_completion_stats

    def _get_batch_progress_counts(self, batch_job: Any) -> tuple[int | None, int | None, int | None, int, bool]:
        completion_stats = self._get_batch_job_field(batch_job, 'completion_stats', 'completionStats')
        successful = self._get_batch_job_field(completion_stats, 'successful_count', 'successfulCount')
        failed = self._get_batch_job_field(completion_stats, 'failed_count', 'failedCount')
        incomplete = self._get_batch_job_field(completion_stats, 'incomplete_count', 'incompleteCount')

        def to_int(value: Any) -> int | None:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        successful_i = to_int(successful)
        failed_i = to_int(failed)
        incomplete_i = to_int(incomplete)
        processed = sum(value for value in (successful_i, failed_i) if value is not None)
        has_completion_stats = any(value is not None for value in (successful_i, failed_i, incomplete_i))
        return successful_i, failed_i, incomplete_i, processed, has_completion_stats

    def _update_batch_progress_bar(
        self,
        progress_bar: tqdm | None,
        batch_job: Any,
        request_count: int | None,
        state_name: str,
        elapsed_seconds: float,
    ) -> None:
        if progress_bar is None:
            return

        successful_i, failed_i, incomplete_i, processed, has_completion_stats = self._get_batch_progress_counts(batch_job)
        if request_count is not None and progress_bar.total != request_count:
            progress_bar.total = request_count

        if has_completion_stats:
            current_progress = getattr(progress_bar, 'n', 0) or 0
            progress_bar.n = max(current_progress, min(processed, request_count or processed))
            progress_bar.set_postfix(
                state=state_name.replace('JOB_STATE_', '') or 'unknown',
                success=successful_i if successful_i is not None else '?',
                failed=failed_i if failed_i is not None else '?',
                incomplete=incomplete_i if incomplete_i is not None else '?',
                elapsed=self._format_elapsed(elapsed_seconds),
            )
        else:
            progress_bar.set_postfix(
                state=state_name.replace('JOB_STATE_', '') or 'unknown',
                processed='unavailable',
                last_known=f'{getattr(progress_bar, "n", 0)}/{request_count}' if request_count else getattr(progress_bar, 'n', 0),
                elapsed=self._format_elapsed(elapsed_seconds),
            )
        progress_bar.refresh()

    def _run_vertex_batch_job(
        self,
        llm: VertexAILLM,
        batch_input_uri: str,
        request_count: int | None = None,
        agent_label: str = 'batch',
        round_index: int | None = None,
        progress_bar: tqdm | None = None,
    ) -> tuple[Any, str, dict[str, dict[str, Any]]]:
        round_label = f' round {round_index}' if round_index is not None else ''
        request_label = f' ({request_count} request(s))' if request_count is not None else ''
        logger.info(f'Starting Vertex batch job: agent={agent_label}{round_label}{request_label} model={llm.model} src={batch_input_uri}')
        batch_job = llm.client.batches.create(
            model=llm.model,
            src=batch_input_uri,
        )
        logger.info(f'Started Vertex batch job: {getattr(batch_job, "name", "unknown")}')

        poll_delay_seconds = 10
        logger.info(f'Polling batch job status every {poll_delay_seconds} seconds')
        job_start_time = time.time()
        last_state_name = None
        last_heartbeat_time = 0.0
        warned_missing_live_progress = False
        heartbeat_seconds = 60
        while True:
            current_job = llm.execute_with_retry(
                llm.client.batches.get,
                name=batch_job.name,
                timeout_seconds=getattr(llm, 'request_timeout_seconds', 90),
            )
            state = getattr(current_job, 'state', None)
            state_name = str(getattr(state, 'name', state) or '').upper()
            elapsed_seconds = time.time() - job_start_time
            self._update_batch_progress_bar(progress_bar, current_job, request_count, state_name, elapsed_seconds)

            if state_name != last_state_name:
                progress_summary, has_completion_stats = self._format_batch_progress(
                    current_job,
                    request_count=request_count,
                    elapsed_seconds=elapsed_seconds,
                )
                logger.info(f'Batch job state: {state_name or state} ({progress_summary})')
                last_state_name = state_name
                last_heartbeat_time = time.time()
                if not has_completion_stats and not warned_missing_live_progress and 'RUNNING' in state_name:
                    logger.info(
                        'Vertex AI Gemini batch jobs do not expose live per-request progress; '
                        'continuing to poll job state and will count output rows after completion.'
                    )
                    warned_missing_live_progress = True
            elif time.time() - last_heartbeat_time >= heartbeat_seconds:
                progress_summary, has_completion_stats = self._format_batch_progress(
                    current_job,
                    request_count=request_count,
                    elapsed_seconds=elapsed_seconds,
                )
                logger.info(f'Batch job heartbeat: state={state_name or state} ({progress_summary})')
                last_heartbeat_time = time.time()

            if 'SUCCEEDED' in state_name or 'COMPLETED' in state_name or 'DONE' in state_name:
                batch_job = current_job
                break

            if 'FAILED' in state_name or 'CANCELLED' in state_name or 'CANCELED' in state_name:
                error = getattr(current_job, 'error', None)
                raise RuntimeError(f'Vertex batch job failed: {error or state_name}')

            time.sleep(poll_delay_seconds)

        batch_output_uri = self._resolve_batch_output_uri(batch_job)
        logger.info(f'Batch output URI: {batch_output_uri}')
        batch_results = self._load_batch_results(batch_output_uri, llm=llm)
        final_progress_summary, _ = self._format_batch_progress(
            batch_job,
            request_count=request_count,
            elapsed_seconds=time.time() - job_start_time,
        )
        logger.info(
            f'Batch job completed: agent={agent_label}{round_label}, '
            f'loaded_results={len(batch_results)}, {final_progress_summary}'
        )
        return batch_job, batch_output_uri, batch_results

    def _record_batch_usage(
        self,
        llm: VertexAILLM,
        prompt_text: str,
        raw_response: str,
        usage_metadata: dict[str, int],
        batch_job: Any,
        batch_input_uri: str,
        batch_output_uri: str,
        sample_key: str,
    ) -> None:
        input_tokens = usage_metadata.get('prompt_tokens') if usage_metadata else None
        output_tokens = usage_metadata.get('completion_tokens') if usage_metadata else None
        total_tokens = usage_metadata.get('total_tokens') if usage_metadata else None

        api_usage = {
            'batch_mode': True,
            'batch_job_name': getattr(batch_job, 'name', None),
            'batch_input_uri': batch_input_uri,
            'batch_output_uri': batch_output_uri,
            'batch_key': sample_key,
            'usage_metadata': usage_metadata or {},
        }

        if input_tokens is not None and output_tokens is None and total_tokens is not None:
            output_tokens = max(total_tokens - input_tokens, 0)
        elif output_tokens is not None and input_tokens is None and total_tokens is not None:
            input_tokens = max(total_tokens - output_tokens, 0)

        llm.track_usage(
            prompt_text,
            raw_response or '',
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            api_usage=api_usage,
        )

    @staticmethod
    def _get_marco_batch_scope(solver_batch_enabled: bool, reflector_batch_enabled: bool) -> str:
        if solver_batch_enabled and reflector_batch_enabled:
            return 'marco_solver_reflector'
        if solver_batch_enabled:
            return 'marco_solver'
        if reflector_batch_enabled:
            return 'marco_reflector'
        return 'none'

    def _run_vertex_batch_marco(self, data: list[tuple[str, int | float | str, pd.Series]], steps: int) -> None:
        if not isinstance(self.system, MARCOSystem):
            raise ValueError('MARCO Solver batch mode requires the MARCO system')
        if self.batch_gcs_uri_input:
            raise ValueError('--batch_gcs_uri_input is only supported for single-system batch runs')

        solver_batch_enabled = self.system.solver is not None and isinstance(self.system.solver.llm, VertexAILLM)
        reflector_batch_enabled = self.system.reflector is not None and isinstance(self.system.reflector.llm, VertexAILLM)

        if not solver_batch_enabled:
            logger.info('MARCO batch requested, but Solver is not Vertex AI; Solver will run normally.')
        if self.system.reflector and not reflector_batch_enabled:
            logger.info('MARCO batch requested, but Reflector is not Vertex AI; Reflector will run normally.')

        task_id = f"{self.dataset}_{self.task}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.task_id = task_id
        task_info = {
            'dataset': self.dataset,
            'task': self.task,
            'system': self.system.__class__.__name__,
            'model_override': self.model_override,
            'samples': len(data),
            'steps': steps,
            'max_history': self.max_his,
            'batch': True,
            'batch_scope': self._get_marco_batch_scope(solver_batch_enabled, reflector_batch_enabled),
            'batch_gcs_uri_prefix': self.batch_gcs_uri_prefix,
        }

        token_tracker.start_task(task_id, task_info)
        duration_tracker.start_task(task_id, task_info)
        self.sample_records = []
        token_tracker.reset_agent_stats(self.system)
        self.before_generate()

        contexts: dict[int, dict[str, Any]] = {}
        records: dict[int, dict[str, Any]] = {}
        prompts_by_sample: dict[int, str] = {}
        active_sample_ids: list[int] = []
        skipped_count = 0

        with tqdm(total=len(data), desc='MARCO setup: prepare active samples') as setup_pbar:
            for sample_idx, (test_data, gt_answer, data_sample) in enumerate(data, start=1):
                logger.info(f"Sample: {sample_idx}/{len(data)}")
                record = {
                    'sample_id': sample_idx,
                    'user_id': data_sample.get('user_id', 'unknown'),
                    '_skipped_no_gt': data_sample.get('_gt_not_in_candidates', False),
                }
                records[sample_idx] = record
                prompts_by_sample[sample_idx] = test_data

                if record['_skipped_no_gt']:
                    logger.info(f"Sample {sample_idx}: GT item not in candidates - skipping system call, counting as automatic failure")
                    self.system.reset(clear=True)
                    answer = [] if self.task == 'sr' else None
                    self.system.finished = False
                    self.system.answer = answer
                    for i in range(steps):
                        record[f'Answer_{i}'] = answer
                    self.after_iteration(answer=answer, gt_answer=gt_answer, record=record, pbar=setup_pbar)
                    setup_pbar.set_description('MARCO setup: prepare active samples')
                    self.sample_records.append(self._build_sample_result(record, gt_answer, data_sample, test_data))
                    skipped_count += 1
                    setup_pbar.set_postfix(active=len(active_sample_ids), skipped=skipped_count)
                    setup_pbar.update(1)
                    continue

                self.system.set_data(input=test_data, context="", gt_answer=gt_answer, data_sample=data_sample)
                self.system._current_sample_idx = sample_idx
                self.system._current_user_id = record['user_id']
                contexts[sample_idx] = self.system.prepare_solver_batch_context()
                active_sample_ids.append(sample_idx)
                setup_pbar.set_postfix(active=len(active_sample_ids), skipped=skipped_count)
                setup_pbar.update(1)

        logger.info(
            f'MARCO batch setup complete: {len(active_sample_ids)} active sample(s), '
            f'{skipped_count} skipped sample(s)'
        )

        with tqdm(
            total=len(data),
            initial=skipped_count,
            desc=f'MARCO batch completion: {skipped_count} skipped, {len(active_sample_ids)} active',
            disable=True,
        ) as pbar:
            round_index = 1
            max_reflections = 1 if self.system.reflector and self.system.enable_reflection_rerun else 0
            pending_sample_ids = active_sample_ids
            completed_sample_ids: set[int] = set()
            batch_uris: list[dict[str, Any]] = []
            pending_rerun_indices: dict[int, int] = {}
            pending_best_states: dict[int, dict[str, Any]] = {}

            while pending_sample_ids:
                next_pending: list[int] = []
                solved_sample_ids: list[int] = []

                if solver_batch_enabled:
                    entries: list[dict[str, Any]] = []
                    prompt_text_by_key: dict[str, str] = {}
                    sample_by_key: dict[str, int] = {}

                    for sample_idx in tqdm(
                        pending_sample_ids,
                        desc=f'Build Solver batch round {round_index}',
                        leave=False,
                    ):
                        self.system.restore_batch_state(contexts[sample_idx])
                        prompt_text = self.system.build_solver_batch_prompt()
                        sample_key = f'sample-{sample_idx}-round-{round_index}'
                        entries.append(self._build_vertex_batch_request(prompt=prompt_text, sample_key=sample_key, llm=self.system.solver.llm))
                        prompt_text_by_key[sample_key] = prompt_text
                        sample_by_key[sample_key] = sample_idx

                    local_input_file = self._write_vertex_batch_entries(entries)
                    try:
                        batch_input_uri = self._build_marco_solver_batch_input_uri(round_index, len(entries), local_input_file)
                    finally:
                        try:
                            os.unlink(local_input_file)
                        except OSError:
                            logger.debug(f'Failed to remove temporary batch input file: {local_input_file}')

                    pbar.set_postfix(batch=f'solver round {round_index}', requests=len(entries))
                    with duration_tracker.track_agent_call('solver'):
                        with tqdm(total=len(entries), desc=f'Waiting Solver batch round {round_index}', leave=False) as batch_pbar:
                            batch_job, batch_output_uri, batch_results = self._run_vertex_batch_job(
                                self.system.solver.llm,
                                batch_input_uri,
                                request_count=len(entries),
                                agent_label='solver',
                                round_index=round_index,
                                progress_bar=batch_pbar,
                            )
                    batch_uris.append({
                        'round': round_index,
                        'agent': 'solver',
                        'batch_input_uri': batch_input_uri,
                        'batch_output_uri': batch_output_uri,
                        'batch_job_name': getattr(batch_job, 'name', None),
                    })

                    pbar.set_postfix(batch=f'apply solver round {round_index}', results=len(sample_by_key))
                    for sample_key, sample_idx in tqdm(
                        sample_by_key.items(),
                        total=len(sample_by_key),
                        desc=f'Apply Solver batch round {round_index}',
                        leave=False,
                    ):
                        _test_data, gt_answer, data_sample = data[sample_idx - 1]
                        record = records[sample_idx]
                        batch_record = batch_results.get(sample_key, {})
                        response_payload = batch_record.get('response') if isinstance(batch_record, dict) else None
                        raw_response = extract_batch_response_text(response_payload if response_payload is not None else batch_record)
                        usage_payload = response_payload if response_payload is not None else batch_record
                        usage_metadata = extract_batch_usage_metadata(usage_payload)

                        self._record_batch_usage(
                            self.system.solver.llm,
                            prompt_text_by_key[sample_key],
                            raw_response,
                            usage_metadata,
                            batch_job,
                            batch_input_uri,
                            batch_output_uri,
                            sample_key,
                        )

                        self.system.restore_batch_state(contexts[sample_idx])
                        self.system.apply_solver_batch_solution(raw_response)
                        if sample_idx in pending_rerun_indices:
                            rerun_index = pending_rerun_indices.pop(sample_idx)
                            rerun_info = self.system.reflection_all_reruns[rerun_index]
                            position_after = self.system._get_ground_truth_position(self.system._last_final_answer)
                            rerun_info['position_after'] = position_after
                            rerun_info['answer_after'] = (
                                self.system.answer.copy() if isinstance(self.system.answer, list) else self.system.answer
                            )
                            position_before = rerun_info.get('position_before', -1)
                            best_state = pending_best_states.pop(sample_idx, {})
                            best_answer = best_state.get('best_answer')
                            best_answer_position = best_state.get('best_answer_position', position_before)
                            if position_before > 0 and position_after > 0 and position_after < position_before:
                                self.system.reflection_improvements.append(rerun_info.copy())
                                best_answer = self.system.answer.copy() if isinstance(self.system.answer, list) else self.system.answer
                                best_answer_position = position_after
                            elif position_before > 0 and position_after > 0 and position_after > position_before and best_answer is not None:
                                self.system.answer = best_answer.copy() if isinstance(best_answer, list) else best_answer
                            current_position = self.system._get_ground_truth_position(self.system.answer)
                            if best_answer is not None and current_position != best_answer_position:
                                self.system.answer = best_answer.copy() if isinstance(best_answer, list) else best_answer
                        record[f'Answer_{round_index - 1}'] = self.system.answer
                        record['batch_key'] = sample_key
                        record['batch_raw_response'] = raw_response
                        if isinstance(batch_record, dict):
                            record['batch_status'] = batch_record.get('status', '')
                            if batch_record.get('status') and not raw_response:
                                record['batch_error'] = batch_record.get('status')

                        contexts[sample_idx] = self.system.snapshot_batch_state()
                        solved_sample_ids.append(sample_idx)
                else:
                    for sample_idx in tqdm(
                        pending_sample_ids,
                        desc=f'Run Solver round {round_index}',
                        leave=False,
                    ):
                        _test_data, gt_answer, data_sample = data[sample_idx - 1]
                        record = records[sample_idx]
                        self.system.restore_batch_state(contexts[sample_idx])
                        self.system._solving_phase()
                        if sample_idx in pending_rerun_indices:
                            rerun_index = pending_rerun_indices.pop(sample_idx)
                            rerun_info = self.system.reflection_all_reruns[rerun_index]
                            position_after = self.system._get_ground_truth_position(self.system._last_final_answer)
                            rerun_info['position_after'] = position_after
                            rerun_info['answer_after'] = (
                                self.system.answer.copy() if isinstance(self.system.answer, list) else self.system.answer
                            )
                            position_before = rerun_info.get('position_before', -1)
                            best_state = pending_best_states.pop(sample_idx, {})
                            best_answer = best_state.get('best_answer')
                            best_answer_position = best_state.get('best_answer_position', position_before)
                            if position_before > 0 and position_after > 0 and position_after < position_before:
                                self.system.reflection_improvements.append(rerun_info.copy())
                                best_answer = self.system.answer.copy() if isinstance(self.system.answer, list) else self.system.answer
                                best_answer_position = position_after
                            elif position_before > 0 and position_after > 0 and position_after > position_before and best_answer is not None:
                                self.system.answer = best_answer.copy() if isinstance(best_answer, list) else best_answer
                            current_position = self.system._get_ground_truth_position(self.system.answer)
                            if best_answer is not None and current_position != best_answer_position:
                                self.system.answer = best_answer.copy() if isinstance(best_answer, list) else best_answer
                        record[f'Answer_{round_index - 1}'] = self.system.answer
                        contexts[sample_idx] = self.system.snapshot_batch_state()
                        solved_sample_ids.append(sample_idx)

                reflector_feedback_by_sample: dict[int, tuple[bool, dict]] = {}
                if self.system.reflector and solved_sample_ids and reflector_batch_enabled:
                    reflector_entries: list[dict[str, Any]] = []
                    reflector_prompt_by_key: dict[str, str] = {}
                    reflector_sample_by_key: dict[str, int] = {}

                    for sample_idx in tqdm(
                        solved_sample_ids,
                        desc=f'Build Reflector batch round {round_index}',
                        leave=False,
                    ):
                        self.system.restore_batch_state(contexts[sample_idx])
                        reflector_prompt = self.system.build_reflector_batch_prompt()
                        reflector_key = f'reflector-sample-{sample_idx}-round-{round_index}'
                        reflector_entries.append(
                            self._build_vertex_batch_request(
                                prompt=reflector_prompt,
                                sample_key=reflector_key,
                                llm=self.system.reflector.llm,
                            )
                        )
                        reflector_prompt_by_key[reflector_key] = reflector_prompt
                        reflector_sample_by_key[reflector_key] = sample_idx

                    local_reflector_input_file = self._write_vertex_batch_entries(reflector_entries)
                    try:
                        reflector_input_uri = self._build_marco_reflector_batch_input_uri(
                            round_index,
                            len(reflector_entries),
                            local_reflector_input_file,
                        )
                    finally:
                        try:
                            os.unlink(local_reflector_input_file)
                        except OSError:
                            logger.debug(f'Failed to remove temporary reflector batch input file: {local_reflector_input_file}')

                    pbar.set_postfix(batch=f'reflector round {round_index}', requests=len(reflector_entries))
                    with duration_tracker.track_agent_call('reflector'):
                        with tqdm(total=len(reflector_entries), desc=f'Waiting Reflector batch round {round_index}', leave=False) as batch_pbar:
                            reflector_job, reflector_output_uri, reflector_results = self._run_vertex_batch_job(
                                self.system.reflector.llm,
                                reflector_input_uri,
                                request_count=len(reflector_entries),
                                agent_label='reflector',
                                round_index=round_index,
                                progress_bar=batch_pbar,
                            )
                    batch_uris.append({
                        'round': round_index,
                        'agent': 'reflector',
                        'batch_input_uri': reflector_input_uri,
                        'batch_output_uri': reflector_output_uri,
                        'batch_job_name': getattr(reflector_job, 'name', None),
                    })

                    pbar.set_postfix(batch=f'apply reflector round {round_index}', results=len(reflector_sample_by_key))
                    for reflector_key, sample_idx in tqdm(
                        reflector_sample_by_key.items(),
                        total=len(reflector_sample_by_key),
                        desc=f'Apply Reflector batch round {round_index}',
                        leave=False,
                    ):
                        reflector_record = reflector_results.get(reflector_key, {})
                        reflector_payload = reflector_record.get('response') if isinstance(reflector_record, dict) else None
                        reflector_response = extract_batch_response_text(
                            reflector_payload if reflector_payload is not None else reflector_record
                        )
                        reflector_usage_payload = reflector_payload if reflector_payload is not None else reflector_record
                        reflector_usage_metadata = extract_batch_usage_metadata(reflector_usage_payload)

                        self._record_batch_usage(
                            self.system.reflector.llm,
                            reflector_prompt_by_key[reflector_key],
                            reflector_response,
                            reflector_usage_metadata,
                            reflector_job,
                            reflector_input_uri,
                            reflector_output_uri,
                            reflector_key,
                        )

                        self.system.restore_batch_state(contexts[sample_idx])
                        reflector_feedback_by_sample[sample_idx] = self.system.apply_reflector_batch_response(reflector_response)
                        contexts[sample_idx] = self.system.snapshot_batch_state()
                elif self.system.reflector and solved_sample_ids:
                    for sample_idx in tqdm(
                        solved_sample_ids,
                        desc=f'Run Reflector round {round_index}',
                        leave=False,
                    ):
                        self.system.restore_batch_state(contexts[sample_idx])
                        if self.system.enable_reflection_rerun:
                            reflector_feedback_by_sample[sample_idx] = self.system._perform_reflection()
                        else:
                            self.system._perform_reflection_logging_only()
                            reflector_feedback_by_sample[sample_idx] = (
                                False,
                                {'planner_correct': True, 'solver_correct': True, 'planner_reason': '', 'solver_reason': ''},
                            )
                        contexts[sample_idx] = self.system.snapshot_batch_state()

                rerun_count_before = len(next_pending)
                for sample_idx in tqdm(
                    solved_sample_ids,
                    desc=f'Finalize round {round_index}',
                    leave=False,
                ):
                    _test_data, gt_answer, data_sample = data[sample_idx - 1]
                    record = records[sample_idx]
                    self.system.restore_batch_state(contexts[sample_idx])
                    rerun_requested = False

                    if self.system.reflector:
                        should_continue_reflecting, feedback_info = reflector_feedback_by_sample.get(
                            sample_idx,
                            (False, {'planner_correct': True, 'solver_correct': True, 'planner_reason': '', 'solver_reason': ''}),
                        )
                        if self.system.enable_reflection_rerun and round_index <= max_reflections and should_continue_reflecting:
                            original_position_before_reflection = getattr(self.system, '_gt_position_before_reflection', -1)
                            best_answer = self.system.answer.copy() if isinstance(self.system.answer, list) else self.system.answer
                            planner_correct = feedback_info.get('planner_correct', True)
                            solver_correct = feedback_info.get('solver_correct', True)
                            planner_reason = feedback_info.get('planner_reason', '')
                            solver_reason = feedback_info.get('solver_reason', '')
                            position_before = original_position_before_reflection

                            if not planner_correct:
                                logger.info(f"Planner feedback triggered: {planner_reason}")
                                self.system.reset(preserve_progress=True)
                                if 'reflections' not in self.system.planner_kwargs:
                                    self.system.planner_kwargs['reflections'] = ""
                                reflection_feedback = "\n=== Planning Improvement Required (Reflection Feedback) ===\n"
                                reflection_feedback += f"{planner_reason}\n"
                                reflection_feedback += "CRITICAL: Revise your plan to address this specific issue.\n"
                                self.system.planner_kwargs['reflections'] += reflection_feedback

                                if not solver_correct:
                                    logger.info(f"Solver feedback also triggered: {solver_reason}")
                                    solver_reflection_feedback = "\n=== Solver Improvement Required (Reflection Feedback) ===\n"
                                    solver_reflection_feedback += f"{solver_reason}\n"
                                    solver_reflection_feedback += "CRITICAL: Adjust your ranking to address this specific issue.\n"
                                    self.system.manager_kwargs['solver_reflections'] = solver_reflection_feedback

                                contexts[sample_idx] = self.system.prepare_solver_batch_context(reset=False)
                                next_pending.append(sample_idx)
                                rerun_requested = True
                                feedback_type = 'both' if not solver_correct else 'planner'

                            elif not solver_correct:
                                logger.info(f"Solver feedback triggered (solver-only reranking): {solver_reason}")
                                previous_ranking = self.system._last_final_answer if hasattr(self.system, '_last_final_answer') else []
                                previous_solution = self.system._last_solution if hasattr(self.system, '_last_solution') else ""
                                feedback_message = f"Previous ranking: {previous_ranking}\n"
                                if previous_solution:
                                    feedback_message += f"Previous solution: {previous_solution}\n\n"
                                feedback_message += f"Feedback: {solver_reason}\n\n"
                                feedback_message += "Required action: Review previous ranking, understand the feedback, re-analyze user preferences and items, and produce an improved ranking that addresses the feedback. The new ranking MUST be different from the previous one.\n"
                                self.system.manager_kwargs['solver_reflections'] = feedback_message
                                contexts[sample_idx] = self.system.snapshot_batch_state()
                                next_pending.append(sample_idx)
                                rerun_requested = True
                                feedback_type = 'solver'

                            if rerun_requested:
                                rerun_info = {
                                    'sample_idx': sample_idx,
                                    'user_id': record['user_id'],
                                    'gt_item': gt_answer,
                                    'position_before': position_before,
                                    'position_after': position_before,
                                    'answer_before': best_answer.copy() if isinstance(best_answer, list) else best_answer,
                                    'answer_after': self.system.answer.copy() if isinstance(self.system.answer, list) else self.system.answer,
                                    'feedback_type': feedback_type,
                                }
                                self.system.reflection_all_reruns.append(rerun_info)
                                pending_rerun_indices[sample_idx] = len(self.system.reflection_all_reruns) - 1
                                pending_best_states[sample_idx] = {
                                    'best_answer': best_answer.copy() if isinstance(best_answer, list) else best_answer,
                                    'best_answer_position': position_before,
                                }

                    if not rerun_requested:
                        completed_sample_ids.add(sample_idx)
                        self.after_iteration(answer=self.system.answer, gt_answer=gt_answer, record=record, pbar=pbar)
                        self.sample_records.append(self._build_sample_result(record, gt_answer, data_sample, prompts_by_sample[sample_idx]))
                        pbar.update(1)

                round_reruns = len(next_pending) - rerun_count_before
                if round_reruns:
                    logger.info(f'MARCO batch round {round_index}: {round_reruns} sample(s) queued for rerun')
                pbar.set_description(
                    f'MARCO batch completion: {len(completed_sample_ids)} active done, '
                    f'{len(next_pending)} rerun pending'
                )
                pending_sample_ids = next_pending
                round_index += 1

            for sample_idx in tqdm(active_sample_ids, desc='Finalize unfinished active samples', leave=False):
                if sample_idx in completed_sample_ids:
                    continue
                _test_data, gt_answer, data_sample = data[sample_idx - 1]
                record = records[sample_idx]
                self.system.restore_batch_state(contexts[sample_idx])
                self.after_iteration(answer=self.system.answer, gt_answer=gt_answer, record=record, pbar=pbar)
                self.sample_records.append(self._build_sample_result(record, gt_answer, data_sample, prompts_by_sample[sample_idx]))
                pbar.update(1)

        token_tracker.collect_system_stats(self.system)
        final_stats = token_tracker.end_task()
        duration_stats = duration_tracker.end_task()
        self._apply_adjusted_duration(final_stats, duration_stats)
        self.final_stats = final_stats
        self.duration_stats = duration_stats

        logger.success("=== Token Usage Summary ===")
        logger.success(f"Task: {self.dataset} {self.task} ({len(data)} samples)")
        logger.success(f"Data file: {self.data_file}")
        logger.success(f"System config: {self.system_config}")
        logger.success(f"Total API calls: {final_stats.get('total_api_calls', 0)}")
        logger.success(f"Total tokens: {final_stats.get('total_tokens', 0)}")
        logger.success(f"Input tokens: {final_stats.get('total_input_tokens', 0)}")
        logger.success(f"Output tokens: {final_stats.get('total_output_tokens', 0)}")
        logger.success(f"Models used: {final_stats.get('models_used', [])}")
        logger.success(f"Duration: {final_stats.get('duration', 0):.2f}s")
        excluded_duration = duration_stats.get('total_excluded_duration', 0) if isinstance(duration_stats, dict) else 0
        if excluded_duration:
            logger.success(f"Excluded retry time: {excluded_duration:.2f}s")
        self.log_run_arguments()
        self._log_per_agent_statistics(final_stats, duration_stats)

        self.after_generate()
        result_payload = self.build_result_payload(final_stats, duration_stats)
        result_payload.setdefault('run_info', {}).setdefault('config', {})['batch'] = True
        result_payload['run_info']['config']['batch_scope'] = self._get_marco_batch_scope(
            solver_batch_enabled,
            reflector_batch_enabled,
        )
        result_payload['run_info']['config']['batch_gcs_uri_prefix'] = self.batch_gcs_uri_prefix
        result_payload['run_info']['config']['batch_rounds'] = batch_uris
        self.save_result_payload(result_payload)

    def generate(self, data: list[tuple[str, int | float | str, pd.Series]], steps: int = 2):
        if self._is_vertex_batch_enabled():
            if isinstance(self.system, SingleSystem):
                self._run_vertex_batch_single(data, steps)
                return
            if isinstance(self.system, MARCOSystem):
                solver_batch_enabled = self.system.solver is not None and isinstance(self.system.solver.llm, VertexAILLM)
                reflector_batch_enabled = self.system.reflector is not None and isinstance(self.system.reflector.llm, VertexAILLM)
                if not solver_batch_enabled and not reflector_batch_enabled:
                    logger.warning(
                        '--batch was requested for MARCO, but neither Solver nor Reflector uses Vertex AI; '
                        'running the normal MARCO workflow.'
                    )
                else:
                    self._run_vertex_batch_marco(data, steps)
                    return
            elif not isinstance(self.system, MARCOSystem):
                raise ValueError('Batch mode is only supported for single and MARCO systems')

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
        self._apply_adjusted_duration(final_stats, duration_stats)

        self.final_stats = final_stats
        self.duration_stats = duration_stats
        
        logger.success("=== Token Usage Summary ===")
        logger.success(f"Task: {self.dataset} {self.task} ({len(data)} samples)")
        logger.success(f"Data file: {self.data_file}")
        logger.success(f"System config: {self.system_config}")
        logger.success(f"Total API calls: {final_stats.get('total_api_calls', 0)}")
        logger.success(f"Total tokens: {final_stats.get('total_tokens', 0)}")
        logger.success(f"Input tokens: {final_stats.get('total_input_tokens', 0)}")
        logger.success(f"Output tokens: {final_stats.get('total_output_tokens', 0)}")
        logger.success(f"Models used: {final_stats.get('models_used', [])}")
        logger.success(f"Duration: {final_stats.get('duration', 0):.2f}s")
        excluded_duration = duration_stats.get('total_excluded_duration', 0) if isinstance(duration_stats, dict) else 0
        if excluded_duration:
            logger.success(f"Excluded retry time: {excluded_duration:.2f}s")
        self.log_run_arguments()
        
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

    def run(self, api_config: str, dataset: str, data_file: str, system: str, system_config: str, task: str, max_his: int, provider: str = None, model: str = None, enable_reflection_rerun: bool = True, batch: bool = False, batch_gcs_uri_prefix: str = '', batch_gcs_uri_input: str = '', no_cache: bool = False):
        if dataset == 'None':
            dataset = os.path.basename(os.path.dirname(data_file))
        self.dataset = dataset
        self.task = task
        self.max_his = max_his
        self.data_file = data_file
        self.system_config = system_config
        self.batch_mode = batch
        self.batch_gcs_uri_prefix = batch_gcs_uri_prefix
        self.batch_gcs_uri_input = batch_gcs_uri_input
        self.no_cache = no_cache
        self.effective_args = {
            'api_config': api_config,
            'dataset': dataset,
            'data_file': data_file,
            'system': system,
            'system_config': system_config,
            'task': task,
            'max_his': max_his,
            'provider': provider,
            'model': model,
            'enable_reflection_rerun': enable_reflection_rerun,
            'batch': batch,
            'batch_gcs_uri_prefix': batch_gcs_uri_prefix,
            'batch_gcs_uri_input': batch_gcs_uri_input,
            'no_cache': no_cache,
        }
        
        data_dir = os.path.dirname(data_file)
        
        init_api(read_json(api_config))

        if self.batch_mode and self.batch_gcs_uri_input and not self.batch_gcs_uri_input.startswith('gs://'):
            raise ValueError('--batch_gcs_uri_input must be a gs:// URI')
        if self.batch_mode and system == 'marco' and self.batch_gcs_uri_input:
            raise ValueError('--batch_gcs_uri_input is only supported for single-system batch runs')
        
        if provider:
            provider_info = self._parse_provider_options(provider, model)
            self.model_override = provider_info['model']
            self.provider = provider_info['provider']
            self.effective_args['provider'] = self.provider
            self.effective_args['model'] = self.model_override
            self.system_kwargs = {
                'task': self.task,
                'leak': False,
                'dataset': self.dataset,
                'data_dir': data_dir,
                'model_override': self.model_override,
                'provider': self.provider,
                'enable_reflection_rerun': enable_reflection_rerun,
                'api_config_path': api_config,
                'no_cache': no_cache,
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
                'no_cache': no_cache,
            }
            logger.info(f"No provider/model specified - using individual agent configurations")
        
        data_df = self.get_data(data_file, max_his)
        
        self.get_system(system, system_config)
        if self.batch_mode and not isinstance(self.system, (SingleSystem, MARCOSystem)):
            raise ValueError('Batch mode is only supported for single and MARCO systems')
        data = self.prompt_data(data_df)
        
        self.setup_task_logger(task=task, dataset=dataset, system=system, num_samples=len(data))
        
        self.generate(data, steps=self.running_steps)
    
    def _parse_provider_options(self, provider: str, model: str = None) -> dict:
        def _get_default(provider_name: str) -> str:
            default_map = {
                'openrouter': 'google/gemini-2.0-flash-001',
                'openai': 'gpt-4o-mini',
                'codexhub': 'oc/deepseek-v4-flash-free',
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
