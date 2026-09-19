import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Callable

from loguru import logger

from marco.utils.duration_tracking import duration_tracker


class LLMCache:
    def __init__(self, cache_dir: str = 'cached') -> None:
        self.cache_dir = cache_dir

    @staticmethod
    def _json_safe(value: Any) -> Any:
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except TypeError:
            if isinstance(value, dict):
                return {str(key): LLMCache._json_safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [LLMCache._json_safe(item) for item in value]
            return str(value)

    def _cache_identity(self, llm: Any, prompt: Any) -> dict[str, Any]:
        return {
            'provider': llm.__class__.__name__,
            'model': getattr(llm, 'model', None),
            'json_mode': getattr(llm, 'json_mode', False),
            'max_tokens': getattr(llm, 'max_tokens', None),
            'temperature': getattr(llm, 'temperature', None),
            'top_p': getattr(llm, 'top_p', None),
            'top_k': getattr(llm, 'top_k', None),
            'prompt': self._json_safe(prompt),
        }

    def cache_path(self, llm: Any, prompt: Any) -> str:
        identity = self._cache_identity(llm, prompt)
        key_payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        cache_key = hashlib.sha256(key_payload.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f'{cache_key}.json')

    @staticmethod
    def _is_successful_response(response: str) -> bool:
        if not isinstance(response, str):
            return False
        return bool(response.strip()) and not response.lstrip().startswith('Error:')

    @staticmethod
    def _quarantine_invalid_cache(path: str) -> None:
        if not os.path.exists(path):
            return

        invalid_dir = os.path.join(os.path.dirname(path), 'invalid')
        os.makedirs(invalid_dir, exist_ok=True)
        target = os.path.join(invalid_dir, os.path.basename(path))

        if os.path.exists(target):
            root, ext = os.path.splitext(target)
            target = f'{root}.{int(time.time() * 1000)}{ext}'

        os.replace(path, target)
        logger.warning(f"Invalid LLM cache moved aside: {path} -> {target}")

    @staticmethod
    def _replay_usage(llm: Any, prompt: Any, response: str, call_info: dict[str, Any]) -> None:
        input_tokens = call_info.get('input_tokens')
        output_tokens = call_info.get('output_tokens')

        if call_info.get('estimated_input', False):
            input_tokens = None
        if call_info.get('estimated_output', False):
            output_tokens = None

        llm.track_usage(
            prompt,
            response,
            input_tokens,
            output_tokens,
            api_usage=call_info.get('api_usage', {}),
        )

    def call(
        self,
        llm: Any,
        prompt: Any,
        invoke: Callable[[], str],
        agent_name: str,
        no_cache: bool = False,
        response_validator: Callable[[str], bool] | None = None,
    ) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        path = self.cache_path(llm, prompt)
        cache_invalid_ignored = False

        if not no_cache and os.path.exists(path):
            cache_start = time.time()
            with open(path, 'r', encoding='utf-8') as cache_file:
                cached = json.load(cache_file)

            response = cached.get('response', '')
            if response_validator is not None and not response_validator(response):
                cache_invalid_ignored = True
                duration_tracker.exclude_agent_duration(agent_name, time.time() - cache_start)
                logger.warning(
                    f"LLM cache ignored invalid response ({agent_name}, {getattr(llm, 'model', 'unknown')}): {path}"
                )
                self._quarantine_invalid_cache(path)
            else:
                self._replay_usage(llm, prompt, response, cached.get('call_info', {}))

                runtime_seconds = float(cached.get('runtime_seconds') or 0.0)
                duration_tracker.add_agent_duration_without_call(agent_name, runtime_seconds)
                duration_tracker.exclude_agent_duration(agent_name, time.time() - cache_start)
                logger.debug(f"LLM cache hit ({agent_name}, {getattr(llm, 'model', 'unknown')}): {path}")
                return response
        call_count_before = len(getattr(llm, 'call_history', []))
        excluded_before = duration_tracker.agent_excluded_durations.get(agent_name, 0.0)
        start_time = time.time()
        response = invoke()
        elapsed = time.time() - start_time
        excluded_after = duration_tracker.agent_excluded_durations.get(agent_name, 0.0)
        runtime_seconds = max(elapsed - max(excluded_after - excluded_before, 0.0), 0.0)

        call_history = getattr(llm, 'call_history', [])
        if (
            (cache_invalid_ignored or not os.path.exists(path))
            and len(call_history) > call_count_before
            and self._is_successful_response(response)
            and (response_validator is None or response_validator(response))
        ):
            call_info = call_history[-1].copy()
            payload = {
                'cache_version': 1,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'provider': llm.__class__.__name__,
                'model': getattr(llm, 'model', None),
                'agent_name': agent_name,
                'json_mode': getattr(llm, 'json_mode', False),
                'prompt': self._json_safe(prompt),
                'response': response,
                'runtime_seconds': runtime_seconds,
                'call_info': self._json_safe(call_info),
                'llm': {
                    'class_name': llm.__class__.__name__,
                    'model': getattr(llm, 'model', None),
                    'max_tokens': getattr(llm, 'max_tokens', None),
                    'temperature': getattr(llm, 'temperature', None),
                    'top_p': getattr(llm, 'top_p', None),
                    'top_k': getattr(llm, 'top_k', None),
                    'agent_context': getattr(llm, 'agent_context', None),
                },
            }
            directory = os.path.dirname(path)
            fd, tmp_path = tempfile.mkstemp(prefix='.tmp-', suffix='.json', dir=directory)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as tmp_file:
                    json.dump(payload, tmp_file, indent=2, ensure_ascii=False)
                os.replace(tmp_path, path)
                logger.debug(f"LLM cache stored ({agent_name}, {getattr(llm, 'model', 'unknown')}): {path}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        return response
