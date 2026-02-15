import os
import random
from typing import Dict, Iterator, Tuple

import numpy as np
from loguru import logger

def _normalize_provider_name(value: str | None) -> str:
    return (value or '').strip().lower()

def _provider_aliases(name: str, config: dict) -> set[str]:
    aliases = {
        _normalize_provider_name(name),
        _normalize_provider_name(config.get('type')),
        _normalize_provider_name(config.get('provider')),
    }
    for alias in config.get('aliases', []):
        aliases.add(_normalize_provider_name(alias))
    return {alias for alias in aliases if alias}

def iter_provider_settings(api_config: dict) -> Iterator[Tuple[str, Dict]]:
    if not isinstance(api_config, dict):
        raise ValueError("API configuration must be a dictionary")

    if 'providers' in api_config:
        providers: dict = api_config.get('providers', {})
        if not providers:
            raise ValueError("Multi-provider API config requires at least one entry in 'providers'")

        default_entry = api_config.get('default_provider')
        yielded: set[str] = set()

        def _yield_entry(entry_name: str) -> Iterator[Tuple[str, Dict]]:
            cfg = providers[entry_name].copy()
            provider = _normalize_provider_name(cfg.get('type') or entry_name)
            cfg.setdefault('provider', provider)
            yielded.add(entry_name)
            yield provider, cfg

        if default_entry and default_entry in providers:
            yield from _yield_entry(default_entry)

        for name in providers:
            if name not in yielded:
                yield from _yield_entry(name)
        return

    provider = _normalize_provider_name(api_config.get('provider', 'openrouter'))
    cfg = api_config.copy()
    cfg.setdefault('provider', provider or 'openrouter')

    yield provider or 'openrouter', cfg

def get_provider_settings(api_config: dict, provider_name: str | None = None) -> Tuple[str, Dict]:
    target = _normalize_provider_name(provider_name)

    if 'providers' in api_config:
        providers: dict = api_config.get('providers', {})
        if not providers:
            raise ValueError("Multi-provider API config requires at least one entry in 'providers'")

        if target:
            for name, cfg in providers.items():
                if target in _provider_aliases(name, cfg):
                    result = cfg.copy()
                    provider = _normalize_provider_name(result.get('type') or name)
                    result.setdefault('provider', provider)
                    return provider, result
            raise ValueError(f"Provider '{provider_name}' not found in API configuration.")

        default_entry = api_config.get('default_provider')
        if default_entry and default_entry in providers:
            result = providers[default_entry].copy()
            provider = _normalize_provider_name(result.get('type') or default_entry)
            result.setdefault('provider', provider)
            return provider, result

        name, cfg = next(iter(providers.items()))
        result = cfg.copy()
        provider = _normalize_provider_name(result.get('type') or name)
        result.setdefault('provider', provider)
        return provider, result

    provider = _normalize_provider_name(api_config.get('provider', 'openrouter'))
    cfg = api_config.copy()

    if target and provider != target:
        raise ValueError(f"Requested provider '{target}' does not match config provider '{provider}'.")

    cfg.setdefault('provider', provider or 'openrouter')
    return cfg['provider'], cfg

def init_gemini_api(api_key: str):
    if not api_key:
        raise ValueError("Gemini API key cannot be empty")

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
    except ImportError as exc:
        raise ImportError(
            "google-generativeai package is required for Gemini API. Install it with: pip install google-generativeai"
        ) from exc

def init_openrouter_api(api_key: str):
    if not api_key:
        raise ValueError("OpenRouter API key cannot be empty")

def init_openai_api(api_key: str, base_url: str | None = None):
    if not api_key:
        raise ValueError("OpenAI API key cannot be empty")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai package is required for OpenAI API. Install it with: pip install openai>=1.0.0"
        ) from exc

    os.environ['OPENAI_API_KEY'] = api_key
    if base_url:
        OpenAI(api_key=api_key, base_url=base_url)
    else:
        OpenAI(api_key=api_key)

def init_huggingface_api(token: str):
    if token:
        os.environ['HF_TOKEN'] = token
        os.environ['HUGGINGFACE_TOKEN'] = token
    logger.debug("HuggingFace provider configured for local inference")

def init_api(api_config: dict):
    initialized = False

    for provider, provider_cfg in iter_provider_settings(api_config):
        if provider == 'openrouter':
            api_keys = provider_cfg.get('api_keys', [])
            if api_keys and isinstance(api_keys, list) and api_keys[0]:
                init_openrouter_api(api_keys[0])
                initialized = True
            else:
                api_key = provider_cfg.get('api_key') or provider_cfg.get('openrouter_api_key')
                if api_key:
                    init_openrouter_api(api_key)
                    initialized = True
                else:
                    logger.info("Skip OpenRouter provider: api_key is empty.")
        elif provider == 'openai':
            api_keys = provider_cfg.get('api_keys', [])
            base_url = provider_cfg.get('base_url')
            if api_keys and isinstance(api_keys, list) and api_keys[0]:
                init_openai_api(api_keys[0], base_url=base_url)
                initialized = True
            else:
                api_key = provider_cfg.get('api_key') or provider_cfg.get('openai_api_key') or provider_cfg.get('openai_key')
                if api_key:
                    init_openai_api(api_key, base_url=base_url)
                    initialized = True
                else:
                    logger.info("Skip OpenAI provider: api_key is empty.")
        elif provider == 'gemini':
            api_keys = provider_cfg.get('api_keys', [])
            if api_keys and isinstance(api_keys, list) and api_keys[0]:
                init_gemini_api(api_keys[0])
                initialized = True
            else:
                api_key = provider_cfg.get('api_key') or provider_cfg.get('gemini_api_key')
                if api_key:
                    init_gemini_api(api_key)
                    initialized = True
                else:
                    logger.info("Skip Gemini provider: api_key is empty.")
        elif provider == 'ollama':
            base_url = provider_cfg.get('base_url')
            if base_url:
                os.environ.setdefault('OLLAMA_BASE_URL', base_url)
            logger.debug(f"Ollama provider configured at {base_url or 'http://localhost:11434'}")
        elif provider == 'huggingface':
            token = provider_cfg.get('token') or provider_cfg.get('api_key') or provider_cfg.get('huggingface_token')
            init_huggingface_api(token)
            logger.info("HuggingFace provider configured for local inference")
        else:
            logger.warning(f"Unsupported API provider '{provider}', skipping.")

    if not initialized:
        logger.warning("No cloud API providers were initialized. Only local models may be available.")

def init_all_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
