#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from loguru import logger

from marco.llms import CodexHubLLM


DEFAULT_BASE_URL = "https://api.codexhub.click/v1"
DEFAULT_MODEL = "oc/deepseek-v4-flash-free"
DEFAULT_CONFIG = "config/api-config.json"


def _flatten_api_key(value):
    if isinstance(value, list):
        for item in value:
            if item:
                return item
        return ""
    return value or ""


def _load_key_from_config(config_path: str) -> str:
    path = Path(config_path)
    if not path.exists():
        return ""

    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    providers = config.get("providers", {})
    codexhub = providers.get("codexhub", {})
    return _flatten_api_key(
        codexhub.get("api_key")
        or codexhub.get("api_keys")
        or config.get("codexhub_api_key")
        or config.get("api_key")
    )


def main():
    parser = argparse.ArgumentParser(description="Smoke test the CodexHub provider.")
    parser.add_argument("--api-key", default="", help="CodexHub API key. Falls back to CODEXHUB_API_KEY and config/api-config.json.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="CodexHub base URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name to test.")
    parser.add_argument("--config-file", default=DEFAULT_CONFIG, help="API config file to read if --api-key is not provided.")
    parser.add_argument("--prompt", default="Hello!", help="Prompt to send.")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("CODEXHUB_API_KEY") or _load_key_from_config(args.config_file)
    if not api_key:
        raise SystemExit("Missing CodexHub API key. Set CODEXHUB_API_KEY, pass --api-key, or add providers.codexhub.api_key in the config file.")

    llm = CodexHubLLM(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        config_file=args.config_file,
        json_mode=False,
        agent_context="CodexHubSmokeTest",
    )

    logger.info(f"Testing CodexHub model={llm.model} base_url={llm.base_url}")
    response = llm(args.prompt)
    print(response)


if __name__ == "__main__":
    main()
