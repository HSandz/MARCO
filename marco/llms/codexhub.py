from typing import Optional

from marco.llms.openai import OpenAILLM

DEFAULT_CODEXHUB_MODEL = 'oc/deepseek-v4-flash-free'
DEFAULT_CODEXHUB_BASE_URL = 'https://api.codexhub.click/v1'


class CodexHubLLM(OpenAILLM):
    def __init__(
        self,
        model: str = DEFAULT_CODEXHUB_MODEL,
        api_key: str = '',
        base_url: str = DEFAULT_CODEXHUB_BASE_URL,
        json_mode: bool = False,
        agent_context: Optional[str] = None,
        config_file: str = 'config/api-config.json',
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            json_mode=json_mode,
            agent_context=agent_context,
            config_file=config_file,
            provider_name='codexhub',
            api_key_env_var='CODEXHUB_API_KEY',
            default_model=DEFAULT_CODEXHUB_MODEL,
            default_base_url=DEFAULT_CODEXHUB_BASE_URL,
            *args,
            **kwargs,
        )
