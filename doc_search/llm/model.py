from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

from doc_search.llm import factory
from doc_search.settings import LLMSetting


@factory.register_builder("ollama")
def build_ollama_as_llm(config: LLMSetting, prompt: str | None = None):
    return Ollama(
        model=config.model,
        system_prompt=prompt or config.system_prompt,
        request_timeout=config.request_timeout,
        base_url=f"http://{config.host}:{config.port}",
    )

