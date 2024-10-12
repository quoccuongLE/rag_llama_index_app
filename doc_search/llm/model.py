from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.azure_inference import AzureAICompletionsModel

from doc_search.llm import factory
from doc_search.settings import LLMSetting


@factory.register_builder("ollama")
def build_ollama_as_llm(config: LLMSetting, prompt: str | None = None):
    return Ollama(
        model=config.model,
        temperature=config.temperature,
        system_prompt=prompt or config.system_prompt,
        request_timeout=config.request_timeout,
        context_window=config.context_window,
        base_url=f"http://{config.host}:{config.port}",
    )


@factory.register_builder("openai")
def build_llm_via_openai(config: LLMSetting, prompt: str | None = None):
    return OpenAI(
        model=config.model, temperature=config.temperature, system_prompt=prompt
    )


@factory.register_builder("azure")
def build_llm_with_azureai(config: LLMSetting, prompt: str | None = None):
    return AzureAICompletionsModel(
        model_name=config.model,
        temperature=config.temperature,
        system_prompt=prompt,
        endpoint="https://models.inference.ai.azure.com",
        credential=config.azure_ai_api_key,
    )
