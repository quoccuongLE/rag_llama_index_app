from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.bridge.pydantic import Field
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from doc_search.embedding import factory
from doc_search.settings import EmbedModelSetting


class InstructOllamaEmbedding(OllamaEmbedding):

    _task_description: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    def get_query_embedding(self, query: str) -> list[float]:
        query = self.get_detailed_instruct(
            task_description=self._task_description, query=query
        )
        return super().get_query_embedding(query)

    def get_general_text_embedding(self, query: str, prompt_name: str | None = None):
        if prompt_name == "query":
            query = self.get_detailed_instruct(
                task_description=self._task_description, query=query
            )

        return super().get_general_text_embedding(prompt=query)


@factory.register_builder('huggingface')
def build_embed_model(config: EmbedModelSetting):
    return HuggingFaceEmbedding(
        model_name=config.name,
        max_length=config.max_seq_length,
        query_instruction=config.query_instruction,
        text_instruction=config.text_instruction,
    )


@factory.register_builder("ollama")
def build_ollama_model(config: EmbedModelSetting):
    if config.name == "rjmalagon/gte-qwen2-1.5b-instruct-embed-f16":
        return InstructOllamaEmbedding(
            model_name=config.name, base_url=f"http://{config.host}:{config.port}"
        )
    else:
        return OllamaEmbedding(
            model_name=config.name, base_url=f"http://{config.host}:{config.port}"
        )


@factory.register_builder("openai")
def build_ollama_model(config: EmbedModelSetting):
    return OpenAIEmbedding(model=config.name)
