from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from sentence_transformers import SentenceTransformer

from doc_search.embedding import factory
from doc_search.settings import EmbedModelSetting
from llama_index.core.bridge.pydantic import Field

class EmbedModel(BaseEmbedding):
    model: SentenceTransformer = Field(
        default_factory=SentenceTransformer, exclude=True
    )

    def __init__(
        self,
        model: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        max_seq_length=131072,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model = SentenceTransformer(model, trust_remote_code=True)
        self.model.max_seq_length = max_seq_length

    def _get_query_embedding(self, query: str) -> list[float]:
        embeddings = self.model.encode([query])
        return embeddings[0].tolist()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        embeddings = self.model.encode([text])
        return embeddings[0].tolist()

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()


@factory.register_builder('huggingface')
def build_embed_model(config: EmbedModelSetting):
    return EmbedModel(model=config.model_name, max_seq_length=config.max_seq_length)


@factory.register_builder("ollama")
def build_ollama_model(config: EmbedModelSetting):
    return OllamaEmbedding(model_name=config.model_name)
