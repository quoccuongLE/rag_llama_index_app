from pydantic import BaseModel, Field
from .prompt import get_system_prompt


class LLMSettings(BaseModel):
    model: str = Field(default="llama3", description="LLM model used in RAG")
    system_prompt: str = Field(default=get_system_prompt(language="eng", is_rag_prompt=False))
    request_timeout: float = Field(default=120.0, description="Timeout for query requesting to Ollama server")


class EmbedLLMSetting(BaseModel):
    name: str = Field(
        default="mxbai-embed-large", description="Embedding model used in RAG"
    )


class RAGSettings(BaseModel):
    index_store: str = Field(
        default="./data/doc_search/index_store", description="Store for doc index"
    )
    file_storage: str = Field(
        default="./data/doc_search/docs", description="Store for docs"
    )

    llm: LLMSettings = LLMSettings()

    embed_model: EmbedLLMSetting= EmbedLLMSetting()
