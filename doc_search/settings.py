import copy
from typing import Sequence

from pydantic import BaseModel, Field

from .prompt import get_system_prompt


class ConfigParams(BaseModel, extra="allow"):

    def override(self, override_dict: dict | BaseModel):
        """The implementation of `override`."""
        for k, v in override_dict.items():
            if k not in self.__dict__.keys():
                super().__setattr__(k, copy.deepcopy(v))
            else:
                if isinstance(v, dict):
                    self.__dict__[k].override(v)
                elif isinstance(v, BaseModel):
                    self.__dict__[k].override(v.__dict__)
                else:
                    self.__dict__[k] = copy.deepcopy(v)


class LLMSetting(ConfigParams):
    model: str = Field(default="llama3", description="LLM model used in RAG")
    system_prompt: str = Field(default=get_system_prompt(language="eng", is_rag_prompt=False))
    request_timeout: float = Field(default=120.0, description="Timeout for query requesting to Ollama server")


class EmbedModelSetting(ConfigParams):
    type: str = Field(
        default="ollama", description="Source of embedding (ollama or huggingface)"
    )
    model_name: str = Field(
        default="mxbai-embed-large", description="Embedding model used in RAG"
    )
    max_seq_length: int = Field(default=8192)
    request_timeout: float = Field(
        default=120.0, description="Timeout for query requesting to Ollama server"
    )
    instruct_prompt: str = Field(
        default="", description="Instruct added into user queries"
    )


class RawBaseSynthesizerConfig(ConfigParams):
    topk: int = Field(default=5)
    sample_length: int = Field(default=200)


class EngineConfig(ConfigParams):
    type: str = Field(default="QA")


class SimpleChatEngineConfig(EngineConfig):
    chat_token_limit: int = Field(default=4000)


class CitationEngineConfig(EngineConfig):
    citation_chunk_size: int = Field(default=512)
    similarity_top_k: int = Field(default=5)
    synthesizer: RawBaseSynthesizerConfig = Field(default_factory=RawBaseSynthesizerConfig)


class QAEngineConfig(EngineConfig):
    similarity_top_k: int = Field(default=12)
    hierarchical: bool = Field(default=False)


class RAGSetting(ConfigParams):
    index_store: str = Field(
        default="./data/doc_search/index_store", description="Store for doc index"
    )
    file_storage: str = Field(
        default="./data/doc_search/docs", description="Store for docs"
    )
    llm: LLMSetting = LLMSetting()
    embed_model: EmbedModelSetting = Field(default_factory=EmbedModelSetting)
    query_engine: EngineConfig = Field(default_factory=QAEngineConfig)


class ReaderConfig(ConfigParams):
    # TODO: To be replaced
    # file_extractor: dict[str, callable] = Field(default_factory=dict)
    # file_extractor: list[str] = Field(default_factory=lambda: [".md"])
    file_extractor: list[str] = Field(default_factory=lambda: [])


class LoaderConfig(ConfigParams):
    reader_config: ReaderConfig = Field(default_factory=ReaderConfig)
    recursive: bool = Field(default=True)

    # LlamaParse(result_type="markdown")
    loader_name: str = Field(default="single_file")
    index_store_name: str = Field(default="vector_store_index")
    result_type: str = Field(default="markdown")


class ParserConfig(ConfigParams):
    chunk_size: int = Field(default=512)
