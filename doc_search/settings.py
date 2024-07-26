import copy

from llama_index.core.llms import LLM
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
    type: str = Field(default="ollama")
    model: str = Field(default="llama3", description="LLM model used in RAG")
    system_prompt: str = Field(default=get_system_prompt(language="eng", is_rag_prompt=False))
    temperature: float = Field(default=0.5, description="The temperature to use for sampling.")
    request_timeout: float = Field(default=120.0, description="Timeout for query requesting to Ollama server")
    host: str = Field(default="localhost")
    port: int = Field(default=11434)


class EmbedModelSetting(ConfigParams):
    type: str = Field(
        default="ollama", description="Source of embedding (ollama or huggingface)"
    )
    name: str = Field(
        default="mxbai-embed-large", description="Embedding model used in RAG"
    )
    host: str = Field(default="localhost")
    port: int = Field(default=11434)
    max_seq_length: int = Field(default=8192)
    request_timeout: float = Field(
        default=120.0, description="Timeout for query requesting to Ollama server"
    )
    instruct_prompt: str = Field(
        default="", description="Instruct added into user queries"
    )


class RawBaseSynthesizerConfig(ConfigParams):
    topk: int = Field(default=5)
    sample_length: int = Field(default=300)


class EngineConfig(ConfigParams):
    type: str = Field(default="QA")

    # SimpleChatEngine
    chat_token_limit: int = Field(default=8000)

    # CitationEngine
    citation_chunk_size: int = Field(default=512)
    similarity_top_k: int = Field(default=5)
    synthesizer: RawBaseSynthesizerConfig = Field(default_factory=RawBaseSynthesizerConfig)

    # QAEngine
    prefix_messages: str = Field(
        default="You are given a context, please answer the question solely on that context."
    )
    hierarchical: bool = Field(default=False)
    context_template: str = Field(default=None)

# TODO: Refacto EngineConfig
class SimpleChatEngineConfig(EngineConfig):
    chat_token_limit: int = Field(default=8000)


class CitationEngineConfig(EngineConfig):

    citation_chunk_size: int = Field(default=512)
    similarity_top_k: int = Field(default=5)
    synthesizer: RawBaseSynthesizerConfig = Field(
        default_factory=RawBaseSynthesizerConfig
    )


class QAEngineConfig(EngineConfig):
    similarity_top_k: int = Field(default=12)
    hierarchical: bool = Field(default=False)


class LoaderConfig(ConfigParams):
    file_extractor: list[str] = Field(default_factory=lambda: [])
    recursive: bool = Field(default=True)
    show_progress: bool = Field(default=False)
    result_type: str = Field(default="markdown")
    parsing_instruction: str = Field(default=None)
    text_summarize: bool = Field(default=False)

    # MarkerPDFReader
    max_pages: int = Field(default=None)
    start_page: int = Field(default=None)
    langs: list[str] = Field(default=None)
    batch_multiplier: int = Field(default=2)
    page_merge: bool = Field(default=False)


class ParserConfig(ConfigParams):
    name: str = Field(default="simple_parser")

    # Loader config
    loader_name: str = Field(default="single_file")
    loader_config: LoaderConfig = Field(default_factory=LoaderConfig)

    # Indexing config
    index_store_name: str = Field(default="vector_store_index")
    instruct_prompt: str = Field(default="")

    # node parser config
    node_parser_name: str = Field(default="markdown_node_parser")
    num_workers: int = Field(default=8)
    llm: LLMSetting = Field(default_factory=LLMSetting)


class RAGSetting(ConfigParams):
    index_store: str = Field(
        default="./data/doc_search/index_store", description="Store for doc index"
    )
    file_storage: str = Field(
        default="./data/doc_search/docs", description="Store for docs"
    )
    llm: LLMSetting = Field(default_factory=LLMSetting)
    embed_model: EmbedModelSetting = Field(default_factory=EmbedModelSetting)
    query_engine: EngineConfig = Field(default_factory=QAEngineConfig)
    parser_config: ParserConfig = Field(default_factory=ParserConfig)
