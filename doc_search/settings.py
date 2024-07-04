import copy

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
    max_seq_length: int = 8192
    request_timeout: float = Field(
        default=120.0, description="Timeout for query requesting to Ollama server"
    )


class RAGSetting(ConfigParams):
    index_store: str = Field(
        default="./data/doc_search/index_store", description="Store for doc index"
    )
    file_storage: str = Field(
        default="./data/doc_search/docs", description="Store for docs"
    )

    llm: LLMSetting = LLMSetting()

    embed_model: EmbedModelSetting = EmbedModelSetting()
