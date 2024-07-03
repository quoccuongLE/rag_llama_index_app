from pathlib import Path
from typing import List, Optional, Tuple

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.prompts import ChatMessage, MessageRole
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import ollama

from .data_store import data_indexing, load_single_doc_into_nodes
from .prompt import get_system_prompt
from .query_tools import get_query_engine_tool, ChatMode
from .settings import RAGSettings


_EMBED_MODELS = [
    "mxbai-embed-large",  # (Recommended for short context (512 max) d = 1024)
    "nomic-embed-text",  # (Recommended for long context (8192 max) d = 768)
    "all-minilm",
    "snowflake-arctic-embed",
]


class DocRetrievalAugmentedGen:
    def __init__(
        self,
        host: str = "127.0.0.1",
        setting: RAGSettings | dict | None = None,
        chat_mode: str = "QA",
    ) -> None:
        self._host = host
        self._language: str = "eng"
        self._system_prompt: str = get_system_prompt("eng", is_rag_prompt=False)
        self._setting = RAGSettings() or setting
        self._model_name: str = "llama3" or self._setting.llm.model
        self._query_engine = None

        Settings.llm = Ollama(
            model=self._setting.llm.model,
            system_prompt=self._system_prompt,
            request_timeout=self._setting.llm.request_timeout,
        )
        Settings.embed_model = OllamaEmbedding(model_name=self._setting.embed_model.name)

        self._query_engine_name: str = ""
        self._files_registry = []
        self._query_engine_tools = {}
        self._file_storage = Path(self._setting.file_storage)
        self._doc_index_stores = {}
        self._doc_ctx_stores = {}
        self._load_index_stores()
        self._chat_mode = ChatMode(chat_mode)

    def get_available_models(self) -> List[str]:
        info_dict = ollama.list()
        _list = [x["name"].replace(":latest", "") for x in info_dict["models"]]
        return [x for x in _list if x not in _EMBED_MODELS]

    @property
    def default_model(self) -> str:
        return "llama3"

    def get_available_embed_models(self) -> list[str]:
        info_dict = ollama.list()
        _list = [x["name"].replace(":latest", "") for x in info_dict["models"]]
        return [x for x in _list if x in _EMBED_MODELS]

    @property
    def default_embed_model(self) -> str:
        return "mxbai-embed-large"

    def _read_doc_and_load_index(
        self, filename: Path, forced_indexing: bool = False
    ) -> Tuple[VectorStoreIndex, StorageContext]:
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=Path(self._setting.index_store) / f"{filename.name}"
            )
            index = load_index_from_storage(storage_context)
            success = True
        except:
            success = False

        if forced_indexing or not success:
            nodes = load_single_doc_into_nodes(filename)
            index, storage_context = data_indexing(
                dirname=filename.name,
                data_runtime=Path(self._setting.index_store) / self.embed_model,
                nodes=nodes,
            )

        return index, storage_context

    def _load_index_stores(self, forced_indexing: bool = False):
        for filename in self._file_storage.glob("**/*"):
            if not filename.is_file():
                continue
            if filename not in self._files_registry:
                _file = Path(filename)
                self._files_registry.append(_file.name)
                index, storage_context = self._read_doc_and_load_index(
                    filename=filename, forced_indexing=forced_indexing
                )
                if self.embed_model not in self._doc_index_stores.keys():
                    self._doc_index_stores[self.embed_model] = {}
                if self.embed_model not in self._doc_ctx_stores.keys():
                    self._doc_ctx_stores[self.embed_model] = {}
                self._doc_index_stores[self.embed_model][_file.name] = index
                self._doc_ctx_stores[self.embed_model][_file.name] = storage_context

    @property
    def embed_model(self) -> str:
        return self._setting.embed_model.name

    @embed_model.setter
    def embed_model(self, model: str):
        self._setting.embed_model.name = model
        Settings.embed_model = OllamaEmbedding(model_name=self._setting.embed_model.name)

    @property
    def model(self) -> str:
        return self._model_name

    @model.setter
    def model(self, model: str):
        self._setting.llm.model = model
        self.set_model()

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, model_name: str):
        self._model_name = model_name
        self._setting.llm.model = model_name

    @property
    def language(self) -> str:
        return self._language

    @language.setter
    def language(self, language: str):
        self._language = language

    @property
    def system_prompt(self):
        return self._system_prompt

    def check_nodes_exist(self):
        # return len(self._query_engine_tools.values()) > 0
        return len(self._doc_index_stores.values()) > 0

    @system_prompt.setter
    def system_prompt(self, system_prompt: Optional[str] = None):
        self._system_prompt = system_prompt or get_system_prompt(
            language=self._language, is_rag_prompt=self.check_nodes_exist()
        )

    def set_model(self):
        Settings.llm = Ollama(
            model=self._setting.llm.model,
            system_prompt=self._system_prompt,
            request_timeout=self._setting.llm.request_timeout,
        )

    def reset_engine(self):
        self._query_engine = get_query_engine_tool(
            index=self._doc_index_stores[self.embed_model][self._query_engine_name],
            storage_context=self._doc_ctx_stores[self.embed_model][
                self._query_engine_name
            ],
            chat_mode=self._chat_mode,
        )

    def clear_conversation(self):
        if self._chat_mode == ChatMode.CHAT:
            self.reset_engine()

    def reset_conversation(self):
        self.reset_engine()
        self._system_prompt = get_system_prompt(
            language=self._language, is_rag_prompt=False
        )

    def store_nodes(self, input_files: List[str] = None) -> None:
        self.add_new_nodes(input_files=input_files)

    def add_new_nodes(self, input_files: List[str] = None) -> None:
        for file in input_files:
            if file not in self._files_registry:
                self._files_registry.append(file)
                _file = Path(file)
                nodes = load_single_doc_into_nodes(_file)
                index, storage_context = data_indexing(
                    dirname=_file.name,
                    data_runtime=Path(self._setting.index_store) / self.embed_model,
                    nodes=nodes,
                )
                self._doc_index_stores[self.embed_model][_file.name] = index
                self._doc_ctx_stores[self.embed_model][_file.name] = storage_context

    def set_chat_mode(
        self,
        system_prompt: Optional[str] = None,
        chat_mode: Optional[str] = None,
        language: Optional[str] = None,
    ):
        if language:
            self.language = self._language or language
            self.system_prompt = system_prompt
        if chat_mode:
            self._chat_mode = ChatMode(chat_mode)
        if system_prompt:
            self.system_prompt = system_prompt
        self.set_model()
        self.reset_engine()

    def get_history(self, chatbot: List[List[str]]):
        history = []
        for chat in chatbot:
            if chat[0]:
                history.append(ChatMessage(role=MessageRole.USER, content=chat[0]))
                history.append(ChatMessage(role=MessageRole.ASSISTANT, content=chat[1]))
        return history

    def query(
        self, mode: str, message: str, chatbot: List[List[str]]
    ) -> StreamingAgentChatResponse:
        if mode == "chat":
            history = self.get_history(chatbot)
            return self._query_engine.stream_chat(message, history)
        else:
            # self._query_engine.reset()
            return self._query_engine.query(message)
            # return self._query_engine(message)
