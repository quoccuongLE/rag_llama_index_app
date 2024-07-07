from pathlib import Path
from typing import List, Optional, Tuple

import ollama
import yaml
from llama_index.core import (Settings, StorageContext, VectorStoreIndex,
                              load_index_from_storage)
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.prompts import ChatMessage, MessageRole

from .data_processing.parser import factory as parser_factory
from .embedding import factory as embedding_factory
from .llm import factory as llm_factory
from .prompt import get_system_prompt
from .query_engine import factory as qengine_factory
from .query_engine.query_tools import ChatMode
from .settings import RAGSetting

_EMBED_MODELS = [
    "ollama/mxbai-embed-large",  # (Recommended for short context (512 max) d = 1024)
    "ollama/nomic-embed-text",  # (Recommended for long context (8192 max) d = 768)
    "ollama/all-minilm",
    "ollama/snowflake-arctic-embed",
    "huggingface/Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "huggingface/intfloat/multilingual-e5-large-instruct",  # instruct added into query
    "huggingface/intfloat/multilingual-e5-small",
]


class DocRetrievalAugmentedGen:

    def __init__(
        self,
        setting: RAGSetting | dict | str | None = None,
        chat_mode: str = "QA",
    ) -> None:
        self._language: str = "eng"
        self._system_prompt: str = get_system_prompt("eng", is_rag_prompt=False)
        self._setting = RAGSetting()

        if isinstance(setting, str):
            with open(setting, "r") as f:
                setting = yaml.safe_load(f)
        self._setting.override(setting)

        self._model_name: str = self._setting.llm.model or "llama3"
        self._query_engine = None
        self._parser = parser_factory.build(
            config=self._setting.parser_config, data_runtime=self._setting.index_store
        )

        Settings.llm = llm_factory.build(config=self._setting.llm)
        Settings.embed_model = embedding_factory.build(self._setting.embed_model)

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
        ollama_list = [
            "ollama/" + x["name"].replace(":latest", "") for x in info_dict["models"]
        ]
        return [x for x in ollama_list if x not in _EMBED_MODELS]

    @property
    def default_model(self) -> str:
        return f"ollama/{self._model_name}"

    def get_available_embed_models(self) -> list[str]:
        return _EMBED_MODELS

    @property
    def default_embed_model(self) -> str:
        return "ollama/mxbai-embed-large"

    def _read_doc_and_load_index(
        self, filename: Path, forced_indexing: bool = False
    ) -> Tuple[VectorStoreIndex, StorageContext]:
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=Path(self._setting.index_store)
                / self.embed_model
                / filename.name
            )
            index = load_index_from_storage(storage_context)
            success = True
        except:
            success = False

        if forced_indexing or not success:
            self._parser.data_runtime = (
                Path(self._setting.index_store) / self.embed_model
            )
            index, storage_context = self._parser.read_file(
                filename=filename, dirname=filename.name
            )

        return index, storage_context

    def _load_index_stores(self, forced_indexing: bool = False):
        for filename in self._file_storage.glob("**/*"):
            if not filename.is_file():
                continue
            if filename not in self._files_registry:
                _filename = Path(filename)
                self._files_registry.append(_filename.name)
                index, storage_context = self._read_doc_and_load_index(
                    filename=filename, forced_indexing=forced_indexing
                )
                if self.embed_model not in self._doc_index_stores.keys():
                    self._doc_index_stores[self.embed_model] = {}
                if self.embed_model not in self._doc_ctx_stores.keys():
                    self._doc_ctx_stores[self.embed_model] = {}
                self._doc_index_stores[self.embed_model][_filename.name] = index
                self._doc_ctx_stores[self.embed_model][_filename.name] = storage_context

    @property
    def embed_model(self) -> str:
        return self._setting.embed_model.name

    @embed_model.setter
    def embed_model(self, model: str):
        model_type, model_name = tuple(model.split("/", 1))
        self._setting.embed_model.override(dict(model_name=model_name, type=model_type))
        Settings.embed_model = embedding_factory.build(self._setting.embed_model)

    @property
    def model(self) -> str:
        return self._model_name

    @model.setter
    def model(self, model: str):
        model_type, model_name = tuple(model.split("/", 1))
        self._setting.llm.model = model_name
        self._setting.llm.type = model_type
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
        Settings.llm = llm_factory.build(config=self._setting.llm)

    def reset_engine(self):
        if self.embed_model not in self._doc_index_stores.keys():
            self._load_index_stores()

        if self._chat_mode in [ChatMode.QA, ChatMode.SEMANTIC_SEARCH]:
            idx = self._doc_index_stores[self.embed_model][self._query_engine_name]
            storage_ctx = self._doc_ctx_stores[self.embed_model][self._query_engine_name]
        else:
            idx, storage_ctx = None, None

        self._query_engine = qengine_factory.build(
            name=self._chat_mode.value,
            config=self._setting.query_engine,
            index=idx,
            storage_context=storage_ctx,
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
                filename = Path(file)
                self._files_registry.append(filename.name)
                index, storage_context = self._read_doc_and_load_index(
                    filename=filename, forced_indexing=True
                )
                self._doc_index_stores[self.embed_model][filename.name] = index
                self._doc_ctx_stores[self.embed_model][filename.name] = storage_context

    def set_chat_mode(
        self,
        chat_mode: Optional[str] = None,
        chat_config: Optional[dict] = None,
        system_prompt: Optional[str] = None,
        language: Optional[str] = None,
    ):
        if language:
            self.language = self._language or language
            self.system_prompt = system_prompt
        if chat_mode:
            self._chat_mode = ChatMode(chat_mode)
            self._setting.query_engine.override(chat_config or {})
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
            # TODO: Setup a system of synchronizing prompts while switching between languages
            return self._query_engine.query(message)
