from pathlib import Path
from typing import List, Optional, Tuple

from llama_index.core import (Settings, StorageContext, VectorStoreIndex,
                              load_index_from_storage)
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.prompts import ChatMessage, MessageRole
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from .data_store import data_indexing, load_single_doc_into_nodes
from .prompt import get_system_prompt
from .query_tools import get_query_engine_tool
from .settings import RAGSettings


class DocRetrievalAugmentedGen:
    def __init__(self, host: str = "127.0.0.1", setting: Optional[dict] = None, chat_mode: str = "QA") -> None:
        self._host = host
        self._language = "eng"
        self._model_name = ""
        self._system_prompt = get_system_prompt("eng", is_rag_prompt=False)
        self._engine = Ollama(
            model="llama3", system_prompt=self._system_prompt, request_timeout=120.0
        )
        self._default_model = Ollama(
            model="llama3", system_prompt=self._system_prompt, request_timeout=120.0
        )
        self._query_engine = None
        self._setting = RAGSettings() or setting

        Settings.llm = Ollama(
            model="llama3", system_prompt=self._system_prompt, request_timeout=120.0
        )
        Settings.embed_model = OllamaEmbedding(model_name="llama3")

        self._files_registry = []
        self._query_engine_tools = {}
        self._file_storage = Path(self._setting.file_storage)
        self._doc_index_stores = {}
        self._load_index_stores()
        self._query_engine_name = None
        self._chat_mode = chat_mode
        # self._update_query_engine()
        self.select_query_engine(list(self._query_engine_tools.keys())[0])

    def _read_doc_and_load_index(
        self, filename: Path, forced_indexing: bool = False
    ) -> Tuple[VectorStoreIndex, StorageContext]:
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=self._setting.index_store / f"{filename.stem}"
            )
            index = load_index_from_storage(storage_context)
            success = True
        except:
            success = False

        if forced_indexing or not success:
            nodes = load_single_doc_into_nodes(filename)
            index, storage_context = data_indexing(
                dirname=filename.name,
                data_runtime=Path(self._setting.index_store),
                nodes=nodes,
            )

        return index, storage_context

    def _load_index_stores(self, forced_indexing: bool = False):
        for filename in self._file_storage.glob("**/*"):
            if not filename.is_file():
                continue
            index, storage_context = self._read_doc_and_load_index(
                filename=filename, forced_indexing=forced_indexing
            )
            if filename not in self._doc_index_stores.keys():
                self._doc_index_stores[filename] = index
            self._query_engine_tools[filename.name] = get_query_engine_tool(
                index=index,
                storage_context=storage_context,
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, model_name: str):
        self._model_name = model_name

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
        return len(self._query_engine_tools.values()) > 0

    @system_prompt.setter
    def system_prompt(self, system_prompt: Optional[str] = None):
        self._system_prompt = system_prompt or get_system_prompt(
            language=self._language, is_rag_prompt=self.check_nodes_exist()
        )

    def set_model(self):
        Settings.llm = Ollama(
            model="llama3", system_prompt=self._system_prompt, request_timeout=120.0
        )
        self._default_model = Settings.llm

    def reset_engine(self):
        if self._chat_mode == "QA":
            self.select_query_engine()
        elif self._chat_mode == "chat":
            self._query_engine = get_query_engine_tool(
                index=None, storage_context=None, chat_mode="chat"
            )

    def clear_conversation(self):
        if self._chat_mode == "chat":
            self._query_engine = get_query_engine_tool(
                index=None, storage_context=None, chat_mode="chat"
            )

    def reset_conversation(self):
        self.reset_engine()
        self._system_prompt = get_system_prompt(language=self._language, is_rag_prompt=False)

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
                    data_runtime=Path(self._setting.index_store),
                    nodes=nodes,
                )
                self._query_engine_tools[_file.name] = get_query_engine_tool(
                    index=index,
                    storage_context=storage_context,
                    tool_name=_file.parent,
                    description="",
                )

    def select_query_engine(self, query_engine_name: Optional[str] = None):
        if query_engine_name:
            self._query_engine_name = query_engine_name
        self._query_engine = self._query_engine_tools[self._query_engine_name]

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
            self._chat_mode = chat_mode
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

    def _update_query_engine(self):
        self._query_engine = RouterQueryEngine.from_defaults(
            query_engine_tools=list(self._query_engine_tools.values()),
            select_multi=False,
        )

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
