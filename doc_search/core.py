from typing import List, Optional, Tuple
from pathlib import Path

from .core import (
    LocalChatEngine,
    LocalRAGModel,
    LocalEmbedding,
    get_system_prompt,
)
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
)
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.prompts import ChatMessage, MessageRole
from llama_index.core.query_engine import RouterQueryEngine
from .data_store import load_single_doc_into_nodes, data_indexing
from .query_tools import get_query_engine_tool
from .settings import RAGSettings

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


class DocRetrievalAugmentedGen:
    def __init__(self, host: str = "127.0.0.1", setting: Optional[dict] = None) -> None:
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

        # Settings.llm = LocalRAGModel.set(host=host)
        # Settings.embed_model = LocalEmbedding.set(host=host)
        Settings.llm = Ollama(
            model="llama3", system_prompt=self._system_prompt, request_timeout=120.0
        )
        Settings.embed_model = OllamaEmbedding(model_name="llama3")

        self._files_registry = []
        self._query_engine_tools = {}
        self._file_storage = Path(setting.file_storage)
        self._load_index_stores()
        self._update_query_engine()

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
                dirname=filename.parent.name,
                data_runtime=self._setting.index_store,
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
            self._query_engine_tools[filename.name] = get_query_engine_tool(
                index=index,
                storage_context=storage_context,
                directory=filename.parent,
                description="",
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

    def get_system_prompt(self):
        return self._system_prompt

    def set_system_prompt(self, system_prompt: str | None = None):
        self._system_prompt = system_prompt or get_system_prompt(
            language=self._language, is_rag_prompt=self._ingestion.check_nodes_exist()
        )

    def set_model(self):
        Settings.llm = LocalRAGModel.set(
            model_name=self._model_name,
            system_prompt=self._system_prompt,
            host=self._host,
        )
        self._default_model = Settings.llm

    def reset_engine(self):
        self._query_engine = self._engine.set_engine(
            llm=self._default_model, nodes=[], language=self._language
        )

    def clear_conversation(self):
        self._query_engine.reset()

    def reset_conversation(self):
        self.reset_engine()
        self.set_system_prompt(
            get_system_prompt(language=self._language, is_rag_prompt=False)
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
                    dirname=_file.parent.name,
                    data_runtime=self._setting.index_store,
                    nodes=nodes,
                )
                self._query_engine_tools[_file.name] = get_query_engine_tool(
                    index=index,
                    storage_context=storage_context,
                    directory=_file.parent,
                    description="",
                )
        self._query_engine = RouterQueryEngine.from_defaults(
            query_engine_tools=self._query_engine_tools,
            select_multi=False,
        )

    def set_chat_mode(self, system_prompt: str | None = None):
        self.set_language(self._language)
        self.set_system_prompt(system_prompt)
        self.set_model()
        self.set_engine()

    def set_engine(self):
        self._query_engine = self._engine.set_engine(
            llm=self._default_model,
            nodes=self._ingestion.get_ingested_nodes(),
            language=self._language,
        )

    def get_history(self, chatbot: List[List[str]]):
        history = []
        for chat in chatbot:
            if chat[0]:
                history.append(ChatMessage(role=MessageRole.USER, content=chat[0]))
                history.append(ChatMessage(role=MessageRole.ASSISTANT, content=chat[1]))
        return history

    def _update_query_engine(self):
        self._query_engine = RouterQueryEngine.from_defaults(
            query_engine_tools=self._query_engine_tools,
            select_multi=False,
        )

    def query(self, message: str) -> StreamingAgentChatResponse:
        return self._query_engine.query(message)
