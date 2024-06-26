from typing import List
from .core import (
    LocalChatEngine,
    LocalDataIngestion,
    LocalRAGModel,
    LocalEmbedding,
    LocalVectorStore,
    get_system_prompt,
)
from llama_index.core import Settings
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.prompts import ChatMessage, MessageRole
from llama_index.core.query_engine import RouterQueryEngine
from .data_store import load_single_doc_into_nodes


class DocRetrievalAugmentedGen:
    def __init__(self, host: str = "127.0.0.1") -> None:
        self._host = host
        self._language = "eng"
        self._model_name = ""
        self._system_prompt = get_system_prompt("eng", is_rag_prompt=False)
        self._engine = LocalChatEngine(host=host)
        self._default_model = LocalRAGModel.set(self._model_name, host=host)
        self._query_engine = None
        # self._ingestion = LocalDataIngestion()
        # self._vector_store = LocalVectorStore(host=host)
        Settings.llm = LocalRAGModel.set(host=host)
        Settings.embed_model = LocalEmbedding.set(host=host)
        self._files_registry = []
        self._single_query_engines = []

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

    def reset_documents(self):
        self._ingestion.reset()

    def clear_conversation(self):
        self._query_engine.reset()

    def reset_conversation(self):
        self.reset_engine()
        self.set_system_prompt(
            get_system_prompt(language=self._language, is_rag_prompt=False)
        )

    def set_embed_model(self, model_name: str):
        Settings.embed_model = LocalEmbedding.set(model_name, self._host)

    def check_exist(self, model_name: str) -> bool:
        return LocalRAGModel.check_model_exist(self._host, model_name)

    def check_exist_embed(self, model_name: str) -> bool:
        return LocalEmbedding.check_model_exist(self._host, model_name)

    def store_nodes(self, input_files: List[str] = None) -> None:
        for file in input_files:
            if file not in self._files_registry:
                self._files_registry.append(file)
                docs = load_single_doc_into_nodes(file)
                data_indexing

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
