from enum import Enum
from typing import Any, Generator, Optional, Sequence

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.response.schema import (RESPONSE_TYPE, Response,
                                                   StreamingResponse)
from llama_index.core.chat_engine import ContextChatEngine, SimpleChatEngine
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import Refine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import (MetadataMode, NodeWithScore, QueryBundle,
                                     QueryType)
from llama_index.core.settings import Settings

from doc_search.prompt.qa_prompt import qa_template
from doc_search.query_engine import factory
from doc_search.settings import (CitationEngineConfig, EngineConfig,
                                 QAEngineConfig, SimpleChatEngineConfig)
from doc_search.translator import Language, TranslationService, Translator

# import gcld3 # TODO: Add language classification/detector


class ChatMode(str, Enum):
    CHAT = "chat"
    QA = "QA"
    SEMANTIC_SEARCH = "semantic search"


def empty_response_generator() -> Generator[str, None, None]:
    yield "Empty Response"


class TranslatorContextChatEngine(ContextChatEngine):

    _translator: Translator = TranslationService.translator
    _src_language: Language | None = None
    _tgt_language: Language = Language("eng")

    @property
    def src_language(self) -> Language:
        return self._src_language

    @src_language.setter
    def src_language(self, language: str | Language):
        if isinstance(language, str):
            language = Language(language)
        self._src_language = language

    @property
    def tgt_language(self) -> Language:
        return self._tgt_language

    @tgt_language.setter
    def tgt_language(self, language: str | Language):
        if isinstance(language, str):
            language = Language(language)
        self._tgt_language = language

    def _generate_context(self, message: str) -> str | list[NodeWithScore]:
        # gcld3
        # TODO: Missing language detector
        nodes = self._retriever.retrieve(message)
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(message)
            )

        if (
            self._tgt_language and self._tgt_language.language_code != self._src_language.language_code
        ):
            draft = []
            for n in nodes:
                text = self._translator.translate(
                    sources=n.node.get_content(metadata_mode=MetadataMode.LLM).strip(),
                    src_lang=self._src_language,
                    tgt_lang=self._tgt_language,
                )
                draft.append(text)
            context_str = "\n\n".join(draft)
        else:
            context_str = "\n\n".join(
                [n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes]
            )

        return self._context_template.format(context_str=context_str), nodes


class RawBaseSynthesizer(Refine):
    def __init__(self, topk: int = 5, sample_length: int = 300, **kwargs) -> None:
        self._topk = topk
        self._sample_length = sample_length
        super().__init__(**kwargs)

    def synthesize(
        self,
        query: QueryType,
        nodes: list[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        if len(nodes) == 0:
            if self._streaming:
                empty_response = StreamingResponse(
                    response_gen=empty_response_generator()
                )
                return empty_response
            else:
                empty_response = Response("Empty Response")
                return empty_response

        if isinstance(query, str):
            query = QueryBundle(query_str=query)

        nodes.sort(key=lambda x: x.get_score(), reverse=True)
        response = self._prepare_response_output(nodes)
        return response

    def _prepare_response_output(self, source_nodes: list[NodeWithScore]):
        response_metadata = self._get_metadata_for_response(
            [node_with_score.node for node_with_score in source_nodes]
        )
        text = ""
        for i, node in enumerate(source_nodes[: min(self._topk, len(source_nodes))]):
            if "original_text" in node.metadata.keys():
                extract = node.metadata["original_text"][: self._sample_length]
            else:
                extract = node.text[: self._sample_length]
            page_number = node.metadata.get("page_label", "n/a")
            text += "".join(
                [
                    f"({i}) - Page {page_number} - Score={node.score:.3f}\n\nText:\t",
                    extract,
                    "\n\n=================================================\n\n",
                ]
            )
        return Response(
            text,
            source_nodes=source_nodes,
            metadata=response_metadata,
        )


@factory.register_builder("semantic search")
def build_semantic_search_engine(
    index: VectorStoreIndex,
    config: EngineConfig,
    postprocessors: Optional[list] = None,
    **kwargs,
) -> RetrieverQueryEngine:
    return RetrieverQueryEngine.from_args(
        retriever=index.as_retriever(similarity_top_k=config.similarity_top_k),
        llm=Settings.llm,
        response_synthesizer=RawBaseSynthesizer(
            topk=config.synthesizer.topk, sample_length=config.synthesizer.sample_length
        ),
        node_postprocessors=postprocessors or [],
    )


@factory.register_builder("QA")
def build_qa_query_engine(
    index: VectorStoreIndex,
    storage_context: StorageContext,
    config: EngineConfig,
    postprocessors: Optional[list] = None,
    **kwargs,
) -> ContextChatEngine:
    if config.hierarchical:
        retriever = AutoMergingRetriever(
            index.as_retriever(similarity_top_k=config.similarity_top_k),
            storage_context=storage_context,
        )
    else:
        retriever = index.as_retriever(similarity_top_k=config.similarity_top_k)

    return TranslatorContextChatEngine(
        prefix_messages=[
            ChatMessage.from_str(content=config.prefix_messages, role="system")
        ],
        retriever=retriever,
        llm=Settings.llm,
        context_template=config.context_template or qa_template,
        memory=ChatMemoryBuffer(token_limit=config.chat_token_limit),
        node_postprocessors=postprocessors or [],
    )


@factory.register_builder("chat")
def build_chat_query_engine(
    config: SimpleChatEngineConfig, **kwargs
) -> SimpleChatEngine:
    return SimpleChatEngine.from_defaults(
        llm=Settings.llm,
        memory=ChatMemoryBuffer(token_limit=config.chat_token_limit),
    )


# NOTE: Config builder
@factory.register_config("semantic search")
def build_semantic_search_engine_config():
    return CitationEngineConfig(type="semantic search")


@factory.register_config("QA")
def build_qa_engine_config():
    return QAEngineConfig(type="QA")


@factory.register_config("chat")
def build_chat_engine_config():
    return SimpleChatEngineConfig(type="chat")
