from enum import Enum
from typing import Any, Generator, Optional, Sequence

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.response.schema import (RESPONSE_TYPE, Response,
                                                   StreamingResponse)
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import (CitationQueryEngine,
                                           RetrieverQueryEngine)
from llama_index.core.response_synthesizers import Refine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, QueryType
from llama_index.core.settings import Settings

from doc_search.prompt.qa_prompt import qa_template
from doc_search.query_engine import factory
from doc_search.settings import (CitationEngineConfig, QAEngineConfig,
                                 SimpleChatEngineConfig)


class ChatMode(str, Enum):
    CHAT = "chat"
    QA = "QA"
    SEMANTIC_SEARCH = "semantic search"


def empty_response_generator() -> Generator[str, None, None]:
    yield "Empty Response"


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
            extract = node.text[: self._sample_length]
            page_number = node.metadata.get("page_label", "n/a")
            text += "".join(
                [
                    f"({i}) - Page {page_number} \nText:\t",
                    extract,
                    "...",
                    f"\nScore:\t {node.score:.3f}\n"
                    "\n[ _______________________________________________ ]\n",
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
    config: CitationEngineConfig,
    postprocessors: Optional[list] = None,
    **kwargs,
) -> CitationQueryEngine:
    return CitationQueryEngine.from_args(
        index,
        response_synthesizer=RawBaseSynthesizer(**config.synthesizer.model_dump()),
        citation_chunk_size=config.citation_chunk_size,
        citation_qa_template=PromptTemplate(qa_template),
        similarity_top_k=config.similarity_top_k,
        node_postprocessors=postprocessors or [],
    )


@factory.register_builder("QA")
def build_qa_query_engine(
    index: VectorStoreIndex,
    storage_context: StorageContext,
    config: QAEngineConfig,
    postprocessors: Optional[list] = None,
    **kwargs,
) -> RetrieverQueryEngine:
    if config.hierarchical:
        retriever = AutoMergingRetriever(
            index.as_retriever(similarity_top_k=config.similarity_top_k),
            storage_context=storage_context,
        )
    else:
        retriever = index.as_retriever(similarity_top_k=config.similarity_top_k)

    return RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=postprocessors or [],
    )


@factory.register_builder("chat")
def build_chat_query_engine(
    config: SimpleChatEngineConfig, **kwargs
) -> RetrieverQueryEngine:
    return SimpleChatEngine.from_defaults(
        llm=Settings.llm,
        memory=ChatMemoryBuffer(token_limit=config.chat_token_limit),
    )


@factory.register_config("semantic search")
def build_semantic_search_engine_config():
    return CitationEngineConfig(type="semantic search")


@factory.register_config("QA")
def build_qa_engine_config():
    return QAEngineConfig(type="QA")


@factory.register_config("chat")
def build_chat_engine_config():
    return SimpleChatEngineConfig(type="chat")
