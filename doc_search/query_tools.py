from typing import Optional

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import RetrieverQueryEngine, CitationQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.settings import Settings, llm_from_settings_or_context
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from llama_index.vector_stores.milvus import MilvusVectorStore

# from llama_index.core.retrievers import (
#     BaseRetriever,
#     QueryFusionRetriever)

from enum import Enum


class ChatMode(str, Enum):
    CHAT = "chat"
    QA = "QA"
    SEMANTIC_SEARCH = "semantic search"


def get_query_engine_tool(
    index: VectorStoreIndex,
    storage_context: StorageContext,
    hierarchical: bool = False,
    chat_mode: ChatMode = ChatMode.QA,
    chat_token_limit: int = 4000,
    postprocessors: Optional[list] = None,
    **kwargs,
) -> QueryEngineTool:
    if chat_mode == ChatMode.QA:
        if hierarchical:
            retriever = AutoMergingRetriever(
                index.as_retriever(similarity_top_k=6), storage_context=storage_context
            )
        else:
            retriever = index.as_retriever(similarity_top_k=12)

        query_engine = RetrieverQueryEngine.from_args(
            retriever,
            node_postprocessors=postprocessors or [],
        )

        return query_engine

    elif chat_mode == ChatMode.CHAT:
        llm = llm_from_settings_or_context(settings=Settings, context=None)
        return SimpleChatEngine.from_defaults(
            llm=llm,
            memory=ChatMemoryBuffer(token_limit=chat_token_limit),
        )
    elif chat_mode == ChatMode.SEMANTIC_SEARCH:
        return CitationQueryEngine.from_args(
            index,
            citation_chunk_size=256,
            similarity_top_k=5,
        )
