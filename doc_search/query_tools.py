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


def get_query_engine_tool(
    index: VectorStoreIndex,
    storage_context: StorageContext,
    hierarchical: bool = False,
    chat_mode: str = "QA",
    chat_token_limit: int = 4000,
    postprocessors: Optional[list] = None,
    **kwargs,
) -> QueryEngineTool:
    if chat_mode == "QA":
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
        # return QueryEngineTool(
        #     query_engine=query_engine,
        #     metadata=ToolMetadata(**kwargs),
        # )
    elif chat_mode == "chat":
        llm = llm_from_settings_or_context(settings=Settings, context=None)
        return SimpleChatEngine.from_defaults(
            llm=llm,
            memory=ChatMemoryBuffer(token_limit=chat_token_limit),
        )
