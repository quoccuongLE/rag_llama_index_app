from typing import Optional

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.tools import QueryEngineTool, ToolMetadata


def get_query_engine_tool(
    index: VectorStoreIndex,
    storage_context: StorageContext,
    directory: str,
    description: str,
    hierarchical: bool = False,
    postprocessors: Optional[list] = None,
) -> QueryEngineTool:
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

    # return QueryEngineTool(
    #     query_engine=query_engine,
    #     metadata=ToolMetadata(name=directory, description=description),
    # )
    return query_engine
