from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import BaseNode, Document
from llama_index.core.storage.docstore import SimpleDocumentStore

from doc_search.data_processing.indexing import factory
from doc_search.settings import ConfigParams


@factory.register_builder("vector_store_index")
def build_vector_store_index(
    dirname: str,
    data_runtime: Path,
    nodes_or_documents: list[BaseNode] | list[Document],
    config: ConfigParams | None = None,
    **kwargs
) -> tuple[VectorStoreIndex, StorageContext | None]:
    if len(nodes_or_documents) == 0:
        return []

    # Indexing
    if isinstance(nodes_or_documents[0], Document):
        index = VectorStoreIndex.from_documents(nodes_or_documents)
    elif isinstance(nodes_or_documents[0], Document):
        index = VectorStoreIndex(nodes_or_documents)

    index.storage_context.persist(persist_dir=data_runtime / dirname)
    storage_context = None

    return index, storage_context


@factory.register_builder("recursive_vector_store_index")
def build_vector_store_index(
    dirname: str,
    data_runtime: Path,
    nodes_or_documents: list[BaseNode] | list[Document],
    config: ConfigParams | None = None,
    **kwargs
) -> tuple[VectorStoreIndex, StorageContext | None]:
    if len(nodes_or_documents) == 0:
        return []

    if isinstance(nodes_or_documents[0], Document):
        node_parser = MarkdownNodeParser()
        nodes = node_parser.get_nodes_from_documents(nodes_or_documents)
        base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
    else:
        base_nodes, objects = node_parser.get_nodes_and_objects(nodes_or_documents)

    recursive_index = VectorStoreIndex(nodes=base_nodes + objects)
    recursive_index.storage_context.persist(persist_dir=data_runtime / dirname)

    return recursive_index, recursive_index.storage_context


@factory.register_builder("hierarchical_vector_store_index")
def build_hierarchical_vector_store_index(
    dirname: str,
    data_runtime: Path,
    nodes: list[BaseNode],
    leaf_nodes: list[BaseNode],
    config: ConfigParams | None = None,
    **kwargs
) -> tuple[VectorStoreIndex, StorageContext | None]:

    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    storage_context = StorageContext.from_defaults(docstore=docstore)

    # Indexing
    index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
    index.storage_context.persist(persist_dir=data_runtime / dirname)

    return index, storage_context
