from copy import deepcopy
from pathlib import Path
from typing import List, Optional

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import (
    NodeParser,
    HierarchicalNodeParser,
    get_leaf_nodes,
    SentenceSplitter,
)
from llama_index.core.schema import Document, MetadataMode, BaseNode, TextNode

from llama_index.core.node_parser import (
    MarkdownElementNodeParser,
    MarkdownNodeParser,
)


def load_docs(filepath: Path, hierarchical: bool = True) -> List[NodeParser]:
    """Load markdown docs from a directory, excluding all other file types."""
    loader = SimpleDirectoryReader(input_dir=filepath)

    documents = loader.load_data()

    if hierarchical:
        # combine all documents into one
        documents = [
            Document(
                text="\n\n".join(
                    document.get_content(metadata_mode=MetadataMode.ALL)
                    for document in documents
                )
            )
        ]

        # chunk into 3 levels
        # majority means 2/3 are retrieved before using the parent
        large_chunk_size = 1536
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[
                large_chunk_size,
                large_chunk_size // 3,
            ]
        )

        nodes = node_parser.get_nodes_from_documents(documents)
        return nodes, get_leaf_nodes(nodes)
    else:
        node_parser = SentenceSplitter.from_defaults()
        nodes = node_parser.get_nodes_from_documents(documents)
        return nodes


def load_single_doc_into_nodes(filename: Path) -> List[BaseNode]:
    # node_parser = MarkdownElementNodeParser()
    node_parser = MarkdownNodeParser()

    def get_page_nodes(docs, separator="\n---\n"):
        """Split each document into page node, by separator."""
        nodes = []
        for doc in docs:
            doc_chunks = doc.text.split(separator)
            for doc_chunk in doc_chunks:
                node = TextNode(
                    text=doc_chunk,
                    metadata=deepcopy(doc.metadata),
                )
                nodes.append(node)

        return nodes

    reader = SimpleDirectoryReader(input_files=[filename])
    documents = reader.load_data()
    page_nodes = get_page_nodes(documents)
    nodes = node_parser.get_nodes_from_documents(documents)
    if isinstance(node_parser, MarkdownElementNodeParser):
        base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
        # For recursive_index
        return base_nodes + objects + page_nodes
    else:
        base_nodes = []
        for node in nodes:
            base_nodes.extend(node_parser.get_nodes_from_node(node))
        # base_nodes = [node_parser.get_nodes_from_node(node) for node in nodes]
        return base_nodes + page_nodes


def data_indexing(
    dirname: str,
    data_runtime: Path,
    nodes: List[BaseNode],
    leaf_nodes: Optional[List[BaseNode]] = None,
):

    if leaf_nodes:
        # If hierarchical
        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)
        storage_context = StorageContext.from_defaults(docstore=docstore)

        # Indexing
        index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
        index.storage_context.persist(persist_dir=data_runtime / dirname)

    else:
        # Indexing
        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=data_runtime / dirname)
        storage_context = None

    return index, storage_context
