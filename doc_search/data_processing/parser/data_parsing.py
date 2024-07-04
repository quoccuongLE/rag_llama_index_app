from copy import deepcopy
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

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
    SimpleFileNodeParser,
)
from llama_index.core import Document, SimpleDirectoryReader

from doc_search.settings import LoaderConfig
from doc_search.data_processing.data_loader import factory as reader_factory
from doc_search.data_processing.indexing import factory as indexer_factory


class SimpleParser:

    # doc_reader: SimpleDirectoryReader
    parser: SimpleFileNodeParser = SimpleFileNodeParser()
    loader_config: LoaderConfig = LoaderConfig()
    data_runtime: Path = Path("./data")

    def __init__(
        self,
        loader_config: LoaderConfig,
        parser: SimpleFileNodeParser,
        data_runtime: Path,
    ) -> None:
        # self.doc_reader = doc_reader
        self.loader_config = loader_config
        self.parser = parser
        self.data_runtime = data_runtime

    def read_file(self, filename: Path, dirname: str):
        assert filename.is_file(), f"Input path {filename} is not a file !"

        doc_loader = reader_factory.build(
            name=self.loader_config.loader_name, file=filename, config=self.loader_config
        )
        documents = doc_loader.load_data()
        index, storage_context = indexer_factory.build(
            name=self.loader_config.index_store_name,
            dirname=dirname,
            data_runtime=self.data_runtime,
            nodes_or_documents=documents,
        )

        return index, storage_context


def _get_page_nodes(
    documents: list[Document], separator: str = "\n---\n"
) -> list[BaseNode]:
    """Split each document into page node, by separator."""
    nodes = []
    for doc in documents:
        doc_chunks = doc.text.split(separator)
        for doc_chunk in doc_chunks:
            node = TextNode(
                text=doc_chunk,
                metadata=deepcopy(doc.metadata),
            )
            nodes.append(node)

    return nodes


def load_docs(filepath: Path, hierarchical: bool = True) -> list[NodeParser]:
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


def load_single_doc_into_nodes(filename: Path) -> list[BaseNode]:
    # node_parser = MarkdownElementNodeParser()
    node_parser = MarkdownNodeParser()
    reader = SimpleDirectoryReader(input_files=[filename])
    documents = reader.load_data()
    page_nodes = _get_page_nodes(documents)
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
