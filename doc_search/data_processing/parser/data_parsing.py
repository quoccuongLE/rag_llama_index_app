from copy import deepcopy
from pathlib import Path

from llama_index.core import (Document, Settings, SimpleDirectoryReader,
                              VectorStoreIndex)
from llama_index.core.node_parser import (HierarchicalNodeParser,
                                          MarkdownElementNodeParser,
                                          MarkdownNodeParser, NodeParser,
                                          SentenceSplitter, get_leaf_nodes)
from llama_index.core.schema import BaseNode, Document, MetadataMode, TextNode

from doc_search.data_processing.data_loader import MultiLingualBaseReader
from doc_search.data_processing.data_loader import factory as loader_factory
from doc_search.data_processing.indexing import factory as indexer_factory
from doc_search.data_processing.parser import factory
from doc_search.llm import factory as llm_factory
from doc_search.settings import ParserConfig


class SimpleParser:

    parser_config: ParserConfig = ParserConfig()
    data_runtime: Path = Path("./data")
    doc_loader: MultiLingualBaseReader | None = None

    def __init__(
        self,
        data_runtime: Path,
        parser_config: ParserConfig,
    ) -> None:
        self.data_runtime = data_runtime
        self.parser_config = parser_config

    def read_file(self, filename: Path, dirname: str):
        assert filename.is_file(), f"Input path {filename} is not a file !"

        self.doc_loader = loader_factory.build(
            name=self.parser_config.loader_name, file=filename, config=self.parser_config
        )
        documents = self.doc_loader.load_data()
        index, storage_context = indexer_factory.build(
            name=self.parser_config.index_store_name,
            config=None,
            dirname=dirname,
            data_runtime=self.data_runtime,
            nodes_or_documents=documents,
        )

        return index, storage_context


class LlamaParser(SimpleParser):

    node_parser: NodeParser = MarkdownNodeParser()

    def __init__(self, node_parser: NodeParser, **kwargs):
        super().__init__(**kwargs)
        self.node_parser = node_parser
        self.doc_loader = None

    def read_file(
        self,
        filename: Path | str,
        dirname: str | None = None,
        indexing: bool = True,
        translate: bool = False,
        src_language: str | None = None,
        tgt_language: str | None = None,
    ):
        if isinstance(filename, str):
            filename = Path(filename)
        assert filename.is_file(), f"Input path {filename} is not a file !"
        if not dirname:
            dirname = filename.name
        if self.doc_loader is None:
            self.doc_loader = loader_factory.build(
                name=self.parser_config.loader_name,
                file=filename,
                config=self.parser_config.loader_config,
            )
        index_documents = self.doc_loader.load_data(
            str(filename),
            translate=translate,
            src_language=src_language,
            tgt_language=tgt_language,
        )
        if indexing:
            # NOTE: This parser doesn't use pre-built indexer
            nodes = self.node_parser.get_nodes_from_documents(index_documents)
            base_nodes, objects = self.node_parser.get_nodes_and_objects(nodes)

            recursive_index = VectorStoreIndex(nodes=base_nodes + objects)
            recursive_index.storage_context.persist(persist_dir=self.data_runtime / dirname)
            return recursive_index, recursive_index.storage_context
        else:
            return None, None


@factory.register_builder("simple_parser")
def build_simple_parser(data_runtime: Path, config: ParserConfig):
    return SimpleParser(data_runtime=data_runtime, parser_config=config)


@factory.register_builder("llama_parser")
def build_llama_parser(data_runtime: Path, config: ParserConfig):
    # TODO: node parser builder
    match config.node_parser_name:
        case "markdown_node_parser":
            node_parser = MarkdownNodeParser()
        case "markdown_element_node_parser":
            llm = llm_factory.build(name="ollama", config=config.llm, prompt=config.instruct_prompt)
            node_parser = MarkdownElementNodeParser(
                llm=llm, num_workers=config.num_workers
            )
        case _:
            raise ValueError(f"{config.node_parser_name} invalid !")
    return LlamaParser(data_runtime=data_runtime, parser_config=config, node_parser=node_parser)


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
