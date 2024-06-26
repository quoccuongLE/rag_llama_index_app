import os
from pathlib import Path
from typing import List, Optional

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    download_loader,
    RAKEKeywordTableIndex,
)

from llama_index.core import StorageContext, QueryBundle, Settings
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.query_engine import RetrieverQueryEngine, RouterQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import load_index_from_storage
from llama_index.core.node_parser import (
    NodeParser,
    HierarchicalNodeParser,
    get_leaf_nodes,
    SentenceSplitter,
)
from llama_index.core.schema import Document, MetadataMode
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.utils import print_text

from transformers import AutoTokenizer

Settings.llm = Ollama(model="llama3", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="llama3")


class LimitRetrievedNodesLength:

    def __init__(self, limit: int = 3000, tokenizer: Optional[callable] = None):
        self._tokenizer = tokenizer
        self.limit = limit

    def postprocess_nodes(self, nodes, query_bundle: QueryBundle):
        included_nodes = []
        current_length = 0

        for node in nodes:
            current_length += len(
                self._tokenizer(node.node.get_content(metadata_mode=MetadataMode.LLM))
            )
            if current_length > self.limit:
                break
            included_nodes.append(node)

        return included_nodes


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


def _get_query_engine_tool(
    directory: str,
    description: str,
    data_runtime: Path,
    hierarchical: bool = True,
    postprocessors: Optional[callable] = None,
) -> QueryEngineTool:
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=data_runtime / f"{os.path.basename(directory)}_index"
        )
        index = load_index_from_storage(storage_context)

        if hierarchical:
            retriever = AutoMergingRetriever(
                index.as_retriever(similarity_top_k=6), storage_context=storage_context
            )
        else:
            retriever = index.as_retriever(similarity_top_k=12)
    except:
        if hierarchical:
            nodes, leaf_nodes = load_docs(directory, hierarchical=hierarchical)

            docstore = SimpleDocumentStore()
            docstore.add_documents(nodes)
            storage_context = StorageContext.from_defaults(docstore=docstore)

            index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
            index.storage_context.persist(
                persist_dir=data_runtime / f"{os.path.basename(directory)}_index"
            )

            retriever = AutoMergingRetriever(
                index.as_retriever(similarity_top_k=12), storage_context=storage_context
            )

        else:
            nodes = load_docs(directory, hierarchical=hierarchical)
            index = VectorStoreIndex(nodes)
            index.storage_context.persist(
                persist_dir=data_runtime / f"{os.path.basename(directory)}_index"
            )

            retriever = index.as_retriever(similarity_top_k=12)

    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=postprocessors or [],
    )

    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(name=directory, description=description),
    )


def main():
    # reader = SimpleDirectoryReader(input_files=["./data/Llama_Getting_Started_Guide.pdf"])
    # data = reader.load_data()

    data_runtime = Path("./data/dev_rag/stores")
    # base_engine = _get_query_engine_tool(
    #     directory="data/dev_rag/docs",
    #     description="Useful for information on various query engines and retrievers, and anything related to querying data.",
    #     data_runtime=data_runtime,
    #     hierarchical=False,
    # ).query_engine

    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=hf_token)

    stopping_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    docs_descriptions = {"data/dev_rag/docs": "Useful for information on how to use the first simple Llama-index applications"}
    query_engine_tools = [
        _get_query_engine_tool(
            directory=d,
            description=des,
            data_runtime=data_runtime,
            hierarchical=False,
            postprocessors=[LimitRetrievedNodesLength(limit=3000, tokenizer=tokenizer)],
        ) for d, des in docs_descriptions.items()
    ]
    query_engine = RouterQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        select_multi=False,
    )

    response = query_engine.query(
        "How do I fine-tune a LLama model?"
    )
    print("\n")
    print_text(response)
    print("\n")


if __name__ == "__main__":
    main()
