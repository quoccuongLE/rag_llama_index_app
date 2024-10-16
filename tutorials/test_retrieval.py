import os
from pathlib import Path
from typing import Callable, List, Optional

from llama_index.core import (Document, QueryBundle, ServiceContext, Settings,
                              SimpleDirectoryReader, StorageContext,
                              VectorStoreIndex, load_index_from_storage,
                              set_global_service_context)
from llama_index.core.node_parser import (HierarchicalNodeParser, NodeParser,
                                          SentenceSplitter, get_leaf_nodes)
from llama_index.core.query_engine import (RetrieverQueryEngine,
                                           RouterQueryEngine,
                                           SubQuestionQueryEngine)
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import Document, MetadataMode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.utils import print_text
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from transformers import AutoTokenizer

from llama_docs_utils.markdown_docs_reader import MarkdownDocsReader

# Replace OpenAI GPT3.5 by Local llama 3 8B model
# service_context = ServiceContext.from_defaults(
#     llm=OpenAI(model="gpt-3.5-turbo-16k", max_tokens=512, temperature=0.1),
#     embed_model="local:BAAI/bge-base-en",
# )

# service_context = ServiceContext.from_defaults(
#     llm=Ollama(model="llama3", request_timeout=120.0),
#     embed_model=OllamaEmbedding(model_name="llama3")
# )

Settings.llm = Ollama(model="llama3", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="llama3")


def load_markdown_docs(filepath: Path, hierarchical: bool = True) -> List[NodeParser]:
    """Load markdown docs from a directory, excluding all other file types."""
    loader = SimpleDirectoryReader(
        input_dir=filepath,
        required_exts=[".md"],
        file_extractor={".md": MarkdownDocsReader()},
        recursive=True,
    )

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


def get_query_engine_tool(
    directory: str, description: str, hierarchical: bool = True, postprocessors: Optional[callable]=None
) -> QueryEngineTool:
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=f"./data/dev/{os.path.basename(directory)}_index"
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
            nodes, leaf_nodes = load_markdown_docs(directory, hierarchical=hierarchical)

            docstore = SimpleDocumentStore()
            docstore.add_documents(nodes)
            storage_context = StorageContext.from_defaults(docstore=docstore)

            index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
            index.storage_context.persist(
                persist_dir=f"./data/dev/{os.path.basename(directory)}_index"
            )

            retriever = AutoMergingRetriever(
                index.as_retriever(similarity_top_k=12), storage_context=storage_context
            )

        else:
            nodes = load_markdown_docs(directory, hierarchical=hierarchical)
            index = VectorStoreIndex(nodes)
            index.storage_context.persist(
                persist_dir=f"./data/dev/{os.path.basename(directory)}_index"
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


hierarchical_engine = get_query_engine_tool(
    directory="data/docs/core_modules/query_modules",
    description="Useful for information on various query engines and retrievers, and anything related to querying data.",
    hierarchical=True,
).query_engine

# rm -rf data/data_query_modules

base_engine = get_query_engine_tool(
    "data/docs/core_modules/query_modules",
    "Useful for information on various query engines and retrievers, and anything related to querying data.",
    hierarchical=False,
).query_engine


hierarchical_nodes = hierarchical_engine.retrieve(
    QueryBundle("How do I setup a query engine?")
)
base_nodes = base_engine.retrieve(QueryBundle("How do I setup a query engine?"))


# from transformers.models.llama.tokenization_llama import LlamaTokenizer
# llama_tokenizer = LlamaTokenizer("../weight_llama/Meta-Llama-3-8B", legacy=False)

hf_token = os.environ.get("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=hf_token)

stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]


class LimitRetrievedNodesLength:

    def __init__(self, limit: int = 3000, tokenizer: Optional[Callable] = None):
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


query_engine = get_query_engine_tool(
    "data/docs/core_modules/query_modules",
    "Useful for information on various query engines and retrievers, and anything related to querying data.",
    hierarchical=True,
    postprocessors=[LimitRetrievedNodesLength(limit=3000, tokenizer=tokenizer)],
).query_engine


hierarchical_nodes = query_engine.retrieve(
    QueryBundle("How do I setup a query engine?")
)
total_length = 0

for node in hierarchical_nodes:
    total_length += len(
        tokenizer(node.node.get_content(metadata_mode=MetadataMode.LLM))
    )
print(f"Total length: {total_length}")


# Here we define the directories we want to index, as well as a description for each
# NOTE: these descriptions are hand-written based on my understanding. We could have also
# used an LLM to write these, maybe a future experiment.
docs_directories = {
    "data/docs/community": "Useful for information on community integrations with other libraries, vector dbs, and frameworks.",
    "data/docs/core_modules/agent_modules": "Useful for information on data agents and tools for data agents.",
    "data/docs/core_modules/data_modules": "Useful for information on data, storage, indexing, and data processing modules.",
    "data/docs/core_modules/model_modules": "Useful for information on LLMs, embedding models, and prompts.",
    "data/docs/core_modules/query_modules": "Useful for information on various query engines and retrievers, and anything related to querying data.",
    "data/docs/core_modules/supporting_modules": "Useful for information on supporting modules, like callbacks, evaluators, and other supporting modules.",
    "data/docs/getting_started": "Useful for information on getting started with LlamaIndex.",
    "data/docs/development": "Useful for information on contributing to LlamaIndex development.",
}

# Build query engine tools
query_engine_tools = [
    get_query_engine_tool(
        directory,
        description,
        hierarchical=True,
        postprocessors=[LimitRetrievedNodesLength(limit=3000, tokenizer=tokenizer)],
    )
    for directory, description in docs_directories.items()
]

# build top-level router -- this will route to multiple sub-indexes and aggregate results
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    verbose=False
)

query_engine = RouterQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    select_multi=True,
)


response = query_engine.query(
    "How do I setup a ChromaDB Vector Store? Give me a code sample please."
)
print_text(response)

response = query_engine.query("How can I customize Document metadata?")
print_text(response)
