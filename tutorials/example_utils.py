from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

from pathlib import Path

from llama_docs_utils.markdown_docs_reader import MarkdownDocsReader
from llama_index.core import SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer


LLAMA_MARKDOWN_DOCS_PATH = "./data/docs"
PERSIST_DATA_DIR = Path("data") / "persist_dirs"


def load_markdown_docs(filepath: Path):
    """Load markdown docs from a directory, excluding all other file types."""
    loader = SimpleDirectoryReader(
        input_dir=filepath,
        required_exts=[".md"],
        file_extractor={".md": MarkdownDocsReader()},
        recursive=True,
    )

    documents = loader.load_data()

    # exclude some metadata from the LLM
    for doc in documents:
        doc.excluded_llm_metadata_keys = ["File Name", "Content Type", "Header Path"]

    return documents


def load_docs():
    getting_started_docs = load_markdown_docs(f"{LLAMA_MARKDOWN_DOCS_PATH}/getting_started")
    community_docs = load_markdown_docs(f"{LLAMA_MARKDOWN_DOCS_PATH}/community")
    data_docs = load_markdown_docs(f"{LLAMA_MARKDOWN_DOCS_PATH}/core_modules/data_modules")
    agent_docs = load_markdown_docs(f"{LLAMA_MARKDOWN_DOCS_PATH}/core_modules/agent_modules")
    model_docs = load_markdown_docs(f"{LLAMA_MARKDOWN_DOCS_PATH}/core_modules/model_modules")
    query_docs = load_markdown_docs(f"{LLAMA_MARKDOWN_DOCS_PATH}/core_modules/query_modules")
    supporting_docs = load_markdown_docs(f"{LLAMA_MARKDOWN_DOCS_PATH}/core_modules/supporting_modules")
    tutorials_docs = load_markdown_docs(f"{LLAMA_MARKDOWN_DOCS_PATH}/end_to_end_tutorials")
    contributing_docs = load_markdown_docs(f"{LLAMA_MARKDOWN_DOCS_PATH}/development")

    return (
        getting_started_docs,
        community_docs,
        data_docs,
        agent_docs,
        model_docs,
        query_docs,
        supporting_docs,
        tutorials_docs,
        contributing_docs,
    )


def create_query_engine(persistent_data_dir: Path = PERSIST_DATA_DIR):
    """Create a query engine."""
    (
        getting_started_docs,
        community_docs,
        data_docs,
        agent_docs,
        model_docs,
        query_docs,
        supporting_docs,
        tutorials_docs,
        contributing_docs,
    ) = load_docs()

    try:
        getting_started_index = load_index_from_storage(
            StorageContext.from_defaults(
                persist_dir=persistent_data_dir / "getting_started_index"
            )
        )
        community_index = load_index_from_storage(
            StorageContext.from_defaults(
                persist_dir=persistent_data_dir / "community_index"
            )
        )
        data_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=persistent_data_dir / "data_index")
        )
        agent_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=persistent_data_dir / "agent_index")
        )
        model_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=persistent_data_dir / "model_index")
        )
        query_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=persistent_data_dir / "query_index")
        )
        supporting_index = load_index_from_storage(
            StorageContext.from_defaults(
                persist_dir=persistent_data_dir / "supporting_index"
            )
        )
        tutorials_index = load_index_from_storage(
            StorageContext.from_defaults(
                persist_dir=persistent_data_dir / "tutorials_index"
            )
        )
        contributing_index = load_index_from_storage(
            StorageContext.from_defaults(
                persist_dir=persistent_data_dir / "contributing_index"
            )
        )
    except Exception:
        print("Indexing: Getting started")
        getting_started_index = VectorStoreIndex.from_documents(getting_started_docs)
        getting_started_index.storage_context.persist(
            persist_dir=persistent_data_dir / "getting_started_index"
        )

        print("Indexing: Community")
        community_index = VectorStoreIndex.from_documents(community_docs)
        community_index.storage_context.persist(
            persist_dir=persistent_data_dir / "community_index"
        )

        print("Indexing: Data")
        data_index = VectorStoreIndex.from_documents(data_docs)
        data_index.storage_context.persist(persist_dir=persistent_data_dir / "data_index")

        print("Indexing: Agent")
        agent_index = VectorStoreIndex.from_documents(agent_docs)
        agent_index.storage_context.persist(
            persist_dir=persistent_data_dir / "agent_index"
        )

        print("Indexing: model")
        model_index = VectorStoreIndex.from_documents(model_docs)
        model_index.storage_context.persist(
            persist_dir=persistent_data_dir / "model_index"
        )

        print("Indexing: Query")
        query_index = VectorStoreIndex.from_documents(query_docs)
        query_index.storage_context.persist(
            persist_dir=persistent_data_dir / "query_index"
        )

        print("Indexing: Supporting")
        supporting_index = VectorStoreIndex.from_documents(supporting_docs)
        supporting_index.storage_context.persist(
            persist_dir=persistent_data_dir / "supporting_index"
        )

        print("Indexing: Toturials")
        tutorials_index = VectorStoreIndex.from_documents(tutorials_docs)
        tutorials_index.storage_context.persist(
            persist_dir=persistent_data_dir / "tutorials_index"
        )

        print("Indexing: Contribution")
        contributing_index = VectorStoreIndex.from_documents(contributing_docs)
        contributing_index.storage_context.persist(
            persist_dir=persistent_data_dir / "contributing_index"
        )

    # create a query engine tool for each folder
    getting_started_tool = QueryEngineTool.from_defaults(
        query_engine=getting_started_index.as_query_engine(),
        name="Getting Started",
        description="Useful for answering questions about installing and running llama index, as well as basic explanations of how llama index works.",
    )

    community_tool = QueryEngineTool.from_defaults(
        query_engine=community_index.as_query_engine(),
        name="Community",
        description="Useful for answering questions about integrations and other apps built by the community.",
    )

    data_tool = QueryEngineTool.from_defaults(
        query_engine=data_index.as_query_engine(),
        name="Data Modules",
        description="Useful for answering questions about data loaders, documents, nodes, and index structures.",
    )

    agent_tool = QueryEngineTool.from_defaults(
        query_engine=agent_index.as_query_engine(),
        name="Agent Modules",
        description="Useful for answering questions about data agents, agent configurations, and tools.",
    )

    model_tool = QueryEngineTool.from_defaults(
        query_engine=model_index.as_query_engine(),
        name="Model Modules",
        description="Useful for answering questions about using and configuring LLMs, embedding modles, and prompts.",
    )

    query_tool = QueryEngineTool.from_defaults(
        query_engine=query_index.as_query_engine(),
        name="Query Modules",
        description="Useful for answering questions about query engines, query configurations, and using various parts of the query engine pipeline.",
    )

    supporting_tool = QueryEngineTool.from_defaults(
        query_engine=supporting_index.as_query_engine(),
        name="Supporting Modules",
        description="Useful for answering questions about supporting modules, such as callbacks, service context, and avaluation.",
    )

    tutorials_tool = QueryEngineTool.from_defaults(
        query_engine=tutorials_index.as_query_engine(),
        name="Tutorials",
        description="Useful for answering questions about end-to-end tutorials and giving examples of specific use-cases.",
    )

    contributing_tool = QueryEngineTool.from_defaults(
        query_engine=contributing_index.as_query_engine(),
        name="Contributing",
        description="Useful for answering questions about contributing to llama index, including how to contribute to the codebase and how to build documentation.",
    )

    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[
            getting_started_tool,
            community_tool,
            data_tool,
            agent_tool,
            model_tool,
            query_tool,
            supporting_tool,
            tutorials_tool,
            contributing_tool,
        ],
        # enable this for streaming
        # response_synthesizer=get_response_synthesizer(streaming=True),
        verbose=True,
        use_async=False
    )

    return query_engine
