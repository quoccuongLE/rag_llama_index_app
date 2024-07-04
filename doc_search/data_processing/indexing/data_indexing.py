from copy import deepcopy
from pathlib import Path
from typing import List, Optional

from llama_index.core import VectorStoreIndex

from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore

from llama_index.core.schema import BaseNode


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
