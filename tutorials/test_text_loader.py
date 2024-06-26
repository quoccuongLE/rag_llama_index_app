from typing import List

# import os
# import sys
# sys.path.append(os.path.join(os.getcwd(), ".."))

from llama_docs_utils.markdown_docs_reader import MarkdownDocsReader
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document, MetadataMode


def load_markdown_docs(filepath) -> List[Document]:
    """Load markdown docs from a directory, excluding all other file types."""
    loader = SimpleDirectoryReader(
        input_dir=filepath, 
        exclude=["*.rst", "*.ipynb", "*.py", "*.bat", "*.txt", "*.png", "*.jpg", "*.jpeg", "*.csv", "*.html", "*.js", "*.css", "*.pdf", "*.json"],
        file_extractor={".md": MarkdownDocsReader()},
        recursive=True
    )

    return loader.load_data()


# load our documents from each folder.
# we keep them seperate for now, in order to create seperate indexes later
getting_started_docs = load_markdown_docs("data/docs/getting_started")
community_docs = load_markdown_docs("data/docs/community")
data_docs = load_markdown_docs("data/docs/core_modules/data_modules")
agent_docs = load_markdown_docs("data/docs/core_modules/agent_modules")
model_docs = load_markdown_docs("data/docs/core_modules/model_modules")
query_docs = load_markdown_docs("data/docs/core_modules/query_modules")
supporting_docs = load_markdown_docs("data/docs/core_modules/supporting_modules")
tutorials_docs = load_markdown_docs("data/docs/end_to_end_tutorials")
contributing_docs = load_markdown_docs("data/docs/development")

print(agent_docs[5].get_content(metadata_mode=MetadataMode.ALL))

text_template = "Content Metadata:\n{metadata_str}\n\nContent:\n{content}"

metadata_template = "{key}: {value},"
metadata_separator = " "

for doc in agent_docs:
    doc.text_template = text_template
    doc.metadata_template = metadata_template
    doc.metadata_seperator = metadata_separator

print(agent_docs[0].get_content(metadata_mode=MetadataMode.ALL))

# Hide the File Name from the LLM
agent_docs[0].excluded_llm_metadata_keys = ["File Name"]
print(agent_docs[0].get_content(metadata_mode=MetadataMode.LLM))

# Hide the File Name from the embedding model
agent_docs[0].excluded_embed_metadata_keys = ["File Name"]
print(agent_docs[0].get_content(metadata_mode=MetadataMode.EMBED))
