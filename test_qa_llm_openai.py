import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_index.llms import ChatMessage


with open("./data/docs/getting_started/starter_example.md", "r") as f:
    text = f.read()

llm = OpenAI(model="gpt-3.5-turbo-16k", temperature=0.0)
# To install LlamaIndex, you need to follow the installation steps provided in the "installation.md" file.

text_qa_template = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}\n"
)

# Test#1
question = "How can I install llama-index?"
prompt = text_qa_template.format(context_str=text, query_str=question)
response = llm.complete(prompt)
print(response.text)

# Test#2
question = "How do I create an index?"
prompt = text_qa_template.format(context_str=text, query_str=question)
response = llm.complete(prompt)
print(response.text)

# Test#3
question = "How do I create an index? Write your answer using only code."
prompt = text_qa_template.format(context_str=text, query_str=question)
response_gen = llm.stream_complete(prompt)
for response in response_gen:
    print(response.delta, end="")

# Test#4
refine_template = PromptTemplate(
    "We have the opportunity to refine the original answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question: {query_str}. "
    "If the context isn't useful, output the original answer again.\n"
    "Original Answer: {existing_answer}"
)

question = "How do I create an index? Write your answer using only code."
existing_answer = """To create an index using LlamaIndex, you need to follow these steps:

1. Download the LlamaIndex repository by cloning it from GitHub.
2. Navigate to the `examples/paul_graham_essay` folder in the cloned repository.
3. Create a new Python file and import the necessary modules: `VectorStoreIndex` and `SimpleDirectoryReader`.
4. Load the documents from the `data` folder using `SimpleDirectoryReader('data').load_data()`.
5. Build the index using `VectorStoreIndex.from_documents(documents)`.
6. To persist the index to disk, use `index.storage_context.persist()`.
7. To reload the index from disk, use the `StorageContext` and `load_index_from_storage` functions.

Note: This answer assumes that you have already installed LlamaIndex and have the necessary dependencies."""
prompt = refine_template.format(context_msg=text, query_str=question, existing_answer=existing_answer)
response = llm.complete(prompt)
print(response.text)
