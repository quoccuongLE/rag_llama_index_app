from typing import Sequence, Tuple
import openai
import os


os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.environ["OPENAI_API_KEY"]


from llama_docs_utils.markdown_docs_reader import MarkdownDocsReader
from llama_index.core import SimpleDirectoryReader


def load_markdown_docs(filepath):
    """Load markdown docs from a directory, excluding all other file types."""
    loader = SimpleDirectoryReader(
        input_dir=filepath, required_exts=[".md"], file_extractor={".md": MarkdownDocsReader()}, recursive=True
    )

    documents = loader.load_data()

    # exclude some metadata from the LLM
    for doc in documents:
        doc.excluded_llm_metadata_keys = ["File Name", "Content Type", "Header Path"]

    return documents


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

from llama_index.core import ServiceContext, set_global_service_context
from llama_index.llms.openai import OpenAI

# create a global service context
service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-16k", temperature=0))
set_global_service_context(service_context)

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

# create a vector store index for each folder
try:
    getting_started_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./data/persist_dirs/getting_started_index")
    )
    community_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./data/persist_dirs/community_index")
    )
    data_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./data/persist_dirs/data_index"))
    agent_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./data/persist_dirs/agent_index"))
    model_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./data/persist_dirs/model_index"))
    query_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./data/persist_dirs/query_index"))
    supporting_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./data/persist_dirs/supporting_index")
    )
    tutorials_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./data/persist_dirs/tutorials_index")
    )
    contributing_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./data/persist_dirs/contributing_index")
    )
except:
    getting_started_index = VectorStoreIndex.from_documents(getting_started_docs)
    getting_started_index.storage_context.persist(persist_dir="./data/persist_dirs/getting_started_index")

    community_index = VectorStoreIndex.from_documents(community_docs)
    community_index.storage_context.persist(persist_dir="./data/persist_dirs/community_index")

    data_index = VectorStoreIndex.from_documents(data_docs)
    data_index.storage_context.persist(persist_dir="./data/persist_dirs/data_index")

    agent_index = VectorStoreIndex.from_documents(agent_docs)
    agent_index.storage_context.persist(persist_dir="./data/persist_dirs/agent_index")

    model_index = VectorStoreIndex.from_documents(model_docs)
    model_index.storage_context.persist(persist_dir="./data/persist_dirs/model_index")

    query_index = VectorStoreIndex.from_documents(query_docs)
    query_index.storage_context.persist(persist_dir="./data/persist_dirs/query_index")

    supporting_index = VectorStoreIndex.from_documents(supporting_docs)
    supporting_index.storage_context.persist(persist_dir="./data/persist_dirs/supporting_index")

    tutorials_index = VectorStoreIndex.from_documents(tutorials_docs)
    tutorials_index.storage_context.persist(persist_dir="./data/persist_dirs/tutorials_index")

    contributing_index = VectorStoreIndex.from_documents(contributing_docs)
    contributing_index.storage_context.persist(persist_dir="./data/persist_dirs/contributing_index")


from llama_index.core.tools import QueryEngineTool

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


from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer


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
    verbose=False,
)

response = query_engine.query("How do I install llama index?")
print(str(response))


from llama_index.core import Document

documents = SimpleDirectoryReader("data/docs", recursive=True, required_exts=[".md"]).load_data()

all_text = ""

for doc in documents:
    all_text += doc.text

giant_document = Document(text=all_text)

import os
import random

random.seed(42)
from llama_index.core import ServiceContext
from llama_index.core.prompts import Prompt
from llama_index.core.evaluation import DatasetGenerator

gpt4_service_context = ServiceContext.from_defaults(llm=OpenAI(llm="gpt-4", temperature=0))

question_dataset = []
question_dataset_file = "data/question_dataset.txt"
if os.path.exists(question_dataset_file):
    with open(question_dataset_file, "r") as f:
        for line in f:
            question_dataset.append(line.strip())

else:
    # generate questions
    data_generator = DatasetGenerator.from_documents(
        documents=[giant_document],
        text_question_template=Prompt(
            "A sample from the LlamaIndex documentation is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Using the documentation sample, carefully follow the instructions below:\n"
            "{query_str}"
        ),
        question_gen_query=(
            "You are an evaluator for a search pipeline. Your task is to write a single question "
            "using the provided documentation sample above to test the search pipeline. The question should "
            "reference specific names, functions, and terms. Restrict the question to the "
            "context information provided.\n"
            "Question: "
        ),
        # set this to be low, so we can generate more questions
        service_context=gpt4_service_context,
    )
    generated_questions = data_generator.generate_questions_from_nodes()

    # randomly pick 40 questions from each dataset
    generated_questions = random.sample(generated_questions, 40)
    question_dataset.extend(generated_questions)

    print(f"Generated {len(question_dataset)} questions.")

    # save the questions!
    with open("question_dataset.txt", "w") as f:
        for question in question_dataset:
            f.write(f"{question.strip()}\n")

print(random.sample(question_dataset, 5))

import asyncio
import time
from llama_index.core import Response
from llama_index.core.query_engine import SubQuestionQueryEngine


def evaluate_query_engine(
    evaluator: QueryResponseEvaluator, query_engine: SubQuestionQueryEngine, questions: Sequence[str]
) -> Tuple[int, Sequence[int]]:
    async def run_query(query_engine, q) -> Response:
        try:
            return await query_engine.aquery(q)
        except:
            return Response(response="Error, query failed.")

    total_correct = 0
    all_results = []
    for batch_size in range(0, len(questions), 5):
        batch_qs = questions[batch_size : batch_size + 5]

        tasks = [run_query(query_engine, q) for q in batch_qs]
        responses = asyncio.run(asyncio.gather(*tasks))
        print(f"finished batch {(batch_size // 5) + 1} out of {len(questions) // 5}")

        for question, response in zip(batch_qs, responses):
            eval_result = 1 if "YES" in evaluator.evaluate(question, response) else 0
            total_correct += eval_result
            all_results.append(eval_result)

        # helps avoid rate limits
        time.sleep(1)

    return total_correct, all_results


from llama_index.core.evaluation import QueryResponseEvaluator

evaluator = QueryResponseEvaluator(service_context=gpt4_service_context)

total_correct, all_results = evaluate_query_engine(evaluator, query_engine, question_dataset)

print(f"Response satisfies the query? Scored {total_correct} out of {len(question_dataset)} questions correctly.")


import numpy as np

unanswered_queries = np.array(question_dataset)[np.array(all_results) == 0]
print(unanswered_queries)


response = query_engine.query(
    "What is the purpose of the `ReActAgent` and how can it be initialized with other agents as tools?"
)
print(str(response))
print("-----------------")
print(response.get_formatted_sources(length=256))
