import os
from typing import Sequence, Tuple

import openai

# os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.environ["OPENAI_API_KEY"]

from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

from tutorials.example_utils import create_query_engine

query_engine = create_query_engine()
response = query_engine.query("How do I install llama index?")
print(str(response))


from llama_index.core import Document

documents = SimpleDirectoryReader(
    "data/docs", recursive=True, required_exts=[".md"]
).load_data()

all_text = ""

for doc in documents:
    all_text += doc.text

giant_document = Document(text=all_text)

import os
import random

random.seed(42)
from llama_index.core import ServiceContext
from llama_index.core.evaluation import DatasetGenerator
from llama_index.core.prompts import Prompt

gpt4_service_context = ServiceContext.from_defaults(
    llm=OpenAI(llm="gpt-4", temperature=0)
)

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
from llama_index.core.evaluation import \
    RelevancyEvaluator  # QueryResponseEvaluator -> RelevancyEvaluator
from llama_index.core.query_engine import SubQuestionQueryEngine


def evaluate_query_engine(
    evaluator: RelevancyEvaluator,
    query_engine: SubQuestionQueryEngine,
    questions: Sequence[str],
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


evaluator = RelevancyEvaluator(service_context=gpt4_service_context)

total_correct, all_results = evaluate_query_engine(
    evaluator, query_engine, question_dataset
)

print(
    f"Response satisfies the query? Scored {total_correct} out of {len(question_dataset)} questions correctly."
)


import numpy as np

unanswered_queries = np.array(question_dataset)[np.array(all_results) == 0]
print(unanswered_queries)


response = query_engine.query(
    "What is the purpose of the `ReActAgent` and how can it be initialized with other agents as tools?"
)
print(str(response))
print("-----------------")
print(response.get_formatted_sources(length=256))
