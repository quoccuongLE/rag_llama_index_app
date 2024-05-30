import openai
import os

os.environ["OPENAI_API_KEY"] = "YOUR_KEY_HERE"
openai.api_key = os.environ["OPENAI_API_KEY"]

from llama_index.embeddings.openai import OpenAIEmbedding

openai_embedding = OpenAIEmbedding()
embed = openai_embedding.get_text_embedding("hello world!")
print(len(embed))
print(embed[:10])

from InstructorEmbedding import INSTRUCTOR

model = INSTRUCTOR("hkunlp/instructor-large")
sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
instruction = "Represent the Science title:"
embeddings = model.encode([[instruction, sentence]])
print(embeddings[:10])
print(embeddings.shape)


from typing import Any, List
from llama_index.core.embeddings import BaseEmbedding


class InstructorEmbeddings(BaseEmbedding):
    def __init__(
        self,
        instructor_model_name: str = "hkunlp/instructor-large",
        instruction: str = "Represent the Computer Science text for retrieval:",
        **kwargs: Any,
    ) -> None:
        self._model = INSTRUCTOR(instructor_model_name)
        self._instruction = instruction
        super().__init__(**kwargs)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = model.encode([[self._instruction, query]])
        return embeddings[0].tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = model.encode([[self._instruction, text]])
        return embeddings[0].tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = model.encode([[self._instruction, text] for text in texts])
        return embeddings.tolist()

instructor_embeddings = InstructorEmbeddings(embed_batch_size=1)


embed = instructor_embeddings.get_text_embedding("How do I create a vector index?")
print(len(embed))
print(embed[:10])


# from llama_index import ServiceContext, set_global_service_context
# from llama_index.llms import OpenAI
from llama_index.core import ServiceContext, set_global_service_context
from llama_index.llms.openai import OpenAI


llm = OpenAI(model="gpt-3.5-turbo-16k", temperature=0)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=instructor_embeddings, chunk_size=512)
set_global_service_context(service_context)


import os
import sys

# sys.path.append(os.path.join(os.getcwd(), ".."))

# from llama_docs_bot.indexing import create_query_engine
from llama_docs_utils.indexing import create_query_engine

# remove any existing indices
# !rm -rf ./*_index

query_engine = create_query_engine()


response = query_engine.query("What is the Sub Question query engine?")
response.print_response_stream()


print(response.get_formatted_sources(length=256))


service_context = ServiceContext.from_defaults(llm=llm, embed_model=OpenAIEmbedding(), chunk_size=512)
set_global_service_context(service_context)

query_engine = create_query_engine()

response = query_engine.query("What is the Sub Question query engine?")
response.print_response_stream()

print(response.get_formatted_sources(length=256))

