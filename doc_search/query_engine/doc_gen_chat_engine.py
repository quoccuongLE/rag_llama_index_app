import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import ollama
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from llama_index.core import PromptTemplate
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.output_parsers import LangchainOutputParser
from llama_index.core.output_parsers.utils import _marshal_llm_to_json
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.settings import Settings

from doc_search.data_processing.data_loader import (
    BaseMarkdownPortfolioReader, JobDescriptionReader)
from doc_search.prompt.qa_prompt import (
    cover_letter_template_given_candidate_bio, multi_select_item_in_resume)
from doc_search.query_engine import factory
from doc_search.settings import EngineConfig

from .base import TranslatorContextChatEngine
from .utils import semantic_search


def _get_ollama_embeddings(documents: list[str]):
    embeddings = []
    response = ollama.embed(model="embeddinggemma", input=documents)
    embeddings.append(response.get("embeddings"))
    res =  np.asarray(embeddings) if embeddings else None
    if res.ndim == 3:
        res = res.squeeze(0)
        return res
    if res.ndim == 2:
        return res
    else:
        raise NotImplementedError


class DocGenChatEngine(TranslatorContextChatEngine):
    expert_domain_str: str = "Tech"
    _selection_template: str = ""
    _topk: int = 3
    _src_document: str = ""
    job_name: str = ""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLM,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        src_document: Path | str | None = None,
        topk: int = 3,
        selection_template: str | None = None,
        node_postprocessors: List[BaseNodePostprocessor] | None = None,
        context_template: str | None = None,
        callback_manager: CallbackManager | None = None,
        job_name: str = "Data Scientist (Artifical Intelligence & Computer Vision)",
        output_token_number: int = 360,
        short_format: bool = False,
    ) -> None:
        """Initializes a DocGenChatEngine instance with the provided parameters.

        Args:
            retriever (BaseRetriever): The retriever used to retrieve relevant documents.
            llm (LLM): The language model used for generating responses.
            memory (BaseMemory): The memory used to store and retrieve context.
            prefix_messages (List[ChatMessage]): The list of initial chat messages to include in the context.
            src_document (Path | str | None, optional): The source document to use. Can be a file path or a string. Defaults to None.
            topk (int, optional): The number of top results to return. Defaults to 3.
            selection_template (str | None, optional): The template to use for selecting relevant text from the source document. Defaults to None.
            node_postprocessors (List[BaseNodePostprocessor] | None, optional): The list of node postprocessors to apply. Defaults to None.
            context_template (str | None, optional): The template to use for generating the context. Defaults to None.
            callback_manager (CallbackManager | None, optional): The callback manager to use. Defaults to None.
            job_name (str, optional): The name of the job. Defaults to "Data Scientist (Artifical Intelligence & Computer Vision)".
        """
        super().__init__(
            retriever,
            llm,
            memory,
            prefix_messages,
            node_postprocessors,
            context_template,
            callback_manager,
        )
        if isinstance(src_document, Path):
            with open(src_document, "r", encoding="utf-8") as f:
                self._src_document = f.read()
        else:
            self._src_document = src_document
        self._topk = topk
        self._selection_template = PromptTemplate(selection_template)
        self.job_name = job_name
        self.output_token_number = output_token_number
        self.short_format = short_format
        self._retrieved_items = None

    def set_source_document(self, document: str):
        """Sets the source document for the DocGenChatEngine instance.

        Args:
            document (str): The source document to use.
        """
        self._src_document = document

    def _retrieve(self, target_document: str) -> str:
        """Retrieves the most relevant qualifications or experiences of the candidate for the given job description.

        Args:
            target_document (str): The job description to use for retrieving the relevant qualifications or experiences.

        Returns:
            List[str]: The list of the most relevant qualifications or experiences of the candidate.
        """
        text_with_template = self._selection_template.format(
            resume=self._src_document,
            job_description=target_document,
            max_outputs=self._topk,
        )
        description = f"One of the most relevant qualifications or experiences of the candidate to apply for the job."
        response_schemas = [
            ResponseSchema(name=f"reason_{i + 1}", description=description)
            for i in range(self._topk)
        ]

        lc_output_parser = StructuredOutputParser.from_response_schemas(
            response_schemas
        )
        output_parser = LangchainOutputParser(lc_output_parser)
        final_query = output_parser.format(text_with_template)
        response = self._llm.complete(final_query)
        json_dict = self._parse(response.text)
        return [v for k, v in json_dict.items()]

    def _parse(self, output: str) -> dict:
        """Attempts to parse the output of an LLM (Large Language Model) response into a JSON object.

        If the output can be directly parsed into JSON, it will return the JSON object. If the output contains a JSON-formatted string, it will extract and return the JSON object from that string. If neither of these approaches work, it will attempt to construct a JSON object from the output text.

        Returns:
            dict: A dictionary representing the parsed JSON object.
        """
        try:
            json_string = _marshal_llm_to_json(output)
            json_obj = json.loads(json_string)
            if not json_obj:
                raise ValueError(f"Failed to convert output to JSON: {output!r}")
            return json_obj
        except:
            if "```" in output:
                json_string = output.split("```")[1]
            else:
                json_string = output
            if 'json\n{\n\t"translated_text": ' in json_string:
                raw_text = json_string.replace('json\n{\n\t"translated_text": ', "")
            else:
                raw_text = (
                    json_string.replace("json\n{\n", "")
                    .replace('"translated_text": ', "")
                    .strip()
                )
            raw_text = raw_text.replace('"', "")
            raw_text = raw_text.replace("}", "")
            raw_text = raw_text[2:]
            json_string = {"translated_text": f"{raw_text}"}
            return json_string

    def _generate_context(self, message: str) -> str | list[NodeWithScore]:
        """Generates the context for a chat-based document generation engine.

        Args:
            message (str): The input message to generate the context for.

        Returns:
            str | list[NodeWithScore]: The generated context as a string, or a list of nodes with scores.
        """
        if (
            self._tgt_language
            and self._translate_node
            and self._tgt_language.language_code != self._src_language.language_code
        ):
            text = self._translator.translate(
                sources=message,
                src_lang=self._src_language,
                tgt_lang=self._tgt_language,
            )
        else:
            text = message

        added_elements = self._retrieve(target_document=text)
        bio = []
        for element in added_elements:
            bio.append(f"* {element}\n")

        self._retrieved_items = bio

        return (
            self._context_template.format(
                expert_domain_str=self.expert_domain_str,
                job_name=self.job_name,
                number_of_words=self.output_token_number,
                short_or_long="short" if self.short_format else "long",
                context_str=text,
                qualifications_str="".join(bio),
            ),
            [],
        )

    def retrieved_items_message(self) -> str:
        extra_info = (
            f"Here is the top-{self._topk} most relevant elements in the"
            " candidate's profile that match to the job description:\n"
        )
        extra_info += "".join(self._retrieved_items)
        extra_info += "".join("\n___________________________________\n")
        return extra_info


class ContextMatchDocGenChatEngine(DocGenChatEngine):
    def __init__(
        self,
        retriever,
        llm,
        memory,
        prefix_messages,
        job_description: str,
        portfolio: str,
        topk: int = 3,
        node_postprocessors=None,
        context_template=None,
        callback_manager=None,
        output_token_number=1200,
        short_format=False,
    ):
        super(DocGenChatEngine, self).__init__(
            retriever,
            llm,
            memory,
            prefix_messages,
            node_postprocessors,
            context_template,
            callback_manager,
        )
        self.portfolio_reader = BaseMarkdownPortfolioReader()
        self.job_description_reader = JobDescriptionReader()
        self.portfolio = portfolio
        self.job_description = job_description
        self._topk = topk
        self.output_token_number = output_token_number
        self.short_format = short_format
        self._retrieved_items = {}
        self._key_criteria = {
            "Education": ["Education"],
            "Professional Experience": ["Professional Experience"],
            "Technical Skills": ["Technical Skills"],
            "Nice to Haves": [
                "Education",
                "Professional Experience",
                "Technical Skills",
            ],
        }
        self._embed_model = Settings.embed_model
        self.matching_threshold = 0.65

    def set_source_document(self, portfolio: str):
        """Sets the source document for the DocGenChatEngine instance.

        Args:
            document (str): The source document to use.
        """
        self.portfolio = portfolio

    def set_job_description(self, job_description: str):
        """Sets the job description for the DocGenChatEngine instance.

        Args:
            job_description (str): The job description to use.
        """
        self.job_description = job_description

    def retrieve(self) -> str:
        self.portfolio_reader.parse(self.portfolio)
        self.job_description_reader.parse(self.job_description)
        for jd_key, portfolio_key in self._key_criteria.items():
            candidate_items = []
            for key in portfolio_key:
                candidate_items.extend(self.portfolio_reader.generate_text(key))
            job_description_items = self.job_description_reader._retrieved_infos[
                "skill and qualification requirements"
            ][jd_key]
            # candidate_embeddings = self._embed_model.get_text_embedding_batch(
            #     candidate_items,
            #     # show_progress=True
            # )
            # job_description_embeddings = self._embed_model.get_text_embedding_batch(
            #     job_description_items,
            #     # show_progress=True
            # )
            candidate_embeddings = _get_ollama_embeddings(candidate_items)
            job_description_embeddings = _get_ollama_embeddings(job_description_items)

            semantic_search_results = semantic_search(
                query_embeddings=job_description_embeddings,
                corpus_embeddings=candidate_embeddings,
                top_k=3,
            )
            corpus_id = semantic_search_results[0][0]["corpus_id"]
            score = semantic_search_results[0][0]["score"]
            if score < self.matching_threshold:
                self._retrieved_items[jd_key] = None
                continue
            retrieved_item = candidate_items[corpus_id]
            self._retrieved_items[jd_key] = retrieved_item

    def get_retrieved_items(self):
        return self._retrieved_items

    def _generate_context(self, message: str) -> str | list[NodeWithScore]:
        if (
            self._tgt_language
            and self._translate_node
            and self._tgt_language.language_code != self._src_language.language_code
        ):
            text = self._translator.translate(
                sources=message,
                src_lang=self._src_language,
                tgt_lang=self._tgt_language,
            )
        else:
            text = message

        bio = []
        for key, value in self._retrieved_items.items():
            if value:
                bio.append(f"### {key}\n")
                bio.append(value)
            # for pair in value:
            #     bio.append(
            #         f"* The candiate possesses {pair[0]} matching to the "
            #         f"criteria found in the job description :{pair[1]}\n"
            #     )
        self.bio = bio

        return (
            self._context_template.format(
                expert_domain_str=self.expert_domain_str,
                job_name=self.job_name,
                number_of_words=self.output_token_number,
                short_or_long="short" if self.short_format else "long",
                context_str=self.job_description,
                qualifications_str="".join(bio),
            ),
            [],
        )

    def generate_cover_letter(self) -> AgentChatResponse:
        if (
            self._tgt_language
            and self._translate_node
            and self._tgt_language.language_code != self._src_language.language_code
        ):
            message += self._translator.translate(
                sources=self._postfix_message + f"{self._tgt_language.english_name}",
                src_lang="eng",
                tgt_lang=self._tgt_language.language_code,
            )
        # self._memory.put(ChatMessage(content=, role="user"))

        context_str_template, nodes = self._generate_context("")
        return self._llm.complete(context_str_template)


@factory.register_builder("cover letter gen legacy")
def build_doc_gen_1(
    config: EngineConfig,
    postprocessors: list | None = None,
    **kwargs,
) -> ContextChatEngine:
    return DocGenChatEngine(
        retriever=None,
        llm=Settings.llm,
        prefix_messages="",
        selection_template=multi_select_item_in_resume,
        topk=config.similarity_top_k,
        memory=ChatMemoryBuffer(token_limit=config.chat_token_limit),
        context_template=cover_letter_template_given_candidate_bio,
    )


@factory.register_builder("cover letter gen")
def build_doc_gen_2(
    config: EngineConfig,
    portfolio: str | None = None,
    job_description: str | None = None,
    postprocessors: list | None = None,
    **kwargs,
) -> ContextChatEngine:
    return ContextMatchDocGenChatEngine(
        retriever=None,
        llm=Settings.llm,
        prefix_messages="",
        portfolio=portfolio,
        job_description=job_description,
        topk=config.similarity_top_k,
        memory=ChatMemoryBuffer(token_limit=config.chat_token_limit),
        context_template=cover_letter_template_given_candidate_bio,
    )
