import json
from pathlib import Path
from typing import Dict, List

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from llama_index.core import PromptTemplate
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.output_parsers import LangchainOutputParser
from llama_index.core.output_parsers.utils import _marshal_llm_to_json
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.settings import Settings

from doc_search.prompt.qa_prompt import (
    cover_letter_template_given_candidate_bio, multi_select_item_in_resume)
from doc_search.query_engine import factory
from doc_search.settings import EngineConfig

from .base import TranslatorContextChatEngine


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
            max_outputs=self._topk
        )
        description = f"One of the most relevant qualifications or experiences of the candidate to apply for the job."
        response_schemas = [
            ResponseSchema(name=f"reason_{i + 1}", description=description) for i in range(self._topk)
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

        return (
            self._context_template.format(
                expert_domain_str=self.expert_domain_str,
                job_name=self.job_name,
                context_str=text,
                qualifications_str="".join(bio),
            ),
            [],
        )


@factory.register_builder("cover letter gen")
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
