import json
from llama_index.core.schema import Document

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from .multilingual_base import MultiLingualBaseReader
from doc_search.translator import Language
from llama_index.core.settings import Settings
from llama_index.core.output_parsers import LangchainOutputParser
from llama_index.core.output_parsers.utils import _marshal_llm_to_json


info_extraction_template = (
    "The following passage is a job description."
    "You must extract all information about {information_str} without adding prior knowledge"
    "\n--------------------\n"
    "{context_str}"
    "\n--------------------\n"
)

class JobDescriptionReader(MultiLingualBaseReader):

    def __init__(
        self, tgt_language: Language = Language("eng"), translator_config: dict = None
    ):
        super().__init__(tgt_language, translator_config)
        self._llm = Settings.llm
        self._template = info_extraction_template
        self._key_items = {
            "skill and qualification requirements": {
                "education": "Required education for the position",
                "experience": "Required experience for the position",
                "technical skills": "Required technical skills for the position",
                "soft skills": "Required soft skills for the position",
                "nice to haves": "Other requirements that make candidate outstanding",
            },
            "about the job": {
                "job name": "Title of the hiring position",
                "responsibilities": "All responsibilities for the position",
            },
            "about the company": {
                "company name": "The name of the company which is hiring"
            },
            "about the team and project": {
                "environment": "General information about the team and project in which the candicate will work"
            },
        }
        self._retrieved_infos = self._init_items()

    def _init_items(self) -> dict:
        info = dict()
        for topic, details in self._key_items.items():
            info[topic] = {k: None for k in details.keys()}
        return info

    def parse(self, text: str):
        for topic, details in self._key_items.items():
            text_with_template = self._template.format(
                information_str=topic, context_str=text
            )
            response_schemas = [
                ResponseSchema(name=k, description=v) for k, v in details.items()
            ]
            lc_output_parser = StructuredOutputParser.from_response_schemas(
                response_schemas
            )
            output_parser = LangchainOutputParser(lc_output_parser)
            final_query = output_parser.format(text_with_template)
            response = self._llm.complete(final_query)
            info_json = self._parse(response.text)
            self._retrieved_infos[topic].update(info_json)
        return

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
            return {} 
