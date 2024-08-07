import json

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from llama_index.core import PromptTemplate
from llama_index.core.output_parsers import LangchainOutputParser
from llama_index.core.output_parsers.utils import _marshal_llm_to_json
from llama_index.core.settings import Settings

from doc_search.settings import TranslatorConfig
from doc_search.translator import factory

from .base import Language, Translator


class LLMTranslator(Translator):
    _template: PromptTemplate = PromptTemplate(
        "Translate the following passage into {language_name}:"
        "\n---------------------\n"
        "{text_str}\n"
        "\n---------------------"
    )

    def __init__(
        self,
        tgt_language: Language,
    ) -> None:
        self.tgt_language = tgt_language
        self.llm = Settings.llm

    def translate(
        self,
        sources: list[str] | str,
        src_lang: str | Language,
        tgt_lang: str | Language | None = None,
    ) -> str:
        if isinstance(src_lang, str):
            src_lang = Language(src_lang)
        if tgt_lang:
            if isinstance(tgt_lang, str):
                tgt_lang = Language(tgt_lang)
        else:
            tgt_lang = self.tgt_language
        response_schemas = [
            ResponseSchema(
                name="translated_text",
                description=(
                    f"The translation in {tgt_lang.english_name} of the original passage"
                ),
            ),
        ]

        lc_output_parser = StructuredOutputParser.from_response_schemas(
            response_schemas
        )
        output_parser = LangchainOutputParser(lc_output_parser)
        final_query = output_parser.format(
            self._template.format(language_name=tgt_lang.english_name, text_str=sources)
        )
        response = self.llm.complete(final_query)
        return self._parse(response.text)["translated_text"]

    def _parse(self, output: str) -> dict:
        json_string = _marshal_llm_to_json(output)
        json_obj = json.loads(json_string)

        if not json_obj:
            raise ValueError(f"Failed to convert output to JSON: {output!r}")
        return json_obj


@factory.register_builder("llm_translator")
def build_llm_translator(tgt_language: str, config: TranslatorConfig | None = None):
    return LLMTranslator(tgt_language=Language(tgt_language))
