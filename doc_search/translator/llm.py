import json

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from llama_index.core import PromptTemplate
from llama_index.core.output_parsers import LangchainOutputParser
from llama_index.core.output_parsers.utils import _marshal_llm_to_json
from llama_index.core.settings import Settings

from doc_search.llm import factory as llm_factory
from doc_search.settings import LLMSetting, TranslatorConfig
from doc_search.translator import factory

from .base import Language, Translator


class LLMTranslator(Translator):
    _template: PromptTemplate = PromptTemplate(
        "Translate the following passage into {language_name}:"
        "\n---------------------\n"
        "{text_str}\n"
        "\n---------------------"
    )
    _title_template: PromptTemplate = PromptTemplate(
        "Translate the following title into {language_name}:"
        "\n---------------------\n"
        "{text_str}\n"
        "\n---------------------"
    )

    def __init__(
        self,
        tgt_language: Language,
        prompt_template: str | PromptTemplate | None = None,
        llm_config: LLMSetting | None = None
    ) -> None:
        self.tgt_language: Language = tgt_language
        if prompt_template:
            if isinstance(prompt_template, str):
                self._template = PromptTemplate(prompt_template)
            else:
                self._template = prompt_template
        if llm_config:
            self.llm = llm_factory.build(config=llm_config)
        else: 
            self.llm = Settings.llm

    def translate(
        self,
        sources: list[str] | str,
        src_lang: str | Language,
        tgt_lang: str | Language | None = None,
        template: PromptTemplate | None = None,
    ) -> str:
        if isinstance(src_lang, str):
            src_lang = Language(src_lang)
        if tgt_lang:
            if isinstance(tgt_lang, str):
                tgt_lang = Language(tgt_lang)
        else:
            tgt_lang = self.tgt_language

        if isinstance(sources, str) and sources[:3] == "\n##":
            # Title translation
            description = (
                f"The translation in {tgt_lang.english_name} of the original title"
            )
            template = self._title_template
        else:
            description = (
                f"The translation in {tgt_lang.english_name} of the original passage"
            )
        response_schemas = [
            ResponseSchema(name="translated_text", description=description),
        ]

        lc_output_parser = StructuredOutputParser.from_response_schemas(
            response_schemas
        )
        output_parser = LangchainOutputParser(lc_output_parser)
        if template:
            text_with_template = template.format(
                language_name=tgt_lang.english_name, text_str=sources
            )
        else:
            text_with_template = self._template.format(
                language_name=tgt_lang.english_name, text_str=sources
            )
        final_query = output_parser.format(text_with_template)
        response = self.llm.complete(final_query)
        json_dict = self._parse(response.text)
        return json_dict.get("translated_text")

    def _parse(self, output: str) -> dict:
        try:
            json_string = _marshal_llm_to_json(output)
            json_obj = json.loads(json_string)
            # try:
            #     json_obj = json.loads(json_string)
            # except json.decoder.JSONDecodeError as e:
            #     # Look for the error position and try inserting a comma
            #     error_pos = e.pos
            #     if error_pos > 0 and json_string[error_pos - 1] in (']', '}', ':'):
            #         fixed_string = json_string[:error_pos] + ',' + json_string[error_pos:]
            #         try:
            #             fixed_string = json.loads(fixed_string)
            #             return fixed_string  # Fixed JSON
            #         except json.decoder.JSONDecodeError:
            #             pass  # Insertion of comma didn't work

            #     return json_string  # Return original string if no fix found
            if not json_obj:
                raise ValueError(f"Failed to convert output to JSON: {output!r}")
            return json_obj
        except:
            if "```" in output:
                json_string = output.split("```")[1]
            else:
                json_string = output
            if 'json\n{\n\t"translated_text": ' in json_string:
                raw_text = json_string.replace('json\n{\n\t"translated_text": ', '')
            else:
                raw_text = json_string.replace('json\n{\n', "").replace('"translated_text": ', '').strip()
            raw_text = raw_text.replace('"', '')
            raw_text = raw_text.replace('}\n', '')
            raw_text = raw_text[2:]
            json_string = {"translated_text": f"{raw_text}"}
            return json_string


@factory.register_builder("llm_translator")
def build_llm_translator(tgt_language: str, config: TranslatorConfig | None = None):
    return LLMTranslator(tgt_language=Language(tgt_language), llm_config=config.llm)
