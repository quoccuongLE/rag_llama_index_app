from llama_index.core import PromptTemplate, Settings
from llama_index.core.readers.base import BaseReader

from doc_search.translator import Translator, Language
from doc_search.translator import factory
from doc_search.settings import TranslatorConfig


class MultiLingualBaseReader(BaseReader):
    _template: PromptTemplate = PromptTemplate(
        # "\nGiven an original text passage is in Markdown format, the answer must "
        # "be in markdown format, and the resulting translation must be complete."
        "Translate the following passage into {language_name}:"
        "\n---------------------\n"
        "{text_str}\n"
        "\n---------------------"
    )
    _translator: Translator = factory.build(
        config=TranslatorConfig(), name="llm_translator", tgt_language="eng"
    )
    _tgt_language: Language = Language("eng")

    def __init__(
        self,
        tgt_language: Language = Language("eng"),
        translator_config: TranslatorConfig | None = None,
    ) -> None:
        self._tgt_language: Language = tgt_language
        if translator_config:
            self._translator = factory.build(
                config=translator_config,
                tgt_language=self._tgt_language,
            )

    @property
    def tgt_language(self) -> Language:
        return self._tgt_language

    @tgt_language.setter
    def tgt_language(self, language: str | Language):
        if isinstance(language, str):
            langcode, _ = tuple(language.split(" - ", 1))
            self._tgt_language = Language(langcode)
        else:
            self._tgt_language = language

    def translate_node_text(self, text: str, src_lang: str, tgt_lang: str | None = None) -> str:
        return self._translator.translate(
            sources=text,
            src_lang=src_lang,
            tgt_lang=tgt_lang or self._tgt_language,
            template=self._template,
        )
