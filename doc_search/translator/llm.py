from llama_index.core import PromptTemplate
from llama_index.core.settings import Settings

from doc_search.settings import TranslatorConfig
from doc_search.translator import factory

from .base import Language, Translator


class LLMTranslator(Translator):
    _template: PromptTemplate = PromptTemplate(
        "Translate the following passage into {language_name}:\n{text_str}\n"
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
        translated_text = self.llm.complete(
            self._template.format(language_name=tgt_lang.english_name, text_str=sources)
        )
        #TODO: post processing to extract refined response
        return translated_text.text


@factory.register_builder("llm_translator")
def build_llm_translator(tgt_language: str, config: TranslatorConfig | None = None):
    return LLMTranslator(tgt_language=Language(tgt_language))
