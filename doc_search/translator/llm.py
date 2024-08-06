from .base import Language, Translator
from llama_index.core.settings import Settings


class LLMTranslator(Translator):
    def __init__(
        self,
        tgt_language: Language,
        device: str | None = None,
        max_length: int | None = None,
        model_id: str | None = None,
    ) -> None:
        super().__init__(tgt_language, device, max_length, model_id)
        self.llm = Settings.llm

    def translate(
        self,
        sources: list[str] | str,
        src_lang: str | Language,
        tgt_lang: str | Language | None = None,
    ) -> str:
        # return super().translate(sources, src_lang, tgt_lang)
        pass
