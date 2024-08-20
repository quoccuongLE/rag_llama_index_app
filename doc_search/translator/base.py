from dataclasses import dataclass

import langcodes
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES

from doc_search.settings import TranslatorConfig
from doc_search.translator import factory


def get_language_name(language_code: str, display_lang: str | None = None) -> str:
    try:
        language = langcodes.Language.get(language_code)
        return (
            language.display_name(display_lang)
            if display_lang
            else language.display_name(language_code)
        )
    except langcodes.LanguageLookupError:
        return f"Unknown language code: {language_code}"


LANGUAGE_CODE_REGISTRY: dict[str, str] = {
    x.split("_")[0]: x.split("_")[1] for x in FAIRSEQ_LANGUAGE_CODES
}


def get_available_languages() -> list[str]:
    return [Language(x).fullname for x in LANGUAGE_CODE_REGISTRY.keys()]


class Language:
    language_code: str
    root_code: str
    name: str
    english_name: str

    def __init__(self, language_code: str) -> None:
        self.language_code = language_code
        self.root_code = LANGUAGE_CODE_REGISTRY[language_code]
        self.name = get_language_name(language_code)
        self.english_name = get_language_name(language_code, display_lang="eng")

    @staticmethod
    def from_fair_langcode(fair_langcode: str):
        lang_code = fair_langcode.split("_")[0]
        return Language(language_code=lang_code)

    @staticmethod
    def from_code_fullname(name: str):
        lang_code = name.split("_")[0].strip()
        return Language(language_code=lang_code)

    @property
    def fullname(self):
        return f"{self.language_code} - {self.name}"

    @property
    def fair_langcode(self):
        return f"{self.language_code}_{self.root_code}"

    def __repr__(self) -> str:
        return self.fullname

    def __str__(self) -> str:
        return self.fullname


class Translator:
    tgt_language: Language = Language("eng")
    model_id: str = "facebook/nllb-200-distilled-600M"
    max_length: int = 256
    device: str = torch.device(torch.cuda.is_available() and "cuda" or "cpu")

    def __init__(
        self,
        tgt_language: Language,
        device: str | None = None,
        max_length: int | None = None,
        model_id: str | None = None,
    ) -> None:
        self.tgt_language = tgt_language
        self.device = device or self.device
        self.max_length = max_length or self.max_length
        self.model_id = model_id or self.model_id
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(self.model_id).to(self.device).eval()
        )

    def get_tokenizer(self, src_language: Language) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(
            self.model_id, src_lang=src_language.fair_langcode
        )

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

        tokenizer = self.get_tokenizer(src_language=src_lang)
        inputs = tokenizer(sources, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang.fair_langcode),
            max_length=self.max_length,
        )
        return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


@factory.register_builder("nllb")
def build_base_translator(
    tgt_language: str,
    config: TranslatorConfig,
):
    return Translator(
        tgt_language=Language(tgt_language),
        max_length=config.max_length,
        model_id=config.hf_model_id,
    )


@dataclass
class _TranslationService:
    _translator: Translator | None = None

    @property
    def translator(self) -> Translator:
        return self._translator

    @translator.setter
    def translator(self, translator: Translator):
        self._translator = translator


TranslationService = _TranslationService()
