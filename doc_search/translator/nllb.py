import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES
import langcodes


def get_language_name(language_code: str) -> str:
    try:
        language = langcodes.Language.get(language_code)
        return language.display_name()
    except langcodes.LanguageLookupError:
        return f"Unknown language code: {language_code}"


LANGUAGE_CODE_REGISTRY: dict[str, str] = {
    x.split("_")[0]: x.split("_")[1] for x in FAIRSEQ_LANGUAGE_CODES
}

class Language:
    language_code: str
    root_code: str
    name: str

    def __init__(self, language_code: str) -> None:
        self.language_code = language_code
        self.root_code = LANGUAGE_CODE_REGISTRY[language_code]
        self.name = get_language_name(language_code)

    @staticmethod
    def from_fair_langcode(fair_langcode: str):
        lang_code, root_lang_code = fair_langcode.split("_")
        return Language(language_code=lang_code, root_code=root_lang_code)

    @property
    def fair_langcode(self):
        return f"{self.language_code}_{self.root_code}"

    def __repr__(self) -> str:
        return f"{self.language_code} - {self.name}"

    def __str__(self) -> str:
        return f"{self.language_code} - {self.name}"


class NLLB:
    src_language: Language = Language.from_fair_langcode("eng_Latn")
    model_id: str = "facebook/nllb-200-distilled-600M"
    max_length: int = 256
    device: str = torch.device(torch.cuda.is_available() and "cuda" or "cpu")

    def __init__(
        self,
        src_language: str,
        device: str | None = None,
        max_length: int | None = None,
        model_id: str | None = None,
    ) -> None:
        self.src_language = src_language
        self.device = device or self.device
        self.max_length = max_length or self.max_length
        self.model_id = model_id or self.model_id
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(self.model_id).to(device).eval()
        )

    def get_tokenizer(self, src_language: str) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.model_id, src_lang=src_language)

    def translate(
        self, sources: list[str], tgt_lang: str, src_lang: str | None = None
    ) -> str:
        tokenizer = self.get_tokenizer(src_lang=src_lang or self.src_language)
        inputs = tokenizer(sources, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=self.max_length,
        )
        return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
