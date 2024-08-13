import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from fitz import Document as FitzDocument
from llama_index.core import PromptTemplate, Settings
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document as LlamaIndexDocument
from marker.models import load_all_models
from marker.settings import settings
from pymupdf4llm import IdentifyHeaders, to_markdown
from tqdm import tqdm

from .utils.marker_pdf import convert_single_pdf_no_images
from doc_search.data_processing.data_loader import factory
from doc_search.settings import LoaderConfig
from marker.postprocessors.markdown import get_full_text


text_summary_template = PromptTemplate(
    "Document is below.\n"
    "---------------------\n"
    "{document_str}\n"
    "---------------------\n"
    "Summary the document following intruction below:\n"
    "---------------------\n"
    "{instruct_str}\n"
    "---------------------\n"
)


class MarkerPDFReader(BaseReader):
    """Read PDF files using Marker PDF library."""

    show_progress: bool = False
    text_summarize: bool = False
    parsing_instruction: str | None = None
    max_pages: int | None = None
    start_page: int | None = None
    langs: list[str] | None = None
    batch_multiplier: int = 1
    page_merge: bool = False

    def __init__(
        self,
        text_summarize: bool = False,
        parsing_instruction: str | None = None,
        max_pages: int | None = None,
        start_page: int | None = None,
        langs: list[str] | None = None,
        batch_multiplier: int = 1,
        page_merge: bool = False,
        show_progress: bool = False,
    ):
        self.text_summarize = text_summarize
        self.parsing_instruction = parsing_instruction
        self.show_progress = show_progress
        self.max_pages = max_pages
        self.start_page = start_page
        self.langs = langs
        self.batch_multiplier = batch_multiplier
        self.page_merge = page_merge
        self._model_list = load_all_models()
        self._chunk_full_text: str = ""

    def load_data(
        self,
        file_path: Path | str,
        extra_info: dict | None = None,
        **kwargs: any,
    ) -> List[LlamaIndexDocument]:
        """Loads list of documents from PDF file and also accepts extra information in dict format.

        Args:
            file_path (Union[Path, str]): The path to the PDF file.
            extra_info (Optional[Dict], optional): A dictionary containing extra information. Defaults to None.
            **load_kwargs (Any): Additional keyword arguments to be passed to the load method.

        Returns:
            List[LlamaIndexDocument]: A list of LlamaIndexDocument objects.
        """
        full_texts, out_meta = convert_single_pdf_no_images(
            fname=file_path,
            model_lst=self._model_list,
            max_pages=self.max_pages,
            langs=self.langs,
            batch_multiplier=self.batch_multiplier, 
            start_page=self.start_page,
            page_merge=self.page_merge,
        )
        self._chunk_full_text = "".join(full_texts)
        index_documents = []
        extra_info = extra_info or {}
        for text in full_texts:
            _extra_info = copy.copy(extra_info)
            _extra_info.update(out_meta)
            if self.text_summarize:
                llm = Settings.llm
                prompt = text_summary_template.format(
                    document_str=text, instruct_str=self.parsing_instruction
                )
                summary = llm.complete(prompt)
                extra_info["original_text"] = text
                index_documents.append(LlamaIndexDocument(text=summary.text, extra_info=extra_info))
            else:
                index_documents.append(LlamaIndexDocument(text=text, extra_info=extra_info))

        return index_documents

    def save_doc(self, filepath: Path):
        with open(filepath, "w+", encoding='utf-8') as f:
            f.write(self._chunk_full_text)


@factory.register_builder("marker_pdf_reader")
def build_pdf_markdown_reader(config: LoaderConfig, **kwargs):
    return MarkerPDFReader(
        show_progress=config.show_progress,
        text_summarize=config.text_summarize,
        parsing_instruction=config.parsing_instruction,
        max_pages=config.max_pages,
        start_page=config.start_page,
        langs=config.langs,
        batch_multiplier=config.batch_multiplier,
        page_merge=config.page_merge
    )
