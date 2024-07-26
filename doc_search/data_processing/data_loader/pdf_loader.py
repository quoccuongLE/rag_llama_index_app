import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import pymupdf as fitz  # available with v1.24.3
except ImportError:
    import fitz

import multiprocessing
from itertools import repeat

from fitz import Document as FitzDocument
from llama_index.core import PromptTemplate, Settings
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document as LlamaIndexDocument
from pymupdf4llm import IdentifyHeaders
from tqdm import tqdm

from doc_search.data_processing.data_loader import factory
from doc_search.settings import LoaderConfig

from .utils.pymupdf import to_markdown

text_summary_template = PromptTemplate(
    "Document is below.\n"
    "---------------------\n"
    "{document_str}\n"
    "---------------------\n"
    "Summary the document following the intruction below:\n"
    "---------------------\n"
    "{instruct_str}\n"
    "---------------------\n"
)


def to_markdown_star(args):
    return to_markdown(*args)


class PDFMarkdownReader(BaseReader):
    """Read PDF files using PyMuPDF library."""

    meta_filter: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    show_progress: bool = False
    text_summarize: bool = False
    parsing_instruction: str | None = None
    num_workers: int = 4

    def __init__(
        self,
        text_summarize: bool = False,
        parsing_instruction: str | None = None,
        show_progress: bool = False,
        meta_filter: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        **kwargs
    ):
        self.text_summarize = text_summarize
        self.parsing_instruction = parsing_instruction
        self.meta_filter = meta_filter
        self.show_progress = show_progress

    def load_data(
        self,
        file_path: Path | str,
        extra_info: dict | None = None,
        **kwargs: Any,
    ) -> List[LlamaIndexDocument]:
        """Loads list of documents from PDF file and also accepts extra information in dict format.

        Args:
            file_path (Union[Path, str]): The path to the PDF file.
            extra_info (Optional[Dict], optional): A dictionary containing extra information. Defaults to None.
            **load_kwargs (Any): Additional keyword arguments to be passed to the load method.

        Returns:
            List[LlamaIndexDocument]: A list of LlamaIndexDocument objects.
        """

        extra_info = extra_info or {}

        # extract text header information
        hdr_info = file_path

        doc: FitzDocument = fitz.open(file_path)
        page_numbers = [[x] for x in range(doc.page_count)]
        extra_info = self._process_doc_meta(doc, file_path, extra_info)
        return self._process_doc_page(
            doc=file_path,
            extra_info=extra_info,
            page_numbers=page_numbers,
            hdr_info=hdr_info,
        )

    def _align_sentence_segments(self, text: str) -> str:
        # Preserve \n\n
        text = text.replace("\n\n", "~~")
        # Preserve **\n
        text = text.replace("**\n", "!!")
        # Remove \n
        text = text.replace("\n", " ")
        # Recover \n\n
        text = text.replace("~~", "\n\n")
        # Recover **\n
        text = text.replace("!!", "**\n")
        return text

    def _process_doc_page(
        self,
        doc: Path | str,
        extra_info: Dict[str, Any],
        page_numbers: list[list[int]],
        hdr_info: str | None = None,
        show_progress: bool = True,
    ) -> LlamaIndexDocument:
        """Processes a single page of a PDF document."""
        extra_info["page_label"] = page_numbers[0][0] + 1

        if self.meta_filter:
            extra_info = self.meta_filter(extra_info)

        if self.num_workers and self.num_workers > 1:
            if self.num_workers > multiprocessing.cpu_count():
                warnings.warn(
                    "Specified num_workers exceed number of CPUs in the system. "
                    "Setting `num_workers` down to the maximum CPU count."
                )

            N = len(page_numbers)
            input_args = zip(repeat(doc), page_numbers, repeat(hdr_info), repeat(False))
            with multiprocessing.get_context("spawn").Pool(self.num_workers) as pool:
                documents = list(tqdm(pool.imap(to_markdown_star, input_args), total=N))

        index_documents = []
        for text in documents:
            text = self._align_sentence_segments(text)
            if self.text_summarize:
                llm = Settings.llm
                prompt = text_summary_template.format(
                    document_str=text, instruct_str=self.parsing_instruction
                )
                summary = llm.complete(prompt)
                extra_info['original_text'] = text
                index_documents.append(LlamaIndexDocument(text=summary.text, extra_info=extra_info))
            else:
                index_documents.append(LlamaIndexDocument(text=text, extra_info=extra_info))

        return index_documents

    def _process_doc_meta(
        self,
        doc: FitzDocument,
        file_path: Union[Path, str],
        extra_info: Optional[Dict] = None,
    ):
        """Processes metas of a PDF document."""
        extra_info.update(doc.metadata)
        extra_info["total_pages"] = len(doc)
        extra_info["file_path"] = str(file_path)

        return extra_info


@factory.register_builder("pdf_markdown_reader")
def build_pdf_markdown_reader(config: LoaderConfig, **kwargs):
    return PDFMarkdownReader(show_progress=config.show_progress, )
