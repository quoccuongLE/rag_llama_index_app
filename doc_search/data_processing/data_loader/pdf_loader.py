import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import pymupdf as fitz  # available with v1.24.3
except ImportError:
    import fitz

import multiprocessing
from functools import reduce
from itertools import repeat

from fitz import Document as FitzDocument
from llama_index.core import PromptTemplate, Settings
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document as LlamaIndexDocument
from pymupdf4llm import IdentifyHeaders
from pymupdf4llm import to_markdown as to_markdown_legacy
from tqdm import tqdm

from doc_search.data_processing.data_loader import factory
from doc_search.settings import LoaderConfig

from .pymupdf_utils import to_markdown

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


class PDFMarkdownReader(BaseReader):
    """Read PDF files using PyMuPDF library."""

    meta_filter: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    show_progress: bool = False
    text_summarize: bool = False
    parsing_instruction: str | None = None

    def __init__(
        self,
        text_summarize: bool = False,
        parsing_instruction: str | None = None,
        show_progress: bool = False,
        meta_filter: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        self.text_summarize = text_summarize
        self.parsing_instruction = parsing_instruction
        self.meta_filter = meta_filter
        self.show_progress = show_progress

    def load_data(
        self,
        file_path: Union[Path, str],
        extra_info: Optional[Dict] = None,
        **load_kwargs: Any,
    ) -> List[LlamaIndexDocument]:
        """Loads list of documents from PDF file and also accepts extra information in dict format.

        Args:
            file_path (Union[Path, str]): The path to the PDF file.
            extra_info (Optional[Dict], optional): A dictionary containing extra information. Defaults to None.
            **load_kwargs (Any): Additional keyword arguments to be passed to the load method.

        Returns:
            List[LlamaIndexDocument]: A list of LlamaIndexDocument objects.
        """
        if not isinstance(file_path, str) and not isinstance(file_path, Path):
            raise TypeError("file_path must be a string or Path.")

        if not extra_info:
            extra_info = {}

        if extra_info and not isinstance(extra_info, dict):
            raise TypeError("extra_info must be a dictionary.")

        # extract text header information
        hdr_info = IdentifyHeaders(file_path)

        doc: FitzDocument = fitz.open(file_path)
        docs = []
        if self.show_progress:
            pages_to_process = tqdm(doc, desc="Loading files", unit="file")
        else:
            pages_to_process = doc

        for page in pages_to_process:
            docs.append(
                self._process_doc_page(
                    doc, extra_info, file_path, page.number, hdr_info
                )
            )
        return docs

    # Helpers
    # ---

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
        doc: FitzDocument,
        extra_info: Dict[str, Any],
        file_path: str,
        page_number: int,
        hdr_info: IdentifyHeaders,
        page_width: int = 612,
        page_height: int | None = None,
    ) -> LlamaIndexDocument:
        """Processes a single page of a PDF document."""
        extra_info = self._process_doc_meta(doc, file_path, page_number, extra_info)

        if self.meta_filter:
            extra_info = self.meta_filter(extra_info)

        # for reflowable documents allow making 1 page for the whole document
        if doc.is_reflowable:
            if hasattr(page_height, "__float__"):
                # accept user page dimensions
                doc.layout(width=page_width, height=page_height)
            else:
                # no page height limit given: make 1 page for whole document
                doc.layout(width=page_width, height=792)
                page_count = doc.page_count
                height = 792 * page_count  # height that covers full document
                doc.layout(width=page_width, height=height)

        if pages is None:  # use all pages if no selection given
            pages = list(range(doc.page_count))

        if hasattr(margins, "__float__"):
            margins = [margins] * 4
        if len(margins) == 2:
            margins = (0, margins[0], 0, margins[1])
        if len(margins) != 4:
            raise ValueError("margins must be a float or a sequence of 2 or 4 floats")
        elif not all([hasattr(m, "__float__") for m in margins]):
            raise ValueError("margin values must be floats")

        pages = [x for x in doc]
        doc_metadata = doc.metadata.copy()
        doc_metadata.update(dict(file_path=doc.name, page_count=doc.page_count))
        # read the Table of Contents
        table_of_contents = doc.get_toc()

        if self.num_workers and self.num_workers > 1:
            if self.num_workers > multiprocessing.cpu_count():
                warnings.warn(
                    "Specified num_workers exceed number of CPUs in the system. "
                    "Setting `num_workers` down to the maximum CPU count."
                )
            with multiprocessing.get_context("spawn").Pool(self.num_workers) as p:
                results = p.starmap(
                    to_markdown,
                    pages,
                    repeat(doc_metadata),
                    repeat(table_of_contents),
                    repeat(hdr_info)
                )
                documents = reduce(lambda x, y: x + y, results)

        # TODO: Here 
        text = to_markdown(
            doc, pages=[page_number], hdr_info=hdr_info, write_images=False
        )
        text = self._align_sentence_segments(text)
        if self.text_summarize:
            llm = Settings.llm
            prompt = text_summary_template.format(
                document_str=text, instruct_str=self.parsing_instruction
            )
            summary = llm.complete(prompt)
            extra_info['original_text'] = text
            return LlamaIndexDocument(text=summary.text, extra_info=extra_info)
        else:
            return LlamaIndexDocument(text=text, extra_info=extra_info)

    def _process_doc_meta(
        self,
        doc: FitzDocument,
        file_path: Union[Path, str],
        page_number: int,
        extra_info: Optional[Dict] = None,
    ):
        """Processes metas of a PDF document."""
        extra_info.update(doc.metadata)
        extra_info["page_label"] = page_number + 1
        extra_info["total_pages"] = len(doc)
        extra_info["file_path"] = str(file_path)

        return extra_info


@factory.register_builder("pdf_markdown_reader")
def build_pdf_markdown_reader(config: LoaderConfig, **kwargs):
    return PDFMarkdownReader(show_progress=config.show_progress, )
