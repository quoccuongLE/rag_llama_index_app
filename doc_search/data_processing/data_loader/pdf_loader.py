from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import pymupdf as fitz  # available with v1.24.3
except ImportError:
    import fitz

from fitz import Document as FitzDocument
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document as LlamaIndexDocument
from pymupdf4llm import IdentifyHeaders, to_markdown
from tqdm import tqdm

from doc_search.data_processing.data_loader import factory
from doc_search.settings import LoaderConfig
from llama_index.core import PromptTemplate, Settings

# text_summary_template = PromptTemplate(
#     "Context information is below.\n"
#     "---------------------\n"
#     "{context_str}\n"
#     "---------------------\n"
#     "Given the context information and not prior knowledge, "
#     "answer the question: {query_str}\n"
# )

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


_INSTRUCT_PROMPT = """This document is an insurance policy.
When a benefits/coverage/exlusion is describe in the document ammend to it add a text in the follwing benefits string format (where coverage could be an exclusion). For {nameofrisk} and in this condition {whenDoesThecoverageApply} the coverage is {coverageDescription}. If the document contain a benefits TABLE that describe coverage amounts, do not ouput it as a table, but instead as a list of benefits string."""


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
        text = text.replace("\n\n", "~~")
        text = text.replace("**\n", "!!")
        text = text.replace("\n", " ")
        text = text.replace("~~", "\n\n")
        text = text.replace("!!", "**\n")
        return text

    def _process_doc_page(
        self,
        doc: FitzDocument,
        extra_info: Dict[str, Any],
        file_path: str,
        page_number: int,
        hdr_info: IdentifyHeaders,
    ):
        """Processes a single page of a PDF document."""
        extra_info = self._process_doc_meta(doc, file_path, page_number, extra_info)

        if self.meta_filter:
            extra_info = self.meta_filter(extra_info)

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
