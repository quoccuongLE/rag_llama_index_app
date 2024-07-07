from pathlib import Path
from doc_search.data_processing.data_loader import factory

from .markdown_docs_reader import MarkdownDocsReader

from llama_index.core import SimpleDirectoryReader

from llama_parse import LlamaParse

from doc_search.settings import LoaderConfig


def get_reader_dict(file_extractor_list: list[str]):
    # TODO: To be replaced
    reader_dict = {}
    for ext in file_extractor_list:
        if ext == ".md":
            reader_dict[ext] = MarkdownDocsReader()

    return reader_dict


@factory.register_builder("directory")
def build_dir_reader(path: Path, config: LoaderConfig):
    return SimpleDirectoryReader(
        input_dir=path,
        required_exts=config.loader_config.file_extractor,
        file_extractor=get_reader_dict(config.loader_config.file_extractor),
        recursive=config.recursive,
    )


@factory.register_builder("single_file")
def build_simple_file_reader(file: Path, config: LoaderConfig):
    assert file.is_file(), f"This path {str(file)} is not a file"
    return SimpleDirectoryReader(
        input_files=[file],
        required_exts=config.loader_config.file_extractor,
        file_extractor=get_reader_dict(config.loader_config.file_extractor),
    )


@factory.register_builder("llama_parse")
def build_llama_parse_reader(file: Path, config: LoaderConfig):
    assert file.is_file(), f"This path {str(file)} is not a file"
    return LlamaParse(result_type=config.result_type)


@factory.register_builder("multiple_files")
def build_multifile_reader(files: list[Path], config: LoaderConfig):
    return SimpleDirectoryReader(
        input_files=files,
        required_exts=config.loader_config.file_extractor,
        file_extractor=get_reader_dict(config.loader_config.file_extractor),
    )
