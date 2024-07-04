from pathlib import Path
from doc_search.data_processing.data_loader import factory

from llama_docs_utils.markdown_docs_reader import MarkdownDocsReader

from llama_index.core import SimpleDirectoryReader

from llama_parse import LlamaParse

from doc_search.settings import LoaderConfig, ReaderConfig


def get_reader_dict(config: ReaderConfig):
    # TODO: To be replaced
    reader_dict = {}
    for ext in config.file_extractor:
        if ext == ".md":
            reader_dict[ext] = MarkdownDocsReader()
    
    return reader_dict


@factory.register_builder("directory")
def build_dir_reader(path: Path, config: LoaderConfig):
    return SimpleDirectoryReader(
        input_dir=path,
        required_exts=list(config.reader_config.file_extractor.keys()),
        file_extractor=get_reader_dict(config.reader_config.file_extractor),
        recursive=config.recursive,
    )


@factory.register_builder("single_file")
def build_dir_reader(file: Path, config: LoaderConfig):
    assert file.is_file(), f"This path {str(file)} is not a file"
    return SimpleDirectoryReader(
        input_files=[file],
        required_exts=list(config.reader_config.file_extractor.keys()),
        file_extractor=get_reader_dict(config.reader_config.file_extractor)
    )


@factory.register_builder("llama_parse")
def build_llama_parse_reader(file: Path, config: LoaderConfig):
    assert file.is_file(), f"This path {str(file)} is not a file"
    return LlamaParse(result_type=config.result_type)


@factory.register_builder("multiple_files")
def build_dir_reader(files: list[Path], config: LoaderConfig):
    return SimpleDirectoryReader(
        input_files=files,
        required_exts=list(config.reader_config.file_extractor.keys()),
        file_extractor=get_reader_dict(config.reader_config.file_extractor)
    )
