from pathlib import Path
import gradio as gr
import yaml

from doc_search.translator import get_available_languages
from doc_search import DocRetrievalAugmentedGen


class TranslatorTab:

    _variant: str = "panel"

    def __init__(
        self,
        rag_engine: DocRetrievalAugmentedGen,
    ) -> None:
        self.rag_engine = rag_engine
        self.create_ui()

    def _preprocess_and_translate(
        self,
        document: str,
        src_lang: str,
        tgt_lang: str,
        export_file: str = "tmp/output.md",
    ):
        tgt_lang, _ = tuple(tgt_lang.split(" - ", 1))
        src_lang, _ = tuple(src_lang.split(" - ", 1))
        file = Path(document)
        self.rag_engine._parser.read_file(
            filename=document,
            indexing=False,
            translate=True,
            src_language=src_lang,
            tgt_language=tgt_lang,
        )
        md_file = (
            Path(self.rag_engine._setting.md_preprocessed_dir) / f"{file.stem}.md"
        )
        self.rag_engine.parser.doc_loader.save_doc(
            filepath=md_file, translated_text=False
        )
        self.rag_engine.parser.doc_loader.save_doc(
            filepath=Path(export_file), translated_text=True
        )

        return gr.update(value=export_file, visible=True), gr.update(
            value=md_file, visible=True
        )

    def _change_src_tgt_languages(self, src_lang: str, tgt_lang: str):
        return gr.update(value=tgt_lang), gr.update(value=src_lang)

    def create_ui(self):
        with gr.Row(variant=self._variant, equal_height=False):
            with gr.Column():
                with gr.Row():
                    src_language = gr.Dropdown(
                        label="Source Language",
                        choices=get_available_languages(),
                        value="eng - English",
                        interactive=True,
                        allow_custom_value=True,
                        # visible=False,
                    )
                    switch_1 = gr.Button(value="Switch ↔")
                with gr.Row():
                    md_documents = gr.Files(
                        label="Preprocessed documents",
                        value=[],
                        file_types=[".md"],
                        file_count="multiple",
                        interactive=True,
                        visible=False,
                    )
                with gr.Row():
                    input_doc = gr.File(
                        label="Input File",
                        file_count="single",
                        file_types=["", ".", ".pdf", "txt"],
                        visible=True,
                    )
                with gr.Row():
                    translate_button = gr.Button("Translate")
            with gr.Column():
                with gr.Row():
                    tgt_language = gr.Dropdown(
                        label="Target Language",
                        choices=get_available_languages(),
                        value="fra - French",
                        interactive=True,
                        allow_custom_value=True,
                    )
                with gr.Row():
                    output_doc = gr.File(
                        label="Output File", value=None, visible=False
                    )

        switch_1.click(
            self._change_src_tgt_languages,
            inputs=[src_language, tgt_language],
            outputs=[src_language, tgt_language],
        )
        translate_button.click(
            self._preprocess_and_translate,
            inputs=[input_doc, src_language, tgt_language],
            outputs=[output_doc, md_documents],
        )
