import os
import shutil
import sys
import gradio as gr

from doc_search import DocRetrievalAugmentedGen
from doc_search.logger import Logger
from doc_search.query_engine.base import ChatMode
from doc_search.translator import get_available_languages
from .defaults import DefaultElement, LLMResponse
from .chat_tab import ChatTab


class QATab(ChatTab):
    _llm_response: LLMResponse = LLMResponse()
    _variant: str = "panel"
    _chat_mode: str = "QA"

    def _change_doc_language(self, language: str):
        self.rag_engine.doc_language = language

    def _change_embed_model(self, model: str):
        if model not in [None, ""]:
            self.rag_engine.embed_model = model
            self.rag_engine.reset_engine()
            gr.Info(f"Change embed model to {model}!")
        return DefaultElement.DEFAULT_STATUS

    def _upload_document(self, document: list[str], list_files: tuple[list[str], dict]):
        if document in [None, []]:
            if isinstance(list_files, list):
                return (list_files, DefaultElement.DEFAULT_DOCUMENT)
            else:
                if list_files.get("files", None):
                    return list_files.get("files")
                return document
        else:
            if isinstance(list_files, list):
                return (document + list_files, DefaultElement.DEFAULT_DOCUMENT)
            else:
                if list_files.get("files", None):
                    return document + list_files.get("files")
                return document

    def _reset_document(self):
        # self.rag_engine.reset_documents()
        gr.Info("Reset all documents!")
        return (
            DefaultElement.DEFAULT_DOCUMENT,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def _show_document_btn(self, document: list[str]):
        visible = False if document in [None, []] else True
        return (gr.update(visible=visible), gr.update(visible=visible))

    def _update_file_list(self):
        return gr.Dropdown(choices=self.rag_engine._files_registry)

    def _change_selected_file(self, filename: str):
        self.rag_engine._query_engine_name = filename
        self.rag_engine.set_chat_mode()

    def _processing_document(
        self, document: list[str], progress=gr.Progress(track_tqdm=False)
    ):
        document = document or []
        if self._host == "127.0.0.1":
            input_files = []
            for file_path in document:
                dest = os.path.join(self._data_dir, file_path.split("/")[-1])
                shutil.move(src=file_path, dst=dest)
                input_files.append(dest)
            self.rag_engine.store_nodes(input_files=input_files)
        else:
            self.rag_engine.store_nodes(input_files=document)
        self.rag_engine.set_chat_mode()
        gr.Info("Processing Completed!")
        return DefaultElement.COMPLETED_STATUS

    def create_ui(self):
        with gr.Row(variant=self._variant, equal_height=False):
            with gr.Column(variant=self._variant, scale=10) as setting:
                with gr.Column():
                    status = gr.Textbox(
                        label="Status", value="Ready!", interactive=False
                    )
                    language = gr.Dropdown(
                        label="User Language",
                        choices=get_available_languages(),
                        value="eng - English",
                        interactive=True,
                        allow_custom_value=True,
                        visible=True,
                    )
                    model = gr.Dropdown(
                        label="Choose LLM:",
                        choices=self.rag_engine.get_available_models(),
                        value=self.rag_engine.default_model,
                        interactive=True,
                        allow_custom_value=False,
                    )
                    embed_model = gr.Dropdown(
                        label="Choose Embed Model:",
                        choices=self.rag_engine.get_available_embed_models(),
                        value=self.rag_engine.default_embed_model,
                        interactive=True,
                        allow_custom_value=False,
                    )
                    doc_language = gr.Dropdown(
                        label="Document Language",
                        choices=get_available_languages(),
                        value="eng - English",
                        interactive=True,
                        allow_custom_value=True,
                        visible=True,
                    )
                    file_list = gr.Dropdown(
                        label="Choose file:",
                        choices=list(self.rag_engine._files_registry),
                        value=None,
                        interactive=True,
                        allow_custom_value=True,
                    )

                    documents = gr.Files(
                        label="Add Documents",
                        value=[],
                        file_types=[".txt", ".pdf", ".csv"],
                        file_count="multiple",
                        height=150,
                        interactive=True,
                    )
                    with gr.Row():
                        upload_doc_btn = gr.UploadButton(
                            label="Upload",
                            value=[],
                            file_types=[".txt", ".pdf", ".csv"],
                            file_count="multiple",
                            min_width=20,
                            visible=False,
                        )
                        reset_doc_btn = gr.Button(
                            "Reset", min_width=20, visible=False
                        )

            with gr.Column(scale=30, variant=self._variant):
                chatbot = gr.Chatbot(
                    layout="bubble",
                    value=[],
                    height=550,
                    scale=2,
                    show_copy_button=True,
                    bubble_full_width=False,
                    avatar_images=self._avatar_images,
                )
                with gr.Row(variant=self._variant):
                    message = gr.MultimodalTextbox(
                        value=DefaultElement.DEFAULT_MESSAGE,
                        placeholder="Enter you message:",
                        show_label=False,
                        scale=6,
                        lines=1,
                    )
                with gr.Row(variant=self._variant):
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=30,
                        value=5,
                        step=1,
                        visible=False,
                        show_label=False,
                    )
                    nb_extract_char = gr.Number(
                        value=300, visible=False, show_label=False
                    )
                    search_update_btn = gr.Button(
                        value="Update", min_width=10, visible=False
                    )
                with gr.Row(variant=self._variant):
                    ui_btn = gr.Button(
                        value="Hide/Show Setting",
                        min_width=20,
                    )
                    undo_btn = gr.Button(value="Undo", min_width=20)
                    clear_btn = gr.Button(value="Clear", min_width=20)
                    reset_btn = gr.Button(value="Reset", min_width=20)

        clear_btn.click(self._clear_chat, outputs=[message, chatbot, status])
        undo_btn.click(self._undo_chat, inputs=[chatbot], outputs=[chatbot])
        reset_btn.click(self._reset_chat, outputs=[message, chatbot, status])
        message.submit(
            self._get_respone,
            inputs=[message, chatbot],
            outputs=[message, chatbot, status],
        )
        language.change(self._change_language, inputs=[language]).then(
            self._clear_chat, outputs=[message, chatbot, status]
        )
        model.change(self._change_model, inputs=[model], outputs=[status])
        ui_btn.click(
            self._show_hide_setting,
            inputs=[self.sidebar_state],
            outputs=[ui_btn, setting, self.sidebar_state],
        )
        # QA
        doc_language.change(self._change_doc_language, inputs=[doc_language])
        embed_model.change(
            self._change_embed_model, inputs=[embed_model], outputs=[status]
        )
        documents.change(
            self._processing_document,
            inputs=[documents],
            outputs=[status],
        ).then(
            self._show_document_btn,
            inputs=[documents],
            outputs=[upload_doc_btn, reset_doc_btn],
        ).then(
            self._update_file_list, outputs=[file_list]
        )
        file_list.change(self._change_selected_file, inputs=[file_list])
