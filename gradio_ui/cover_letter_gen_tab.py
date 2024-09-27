import os
from pathlib import Path
import shutil
import sys
from doc_search.core import DocRetrievalAugmentedGen
from doc_search.query_engine.base import ChatMode
from .qa_tab import QATab
import gradio as gr

from doc_search.translator import get_available_languages
from .defaults import DefaultElement


class CoverLetterGenTab(QATab):
    _resume_dict: dict[str, str] = {
        "my_portfolio.md": Path(
            "./data/cover_letter_gen/docs/my_portfolio.md"
        )
    }

    def __init__(
        self,
        rag_engine: DocRetrievalAugmentedGen,
        chat_mode: str | None = None,
        avatar_images: list[str] = ["./assets/user.png", "./assets/bot.png"],
        data_dir: str = "data/cover_letter_gen/docs",
        logfile: str = "logging.log",
    ) -> None:
        super().__init__(rag_engine, chat_mode, avatar_images, data_dir, logfile)
        self.rag_engine.set_chat_mode(
            chat_mode=ChatMode.COVERLETTER_GEN, chat_config=dict(type=self.chat_mode)
        )
        self._change_selected_file("my_portfolio.md")

    def _processing_document(
        self, document: list[str], progress=gr.Progress(track_tqdm=False)
    ):
        document = document or []
        if self._host == "127.0.0.1":
            for file_path in document:
                dest = os.path.join(self._data_dir, file_path.split("/")[-1])
                shutil.move(src=file_path, dst=dest)
                file_0 = Path(dest)
                self._resume_dict.update({file_0.name: file_0})

        self.rag_engine.set_chat_mode(
            chat_mode=ChatMode.COVERLETTER_GEN, chat_config=dict(type=self.chat_mode)
        )
        gr.Info("Processing Completed!")
        return DefaultElement.COMPLETED_STATUS

    def _update_file_list(self):
        return gr.Dropdown(choices=list(self._resume_dict))

    def _change_selected_file(self, filename: str):
        with open(self._resume_dict[filename], "r", encoding="utf-8") as f:
            document = f.read()
        self.rag_engine._query_engine.set_source_document(document)

    def check_and_update_chat_mode(self, topk: int = 3, job_name: str = ""):
        self.rag_engine._query_engine._topk = topk
        self.rag_engine._query_engine.job_name = job_name

    def _get_respone(
        self,
        message: dict[str, str],
        chatbot: list[list[str, str]],
        progress=gr.Progress(track_tqdm=False),
    ):
        if message["text"] in [None, ""]:
            for m in self._llm_response.empty_message():
                yield m
        else:
            console = sys.stdout
            sys.stdout = self._logger
            response = self.rag_engine.query(self.chat_mode, message["text"], chatbot)
            # Yield response
            for m in self._llm_response.stream_response(
                message["text"], chatbot, response
            ):
                yield m
            sys.stdout = console

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
                        choices=list(self._resume_dict.keys()),
                        value="my_portfolio.md",
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
                            label="Upload your portfolio",
                            value=[],
                            file_types=[".txt", ".md"],
                            file_count="multiple",
                            min_width=20,
                            visible=False,
                        )
                        reset_doc_btn = gr.Button("Reset", min_width=20, visible=False)

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
                        placeholder="Paste the job description here:",
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
                        show_label=False,
                    )
                    # nb_extract_char = gr.Number(value=300, show_label=False)
                    job_name = gr.Text(show_label="Job name", value="Data Scientist")
                    search_update_btn = gr.Button(value="Update", min_width=10)
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
        search_update_btn.click(
            self.check_and_update_chat_mode, inputs=[top_k, job_name]
        )
