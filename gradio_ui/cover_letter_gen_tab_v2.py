import os
import shutil
from pathlib import Path

import gradio as gr

from doc_search.core import DocRetrievalAugmentedGen
from doc_search.query_engine.base import ChatMode
from doc_search.translator import get_available_languages

from .defaults import DefaultElement
from .qa_tab import QATab


class CoverLetterGenTabV2(QATab):
    _resume_dict: dict[str, str] = {
        "my_portfolio.md": Path(
            "/media/oracle/workspace/rag_llama_index_app/data/cover_letter_gen/docs/my_portfolio.md"
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
        portfolio_text = self._load_portfolio("my_portfolio.md")
        self._update_portfolio(portfolio_text)

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

    def _update_portfolio(self, porfolio: str):
        self.rag_engine._query_engine.set_source_document(porfolio)
        gr.Info("Update portfolio!")

    def _load_portfolio(self, filename: str) -> str:
        with open(self._resume_dict[filename], "r", encoding="utf-8") as f:
            document = f.read()
        return document

    def _get_portfolio(self):
        return self.rag_engine._query_engine.portfolio

    def _update_job_description(self, job_description: str):
        self.rag_engine._query_engine.set_job_description(job_description)
        gr.Info("Update job description!")

    def retrieve(self):
        self.rag_engine._query_engine.retrieve()
        return self.rag_engine._query_engine.get_retrieved_items()

    def generate_cover_letter(self):
        response = self.rag_engine._query_engine.generate_cover_letter()
        return response.text

    def create_ui(self):
        with gr.Row(variant=self._variant, equal_height=False):
            with gr.Column(variant=self._variant, scale=50) as setting:
                with gr.Row(variant=self._variant):
                    status = gr.Textbox(
                        label="Status", value="Ready!", interactive=False
                    )
                    file_list = gr.Dropdown(
                            label="Choose file:",
                            choices=list(self._resume_dict.keys()),
                            value="my_portfolio.md",
                            interactive=True,
                            allow_custom_value=True,
                        )
                with gr.Row(variant=self._variant):
                    update_portfolio_btn = gr.Button(value="Update Portfolio")
                    get_portfolio_btn = gr.Button(value="Get Portfolio")
                candidate_porfolio = gr.Code(
                    label="Candidate Portfolio",
                    language="markdown",
                    lines=50,
                    # wrap_lines=True,
                    # max_lines=50,
                    interactive=True,
                    show_label=True,
                )
                with gr.Row(variant=self._variant):
                    language = gr.Dropdown(
                        label="User Language",
                        choices=get_available_languages(),
                        value="eng - English",
                        interactive=True,
                        allow_custom_value=True,
                        visible=True,
                    )
                    doc_language = gr.Dropdown(
                        label="Document Language",
                        choices=get_available_languages(),
                        value="eng - English",
                        interactive=True,
                        allow_custom_value=True,
                        visible=True,
                    )
                with gr.Row(variant=self._variant):
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

            with gr.Column(scale=50, variant=self._variant):
                job_description = gr.Textbox(
                    label="Job description",
                    lines=50,
                    max_lines=100,
                    interactive=True,
                    show_label=True,
                )
                with gr.Row(variant=self._variant):
                    update_jd_btn = gr.Button(value="1 - Update Job Description")
                    find_matches_btn = gr.Button(value="2 - Find Matches")
                    gen_btn = gr.Button(value="3 - Generate Cover Letter")
                cover_letter = gr.Textbox(
                    label="Cover Letter",
                    value="",
                    interactive=False,
                    lines=10,
                    max_lines=50,
                )

        language.change(self._change_language, inputs=[language])
        model.change(self._change_model, inputs=[model], outputs=[status])
        doc_language.change(self._change_doc_language, inputs=[doc_language])
        embed_model.change(
            self._change_embed_model, inputs=[embed_model], outputs=[status]
        )
        update_portfolio_btn.click(self._update_portfolio, inputs=[candidate_porfolio])
        file_list.change(
            self._load_portfolio, inputs=[file_list], outputs=[candidate_porfolio]
        ).then()
        update_jd_btn.click(self._update_job_description, inputs=[job_description])
        gen_btn.click(self.generate_cover_letter, outputs=[cover_letter])
        get_portfolio_btn.click(self._get_portfolio, outputs=[candidate_porfolio])
        find_matches_btn.click(self.retrieve, outputs=[status])
