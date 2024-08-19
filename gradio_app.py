import os
from pathlib import Path
import shutil
import sys
import time
import yaml
import gradio as gr
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Tuple
from llama_index.core.chat_engine.types import StreamingAgentChatResponse

from doc_search import DocRetrievalAugmentedGen
from doc_search.logger import Logger
from doc_search.translator import get_available_languages

import fire

LOG_FILE = "logging.log"
DATA_DIR = "data/data"
AVATAR_IMAGES = ["./assets/user.png", "./assets/bot.png"]

JS_LIGHT_THEME = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

CSS = """
.btn {
    background-color: #64748B;
    color: #FFFFFF;
    }

.stop_btn {
    background-color: #ff7373;
    color: #FFFFFF;
    }
"""


@dataclass
class DefaultElement:
    # DEFAULT_MESSAGE: ClassVar[dict] = {"text": "How to fine-tune a LLama model?"}
    # DEFAULT_MESSAGE: ClassVar[dict] = {"text": "Comment fine-tune un modèle LLama?"}
    # DEFAULT_MESSAGE: ClassVar[dict] = {"text": "Tell me about the repatriation policy in the insurance contract."}
    DEFAULT_MESSAGE: ClassVar[dict] = {"text": ""}
    DEFAULT_MODEL: str = ""
    DEFAULT_HISTORY: ClassVar[list] = []
    DEFAULT_DOCUMENT: ClassVar[list] = []

    HELLO_MESSAGE: str = "Hi 👋, how can I help you today?"
    SET_MODEL_MESSAGE: str = "You need to choose LLM model 🤖 first!"
    EMPTY_MESSAGE: str = "You need to enter your message!"
    DEFAULT_STATUS: str = "Ready!"
    CONFIRM_PULL_MODEL_STATUS: str = "Confirm Pull Model!"
    PULL_MODEL_SCUCCESS_STATUS: str = "Pulling model 🤖 completed!"
    PULL_MODEL_FAIL_STATUS: str = "Pulling model 🤖 failed!"
    MODEL_NOT_EXIST_STATUS: str = "Model doesn't exist!"
    PROCESS_DOCUMENT_SUCCESS_STATUS: str = "Processing documents 📄 completed!"
    PROCESS_DOCUMENT_EMPTY_STATUS: str = "Empty documents!"
    ANSWERING_STATUS: str = "Answering!"
    COMPLETED_STATUS: str = "Completed!"


class LLMResponse:
    def __init__(self) -> None:
        pass

    def _yield_string(self, message: str):
        for i in range(len(message)):
            time.sleep(0.01)
            yield (
                DefaultElement.DEFAULT_MESSAGE,
                [[None, message[: i + 1]]],
                DefaultElement.DEFAULT_STATUS,
            )

    def welcome(self):
        yield from self._yield_string(DefaultElement.HELLO_MESSAGE)

    def set_model(self):
        yield from self._yield_string(DefaultElement.SET_MODEL_MESSAGE)

    def empty_message(self):
        yield from self._yield_string(DefaultElement.EMPTY_MESSAGE)

    def stream_response(
        self,
        message: str,
        history: List[List[str]],
        response: StreamingAgentChatResponse,
    ):
        answer = []
        _response = (
            response.response_gen
            if isinstance(response, StreamingAgentChatResponse)
            else response.response
        )
        for text in _response:
            answer.append(text)
            yield (
                DefaultElement.DEFAULT_MESSAGE,
                history + [[message, "".join(answer)]],
                DefaultElement.ANSWERING_STATUS,
            )
        yield (
            DefaultElement.DEFAULT_MESSAGE,
            history + [[message, "".join(answer)]],
            DefaultElement.COMPLETED_STATUS,
        )


class LocalChatbotUI:
    def __init__(
        self,
        logger: Logger,
        host: str = "127.0.0.1",
        data_dir: str = "data/doc_search/docs",
        avatar_images: List[str] = ["./assets/user.png", "./assets/bot.png"],
        rag_yaml_config: Path | None = None,
    ):
        self._rag_engine = DocRetrievalAugmentedGen(setting=rag_yaml_config)
        self._logger = logger
        self._host = host
        self._data_dir = os.path.join(os.getcwd(), data_dir)
        self._avatar_images = [
            os.path.join(os.getcwd(), image) for image in avatar_images
        ]
        self._variant = "panel"
        self._llm_response = LLMResponse()

    def _get_respone(
        self,
        chat_mode: str,
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
            response = self._rag_engine.query(chat_mode, message["text"], chatbot)
            # Yield response
            for m in self._llm_response.stream_response(
                message["text"], chatbot, response
            ):
                yield m
            sys.stdout = console

    def _change_model(self, model: str):
        if model not in [None, ""]:
            self._rag_engine.model = model
            self._rag_engine.reset_engine()
            gr.Info(f"Change model to {model}!")
        return DefaultElement.DEFAULT_STATUS

    def _change_embed_model(self, model: str):
        if model not in [None, ""]:
            self._rag_engine.embed_model = model
            self._rag_engine.reset_engine()
            gr.Info(f"Change embed model to {model}!")
        return DefaultElement.DEFAULT_STATUS

    def _upload_document(self, document: List[str], list_files: Tuple[List[str], dict]):
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
        # self._rag_engine.reset_documents()
        gr.Info("Reset all documents!")
        return (
            DefaultElement.DEFAULT_DOCUMENT,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def _show_document_btn(self, document: List[str]):
        visible = False if document in [None, []] else True
        return (gr.update(visible=visible), gr.update(visible=visible))

    def _processing_document(
        self, document: List[str], progress=gr.Progress(track_tqdm=False)
    ):
        document = document or []
        if self._host == "127.0.0.1":
            input_files = []
            for file_path in document:
                dest = os.path.join(self._data_dir, file_path.split("/")[-1])
                shutil.move(src=file_path, dst=dest)
                input_files.append(dest)
            self._rag_engine.store_nodes(input_files=input_files)
        else:
            self._rag_engine.store_nodes(input_files=document)
        self._rag_engine.set_chat_mode()
        gr.Info("Processing Completed!")
        return (self._rag_engine.system_prompt, DefaultElement.COMPLETED_STATUS)

    def _change_system_prompt(self, sys_prompt: str):
        self._rag_engine.system_prompt = sys_prompt
        self._rag_engine.set_chat_mode()
        gr.Info("System prompt updated!")

    def _change_language(self, language: str):
        self._rag_engine.set_chat_mode(language=language)
        gr.Info(f"Change language to {language}")

    def _change_doc_language(self, language: str):
        self._rag_engine.doc_language = language

    def _change_chat_mode(self, chat_mode: str, topk: int, nb_extract_char: int):
        chat_config = dict(type=chat_mode)
        (
            _topk,
            _nb_extract_char,
            _search_update_btn,
            _language,
            _embed_model,
            _file_list,
            _document,
        ) = (False, False, False, False, False, False, False)
        match chat_mode:
            case "semantic search":
                chat_config.update(
                    dict(synthesizer=dict(topk=topk, sample_length=nb_extract_char))
                )
                _topk = True
                _nb_extract_char = True
                _search_update_btn = True
                _embed_model = True
                _file_list = True
                _document = True

            case "chat":
                _language = True

            case "QA":
                _language = True
                _embed_model = True
                _file_list = True
                _document = True

        self._rag_engine.set_chat_mode(chat_mode=chat_mode, chat_config=chat_config)
        gr.Info(f"Change chat mode to {chat_mode}")

        return (
            gr.update(visible=_topk),
            gr.update(visible=_nb_extract_char),
            gr.update(visible=_search_update_btn),
            gr.update(visible=_language),
            gr.update(visible=_embed_model),
            gr.update(visible=_file_list),
            gr.update(visible=_document),
        )

    def _change_selected_file(self, filename: str):
        self._rag_engine._query_engine_name = filename
        self._rag_engine.set_chat_mode()

    def _undo_chat(self, history: List):
        if len(history) > 0:
            history.pop(-1)
            return history
        return DefaultElement.DEFAULT_HISTORY

    def _reset_chat(self):
        self._rag_engine.reset_conversation()
        gr.Info("Reset chat!")
        return (
            DefaultElement.DEFAULT_MESSAGE,
            DefaultElement.DEFAULT_HISTORY,
            DefaultElement.DEFAULT_DOCUMENT,
            DefaultElement.DEFAULT_STATUS,
        )

    def _clear_chat(self):
        self._rag_engine.clear_conversation()
        gr.Info("Clear chat!")
        return (
            DefaultElement.DEFAULT_MESSAGE,
            DefaultElement.DEFAULT_HISTORY,
            DefaultElement.DEFAULT_STATUS,
        )

    def _show_hide_setting(self, state):
        state = not state
        label = "Hide Setting" if state else "Show Setting"
        return (label, gr.update(visible=state), state)

    def _welcome(self):
        for m in self._llm_response.welcome():
            yield m

    def _update_file_list(self):
        return gr.Dropdown(choices=self._rag_engine._files_registry)

    def _retrieve_rag_cfg(self) -> dict:
        return gr.update(value=yaml.dump(self._rag_engine.config))

    def _set_rag_cfg(self, config: str):
        self._rag_engine.config = yaml.safe_load(config)
        gr.Info(f"Updated full configuration!")

    def _change_src_tgt_languages(self, src_lang: str, tgt_lang: str):
        return gr.update(value=tgt_lang), gr.update(value=src_lang)

    def _preprocess(
        self,
        document: str,
        src_lang: str,
        tgt_lang: str,
        export_file: str = "tmp/output.md",
    ):
        tgt_lang, _ = tuple(tgt_lang.split(" - ", 1))
        src_lang, _ = tuple(src_lang.split(" - ", 1))
        self._rag_engine._parser.read_file(
            filename=document,
            indexing=False,
            translate=True,
            src_language=src_lang,
            tgt_language=tgt_lang,
        )
        try:
            chunk_full_text = self._rag_engine._parser.doc_loader._chunk_full_text
            translated = self._rag_engine._translator.translate(
                chunk_full_text, tgt_lang=tgt_lang, src_lang=src_lang
            )
            with open(export_file, "w+", encoding="utf-8") as f:
                f.write(translated)
        except:
            raise NotImplementedError

        return gr.update(value=export_file, visible=True)

    def build(self):
        with gr.Blocks(theme="ParityError/Interstellar") as demo:
            gr.Markdown("## QA semantic search powered by LLM")
            with gr.Tab("Interface"):
                sidebar_state = gr.State(True)
                with gr.Row(variant=self._variant, equal_height=False):
                    with gr.Column(
                        variant=self._variant, scale=10, visible=sidebar_state.value
                    ) as setting:
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
                                choices=self._rag_engine.get_available_models(),
                                value=self._rag_engine.default_model,
                                interactive=True,
                                allow_custom_value=False,
                            )
                            embed_model = gr.Dropdown(
                                label="Choose Embed Model:",
                                choices=self._rag_engine.get_available_embed_models(),
                                value=self._rag_engine.default_embed_model,
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
                                choices=list(self._rag_engine._files_registry),
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
                            chat_mode = gr.Dropdown(
                                choices=["QA", "chat", "semantic search"],
                                value="QA",
                                min_width=50,
                                show_label=False,
                                interactive=True,
                                allow_custom_value=False,
                            )
                            message = gr.MultimodalTextbox(
                                value=DefaultElement.DEFAULT_MESSAGE,
                                placeholder="Enter you message:",
                                show_label=False,
                                scale=6,
                                lines=1,
                            )
                        with gr.Row():
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
                                value=(
                                    "Hide Setting"
                                    if sidebar_state.value
                                    else "Show Setting"
                                ),
                                min_width=20,
                            )
                            undo_btn = gr.Button(value="Undo", min_width=20)
                            clear_btn = gr.Button(value="Clear", min_width=20)
                            reset_btn = gr.Button(value="Reset", min_width=20)

            with gr.Tab("Setting"):
                with gr.Row(variant=self._variant, equal_height=False):
                    with gr.Column():
                        system_prompt = gr.Textbox(
                            label="System Prompt",
                            value=self._rag_engine.system_prompt,
                            interactive=True,
                            lines=10,
                            max_lines=50,
                        )
                        sys_prompt_btn = gr.Button(value="Set System Prompt")
                    with gr.Column(variant=self._variant):
                        system_config = gr.Code(
                            value=None,
                            language="yaml",
                            interactive=True,
                            show_label=True,
                        )
                        sys_cfg_btn = gr.Button(value="Get current system config")
                        apply_cfg_btn = gr.Button(value="Apply config")

            with gr.Tab("Translator"):
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
                                # visible=False,
                            )
                        with gr.Row():
                            output_doc = gr.File(
                                label="Output File", value=None, visible=False
                            )

            with gr.Tab("Output"):
                with gr.Row(variant=self._variant):
                    log = gr.Code(
                        label="", language="markdown", interactive=False, lines=30
                    )
                    demo.load(
                        self._logger.read_logs,
                        outputs=[log],
                        every=1,
                        show_progress="hidden",
                        scroll_to_output=True,
                    )

            clear_btn.click(self._clear_chat, outputs=[message, chatbot, status])
            undo_btn.click(self._undo_chat, inputs=[chatbot], outputs=[chatbot])
            reset_btn.click(
                self._reset_chat, outputs=[message, chatbot, documents, status]
            )
            message.submit(
                self._get_respone,
                inputs=[chat_mode, message, chatbot],
                outputs=[message, chatbot, status],
            )
            chat_mode.change(
                self._change_chat_mode,
                inputs=[chat_mode, top_k, nb_extract_char],
                outputs=[
                    top_k,
                    nb_extract_char,
                    search_update_btn,
                    language,
                    embed_model,
                    file_list,
                    documents,
                ],
            ).then(self._rag_engine.reset_conversation).then(
                self._clear_chat, outputs=[message, chatbot, status]
            )
            search_update_btn.click(
                self._change_chat_mode, inputs=[chat_mode, top_k, nb_extract_char]
            )
            language.change(self._change_language, inputs=[language]).then(
                self._clear_chat, outputs=[message, chatbot, status]
            )
            doc_language.change(self._change_doc_language, inputs=[doc_language])
            embed_model.change(
                self._change_embed_model, inputs=[embed_model], outputs=[status]
            )
            model.change(self._change_model, inputs=[model], outputs=[status])
            documents.change(
                self._processing_document,
                inputs=[documents],
                outputs=[system_prompt, status],
            ).then(
                self._show_document_btn,
                inputs=[documents],
                outputs=[upload_doc_btn, reset_doc_btn],
            ).then(
                self._update_file_list, outputs=[file_list]
            )

            file_list.change(self._change_selected_file, inputs=[file_list])

            ui_btn.click(
                self._show_hide_setting,
                inputs=[sidebar_state],
                outputs=[ui_btn, setting, sidebar_state],
            )
            upload_doc_btn.upload(
                self._upload_document,
                inputs=[documents, upload_doc_btn],
                outputs=[documents, upload_doc_btn],
            )
            reset_doc_btn.click(
                self._reset_document, outputs=[documents, upload_doc_btn, reset_doc_btn]
            )
            # Setting tab
            sys_prompt_btn.click(self._change_system_prompt, inputs=[system_prompt])
            sys_cfg_btn.click(self._retrieve_rag_cfg, outputs=[system_config])
            apply_cfg_btn.click(self._set_rag_cfg, inputs=[system_config])
            demo.load(self._welcome, outputs=[message, chatbot, status])

            # Translator Tab
            switch_1.click(
                self._change_src_tgt_languages,
                inputs=[src_language, tgt_language],
                outputs=[src_language, tgt_language],
            )
            # upload_doc_btn_2.upload(
            #     self._upload_document,
            #     inputs=[input_doc, upload_doc_btn_2],
            #     outputs=[input_doc, upload_doc_btn_2],
            # )
            translate_button.click(
                self._preprocess,
                inputs=[input_doc, src_language, tgt_language],
                outputs=[output_doc],
            )

        return demo


def main(
    config: str | None = None,
    host: str = "127.0.0.1",
    share: bool = False,
    debug: bool = False,
    show_api: bool = False,
):
    logger = Logger(LOG_FILE)
    logger.reset_logs()
    ui = LocalChatbotUI(logger=logger, host=host, rag_yaml_config=config)

    ui.build().launch(share=share, server_name=host, debug=debug, show_api=show_api)


if __name__ == "__main__":
    fire.Fire(main)
