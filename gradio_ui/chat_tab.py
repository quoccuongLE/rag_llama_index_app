import os
import sys
import gradio as gr

from doc_search import DocRetrievalAugmentedGen
from doc_search.logger import Logger
from doc_search.translator import get_available_languages
from .defaults import DefaultElement, LLMResponse


class ChatTab:
    _llm_response: LLMResponse = LLMResponse()
    _variant: str = "panel"
    chat_mode: str = "chat"

    def __init__(
        self,
        rag_engine: DocRetrievalAugmentedGen,
        chat_mode: str | None = None,
        avatar_images: list[str] = ["./assets/user.png", "./assets/bot.png"],
        logfile: str = "logging.log"
    ) -> None:
        if chat_mode:
            self.chat_mode = chat_mode
        self._logger = Logger(logfile)
        self._logger.reset_logs()
        self._avatar_images = [
            os.path.join(os.getcwd(), image) for image in avatar_images
        ]
        self.sidebar_state = gr.State(True)
        self.rag_engine = rag_engine
        self.check_and_update_chat_mode()
        self.create_ui()

    def check_and_update_chat_mode(self, topk: int = -1, nb_extract_char: int = -1):
        if self.chat_mode == self.rag_engine._chat_mode:
            return
        chat_config = dict(type=self.chat_mode)
        if self.chat_mode == "semantic search":
            chat_config.update(
                dict(synthesizer=dict(topk=topk, sample_length=nb_extract_char))
            )

        self.rag_engine.set_chat_mode(chat_mode=self.chat_mode, chat_config=chat_config)

    def _clear_chat(self) -> tuple[str]:
        self.rag_engine.clear_conversation()
        gr.Info("Clear chat!")
        return (
            DefaultElement.DEFAULT_MESSAGE,
            DefaultElement.DEFAULT_HISTORY,
            DefaultElement.DEFAULT_STATUS,
        )

    def _undo_chat(self, history: list[str]) -> str:
        if len(history) > 0:
            history.pop(-1)
            return history
        return DefaultElement.DEFAULT_HISTORY

    def _get_respone(
        self,
        message: dict[str, str],
        chatbot: list[list[str, str]],
        progress=gr.Progress(track_tqdm=False),
    ):
        self.check_and_update_chat_mode()
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

    def _change_model(self, model: str) -> str:
        if model not in [None, ""]:
            self.rag_engine.model = model
            self.rag_engine.reset_engine()
            gr.Info(f"Change model to {model}!")
        return DefaultElement.DEFAULT_STATUS

    def _change_language(self, language: str):
        self.rag_engine.set_chat_mode(language=language)
        gr.Info(f"Change language to {language}")

    def _show_hide_setting(self, state) -> tuple:
        state = not state
        label = "Hide Setting" if state else "Show Setting"
        return (label, gr.update(visible=state), state)

    def _reset_chat(self) -> tuple[str]:
        self._rag_engine.reset_conversation()
        gr.Info("Reset chat!")
        return (
            DefaultElement.DEFAULT_MESSAGE,
            DefaultElement.DEFAULT_HISTORY,
            DefaultElement.DEFAULT_DOCUMENT,
            DefaultElement.DEFAULT_STATUS,
        )

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
                    ui_btn = gr.Button(
                        value="Hide/Show Setting",
                        min_width=20,
                    )
                    undo_btn = gr.Button(value="Undo", min_width=20)
                    clear_btn = gr.Button(value="Clear", min_width=20)
                    reset_btn = gr.Button(value="Reset", min_width=20)

        clear_btn.click(self._clear_chat, outputs=[message, chatbot, status])
        undo_btn.click(self._undo_chat, inputs=[chatbot], outputs=[chatbot])
        reset_btn.click(
            self._reset_chat, outputs=[message, chatbot, status]
        )
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
