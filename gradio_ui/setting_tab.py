import gradio as gr
import yaml

from doc_search import DocRetrievalAugmentedGen


class SettingTab:
    _variant: str = "panel"

    def __init__(
        self,
        rag_engine: DocRetrievalAugmentedGen,
    ) -> None:
        self.rag_engine = rag_engine
        self.create_ui()

    def _retrieve_rag_cfg(self) -> dict:
        return gr.update(value=yaml.dump(self.rag_engine.config))

    def _set_rag_cfg(self, config: str):
        self.rag_engine.config = yaml.safe_load(config)
        gr.Info(f"Updated full configuration!")

    def _change_system_prompt(self, sys_prompt: str):
        self.rag_engine.system_prompt = sys_prompt
        self.rag_engine.set_chat_mode()
        gr.Info("System prompt updated!")

    def create_ui(self):
        with gr.Row(variant=self._variant, equal_height=False):
            with gr.Column():
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value=self.rag_engine.system_prompt,
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

        sys_prompt_btn.click(self._change_system_prompt, inputs=[system_prompt])
        sys_cfg_btn.click(self._retrieve_rag_cfg, outputs=[system_config])
        apply_cfg_btn.click(self._set_rag_cfg, inputs=[system_config])
