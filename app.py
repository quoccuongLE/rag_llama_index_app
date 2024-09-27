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
from gradio_ui import ChatTab, DefaultElement

import fire


def main(config: str):
    rag_engine = DocRetrievalAugmentedGen(setting=config)
    title = "Multilingual QA semantic search powered by LLM"
    DESCRIPTION = """
    #### This is an official demo for Multilingual QA semantic search engine. \n
    """
    with gr.Blocks(
        theme="ParityError/Interstellar", analytics_enabled=False, title=title
    ) as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Tabs("Interface"):
            sidebar_state = gr.State(True)
            with gr.TabItem("Chat"):
                ChatTab(rag_engine=rag_engine)
            # with gr.TabItem("QA"):
            #     QATab()
            # with gr.TabItem("Summarization"):
            #     SummarizationTab()
            # with gr.TabItem("Setting"):
            #     pass
            # with gr.TabItem("Translation"):
            #     pass
            # with gr.TabItem("Semantic search"):
            #     pass
    demo.queue().launch(share=False)


if __name__ == "__main__":
    fire.Fire(main)
