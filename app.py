import gradio as gr

from doc_search import DocRetrievalAugmentedGen
from gradio_ui import ChatTab, QATab, SettingTab, TranslatorTab

import fire


def main(
    config: str,
    host: str = "127.0.0.1",
    share: bool = False,
    debug: bool = False,
    show_api: bool = False,
):
    rag_engine = DocRetrievalAugmentedGen(setting=config)
    title = "Multilingual QA semantic search powered by LLM"
    DESCRIPTION = """
    # This is an official demo for Multilingual QA semantic search engine. \n
    """
    with gr.Blocks(
        theme="ParityError/Interstellar", analytics_enabled=False, title=title
    ) as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Tabs("Interface"):
            with gr.TabItem("Chat"):
                ChatTab(rag_engine=rag_engine)
            with gr.TabItem("QA"):
                QATab(rag_engine=rag_engine)
            with gr.TabItem("Summarization"):
                ChatTab(rag_engine=rag_engine, chat_mode="summarization")
            with gr.TabItem("Semantic search"):
                QATab(rag_engine=rag_engine, chat_mode="semantic search")
            with gr.TabItem("Cover letter generation"):
                ChatTab(rag_engine=rag_engine, chat_mode="cover letter gen")
            with gr.TabItem("Setting"):
                SettingTab(rag_engine=rag_engine)
            with gr.TabItem("Translation"):
                TranslatorTab(rag_engine=rag_engine)
    demo.queue().launch(share=share, server_name=host, debug=debug, show_api=show_api)


if __name__ == "__main__":
    fire.Fire(main)
