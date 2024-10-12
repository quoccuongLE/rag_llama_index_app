from enum import Enum
from typing import Any, Generator, List, Optional, Sequence

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.response.schema import (RESPONSE_TYPE, Response,
                                                   StreamingResponse)
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.chat_engine import ContextChatEngine, SimpleChatEngine
from llama_index.core.chat_engine.types import (StreamingAgentChatResponse,
                                                ToolOutput)
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import Refine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import (MetadataMode, NodeWithScore, QueryBundle,
                                     QueryType)
from llama_index.core.settings import Settings
from llama_index.core.types import Thread
from tqdm import tqdm

from doc_search.prompt.qa_prompt import qa_template, summarization_template
from doc_search.query_engine import factory
from doc_search.settings import EngineConfig, SimpleChatEngineConfig
from doc_search.translator import Language, TranslationService, Translator

# import gcld3 # TODO: Add language classification/detector


class ChatMode(str, Enum):
    CHAT = "chat"
    QA = "QA"
    SEMANTIC_SEARCH = "semantic search"
    SUMMARIZATION = "summarization"
    COVERLETTER_GEN = "cover letter gen"


def empty_response_generator() -> Generator[str, None, None]:
    yield "Empty Response"


class TranslatorContextChatEngine(ContextChatEngine):
    """The `TranslatorContextChatEngine` class is an extension of the
    `ContextChatEngine` class that adds translation capabilities to the chat
    engine. It allows the chat engine to translate the user's message, the
    context, and the prefix messages to a target language if specified. The
    class also provides properties to set the source and target languages, and
    a method to generate the context for the chat engine by retrieving relevant
    nodes from the retriever and optionally translating the node content."""

    _translator: Translator = TranslationService.translator
    _src_language: Language | None = None
    _tgt_language: Language = Language("eng")
    _show_process: bool = True
    _translate_node: bool = True
    _postfix_message: str = " Please answer in "

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLM,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        node_postprocessors: List[BaseNodePostprocessor] | None = None,
        context_template: str | None = None,
        callback_manager: CallbackManager | None = None,
    ) -> None:
        super().__init__(
            retriever,
            llm,
            memory,
            prefix_messages,
            node_postprocessors,
            context_template,
            callback_manager,
        )
        self._translator = TranslationService.translator
        # self._transtate_prefix_messages()

    def _transtate_prefix_messages(self):
        if self._tgt_language.language_code == "eng":
            return

        self._context_template = self._translator.translate(
            sources=self._context_template,
            src_lang="eng",
            tgt_lang=self._tgt_language.language_code,
        )

        for message in self._prefix_messages:
            if len(message.content) == 0:
                continue
            message.content = self._translator.translate(
                sources=message.content,
                src_lang="eng",
                tgt_lang=self._tgt_language.language_code,
            )

    @property
    def src_language(self) -> Language:
        return self._src_language

    @src_language.setter
    def src_language(self, language: str | Language):
        if isinstance(language, str):
            language = Language(language)
        self._src_language = language

    @property
    def tgt_language(self) -> Language:
        return self._tgt_language

    @tgt_language.setter
    def tgt_language(self, language: str | Language):
        if isinstance(language, str):
            language = Language(language)
        self._tgt_language = language
        # self._transtate_prefix_messages()

    def _get_prefix_messages_with_context(self, context_str: str) -> List[ChatMessage]:
        """Get the prefix messages with context."""
        # ensure we grab the user-configured system prompt
        system_prompt = ""
        prefix_messages = self._prefix_messages
        if (
            len(self._prefix_messages) != 0
            and self._prefix_messages[0].role == MessageRole.SYSTEM
        ):
            system_prompt = str(self._prefix_messages[0].content)
            prefix_messages = self._prefix_messages[1:]

        context_str_w_sys_prompt = system_prompt.strip() + "\n" + context_str
        return [
            ChatMessage(
                content=context_str_w_sys_prompt, role=self._llm.metadata.system_role
            ),
            *prefix_messages,
        ]

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        """
        Streams a chat response using the provided message, chat history, and
        the generated context.
        
        Args:
            message (str): The user's message to be processed.
            chat_history (Optional[List[ChatMessage]]): The chat history to be
            used in the context generation.
        
        Returns:
            StreamingAgentChatResponse: The streaming chat response, including
            the chat stream, sources, and source nodes.
        """    
        if (
            self._tgt_language
            and self._translate_node
            and self._tgt_language.language_code != self._src_language.language_code
        ):
            message += self._translator.translate(
                sources=self._postfix_message + f"{self._tgt_language.english_name}",
                src_lang="eng",
                tgt_lang=self._tgt_language.language_code,
            )
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        context_str_template, nodes = self._generate_context(message)
        prefix_messages = self._get_prefix_messages_with_context(context_str_template)
        initial_token_count = len(
            self._memory.tokenizer_fn(
                " ".join([(m.content or "") for m in prefix_messages])
            )
        )
        all_messages = prefix_messages + self._memory.get(
            initial_token_count=initial_token_count
        )

        chat_response = StreamingAgentChatResponse(
            chat_stream=self._llm.stream_chat(all_messages),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(prefix_messages[0]),
                    raw_input={"message": message},
                    raw_output=prefix_messages[0],
                )
            ],
            source_nodes=nodes,
        )
        thread = Thread(
            target=chat_response.write_response_to_history, args=(self._memory,)
        )
        thread.start()

        return chat_response

    def _generate_context(self, message: str) -> str | list[NodeWithScore]:
        """Generates the context for the chat engine by retrieving relevant
        nodes from the retriever, and optionally translating the node content if
        a target language is specified and the source and target languages differ.
        
        Args:
            message (str): The user's message to be processed.
        
        Returns:
            str | list[NodeWithScore]: The generated context string and the list
                of retrieved nodes.
        """
        # gcld3
        # TODO: Missing language detector
        nodes = self._retriever.retrieve(message)
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(message)
            )

        if (
            self._tgt_language
            and self._translate_node
            and self._tgt_language.language_code != self._src_language.language_code
        ):
            draft = []
            if self._show_process:
                node_iterator = tqdm(nodes, desc="Translating text ...")
            else:
                node_iterator
            for n in node_iterator:
                text = self._translator.translate(
                    sources=n.node.get_content(metadata_mode=MetadataMode.LLM).strip(),
                    src_lang=self._src_language,
                    tgt_lang=self._tgt_language,
                )
                draft.append(text)
            context_str = "\n\n".join(draft)
        else:
            context_str = "\n\n".join(
                [
                    n.node.get_content(metadata_mode=MetadataMode.LLM).strip()
                    for n in nodes
                ]
            )

        return self._context_template.format(context_str=context_str), nodes


class SummarizationChatEngine(TranslatorContextChatEngine):
    _expert_domain_str: str = "Science"

    def _generate_context(self, message: str) -> str | list[NodeWithScore]:
        """
        Generates the context for the chat engine by retrieving the provided
        message and optionally translating it if a target language is specified
        and the source and target languages differ.
        
        Args:
            message (str): The user's message to be processed.
        
        Returns:
            str | list[NodeWithScore]: The generated context string and an empty
                list of retrieved nodes.
        """
        if (
            self._tgt_language
            and self._translate_node
            and self._tgt_language.language_code != self._src_language.language_code
        ):
            text = self._translator.translate(
                sources=message,
                src_lang=self._src_language,
                tgt_lang=self._tgt_language,
            )
        else:
            text = message
        return self._context_template.format(context_str=text, expert_domain_str=self._expert_domain_str), []


class RawBaseSynthesizer(Refine):
    def __init__(self, topk: int = 5, sample_length: int = 300, **kwargs) -> None:
        self._topk = topk
        self._sample_length = sample_length
        super().__init__(**kwargs)

    def synthesize(
        self,
        query: QueryType,
        nodes: list[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        """
        Synthesizes a response based on the provided query and list of nodes.
        
        Args:
            query (QueryType): The query to be processed.
            nodes (list[NodeWithScore]): The list of nodes to be used in the
                response synthesis.
            additional_source_nodes (Optional[Sequence[NodeWithScore]]):
                Additional source nodes to be used in the response synthesis.
            **response_kwargs (Any): Additional keyword arguments to be passed
                to the response synthesis.
        
        Returns:
            RESPONSE_TYPE: The synthesized response.
        """
                
        if len(nodes) == 0:
            if self._streaming:
                empty_response = StreamingResponse(
                    response_gen=empty_response_generator()
                )
                return empty_response
            else:
                empty_response = Response("Empty Response")
                return empty_response

        if isinstance(query, str):
            query = QueryBundle(query_str=query)

        nodes.sort(key=lambda x: x.get_score(), reverse=True)
        response = self._prepare_response_output(nodes)
        return response

    def _prepare_response_output(self, source_nodes: list[NodeWithScore]):
        """
        Prepares the response output by formatting the source nodes and their
            metadata.
        
        Args:
            source_nodes (list[NodeWithScore]): The list of source nodes to be
                used in the response.
        
        Returns:
            Response: The formatted response containing the text, source nodes,
                and metadata.
        """
        response_metadata = self._get_metadata_for_response(
            [node_with_score.node for node_with_score in source_nodes]
        )
        text = ""
        for i, node in enumerate(source_nodes[: min(self._topk, len(source_nodes))]):
            if "original_text" in node.metadata.keys():
                extract = node.metadata["original_text"][: self._sample_length]
            else:
                extract = node.text[: self._sample_length]
            page_number = node.metadata.get("page_label", "n/a")
            text += "".join(
                [
                    f"({i}) - Page {page_number} - Score={node.score:.3f}\n\nText:\t",
                    extract,
                    "\n\n=================================================\n\n",
                ]
            )
        return Response(
            text,
            source_nodes=source_nodes,
            metadata=response_metadata,
        )


@factory.register_builder("semantic search")
def build_semantic_search_engine(
    index: VectorStoreIndex,
    config: EngineConfig,
    postprocessors: Optional[list] = None,
    **kwargs,
) -> RetrieverQueryEngine:
    return RetrieverQueryEngine.from_args(
        retriever=index.as_retriever(similarity_top_k=config.similarity_top_k),
        llm=Settings.llm,
        response_synthesizer=RawBaseSynthesizer(
            topk=config.synthesizer.topk, sample_length=config.synthesizer.sample_length
        ),
        node_postprocessors=postprocessors or [],
    )


@factory.register_builder("QA")
def build_qa_query_engine(
    index: VectorStoreIndex,
    storage_context: StorageContext,
    config: EngineConfig,
    postprocessors: Optional[list] = None,
    **kwargs,
) -> ContextChatEngine:
    if config.hierarchical:
        retriever = AutoMergingRetriever(
            index.as_retriever(similarity_top_k=config.similarity_top_k),
            storage_context=storage_context,
        )
    else:
        retriever = index.as_retriever(similarity_top_k=config.similarity_top_k)

    return TranslatorContextChatEngine(
        prefix_messages=[
            ChatMessage.from_str(content=config.prefix_messages, role="system")
        ],
        retriever=retriever,
        llm=Settings.llm,
        context_template=config.context_template or qa_template,
        memory=ChatMemoryBuffer(token_limit=config.chat_token_limit),
        node_postprocessors=postprocessors or [],
    )


@factory.register_builder("summarization")
def build_summarization_engine(
    config: SimpleChatEngineConfig, postprocessors: Optional[list] = None, **kwargs
) -> SimpleChatEngine:

    return SummarizationChatEngine(
        prefix_messages=[
            ChatMessage.from_str(content=config.prefix_messages, role="system")
        ],
        retriever=None,
        llm=Settings.llm,
        context_template=config.context_template or summarization_template,
        memory=ChatMemoryBuffer(token_limit=config.chat_token_limit),
        node_postprocessors=postprocessors or [],
    )


@factory.register_builder("chat")
def build_chat_query_engine(
    config: SimpleChatEngineConfig, **kwargs
) -> SimpleChatEngine:
    return SimpleChatEngine.from_defaults(
        llm=Settings.llm,
        memory=ChatMemoryBuffer(token_limit=config.chat_token_limit),
    )
