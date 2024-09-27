import time
from dataclasses import dataclass
from typing import ClassVar

from llama_index.core.chat_engine.types import StreamingAgentChatResponse

LOG_FILE = "logging.log"
DATA_DIR = "data/data"
AVATAR_IMAGES = ["./assets/user.png", "./assets/bot.png"]


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
        history: list[list[str]],
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
    PROCESS_DOCUMENT_SUCCESS_STATUS: str = "Processing documents 📄 completed!"
    PROCESS_DOCUMENT_EMPTY_STATUS: str = "Empty documents!"
    ANSWERING_STATUS: str = "Answering!"
    COMPLETED_STATUS: str = "Completed!"
