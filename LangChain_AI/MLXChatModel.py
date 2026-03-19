# 커스텀 래퍼 클래스
"""
서버 없이 mlx_lm을 직접 python에서 호출하고 싶으면,
LangChain의 BaseChatModel을 상속하여 커스텀 래퍼를 만들 수 있다.
이 방법은 서버 오버헤드가 없어 더 빠르지만, 동시 요청 처리가 어렵다.

"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from mlx_lm import load, generate
from typing import List, Optional, Any


class MLXChatModel(BaseChatModel):
    model_path: str = "mlx-community/Qwen3.5-9B-4bit"
    _model: Any = None
    _tokenizer: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 모델을 메모리에 직접 로드
        self._model, self.tokenizer = load(self.model_path)

    @property
    def _llm_type(self) -> str:
        return "mlx-local"

    def _generate(
        self, messages: List[BaseMessage], stop: list[str] | None = None, **kwarg
    ) -> ChatResult:
        prompt = self._tokenizer.apply_chat_template(
            [{"role": m.type, "content": m.content} for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=kwargs.get("max_tokens", 1024),
        )

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
