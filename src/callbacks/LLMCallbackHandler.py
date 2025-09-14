from typing import Any
from langchain_core.callbacks import BaseCallbackHandler


class LLMCallbackHandler(BaseCallbackHandler):
    """自定义LLM回调处理器"""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """处理新的token输出

        Args:
            token: 新生成的token
            **kwargs: 其他参数
        """
        print(token, end="", flush=True)
