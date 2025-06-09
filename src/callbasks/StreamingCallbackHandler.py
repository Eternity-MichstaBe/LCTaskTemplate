from langchain_core.callbacks import BaseCallbackHandler
from typing import Any


class StreamingCallbackHandler(BaseCallbackHandler):
    """处理流式输出的回调处理器"""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """处理新的token输出

        Args:
            token: 新生成的token
            **kwargs: 其他参数
        """
        print(token, end="", flush=True)
