#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自定义硅基流动(SiliconFlow)聊天模型类
"""

import requests
import json
from typing import List, Optional, Dict, Any, Iterator
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI


class ChatSiliconFlow(SimpleChatModel):

    api_key: str
    model: str
    base_url: str = "https://api.siliconflow.cn/v1"
    temperature: float = 0.3
    timeout: int = 60
    streaming: bool = False

    @property
    def _llm_type(self) -> str:
        return "SiliconFlow"

    @staticmethod
    def _msg_to_dict(m: BaseMessage) -> Dict[str, Any]:
        """
        将 LangChain 的消息映射到 OpenAI/SiliconFlow role
        """
        t = m.type
        if t == "human":
            role = "user"
        elif t == "ai":
            role = "assistant"
        elif t == "system":
            role = "system"
        elif t == "tool":
            role = "tool"
        else:
            role = "user"
        return {"role": role, "content": m.content}

    def _call(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> str:
        """
        非流式调用
        """
        payload = {
            "model": kwargs.get("model", self.model),
            "temperature": kwargs.get("temperature", self.temperature),
            "messages": [self._msg_to_dict(m) for m in messages],
            "stream": False,
        }

        if stop:
            payload["stop"] = stop

        # 透传工具调用/输出格式等参数
        for key in (
                "tools", "tool_choice", "response_format",
                "max_tokens", "top_p", "n", "presence_penalty", "frequency_penalty"
        ):
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        r = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        """
        流式调用
        """
        payload = {
            "model": kwargs.get("model", self.model),
            "temperature": kwargs.get("temperature", self.temperature),
            "messages": [self._msg_to_dict(m) for m in messages],
            "stream": True,
        }

        if stop:
            payload["stop"] = stop

        for key in (
                "tools", "tool_choice", "response_format",
                "max_tokens", "top_p", "n", "presence_penalty", "frequency_penalty"
        ):
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
                timeout=self.timeout,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                if not line.startswith(b"data: "):
                    continue
                data_str = line[6:].decode("utf-8").strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                    delta = obj["choices"][0]["delta"]
                    # content token
                    if "content" in delta and delta["content"]:
                        yield ChatGenerationChunk(message=AIMessageChunk(content=delta["content"]))
                    # 如需支持工具调用流式，这里可解析 delta.get("tool_calls")
                except Exception:
                    continue
