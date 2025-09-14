#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自定义硅基流动(SiliconFlow)嵌入模型类
"""

import requests
from typing import List
from langchain.embeddings.base import Embeddings


class SiliconFlowEmbeddings(Embeddings):

    def __init__(self, api_key: str, base_url: str = "https://api.siliconflow.cn/v1",
                 model: str = "Qwen/Qwen3-Embedding-4B"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        嵌入文档列表
        """
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询
        """
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "input": text
            }
        )

        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
