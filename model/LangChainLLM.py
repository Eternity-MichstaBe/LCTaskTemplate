import os
import sys
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_community.cache import SQLiteCache
from typing import List, Dict, Any
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableMap
from functools import reduce

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.LLMConfig import LLMConfig, get_llm_configs, deepseek_api_key, deepseek_api_base
from src.LLMChainBuilder import LLMChainBuilder


class BaseLLM:
    """
    提供基础LLM调用，提供记忆功能

    Attributes:
        config: LLM配置对象
        chain_builder: 链式处理构建器
        base_chain: 基础处理链
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.chain_builder = LLMChainBuilder()
        self.base_chain = self.chain_builder.create_chain(config)
        self.history = []

    def query(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        发送查询到LLM
        Args:
            inputs: 用户输入
        Returns:
            str: LLM的回答
        """
        try:
            inputs = self._preprocess_inputs(inputs)
            result = self.base_chain.invoke(inputs)

            q = inputs["question"]
            # 记录本次对话
            message = f"input: {q}\noutput: {result}"
            self.history.append(message)
            return result
        except Exception as e:
            raise RuntimeError(f"Query failed: {str(e)}")

    def _preprocess_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        预处理用户输入
        """
        if self.history:
            inputs["chat_history"] = "\n".join(self.history)
        else:
            inputs["chat_history"] = "无历史对话"
        return inputs


llmconfig = get_llm_configs(
    model_name="gpt-4o",
    temperature=0.2,
    system_prompt="你是一个AI助手，请根据用户的问题给出回答。",
    output_parser=StrOutputParser()
)

llm = BaseLLM(llmconfig)

result = llm.query({"question": "你好"})
print(result)


