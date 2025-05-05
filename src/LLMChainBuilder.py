import os
import sys
from functools import reduce
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableMap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.LLMConfig import LLMConfig
from src.PromptBuilder import PromptBuilder


class LLMChainBuilder:
    """链式处理构建器"""

    @staticmethod
    def _init_chat_model(config: LLMConfig) -> BaseChatModel:
        """初始化聊天模型"""
        try:
            if not config.local:
                return ChatOpenAI(
                    temperature=config.temperature,
                    model_name=config.model_name,
                    openai_api_key=config.api_key,
                    openai_api_base=config.api_base
                )
            return ChatOllama(
                model=config.model_name,
                temperature=config.temperature
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize chat model: {str(e)}")

    def create_chain(self, config: LLMConfig, sequence: str = None) -> RunnableParallel:
        """基于configs创建基础处理链"""
        chat_model = self._init_chat_model(config)
        prompt = PromptBuilder.create_basic_template(
            system_prompt=config.system_prompt,
            model_name=config.model_name,
            examples=config.examples,
            is_memory=config.is_memory,
            sequence=sequence
        )

        if config.format is not None:
            return prompt | chat_model | config.format
        
        return prompt | chat_model
