import os
import sys
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableParallel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.llm.LLMConfig import LLMConfig
from src.llm.PromptBuilder import PromptBuilder


class LLMChainBuilder:
    """链式处理构建器"""

    @staticmethod
    def init_chat_model(config: LLMConfig) -> BaseChatModel:
        """初始化聊天模型"""
        try:
            if not config.local:
                return ChatOpenAI(
                    temperature=config.temperature,
                    model_name=config.model_name,
                    api_key=config.api_key,
                    base_url=config.base_url,
                    streaming=config.streaming
                )
            return ChatOllama(
                model=config.model_name,
                temperature=config.temperature,
                streaming=config.streaming
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize chat model: {str(e)}")

    def _create_chain(self, config: LLMConfig, prompt) -> RunnableParallel:
        """创建通用处理链"""
        chat_model = self.init_chat_model(config)
        output_parser = config.output_parser or StrOutputParser()
        return prompt | chat_model | output_parser

    def create_prompt_chain(self, config: LLMConfig) -> RunnableParallel:
        """创建基础提示处理链"""
        prompt = PromptBuilder.create_prompt_template(
            system_prompt=config.system_prompt
        )
        return self._create_chain(config, prompt)

    def create_few_shot_prompt_chain(self, config: LLMConfig) -> RunnableParallel:
        """创建少样本提示处理链"""
        prompt = PromptBuilder.create_few_shot_prompt_template(
            system_prompt=config.system_prompt,
            examples=config.examples
        )
        return self._create_chain(config, prompt)

    def create_few_shot_prompt_chain_with_selector(self, config: LLMConfig, example_num: int) -> RunnableParallel:
        """创建带选择器的少样本提示处理链"""
        prompt = PromptBuilder.create_few_shot_prompt_template_with_selector(
            system_prompt=config.system_prompt,
            examples=config.examples,
            example_num=example_num
        )
        return self._create_chain(config, prompt)

    def create_chat_chain(self, config: LLMConfig) -> RunnableParallel:
        """创建聊天提示处理链"""
        prompt = PromptBuilder.create_chat_prompt_template(
            system_prompt=config.system_prompt,
            is_agent=config.agent,
            is_memory=config.memory
        )
        return self._create_chain(config, prompt)

    def create_few_shot_chat_chain(self, config: LLMConfig) -> RunnableParallel:
        """创建少样本聊天处理链"""
        prompt = PromptBuilder.create_few_shot_chat_prompt_template(
            system_prompt=config.system_prompt,
            examples=config.examples,
            is_agent=config.agent,
            is_memory=config.memory
        )
        return self._create_chain(config, prompt)

    def create_few_shot_chat_chain_with_selector(self, config: LLMConfig, example_num: int) -> RunnableParallel:
        """创建带选择器的少样本聊天处理链"""
        prompt = PromptBuilder.create_few_shot_chat_prompt_template_with_selector(
            system_prompt=config.system_prompt,
            examples=config.examples,
            example_num=example_num,
            is_agent=config.agent,
            is_memory=config.memory
        )
        return self._create_chain(config, prompt)
    
    def create_multimodal_chat_chain(self, config: LLMConfig) -> RunnableParallel:
        """创建多模态聊天处理链"""
        prompt = PromptBuilder.create_multimodal_chat_prompt_template(
            system_prompt=config.system_prompt,
            is_agent=config.agent,
            is_memory=config.memory
        )
        return self._create_chain(config, prompt)
    
    def create_few_shot_multimodal_chat_chain(self, config: LLMConfig) -> RunnableParallel:
        """创建少样本多模态聊天处理链"""
        prompt = PromptBuilder.create_few_shot_multimodal_chat_prompt_template(
            system_prompt=config.system_prompt,
            examples=config.examples,
            is_agent=config.agent,
            is_memory=config.memory
        )
        return self._create_chain(config, prompt)
