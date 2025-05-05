from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
    MessagesPlaceholder
)
from typing import List, Dict, Any, Optional, Tuple

class PromptBuilder:
    """提示模板构建器"""

    @staticmethod
    def create_prompt_template(
        system_prompt: str,
    ) -> PromptTemplate:
        """
        创建基础提示模板
        
        Args:
            system_prompt: 系统提示词
        Returns:
            PromptTemplate: 提示模板
        """

        return PromptTemplate.from_template(system_prompt)
    

    @staticmethod
    def create_chat_prompt_template(
        system_prompt: str,
        examples: Optional[List[Tuple[str, str]]] = None,
    ) -> ChatPromptTemplate:
        """
        创建聊天提示模板
        
        Args:
            system_prompt: 系统提示词
            examples: 示例对话列表
        Returns:
            ChatPromptTemplate: 聊天提示模板
        """
        messages = []

        # 根据模型处理系统提示词
        if system_prompt:
            messages.append(("system", system_prompt))

        if examples:
            for example in examples:
                messages.append(("human", example["question"]))
                messages.append(("ai", example["answer"]))

        # 添加问题占位符
        messages.append(("human", "{{question}}"))

        return ChatPromptTemplate.from_messages(messages)
    

    # @staticmethod
    # def create_chat_prompt_template(
    #     system_prompt: str,
    #     model_name: str,
    #     examples: Optional[List[Tuple[str, str]]] = None,
    #     is_memory: bool = False,
    #     sequence: Optional[str] = None
    # ) -> ChatPromptTemplate:
    #     """
    #     创建聊天提示模板
        
    #     Args:
    #         system_prompt: 系统提示词
    #         model_name: 模型名称
    #         examples: 示例对话列表
    #         is_question: 是否为问题模式
    #         is_memory: 是否需要记忆功能
    #         sequence: 序列标识
            
    #     Returns:
    #         ChatPromptTemplate: 聊天提示模板
    #     """
    #     messages = []

    #     # 根据模型处理系统提示词
    #     if system_prompt:
    #         role = "human" if model_name == "o1-mini" else "system"
    #         messages.append((role, system_prompt))

    #     # 使用FewShotChatMessagePromptTemplate处理示例对话
    #     if examples:
    #         example_messages = []
    #         for question, answer in examples:
    #             example_messages.append({"role": "human", "content": question})
    #             example_messages.append({"role": "assistant", "content": answer})
            
    #         few_shot_prompt = FewShotChatMessagePromptTemplate(
    #             example_messages=example_messages,
    #             input_variables=[],
    #             prefix="以下是一些示例对话：",
    #             suffix="现在请基于以上示例回答问题。"
    #         )
    #         messages.append(few_shot_prompt)

    #     # 添加历史记录
    #     if is_memory:
    #         messages.append(MessagesPlaceholder(variable_name="chat_history"))

    #     # 添加问题占位符
    #     question_key = f"question_{sequence}" if sequence else "question"
    #     messages.append(("human", f"{{{question_key}}}"))

    #     return ChatPromptTemplate.from_messages(messages)