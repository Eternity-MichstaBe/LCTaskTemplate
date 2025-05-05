from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any, Optional, Tuple

class PromptBuilder:
    """提示模板构建器"""

    @staticmethod
    def create_basic_template(
        system_prompt: str,
        model_name: str,
        examples: Optional[List[Tuple[str, str]]] = None,
        is_memory: bool = False,
        sequence: Optional[str] = None
    ) -> ChatPromptTemplate:
        """
        创建基础提示模板
        
        Args:
            system_prompt: 系统提示词
            model_name: 模型名称
            examples: 示例对话列表
            is_question: 是否为问题模式
            is_memory: 是否需要记忆功能
            sequence: 序列标识
            
        Returns:
            ChatPromptTemplate: 聊天提示模板
        """
        messages = []

        # 根据模型处理系统提示词
        if system_prompt:
            role = "human" if model_name == "o1-mini" else "system"
            messages.append((role, system_prompt))

        # 添加示例对话
        if examples:
            for question, answer in examples:
                messages.extend([
                    ("human", question),
                    ("ai", answer)
                ])

        # 添加历史记录
        if is_memory:
            messages.append(("human", "chat_history:\n{chat_history}"))

        # 添加问题占位符
        question_key = f"question_{sequence}" if sequence else "question"
        messages.append(("human", f"{{{question_key}}}"))

        return ChatPromptTemplate.from_messages(messages)