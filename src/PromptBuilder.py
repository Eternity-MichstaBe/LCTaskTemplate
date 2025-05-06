import os
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
)
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


class PromptBuilder:
    """提示模板构建器，用于创建各类提示模板"""

    @staticmethod
    def create_prompt_template(system_prompt: str) -> PromptTemplate:
        """
        创建基础提示模板
        
        Args:
            system_prompt (str): 系统提示词
        Returns:
            PromptTemplate: 基础提示模板
        """
        if not system_prompt:
            raise ValueError("系统提示词不能为空")
            
        return PromptTemplate.from_template(system_prompt)

    @staticmethod
    def create_chat_prompt_template(
        system_prompt: str,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> ChatPromptTemplate:
        """
        创建聊天提示模板
        
        Args:
            system_prompt (str): 系统提示词
            examples (Optional[List[Dict[str, str]]]): 示例对话列表，每个示例包含question和answer
        Returns:
            ChatPromptTemplate: 聊天提示模板
        """
        if not system_prompt:
            raise ValueError("系统提示词不能为空")

        messages = [("system", system_prompt)]

        if examples:
            for example in examples:
                if "question" not in example or "answer" not in example:
                    raise ValueError("示例对话格式错误，必须包含question和answer字段")
                messages.extend([
                    ("human", example["question"]),
                    ("ai", example["answer"])
                ])

        messages.append(("human", "{question}"))
        return ChatPromptTemplate.from_messages(messages)

    @staticmethod
    def _validate_inputs(system_prompt: str, examples: Optional[List[Dict[str, str]]] = None, example_num: int = 1):
        """
        验证输入参数
        
        Args:
            system_prompt (str): 系统提示词
            examples (Optional[List[Dict[str, str]]]): 示例列表
            example_num (int): 示例数量
        """
        if not system_prompt:
            raise ValueError("系统提示词不能为空")
        if not examples:
            raise ValueError("示例列表不能为空")
        if example_num and example_num < 1:
            raise ValueError("示例数量必须大于0")
        
        # 检查示例格式
        for example in examples:
            if "question" not in example or "answer" not in example:
                raise ValueError("示例对话格式错误，必须包含question和answer字段")

    @staticmethod
    def create_few_shot_prompt_template(
        system_prompt: str,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> FewShotPromptTemplate:
        """
        创建少样本提示模板
        
        Args:
            system_prompt (str): 系统提示词
            examples (Optional[List[Dict[str, str]]]): 示例列表，每个示例包含question和answer
        Returns:
            FewShotPromptTemplate: 少样本提示模板
        """
        PromptBuilder._validate_inputs(system_prompt, examples)

        example_prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template="question: {question}\nanswer: {answer}"
        )
        
        return FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            suffix=f"分析以上问答示例，请回答以下问题\n{system_prompt}"
        )

    @staticmethod
    def create_few_shot_prompt_template_with_selector(
        system_prompt: str,
        examples: Optional[List[Dict[str, str]]] = None,
        example_num: int = 1
    ) -> FewShotPromptTemplate:
        """
        创建带有语义相似度选择器的少样本提示模板
        
        Args:
            system_prompt (str): 系统提示词
            examples (Optional[List[Dict[str, str]]]): 示例列表，每个示例包含question和answer
            example_num (int): 选择的示例数量
        Returns:
            FewShotPromptTemplate: 带有选择器的少样本提示模板
        """
        PromptBuilder._validate_inputs(system_prompt, examples, example_num)

        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples=examples,
            embeddings=OpenAIEmbeddings(),
            vectorstore_cls=Chroma,
            k=example_num
        )

        example_prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template="question: {question}\nanswer: {answer}"
        )

        return FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            suffix=f"分析以上问答示例，请回答以下问题\n{system_prompt}"
        )
    
    @staticmethod
    def create_few_shot_chat_prompt_template(
        system_prompt: str,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> ChatPromptTemplate:
        """
        创建few-shot聊天提示模板
        
        Args:
            system_prompt (str): 系统提示词
            examples (Optional[List[Dict[str, str]]]): 示例对话列表，每个示例包含question和answer
        Returns:
            ChatPromptTemplate: 包含系统提示词和few-shot示例的聊天提示模板
        """
        PromptBuilder._validate_inputs(system_prompt, examples)

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{question}"),
            ("ai", "{answer}")
        ])

        few_shot_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples
        )

        # 创建完整的聊天提示模板，包含系统提示词和few-shot示例
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            few_shot_template,
            ("human", "{question}")
        ])

        return chat_prompt
    
    @staticmethod
    def create_few_shot_chat_prompt_template_with_selector(
        system_prompt: str,
        examples: Optional[List[Dict[str, str]]] = None,
        example_num: int = 3
    ) -> ChatPromptTemplate:
        """
        创建带有示例选择器的few-shot聊天提示模板
        
        Args:
            system_prompt (str): 系统提示词
            examples (Optional[List[Dict[str, str]]]): 示例对话列表，每个示例包含question和answer
            example_num (int): 要选择的示例数量，默认为3
        Returns:
            ChatPromptTemplate: 包含系统提示词和few-shot示例的聊天提示模板
        """
        PromptBuilder._validate_inputs(system_prompt, examples, example_num)

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{question}"),
            ("ai", "{answer}")
        ])

        try:
            example_selector = SemanticSimilarityExampleSelector(
                vectorstore=Chroma.from_texts(
                    [ex["question"] for ex in examples],
                    embedding=OpenAIEmbeddings(),
                    metadatas=examples
                ),
                k=example_num
            )

            few_shot_template = FewShotChatMessagePromptTemplate(
                example_prompt=example_prompt,
                example_selector=example_selector
            )

            # 创建完整的聊天提示模板，包含系统提示词和few-shot示例
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                few_shot_template,
                ("human", "{question}")
            ])

            return chat_prompt
        except ImportError:
            raise ImportError("请安装必要的依赖: pip install langchain-openai chromadb")