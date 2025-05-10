from typing import List, Dict, Any, Optional, Tuple, Union
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
    MessagesPlaceholder
)
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


class PromptBuilder:
    """提示模板构建器，用于创建各类提示模板"""
    def create_multi_image_content(question, image_datas):
        content = [{"type": "text", "text": question}]
        for image_data in image_datas:
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            })
        return content
    

    @staticmethod
    def create_prompt_template(system_prompt: str) -> PromptTemplate:
        """
        创建基础提示模板
        
        Args:
            system_prompt (str): 系统提示词
        
        Returns:
            PromptTemplate: 基础提示模板
        
        Raises:
            ValueError: 当系统提示词为空时抛出
        """
        if not system_prompt:
            raise ValueError("系统提示词不能为空")
            
        return PromptTemplate.from_template(system_prompt)


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
        if not system_prompt:
            raise ValueError("系统提示词不能为空")

        if not examples:
            return PromptBuilder.create_prompt_template(system_prompt)
        
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
        if not system_prompt:
            raise ValueError("系统提示词不能为空")

        if not examples:
            return PromptBuilder.create_prompt_template(system_prompt)
        
        if example_num < 1:
            raise ValueError("至少选择1个示例")

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
    def create_chat_prompt_template(
        system_prompt: str,
        is_agent: bool = False,
        is_memory: bool = False
    ) -> ChatPromptTemplate:
        """
        创建聊天提示模板
        
        Args:
            system_prompt (str): 系统提示词
            is_agent (bool): 是否为Agent模式
            is_memory (bool): 是否存储历史
        
        Returns:
            ChatPromptTemplate: 聊天提示模板
        """
        if not system_prompt:
            raise ValueError("系统提示词不能为空")

        messages = [("system", system_prompt)]

        if is_memory:
            messages.append(MessagesPlaceholder(variable_name="history"))

        messages.append(("human", "{question}"))

        if is_agent:
            messages.append(MessagesPlaceholder(variable_name="agent_scratchpad"))

        return ChatPromptTemplate.from_messages(messages)
    

    @staticmethod
    def create_few_shot_chat_prompt_template(
        system_prompt: str,
        examples: Optional[List[Dict[str, str]]] = None,
        is_agent: bool = False,
        is_memory: bool = False
    ) -> ChatPromptTemplate:
        """
        创建few-shot聊天提示模板
        
        Args:
            system_prompt (str): 系统提示词
            examples (Optional[List[Dict[str, str]]]): 示例对话列表，每个示例包含question和answer
            is_agent (bool): 是否为Agent模式
            is_memory (bool): 是否存储历史
        
        Returns:
            ChatPromptTemplate: 包含系统提示词和few-shot示例的聊天提示模板
        """
        if not system_prompt:
            raise ValueError("系统提示词不能为空")

        if not examples:
            # 如果没有示例，则直接返回普通聊天模板
            return PromptBuilder.create_chat_prompt_template(system_prompt, is_agent, is_memory)

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{question}"),
            ("ai", "{answer}")
        ])

        few_shot_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples
        )

        messages = [
            ("system", system_prompt),
            few_shot_template
        ]

        if is_memory:
            messages.append(MessagesPlaceholder(variable_name="history"))

        messages.append(("human", "{question}"))

        if is_agent:
            messages.append(MessagesPlaceholder(variable_name="agent_scratchpad"))

        return ChatPromptTemplate.from_messages(messages)
    

    @staticmethod
    def create_few_shot_chat_prompt_template_with_selector(
        system_prompt: str,
        examples: Optional[List[Dict[str, str]]] = None,
        example_num: int = 3,
        is_agent: bool = False,
        is_memory: bool = False
    ) -> ChatPromptTemplate:
        """
        创建带有示例选择器的few-shot聊天提示模板
        
        Args:
            system_prompt (str): 系统提示词
            examples (Optional[List[Dict[str, str]]]): 示例对话列表，每个示例包含question和answer
            example_num (int): 要选择的示例数量，默认为3
            is_agent (bool): 是否为Agent模式
            is_memory (bool): 是否存储历史
        
        Returns:
            ChatPromptTemplate: 包含系统提示词和few-shot示例的聊天提示模板
        """
        if not system_prompt:
            raise ValueError("系统提示词不能为空")

        if not examples:
            return PromptBuilder.create_chat_prompt_template(system_prompt, is_agent, is_memory)
        
        if example_num < 1:
            raise ValueError("至少选择1个示例")

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{question}"),
            ("ai", "{answer}")
        ])

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

        messages = [
            ("system", system_prompt),
            few_shot_template,
        ]

        if is_memory:
            messages.append(MessagesPlaceholder(variable_name="history"))

        messages.append(("human", "{question}"))

        if is_agent:
            messages.append(MessagesPlaceholder(variable_name="agent_scratchpad"))

        return ChatPromptTemplate.from_messages(messages)
    

    @staticmethod
    def create_multimodal_chat_prompt_template(
        system_prompt: str,
        is_agent: bool = False,
        is_memory: bool = False
    ) -> ChatPromptTemplate:
        """
        创建多模态提示模板，支持图片和文本
        
        Args:
            system_prompt (str): 系统提示词
            is_agent (bool): 是否为Agent模式，默认为False
            is_memory (bool): 是否包含历史记忆，默认为False
        
        Returns:
            ChatPromptTemplate: 多模态提示模板
        """
        if not system_prompt:
            raise ValueError("系统提示词不能为空")
        
        messages = [("system", system_prompt)]
        
        if is_memory:
            messages.append(MessagesPlaceholder(variable_name="history"))
        
        messages.append(("human", [
            {"type": "text", "text": "{question}"}, 
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_data}"}}]
        ))
        
        if is_agent:
            messages.append(MessagesPlaceholder(variable_name="agent_scratchpad"))
        
        return ChatPromptTemplate.from_messages(messages)
    

    @staticmethod
    def create_few_shot_multimodal_chat_prompt_template(
        system_prompt: str,
        examples: Optional[List[Dict[str, Any]]] = None,
        is_agent: bool = False,
        is_memory: bool = False
    ) -> ChatPromptTemplate:
        """
        创建带有示例的多模态聊天提示模板
        
        Args:
            system_prompt (str): 系统提示词
            examples (Optional[List[Dict[str, Any]]]): 示例列表，每个示例包含question、image_url和answer
            is_agent (bool): 是否为Agent模式，默认为False
            is_memory (bool): 是否包含历史记忆，默认为False
        
        Returns:
            ChatPromptTemplate: 多模态聊天提示模板
        """
        if not system_prompt:
            raise ValueError("系统提示词不能为空")
        
        if not examples:
            return PromptBuilder.create_multimodal_chat_prompt_template(system_prompt, is_agent, is_memory)
        
        messages = [("system", system_prompt)]
        
        for example in examples:
            if "question" not in example or "answer" not in example or "image_data" not in example:
                raise ValueError("示例格式错误，必须包含question、image_data和answer字段")
            
            image_data = example["image_data"]

            human_content = [
                {"type": "text", "text": example["question"]},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
            
            messages.extend([
                ("human", human_content),
                ("ai", example["answer"])
            ])
        
        # 添加历史记忆
        if is_memory:
            messages.append(MessagesPlaceholder(variable_name="history"))
        
        # 添加当前问题和图片
        human_content = [
            {"type": "text", "text": "{question}"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_data}"}}
        ]
        messages.append(("human", human_content))
        
        if is_agent:
            messages.append(MessagesPlaceholder(variable_name="agent_scratchpad"))
        
        return ChatPromptTemplate.from_messages(messages)
    

    @staticmethod
    def create_multimodal_chat_prompt_template_with_more_images(
        system_prompt: str,
        is_agent: bool = False,
        is_memory: bool = False
    ) -> ChatPromptTemplate:
        """
        创建支持多张图片的多模态提示模板
        
        Args:
            system_prompt (str): 系统提示词
            is_agent (bool): 是否为Agent模式，默认为False
            is_memory (bool): 是否包含历史记忆，默认为False
        
        Returns:
            ChatPromptTemplate: 支持多张图片的多模态提示模板
        """
        if not system_prompt:
            raise ValueError("系统提示词不能为空")
        
        messages = [("system", system_prompt)]
        
        if is_memory:
            messages.append(MessagesPlaceholder(variable_name="history"))
        
        # 添加当前问题和图片
        human_content = [
            {"type": "text", "text": "{question}"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_data1}"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_data2}"}}
        ]
        messages.append(("human", human_content))
        
        if is_agent:
            messages.append(MessagesPlaceholder(variable_name="agent_scratchpad"))
        
        return ChatPromptTemplate.from_messages(messages)
    

    @staticmethod
    def create_few_shot_multimodal_chat_prompt_template_with_more_images(
        system_prompt: str,
        examples: List[Dict[str, Any]] = None,
        is_agent: bool = False,
        is_memory: bool = False
    ) -> ChatPromptTemplate:
        """
        创建支持少样本提示功能的多模态提示模板
        
        Args:
            system_prompt (str): 系统提示词
            examples (List[Dict[str, Any]]): 少样本示例列表，每个示例包含问题和回答
            is_agent (bool): 是否为Agent模式，默认为False
            is_memory (bool): 是否包含历史记忆，默认为False
        
        Returns:
            ChatPromptTemplate: 支持多张图片的多模态提示模板
        """
        if not system_prompt:
            raise ValueError("系统提示词不能为空")
        
        if not examples:
            return PromptBuilder.create_multimodal_chat_prompt_template_with_more_images(
                system_prompt, is_agent, is_memory
            )        
        
        messages = [("system", system_prompt)]
        
        for example in examples:
            if "question" not in example or "answer" not in example or "image_data1" not in example or "image_data2" not in example:
                raise ValueError("示例格式错误，必须包含question、image_data1/2和answer字段")
            
            image_data1 = example["image_data1"]
            image_data2 = example["image_data2"]
            
            human_content = [
                {"type": "text", "text": example["question"]},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data1}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data2}"}}
            ]
            messages.extend([
                ("human", human_content),
                ("ai", example["answer"])
            ])
        
        if is_memory:
            messages.append(MessagesPlaceholder(variable_name="history"))
        
        human_content = [
            {"type": "text", "text": "{question}"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_data1}"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_data2}"}}
        ]
        messages.append(("human", human_content))
        
        if is_agent:
            messages.append(MessagesPlaceholder(variable_name="agent_scratchpad"))
        
        return ChatPromptTemplate.from_messages(messages)