from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from typing import Any, Dict


class StreamingCallbackHandler(BaseCallbackHandler):
    """处理流式输出的回调处理器"""
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """处理新的token输出
        
        Args:
            token: 新生成的token
            **kwargs: 其他参数
        """
        print(token, end="", flush=True)


def create_streaming_chain(animal: str) -> str:
    """创建并执行流式输出链
    
    Args:
        animal: 动物名称
        
    Returns:
        str: 生成的响应
    """
    # 创建回调处理器
    callbacks = [StreamingCallbackHandler()]
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个幽默的AI助手"),
        ("human", "Tell me a joke about {animal}")
    ])
    
    # 创建模型和链
    model = ChatOpenAI(streaming=True)
    chain = prompt | model
    
    # 添加回调并执行
    chain_with_callbacks = chain.with_config(callbacks=callbacks)
    return chain_with_callbacks.invoke({"animal": animal})


if __name__ == "__main__":
    response = create_streaming_chain("bears")