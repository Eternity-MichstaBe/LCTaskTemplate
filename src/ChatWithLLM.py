"""
聊天LLM、支持历史跟踪、记忆缓存
"""

import os
import sys
import asyncio
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_community.chat_message_histories import RedisChatMessageHistory

# 添加项目根目录到路径
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

# 导入自定义模块
from src.llm.LLMConfig import get_llm_configs, cfg
from src.llm.LLMChainBuilder import LLMChainBuilder
from src.utils.utils import custom_trim_messages


def get_llm_history(user_id: str, session_id: str) -> RedisChatMessageHistory:
    """获取或创建llm会话历史"""
    history = RedisChatMessageHistory(session_id=user_id + "_" + session_id, url=REDIS_URL)
    messages = history.messages
    save_messages = custom_trim_messages(messages)

    history.clear()
    for message in save_messages:
        history.add_message(message)

    return history


# ===== LLM 配置 =====
def setup_llm():
    """设置并返回LLM"""

    # 配置LLM - 普通聊天模式
    llmConfig = get_llm_configs(
        model_name="deepseek-ai/DeepSeek-V3.1",
        api_key=os.getenv("SF_API_KEY", ""),
        base_url=cfg.get("siliconflow", "base_url", fallback=""),
        streaming=True,
        system_prompt="你是一个AI助手，请根据用户的问题给出回答。",
        memory=True  # 启用记忆功能
    )

    # 创建聊天链
    llm = LLMChainBuilder().create_chat_chain(llmConfig)

    chat_with_history = RunnableWithMessageHistory(
        llm,
        get_llm_history,
        input_messages_key="question",
        history_messages_key="history",
        history_factory_config=[
            ConfigurableFieldSpec(
                id="user_id",
                annotation=str,
                name="User ID",
                description="用户唯一标识",
                default="",
                is_shared=True
            ),
            ConfigurableFieldSpec(
                id="session_id",
                annotation=str,
                name="Session ID",
                description="会话唯一标识",
                default="",
                is_shared=True
            )
        ]
    )

    return chat_with_history


# ===== 聊天会话 =====
async def async_stream_chat_session():
    """异步流式聊天会话"""
    print("欢迎使用LLM聊天助手（异步流式版本），输入'退出'结束对话")
    llm_with_history = setup_llm()

    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("感谢使用，再见！")
            break

        print("AI助手: ", end="", flush=True)

        # 使用普通聊天模式
        async for chunk in llm_with_history.astream(
                {"question": user_input},
                config={"configurable": {"user_id": user_id, "session_id": session_id_1}}
        ):
            print(chunk, end="", flush=True)


# ===== 主入口 =====
if __name__ == "__main__":
    # ===== 配置 =====
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "LLMTest"

    user_id = "zby"
    session_id_1 = "llm_session_1"
    REDIS_URL = "redis://localhost:6379/0"

    asyncio.run(async_stream_chat_session())
