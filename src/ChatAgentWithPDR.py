"""
聊天Agent、支持工具调用、历史跟踪、RAG、记忆缓存
"""

import os
import sys
import asyncio
from typing import List, Any
# LangChain 相关导入
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.tools import StructuredTool, ToolException
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_message_histories import RedisChatMessageHistory
from pydantic import BaseModel, Field
from langchain_core.messages import trim_messages
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain.schema import AgentAction

# 添加项目根目录到路径
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

# 导入自定义模块
from src.llm.LLMConfig import get_llm_configs
from src.llm.LLMAgentBuilder import LLMAgentBuilder
from src.llm.PromptManager import SystemPromptManager
from src.rag.ParentDocumentRetriever import VectorDBManager


# ===== 配置 =====
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Test"

vectorManager = VectorDBManager()


class CustomHandler(BaseCallbackHandler):
    """自定义回调处理器，用于工具调用前的人工确认"""

    def on_agent_action(self, action: AgentAction, **kwargs: Any, ) -> None:
        # print(f"\n🧠 Agent 计划调用工具: {action.tool}")
        # print(f"📦 工具输入: {action.tool_input}")

        # choice = input("❓是否继续调用工具？(y/n): ").strip().lower()
        # if choice != "y":
        #     print("⛔ 用户取消了工具调用")
        #     # 返回一个空结果，让Agent继续执行
        #     raise "用户取消了工具调用，使用自身知识回答"
        pass


# ===== 模型定义 =====
class SearchInput(BaseModel):
    """搜索工具的输入验证模型"""
    query: str = Field(description="查询的关键词或内容")


class RAGInput(BaseModel):
    """知识库查询工具的输入验证模型"""
    query: str = Field(description="需要在知识库中查询的问题")


# ===== 工具函数 =====
def _handle_error(error: ToolException) -> str:
    """统一的工具错误处理函数"""
    return f"工具执行期间发生错误：{str(error)}"


async def search_tool(query: str) -> str:
    """搜索引擎工具"""
    try:
        search = TavilySearchResults(max_results=3)
        results = await search.ainvoke(query)

        if not results:
            return "未找到相关信息"

        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"结果{i}:\n"
                f"标题: {result.get('title', '无标题')}\n"
                f"内容: {result.get('content', '无内容')}\n"
                f"链接: {result.get('url', '无链接')}\n"
            )

        return "搜索结果:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"搜索出错: {str(e)}"


async def interactive_search_tool(query: str) -> str:
    """搜索引擎工具的交互式包装函数"""
    print(f"\n🧠 Agent 计划调用工具：search_engine")
    print(f"📦 工具输入：{query}")

    choice = input("❓是否继续调用工具？(y/n): ").strip().lower()
    if choice != "y":
        print("🚫 用户取消了工具调用，返回空结果\n")
        return "用户取消了工具调用，未实际执行，仅使用自身知识进行回答\n"

    # 继续执行原始工具逻辑
    return await search_tool(query)


async def rag_tool(query: str) -> str:
    """知识库查询工具"""
    try:
        search = await vectorManager.get_retriever(
            "ParentDocumentRetrieverDB",
            3, 0.6, 0.9,
            use_ensemble=True
        )
        docs = await search.ainvoke(query)

        if not docs:
            return "知识库中未找到相关信息"

        results = []
        for i, doc in enumerate(docs, 1):
            results.append(
                f"结果{i}:\n"
                f"内容: {doc.page_content}\n"
                f"来源: {doc.metadata.get('source', '未知')}\n"
                f"检索器: {doc.metadata.get('retriever', 'vector')}\n"
            )

        return "知识库查询结果:\n" + "\n".join(results)

    except Exception as e:
        return f"知识库查询出错: {str(e)}"


async def interactive_rag_tool(query: str) -> str:
    """知识库查询工具的交互式包装函数"""
    print(f"\n🧠 Agent 计划调用工具：knowledge_query")
    print(f"📦 工具输入：{query}")

    choice = input("❓是否继续调用工具？(y/n): ").strip().lower()
    if choice != "y":
        print("🚫 用户取消了工具调用，返回空结果\n")
        return "用户取消了工具调用，未实际执行，仅使用自身知识进行回答\n"

    # 继续执行原始工具逻辑
    return await rag_tool(query)


def get_agent_history(user_id: str, session_id: str) -> RedisChatMessageHistory:
    """获取或创建Agent会话历史"""
    history = RedisChatMessageHistory(session_id=user_id + "-" + session_id, url=vectorManager.redis_url)
    messages = history.messages

    save_messages = trim_messages(
        messages,
        strategy="last",
        token_counter=len,
        max_tokens=20,
        start_on="human",
        end_on=("ai", "tool"),
        include_system=True,
    )

    history.clear()
    for message in save_messages:
        history.add_message(message)

    return history


# ===== 工具定义 =====
def create_tools() -> List[StructuredTool]:
    """创建并返回工具列表"""
    return [
        StructuredTool.from_function(
            name="search_engine",
            func=interactive_search_tool,
            description="用于在网页中搜索获取实时信息，输入应为字符串形式的问题或关键字",
            args_schema=SearchInput,
            handle_tool_error=_handle_error
        ),
        StructuredTool.from_function(
            name="knowledge_query",
            # func=interactive_rag_tool,
            coroutine=interactive_rag_tool,
            description="用于在本地知识库中查询信息，适合查询专业领域知识",
            args_schema=RAGInput,
            handle_tool_error=_handle_error
        )
    ]


# ===== Agent 配置 =====
def setup_agent():
    """设置并返回Agent"""
    agent_config = get_llm_configs(
        system_prompt=SystemPromptManager.get_agent_system_prompt(),
        agent=True,
        memory=True
    )

    llm_agent_builder = LLMAgentBuilder(agent_config)

    tools = create_tools()
    llm_agent_builder.add_tools(tools)

    # 创建带有自定义回调的Agent执行器
    agent_executor = llm_agent_builder.create_agent_executor(
        verbose=True,
        callback_manager=CallbackManager([CustomHandler()])
    )

    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        get_agent_history,
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

    return agent_with_history, llm_agent_builder


# ===== 聊天会话 =====
async def async_stream_chat_session():
    """异步流式聊天会话"""
    print("欢迎使用Agent聊天助手（异步流式版本），输入'退出'结束对话")

    user_id = "zby"
    session_id = "session_2"

    agent_with_history, llm_agent_builder = setup_agent()

    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("感谢使用，再见！")
            break

        print("\nAI助手: ", end="", flush=True)

        tools_desc = llm_agent_builder.get_tools_description()
        response = await agent_with_history.ainvoke(
            {"tools": tools_desc, "question": user_input},
            config={"configurable": {"user_id": user_id, "session_id": session_id}}
        )
        print(response['output'])


# ===== 主入口 =====
if __name__ == "__main__":
    asyncio.run(async_stream_chat_session())
