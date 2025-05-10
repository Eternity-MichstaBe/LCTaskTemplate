"""
聊天Agent、支持工具调用、历史跟踪、RAG
"""

import os
import sys
import asyncio
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool, StructuredTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS, Redis
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_redis import RedisVectorStore
from redis import Redis as RedisClient

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

from src.LLMChainBuilder import LLMChainBuilder
from src.LLMConfig import get_llm_configs
from src.LLMAgentBuilder import LLMAgentBuilder
from src.PromptManager import SystemPromptManager

# 启用 LangChain 追踪
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Test"

# Redis配置
REDIS_URL = "redis://localhost:6379/0"
REDIS_INDEX_NAME = "LLM-EVENT-EXTRACT-KNOWLEDGE"

# 创建RAG知识库
def create_rag_tool():
    """创建向量数据库"""
    try:
        # 初始化Redis客户端
        redis_client = RedisClient.from_url(REDIS_URL)
        embeddings = OpenAIEmbeddings()
        
        # 检查Redis中是否已存在向量库
        if redis_client.exists(f"{REDIS_INDEX_NAME}:schema"):
            # 加载现有的Redis向量库
            vector_db = RedisVectorStore.from_existing_index(
                embedding=embeddings,
                index_name=REDIS_INDEX_NAME,
                redis_url=REDIS_URL
            )
            return vector_db.as_retriever()
        
        # 加载文档
        loader = DirectoryLoader(os.path.join(root, "knowledge"), glob="**/*.pdf")
        documents = loader.load()
        
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # 创建Redis向量数据库
        vector_db = RedisVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            redis_url=REDIS_URL,
            index_name=REDIS_INDEX_NAME
        )
        
        return vector_db.as_retriever()
    except Exception as e:
        print(f"创建向量数据库出错: {str(e)}")
        return None
    
# 初始化向量数据库
vector_query_tool = create_rag_tool()

# 定义pydantic输入验证
class SearchInput(BaseModel):
    query: str = Field(description="查询的关键词或内容")

class RAGInput(BaseModel):
    query: str = Field(description="需要在知识库中查询的问题")

# 定义工具函数
def search_tool(query: str) -> str:
    """搜索引擎工具"""
    try:
        search = TavilySearchResults(max_results=3)
        results = search.invoke(query)
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

def rag_tool(query: str) -> str:
    """知识库查询工具"""
    try:
        # 初始化Redis客户端
        redis_client = RedisClient.from_url(REDIS_URL)
        embeddings = OpenAIEmbeddings()
        
        # 检查Redis中是否已存在向量库
        if not redis_client.exists(f"{REDIS_INDEX_NAME}:schema"):
            return f"向量数据库中不存在{REDIS_INDEX_NAME}数据"

        # 加载现有的Redis向量库
        vector_db = RedisVectorStore.from_existing_index(
            embedding=embeddings,
            index_name=REDIS_INDEX_NAME,
            redis_url=REDIS_URL
        )
        # 创建检索器并指定返回的文档数量
        rag_tool = vector_db.as_retriever(search_kwargs={"k": 5})
        
        # 执行相似性搜索
        docs = rag_tool.invoke(query)
        
        if not docs:
            return "知识库中未找到相关信息"
        
        # 格式化结果
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(
                f"结果{i}:\n"
                f"内容: {doc.page_content}\n"
                f"来源: {doc.metadata.get('source', '未知')}\n"
            )
        
        return "知识库查询结果:\n" + "\n".join(results)
    except Exception as e:
        return f"知识库查询出错: {str(e)}"

# 配置LLM - Agent模式
agent_config = get_llm_configs(
    system_prompt=SystemPromptManager.get_agent_system_prompt(),
    agent=True,
    memory=True  # 启用记忆功能
)

# 创建Agent构建器
llm_agent_builder = LLMAgentBuilder(agent_config)

def _handle_error(error: ToolException) -> str:
    return f"工具执行期间发生错误：{error.args[0]}"

# 定义工具列表
tools = [
    StructuredTool.from_function(
        name="search_engine",
        func=search_tool,
        description="用于在网页中搜索获取实时信息，输入应为字符串形式的问题或关键字",
        args_schema=SearchInput,
        # return_direct=True,
        handle_tool_error=_handle_error
    ),
    StructuredTool.from_function(
        name="knowledge_query",
        func=rag_tool,
        description="用于在本地知识库中查询信息，适合查询专业领域知识",
        args_schema=RAGInput,
        handle_tool_error=_handle_error
    )
]

# 添加工具
llm_agent_builder.add_tools(tools)

# 创建Agent执行器
agent_executor = llm_agent_builder.create_agent_executor(verbose=True)

def get_agent_history(user_id: str, session_id: str) -> RedisChatMessageHistory:
    """获取或创建Agent会话历史"""
    history = RedisChatMessageHistory(session_id=user_id + "-" + session_id, url=REDIS_URL)
    # 限制历史记录数量，只保留最近的10轮对话
    messages = history.messages
    if len(messages) > 20:
        # 清空所有消息
        history.clear()
        # 只添加最近的10条消息
        for message in messages[-20:]:
            history.add_message(message)

    return history

# 创建带有消息历史的可运行对象 - Agent
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

async def async_stream_chat_session():
    """异步流式聊天会话"""
    print("欢迎使用Agent聊天助手（异步流式版本），输入'退出'结束对话")
    
    user_id_1 = "zby"
    session_id_1 = "session_2"
    
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("感谢使用，再见！")
            break
        
        print("\nAI助手: ", end="", flush=True)

        tools_desc = llm_agent_builder.get_tools_description()
        response = await agent_with_history.ainvoke(
            {"tools": tools_desc, "question": user_input},
            config={"configurable": {"user_id": user_id_1, "session_id": session_id_1}}
        )
        print(response['output'])

asyncio.run(async_stream_chat_session())
