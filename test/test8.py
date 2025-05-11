import os  
import tempfile  
import streamlit as st  
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  
from langchain.document_loaders import PyPDFLoader  
from langchain.memory import ConversationBufferMemory  
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory  
from langchain.callbacks.base import BaseCallbackHandler  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_redis import RedisVectorStore
from redis import Redis as RedisClient
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import asyncio
from typing import Dict, Any, List

st.set_page_config(page_title="PDF文档问答", page_icon="🦜")  
st.title("🦜 PDF文档问答")  
  
  
@st.cache_resource(ttl="1h")  
def configure_retriever(uploaded_files):  
    # Read documents  
    docs = []  
    temp_dir = tempfile.TemporaryDirectory()  
    for file in uploaded_files:  
        temp_filepath = os.path.join(temp_dir.name, file.name)  
        with open(temp_filepath, "wb") as f:  
            f.write(file.getvalue())  
        loader = PyPDFLoader(temp_filepath)  
        docs.extend(loader.load())  
  
    # Split documents  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)  
    splits = text_splitter.split_documents(docs)  
  
    # Create embeddings and store in Redis vectordb
    embeddings = OpenAIEmbeddings()
    redis_url = "redis://localhost:6379/0"
    redis_client = RedisClient.from_url(redis_url, decode_responses=True)
    
    # 使用唯一索引名称
    index_name = "pdf_qa_index"
    
    # 检查并清理可能存在的旧索引
    keys = redis_client.keys(f"{index_name}:*")
    if keys:
        redis_client.delete(*keys)
    try:
        redis_client.execute_command(f"FT.DROPINDEX {index_name} DD")
    except:
        pass
    
    # 创建Redis向量数据库
    vectordb = RedisVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        redis_url=redis_url,
        index_name=index_name
    )
  
    # Define retriever  
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})  
  
    return retriever  
  
  
class StreamHandler(BaseCallbackHandler):  
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):  
        self.container = container  
        self.text = initial_text  
        self.run_id_ignore_token = None  
  
    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):  
        # Workaround to prevent showing the rephrased question as output  
        if prompts[0].startswith("Human"):  
            self.run_id_ignore_token = kwargs.get("run_id")  
  
    def on_llm_new_token(self, token: str, **kwargs) -> None:  
        if self.run_id_ignore_token == kwargs.get("run_id", False):  
            return  
        self.text += token  
        self.container.markdown(self.text)  
  
  
class PrintRetrievalHandler(BaseCallbackHandler):  
    def __init__(self, container):  
        self.status = container.status("**Context Retrieval**")  
  
    def on_retriever_start(self, serialized: dict, query: str, **kwargs):  
        self.status.write(f"**Question:** {query}")  
        self.status.update(label=f"**Context Retrieval:** {query}")  
  
    def on_retriever_end(self, documents, **kwargs):  
        for idx, doc in enumerate(documents):  
            source = os.path.basename(doc.metadata["source"])  
            self.status.write(f"**Document {idx} from {source}**")  
            self.status.markdown(doc.page_content)  
        self.status.update(state="complete")  
  
# 初始化session_state
if "response" not in st.session_state:
    st.session_state.response = None
if "msgs" not in st.session_state:
    st.session_state.msgs = StreamlitChatMessageHistory()

uploaded_files = st.sidebar.file_uploader(  
    label="上传PDF文件", type=["pdf"], accept_multiple_files=True  
)  
if not uploaded_files:  
    st.info("上传PDF文档后使用")  
    st.stop()  
  
retriever = configure_retriever(uploaded_files)  
  
# Setup memory for contextual conversation  
msgs = st.session_state.msgs
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)  
  
# 创建检索工具
async def retrieve_docs(query):
    docs = await retriever.ainvoke(query)
    return "\n\n".join([f"Document {i}: {doc.page_content}" for i, doc in enumerate(docs)])

retrieval_tool = Tool(
    name="pdf_search",
    description="搜索PDF文档中的信息，用于回答关于文档内容的问题",
    func=lambda query: asyncio.run(retrieve_docs(query))
)

# 设置Agent提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的PDF文档助手，可以回答用户关于上传文档的问题。使用提供的工具来搜索文档内容，并给出详细、准确的回答。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 设置LLM
llm = ChatOpenAI(temperature=0, streaming=True)

# 创建Agent
agent = create_openai_tools_agent(llm, [retrieval_tool], prompt)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=[retrieval_tool],
    memory=memory,
    verbose=True,
    return_intermediate_steps=True
)
  
if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):  
    msgs.clear()  
    msgs.add_ai_message("我可以帮你回答关于上传PDF文档的问题。")  
  
avatars = {"human": "user", "ai": "assistant"}  
for msg in msgs.messages:  
    st.chat_message(avatars[msg.type]).write(msg.content)  
  
if user_query := st.chat_input(placeholder="请问任何关于文档的问题!"):  
    # 显示用户问题
    st.chat_message("user").write(user_query)  
    
    # 将用户问题添加到消息历史中（只添加一次）
    msgs.add_user_message(user_query)
  
    with st.chat_message("assistant"):  
        retrieval_container = st.container()
        response_placeholder = st.empty()
        stream_handler = StreamHandler(response_placeholder)  
        
        # 创建一个状态指示器
        with st.status("正在处理您的问题...", expanded=True) as status:
            st.write("正在搜索相关文档并生成回答...")
            
            # 使用异步方式执行Agent查询
            async def process_query():
                # 使用stream_handler立即将生成的内容显示到前端
                response = await agent_executor.ainvoke(
                    {"input": user_query},
                    {"callbacks": [stream_handler]}
                )
                
                # 实时显示检索到的文档（如果有）
                for step in response["intermediate_steps"]:
                    if step[0].tool == "pdf_search":
                        with retrieval_container:
                            st.write("**检索到的文档内容:**")
                            st.markdown(step[1])
                
                status.update(label="处理完成!", state="complete")
                
                # 将AI回答添加到消息历史中
                if "output" in response:
                    # 实时更新历史信息
                    msgs.add_ai_message(response["output"])
                    # 强制刷新前端显示
                    st.rerun()
                
                return response
            
            # 在Streamlit中运行异步函数并立即展示结果
            response = asyncio.run(process_query())
            st.session_state.response = response
