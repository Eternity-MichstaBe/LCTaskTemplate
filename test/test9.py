from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_redis import RedisVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
import json
from langchain.schema import Document

# 1. 加载原始文档（父文档）
loader = TextLoader("test.txt", encoding="utf-8")  # 换成你自己的文档路径
documents = loader.load()

# 2. 定义子文档切分器（用于向量化）
child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

# 3. 定义父文档切分器（默认不切，整个文本当作一个父文档）
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# 4. 初始化嵌入模型
embeddings = OpenAIEmbeddings()  # 替换为你自己的 API key

# 5. 创建 Redis 文档存储（用于保存父文档）
docstore = InMemoryStore()

# 6. 创建向量存储（用于保存子文档向量）
# 注意：不要在初始化时传入文档，而是在 retriever.add_documents 中添加
vectorstore = RedisVectorStore(
    embeddings=embeddings,
    index_name="parent_index",
    redis_url="redis://localhost:6379/0"
)

# 7. 创建 ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

# 8. 添加父文档（自动拆分、向量化子文档并存入 vectorstore）
retriever.add_documents(documents)

# 获取 store 的所有键值对
all_data = docstore.mget(docstore.yield_keys())

# 转为可序列化形式（只存文本内容）
serializable = {
    k: {"page_content": v.page_content, "metadata": v.metadata}
    for k, v in zip(docstore.yield_keys(), all_data)
}

# 保存为 JSON 文件
with open("docstore.json", "w", encoding="utf-8") as f:
    json.dump(serializable, f, ensure_ascii=False, indent=2)


# 加载 JSON 文件
with open("docstore.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 构建新 InMemoryStore
from langchain.storage import InMemoryStore
store = InMemoryStore()

# 写入
store.mset({
    k: Document(page_content=v["page_content"], metadata=v["metadata"])
    for k, v in data.items()
})