import os
import json
import aiofiles
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.storage import InMemoryStore
from typing import List
import hashlib
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline


# 文档存储相关函数
def get_docStore_path(cache_dir, index_name: str) -> str:
    """
    获取文档存储文件路径
    Args:
        index_name (str): 索引名称
        cache_dir: 缓存目录
    Returns:
        str: 文档存储文件路径
    """
    return os.path.join(cache_dir, f"{index_name}_docStore.json")


async def save_docStore(cache_dir, doc_store, index_name: str) -> None:
    """
    异步将文档存储保存到本地文件
    Args:
        doc_store (InMemoryStore): 文档存储
        index_name (str): 索引名称
        cache_dir: 缓存目录
    """
    docStore_path = get_docStore_path(cache_dir, index_name)
    os.makedirs(os.path.dirname(docStore_path), exist_ok=True)
    # 获取 store 的所有键值对
    all_data = doc_store.mget(list(doc_store.yield_keys()))
    # 转为可序列化形式（只存文本内容）
    serializable = {
        k: {"page_content": v.page_content, "metadata": v.metadata}
        for k, v in zip(doc_store.yield_keys(), all_data)
    }
    # 异步保存为 JSON 文件
    async with aiofiles.open(docStore_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(serializable, ensure_ascii=False, indent=2))


async def load_docStore(cache_dir, index_name: str):
    """
    异步从本地文件加载文档存储
    Args:
        index_name (str): 索引名称
        cache_dir: 缓存目录
    """
    docStore_path = get_docStore_path(cache_dir, index_name)
    os.makedirs(os.path.dirname(docStore_path), exist_ok=True)
    docStore = InMemoryStore()
    if os.path.exists(docStore_path):
        # 异步加载JSON文件
        async with aiofiles.open(docStore_path, "r", encoding="utf-8") as f:
            data = json.loads(await f.read())
        # 写入docStore
        docStore.mset([
            (k, Document(page_content=v["page_content"], metadata=v["metadata"]))
            for k, v in data.items()
        ])
    return docStore


# 文档加载和处理相关函数
async def load_documents(document_path: str, file_glob: str = "**/*.pdf") -> List[Document]:
    """
    异步加载指定路径下的文档文件

    Args:
        document_path (str): 文档所在目录路径
        file_glob (str): 文件匹配模式，默认匹配所有PDF文件

    Returns:
        List[Document]: 加载的文档列表，每个文档包含文本内容和元数据

    Raises:
        ValueError: 当指定的文档路径不存在时抛出异常
    """
    if not os.path.exists(document_path):
        raise ValueError(f"文档路径 {document_path} 不存在")

    loader = DirectoryLoader(document_path, glob=file_glob)
    documents = await loader.aload()

    return documents


async def get_all_documents(redis_client, index_name: str, retriever: str = "bm25") -> List[Document]:
    """
    异步获取索引中的所有文档

    Args:
        redis_client: Redis客户端实例，用于连接Redis数据库
        index_name (str): 索引名称，用于标识要获取的文档集合
        retriever (str): 检索器类型，默认为"bm25"，用于标识文档的检索方式

    Returns:
        List[Document]: 文档列表，每个文档包含文本内容和元数据

    Note:
        该方法会从Redis数据库中获取指定索引下的所有文档，并将它们转换为Document对象。
        每个文档都包含文本内容和元数据（如来源和检索器类型）。
    """
    all_docs = []
    keys = await redis_client.keys(f"{index_name}:*")
    for key in keys:
        doc_data = await redis_client.hgetall(key)
        if b'text' in doc_data:
            content = doc_data[b'text'].decode('utf-8')
            metadata = {}
            if b'source' in doc_data:
                metadata['source'] = doc_data[b'source'].decode('utf-8')
                metadata['retriever'] = retriever
            all_docs.append(Document(page_content=content, metadata=metadata))

    return all_docs


# 文本处理和检索相关函数
def get_text_splitter(chunk_size: int = 400, chunk_overlap: int = 100, mode: str = "VectorStoreRetriever"):
    """
    创建文本分割器，用于将长文本分割成较小的文本块

    Args:
        chunk_size (int): 文本块大小，默认400个字符
        chunk_overlap (int): 文本块重叠大小，默认100个字符，用于保持上下文连贯性
        mode: 检索器类型

    Returns:
        RecursiveCharacterTextSplitter: 文本分割器实例，支持递归分割文本
    """
    if mode == "VectorStoreRetriever":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    elif mode == "ParentDocumentRetriever":
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 2,
            chunk_overlap=chunk_overlap
        )

        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        return parent_splitter, child_splitter


def create_compression_retriever(base_embeddings, base_retriever, threshold1: float = 0.7, threshold2: float = 0.7):
    """
    创建压缩检索器，用于优化检索结果的质量和相关性

    Args:
        base_embeddings: 基础嵌入模型，用于计算文本相似度
        base_retriever: 基础检索器，用于初步检索相关文档
        threshold1 (float): 相似性阈值，默认0.7，用于过滤相似度较低的文档
        threshold2 (float): 相似性阈值，默认0.7，用于过滤高度重复的文档

    Returns:
        ContextualCompressionRetriever: 压缩检索器实例，包含冗余过滤和相似度过滤功能
    """
    # 创建相似度过滤器，用于过滤掉与查询相关性较低的文档
    embeddings_filter = EmbeddingsFilter(
        embeddings=base_embeddings,
        similarity_threshold=threshold1
    )

    # 创建冗余过滤器，用于去除重复或高度相似的文档
    redundant_filter = EmbeddingsRedundantFilter(
        embeddings=base_embeddings,
        similarity_threshold=threshold2
    )

    # 创建文档压缩管道，组合多个过滤器
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[embeddings_filter, redundant_filter]
    )
    # 返回上下文压缩检索器，结合基础检索器和压缩管道
    return ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=base_retriever
    )


def get_doc_hash(doc: Document) -> str:
    """计算文档内容的哈希值"""
    return hashlib.md5(doc.page_content.strip().encode('utf-8')).hexdigest()


def normalize_newlines(text: str) -> str:
    # 替换所有 \r\n 和 \r 为 \n，然后将多个换行替换为一个
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+', ' ', text)  # 多个空格合并为一个
    text = re.sub(r'\n+', '\n', text)
    return text.strip()
