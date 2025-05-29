"""
父文本检索器实现
使用Redis作为向量存储后端，支持文档的分块、嵌入和检索。
"""

import os
import sys
from langchain_openai import OpenAIEmbeddings
from langchain_redis import RedisVectorStore
from redis.asyncio import Redis as RedisClient
from typing import Optional, Any
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain.storage import InMemoryStore
from redisvl.exceptions import RedisSearchError

# 添加项目根目录到Python路径，确保可以导入项目中的其他模块
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)

from src.utils.utils import (
    load_documents, get_text_splitter, create_compression_retriever, get_all_documents,
    get_docStore_path, save_docStore, load_docStore, get_doc_hash, normalize_newlines
)


class VectorDBManager:
    """
    向量数据库管理器类，负责文档的向量化存储和检索。
    
    主要功能：
    1. 文档加载和向量化
    2. 文档追加
    3. 向量数据库删除
    4. 文档检索器获取
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", embedding_model: Optional[Any] = None):
        """
        初始化向量数据库管理器
        
        Args:
            redis_url: Redis服务器URL
            embedding_model: 可选的嵌入模型，默认使用OpenAIEmbeddings
        """
        # 初始化Redis连接和嵌入模型
        self.redis_url = redis_url
        self.redis_client = RedisClient.from_url(redis_url, decode_responses=False)
        self.base_embeddings = embedding_model or OpenAIEmbeddings()

        # 设置缓存目录和文件存储
        self.cache_dir = os.path.abspath(
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pdr_cache"))
        os.makedirs(self.cache_dir, exist_ok=True)
        self.store = LocalFileStore(self.cache_dir)

        # 创建带缓存的嵌入模型
        self.embeddings = CacheBackedEmbeddings.from_bytes_store(
            self.base_embeddings,
            self.store,
            namespace=self.base_embeddings.model
        )

    async def load_documents_to_vector_db(self, document_path: str, index_name: str,
                                          chunk_size: int = 400, chunk_overlap: int = 100,
                                          file_glob: str = "**/*.pdf"):
        """
        将文档加载到向量数据库中
        
        Args:
            document_path: 文档路径
            index_name: 索引名称
            chunk_size: 文本分块大小
            chunk_overlap: 文本分块重叠大小
            file_glob: 文件匹配模式
        """
        # 加载并预处理文档
        documents = await load_documents(document_path, file_glob)
        if not documents:
            raise RuntimeError(f"documents load failed: {document_path}")

        # 标准化文档内容中的换行符
        for doc in documents:
            doc.page_content = normalize_newlines(doc.page_content)

        # 创建文档分割器
        parent_splitter, child_splitter = get_text_splitter(chunk_size, chunk_overlap,
                                                            mode="ParentDocumentRetriever")

        try:
            # 尝试加载现有索引
            vectorstore = RedisVectorStore.from_existing_index(
                embedding=self.embeddings,
                index_name=index_name,
                redis_url=self.redis_url
            )
            doc_store = await load_docStore(self.cache_dir, index_name)
            is_new_index = False
        except RedisSearchError:
            # 如果索引不存在，创建新索引
            vectorstore = RedisVectorStore.from_documents(
                documents=[],
                embedding=self.embeddings,
                redis_url=self.redis_url,
                index_name=index_name
            )
            doc_store = InMemoryStore()
            is_new_index = True

        # 创建父文档检索器
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=doc_store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )

        try:
            if is_new_index:
                # 新索引直接添加所有文档
                retriever.add_documents(
                    documents,
                    ids=[get_doc_hash(doc) for doc in parent_splitter.split_documents(documents)]
                )
                await save_docStore(self.cache_dir, doc_store, index_name)
            else:
                # 现有索引只添加新文档
                new_documents = []
                existing_keys = set(doc_store.yield_keys())
                for doc in parent_splitter.split_documents(documents):
                    doc_hash = get_doc_hash(doc)
                    if doc_hash not in existing_keys:
                        new_documents.append(doc)

                if new_documents:
                    retriever.add_documents(
                        new_documents,
                        ids=[get_doc_hash(doc) for doc in new_documents]
                    )
                    await save_docStore(self.cache_dir, doc_store, index_name)

        except Exception as e:
            raise RuntimeError("Failed to load documents: " + str(e))

    async def append_documents_to_vector_db(self, document_path: str, index_name: str,
                                            chunk_size: int = 400, chunk_overlap: int = 100,
                                            file_glob: str = "**/*.pdf"):
        """
        向现有向量数据库追加文档
        
        Args:
            document_path: 文档路径
            index_name: 索引名称
            chunk_size: 文本分块大小
            chunk_overlap: 文本分块重叠大小
            file_glob: 文件匹配模式
        """
        # 加载并预处理文档
        documents = await load_documents(document_path, file_glob)
        if not documents:
            raise RuntimeError(f"documents load failed: {document_path}")

        # 标准化文档内容
        for doc in documents:
            doc.page_content = normalize_newlines(doc.page_content)

        # 创建文档分割器
        parent_splitter, child_splitter = get_text_splitter(chunk_size, chunk_overlap,
                                                            mode="ParentDocumentRetriever")

        # 加载现有索引和文档存储
        vectorstore = RedisVectorStore.from_existing_index(
            embedding=self.embeddings,
            index_name=index_name,
            redis_url=self.redis_url
        )

        doc_store = await load_docStore(self.cache_dir, index_name)

        # 创建父文档检索器
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=doc_store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )

        try:
            # 只添加新文档
            new_documents = []
            existing_keys = set(doc_store.yield_keys())
            for doc in parent_splitter.split_documents(documents):
                doc_hash = get_doc_hash(doc)
                if doc_hash not in existing_keys:
                    new_documents.append(doc)

            if new_documents:
                retriever.add_documents(
                    new_documents,
                    ids=[get_doc_hash(doc) for doc in new_documents]
                )
                await save_docStore(self.cache_dir, doc_store, index_name)

        except Exception as e:
            raise RuntimeError("Failed to load documents: " + str(e))

    async def delete_vector_db(self, index_name: str):
        """
        删除向量数据库
        
        Args:
            index_name: 要删除的索引名称
        """
        try:
            # 删除Redis中的所有相关键
            keys = await self.redis_client.keys(f"{index_name}:*")
            if keys:
                await self.redis_client.delete(*keys)
            # 删除Redis搜索索引
            await self.redis_client.execute_command(f"FT.DROPINDEX {index_name} DD")
            # 删除本地文档存储文件
            doc_store_path = get_docStore_path(self.cache_dir, index_name)
            if os.path.exists(doc_store_path):
                os.remove(doc_store_path)

        except Exception as e:
            raise RuntimeError(f"Failed to delete vector database: {str(e)}")

    async def get_retriever(self, index_name: str, k: int = 3, threshold1: float = 0.5, threshold2: float = 0.7,
                            chunk_size: int = 400, chunk_overlap: int = 100,
                            use_ensemble: bool = False):
        """
        获取文档检索器
        
        Args:
            index_name: 索引名称
            k: 检索的文档数量
            threshold1: 相似度阈值
            threshold2: 相似度阈值
            chunk_size: 文本分块大小
            chunk_overlap: 文本分块重叠大小
            use_ensemble: 是否使用集成检索器
            
        Returns:
            文档检索器实例
        """
        try:
            # 加载现有索引
            vectorstore = RedisVectorStore.from_existing_index(
                embedding=self.embeddings,
                index_name=index_name,
                redis_url=self.redis_url
            )

            doc_store = await load_docStore(self.cache_dir, index_name)

            parent_splitter, child_splitter = get_text_splitter(chunk_size, chunk_overlap,
                                                                mode="ParentDocumentRetriever")

            parent_retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=doc_store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                search_kwargs={"k": k}
            )

            compression_retriever = create_compression_retriever(
                self.base_embeddings,
                parent_retriever,
                threshold1,
                threshold2
            )

            if not use_ensemble:
                return compression_retriever
            else:
                all_docs = await get_all_documents(self.redis_client, index_name)
                bm25_retriever = BM25Retriever.from_documents(all_docs)
                bm25_retriever.k = 2
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, compression_retriever],
                    weights=[0.3, 0.7]
                )
                return ensemble_retriever

        except Exception as e:
            raise RuntimeError(f"Failed to get retriever: {str(e)}")
