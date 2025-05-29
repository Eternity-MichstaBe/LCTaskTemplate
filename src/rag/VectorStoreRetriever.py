"""
向量存储检索器实现
使用Redis作为向量存储后端，支持文档的向量化存储和检索。
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
from langchain.retrievers import EnsembleRetriever

# 添加项目根目录到路径
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)

from src.utils.utils import load_documents, get_text_splitter, create_compression_retriever, get_all_documents


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
        self.redis_url = redis_url
        self.redis_client = RedisClient.from_url(redis_url, decode_responses=False)
        self.base_embeddings = embedding_model or OpenAIEmbeddings()

        # 使用绝对路径创建缓存目录
        self.cache_dir = os.path.abspath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "vsr_cache"
        ))
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)

        self.store = LocalFileStore(self.cache_dir)
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
        try:
            # 加载文档
            documents = await load_documents(document_path, file_glob)

            if not documents:
                raise RuntimeError(f"documents load failed: {document_path}")

            # 文本分割
            text_splitter = get_text_splitter(chunk_size, chunk_overlap)
            chunks = text_splitter.split_documents(documents)

            # 创建Redis向量数据库
            RedisVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                redis_url=self.redis_url,
                index_name=index_name
            )

        except Exception as e:
            raise RuntimeError(f"加载文档到向量数据库失败: {str(e)}")

    async def append_documents_to_vector_db(self, document_path: str, index_name: str,
                                            chunk_size: int = 400, chunk_overlap: int = 100,
                                            file_glob: str = "**/*.pdf") -> None:
        """
        向现有向量数据库追加文档
        
        Args:
            document_path: 文档路径
            index_name: 索引名称
            chunk_size: 文本分块大小
            chunk_overlap: 文本分块重叠大小
            file_glob: 文件匹配模式
        """
        try:
            # 加载文档
            documents = await load_documents(document_path, file_glob)

            if not documents:
                raise RuntimeError(f"documents load failed: {document_path}")

            # 文本分割
            text_splitter = get_text_splitter(chunk_size, chunk_overlap)
            chunks = text_splitter.split_documents(documents)

            # 加载现有的Redis向量库
            vector_db = RedisVectorStore.from_existing_index(
                embedding=self.embeddings,
                index_name=index_name,
                redis_url=self.redis_url
            )

            # 添加新文档
            await vector_db.add_documents(chunks)

        except Exception as e:
            raise RuntimeError(f"追加文档到向量数据库失败: {str(e)}")

    async def delete_vector_db(self, index_name: str) -> None:
        """
        删除向量数据库
        
        Args:
            index_name: 要删除的索引名称
        """
        try:
            # 获取所有与该索引相关的键
            keys = await self.redis_client.keys(f"{index_name}:*")
            if keys:
                await self.redis_client.delete(*keys)
            await self.redis_client.execute_command(f"FT.DROPINDEX {index_name} DD")
        except Exception as e:
            raise RuntimeError(f"删除向量数据库失败: {str(e)}")

    async def get_retriever(self, index_name: str, k: int = 3,
                            threshold1: float = 0.75, threshold2: float = 0.75,
                            use_ensemble: bool = False):
        """
        获取文档检索器
        
        Args:
            index_name: 索引名称
            k: 检索的文档数量
            threshold1: 相似度阈值1
            threshold2: 相似度阈值2
            use_ensemble: 是否使用集成检索器
            
        Returns:
            文档检索器实例
        """
        try:
            # 加载现有的Redis向量库
            vector_db = RedisVectorStore.from_existing_index(
                embedding=self.embeddings,
                index_name=index_name,
                redis_url=self.redis_url
            )

            # 创建向量检索器
            vector_retriever = vector_db.as_retriever(search_kwargs={"k": k})

            # 创建压缩检索器
            compression_retriever = create_compression_retriever(
                self.base_embeddings,
                vector_retriever,
                threshold1,
                threshold2
            )

            if not use_ensemble:
                return compression_retriever
            else:
                # 获取所有文档以创建BM25检索器
                all_docs = await get_all_documents(self.redis_client, index_name)

                # 创建BM25检索器
                bm25_retriever = BM25Retriever.from_documents(all_docs)
                bm25_retriever.k = 2

                # 创建组合检索器
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, compression_retriever],
                    weights=[0.3, 0.7]
                )

                return ensemble_retriever

        except Exception as e:
            raise RuntimeError(f"获取检索器失败: {str(e)}")
