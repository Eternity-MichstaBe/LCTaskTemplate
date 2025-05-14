from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_redis import RedisVectorStore
from redis import Redis as RedisClient
import time
import os
from typing import Optional, List, Union, Dict, Any
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document

class VectorDBManager:
    """向量数据库管理器，用于管理Redis向量数据库的文档加载、检索和删除操作"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", embedding_model: Optional[Any] = None):
        """
        初始化向量数据库管理器
        
        Args:
            redis_url (str): Redis连接URL
            embedding_model: 嵌入模型，默认使用OpenAIEmbeddings
        """
        self.redis_url = redis_url
        self.redis_client = RedisClient.from_url(redis_url, decode_responses=False)
        self.base_embeddings = embedding_model or OpenAIEmbeddings()
        
        # 使用绝对路径创建缓存目录
        cache_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache"))
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        self.store = LocalFileStore(cache_dir)
        self.embeddings = CacheBackedEmbeddings.from_bytes_store(
            self.base_embeddings, 
            self.store, 
            namespace=self.base_embeddings.model
        )

    def _is_schema_exist(self, index_name: str) -> bool:
        """检查索引schema是否存在"""
        try:
            self.redis_client.execute_command(f"FT.INFO {index_name}")
            return True
        except Exception:
            return False
    
    def _check_and_clean_inconsistent_data(self, index_name: str) -> None:
        """
        检查并清理不一致的数据
        
        Args:
            index_name (str): 索引名称
        """
        keys = self.redis_client.keys(f"{index_name}:*")
        schema_exists = self._is_schema_exist(index_name)
        
        if keys and not schema_exists:
            self.redis_client.delete(*keys)
    
    def load_documents_to_vector_db(self, document_path: str, index_name: str, 
                                   chunk_size: int = 1000, chunk_overlap: int = 200, 
                                   file_glob: str = "**/*.pdf") -> str:
        """
        加载文档并转换为向量存入Redis数据库
        
        Args:
            document_path (str): 文档路径
            index_name (str): 索引名称
            chunk_size (int): 文本块大小
            chunk_overlap (int): 文本块重叠大小
            file_glob (str): 文件匹配模式
            
        Returns:
            str: 操作结果信息
            
        Raises:
            ValueError: 当文档路径不存在时
            RuntimeError: 当加载文档到向量数据库失败时
        """
        try:
            # 检查文档路径是否存在
            if not os.path.exists(document_path):
                raise ValueError(f"文档路径 {document_path} 不存在")
                
            # 检查是否存在相关键但schema不存在的情况
            self._check_and_clean_inconsistent_data(index_name)
            
            # 加载文档
            loader = DirectoryLoader(document_path, glob=file_glob)
            documents = loader.load()
            
            if not documents:
                return
                
            # 文本分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
                
            # 创建Redis向量数据库
            RedisVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                redis_url=self.redis_url,
                index_name=index_name
            )
            
            # 验证schema是否成功创建
            time.sleep(2)
            if not self._is_schema_exist(index_name):
                # 检查其他键是否存在
                _ = self.redis_client.keys(f"{index_name}:*")
                
        except Exception as e:
            # 检查是否有部分键被创建但操作失败，进行清理
            keys = self.redis_client.keys(f"{index_name}:*")
            schema_exists = self._is_schema_exist(index_name)

            if keys:
                self.redis_client.delete(*keys)
            if schema_exists:
                self.redis_client.execute_command(f"FT.DROPINDEX {index_name} DD")

            raise RuntimeError(f"加载文档到向量数据库失败: {str(e)}")
    
    def append_documents_to_vector_db(self, document_path: str, index_name: str, 
                                     chunk_size: int = 1000, chunk_overlap: int = 200, 
                                     file_glob: str = "**/*.pdf") -> None:
        """
        追加文档到现有的向量数据库
        
        Args:
            document_path (str): 文档路径
            index_name (str): 索引名称
            chunk_size (int): 文本块大小
            chunk_overlap (int): 文本块重叠大小
            file_glob (str): 文件匹配模式
            
        Raises:
            ValueError: 当文档路径不存在或索引状态不一致时
            RuntimeError: 当追加文档失败时
        """
        try:
            # 检查文档路径是否存在
            if not os.path.exists(document_path):
                raise ValueError(f"文档路径 {document_path} 不存在")
                
            # 检查索引状态
            keys = self.redis_client.keys(f"{index_name}:*")
            schema_exists = self._is_schema_exist(index_name)
            
            # 如果有相关键但schema不存在，先清理这些键并提示重新创建
            if keys and not schema_exists:
                self.redis_client.delete(*keys)
                raise ValueError(f"索引 {index_name} 数据不一致已清理，请重新创建索引")
            
            # 检查索引是否存在
            if not schema_exists:
                raise ValueError(f"索引 {index_name} 不存在，请先创建索引")
            
            # 加载文档
            loader = DirectoryLoader(document_path, glob=file_glob)
            documents = loader.load()
            
            if not documents:
                return
            
            # 文本分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
            
            # 加载现有的Redis向量库
            vector_db = RedisVectorStore.from_existing_index(
                embedding=self.embeddings,
                index_name=index_name,
                redis_url=self.redis_url
            )
            
            # 添加新文档
            vector_db.add_documents(chunks)
        except Exception as e:
            raise RuntimeError(f"追加文档到向量数据库失败: {str(e)}")
    
    def get_retriever(self, index_name: str, k: int = 3, similarity_threshold: float = 0.75, use_ensemble: bool = False):
        """
        获取向量检索器
        
        Args:
            index_name (str): 索引名称
            k (int): 返回的文档数量
            similarity_threshold (float): 相似性阈值
            use_ensemble (bool): 是否使用BM25和向量检索的组合方式

        Returns:
            Retriever: 向量检索器
            
        Raises:
            RuntimeError: 当获取检索器失败时
        """
        try:
            self._check_and_clean_inconsistent_data(index_name)
            
            # 检查索引是否存在
            if not self._is_schema_exist(index_name):
                raise ValueError(f"索引 {index_name} 不存在，请先创建索引")
            
            # 加载现有的Redis向量库
            vector_db = RedisVectorStore.from_existing_index(
                embedding=self.embeddings,
                index_name=index_name,
                redis_url=self.redis_url
            )
            
            # 创建向量检索器
            vector_retriever = vector_db.as_retriever(search_kwargs={"k": k})
            
            # 创建混合压缩器管道
            redundant_filter = EmbeddingsRedundantFilter(embeddings=self.base_embeddings, similarity_threshold=similarity_threshold)
            embeddings_filter = EmbeddingsFilter(embeddings=self.base_embeddings, similarity_threshold=similarity_threshold)
            pipeline_compressor = DocumentCompressorPipeline(
                transformers=[redundant_filter, embeddings_filter]
            )
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=pipeline_compressor, base_retriever=vector_retriever
            )
            
            if not use_ensemble:
                return compression_retriever
            else:
                # 获取所有文档以创建BM25检索器
                all_docs = []
                for key in self.redis_client.keys(f"{index_name}:*"):
                    doc_data = self.redis_client.hgetall(key)
                    if b'text' in doc_data:
                        content = doc_data[b'text'].decode('utf-8')
                        metadata = {}
                        if b'source' in doc_data:
                            metadata['source'] = doc_data[b'source'].decode('utf-8')
                            metadata['retriever'] = "bm25"

                        all_docs.append(Document(page_content=content, metadata=metadata))
                
                # 创建BM25检索器
                bm25_retriever = BM25Retriever.from_documents(all_docs)
                bm25_retriever.k = k
                
                # 创建组合检索器
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, compression_retriever],
                    weights=[0.2, 0.8]
                )
                
                return ensemble_retriever
            
        except Exception as e:
            raise RuntimeError(f"获取检索器失败: {str(e)}")
    
    def delete_vector_db(self, index_name: str) -> None:
        """
        删除向量数据库
        
        Args:
            index_name (str): 索引名称
            
        Raises:
            RuntimeError: 当删除向量数据库失败时
        """
        try:
            # 获取所有与该索引相关的键
            keys = self.redis_client.keys(f"{index_name}:*")
            
            # 检查索引是否存在
            schema_exists = self._is_schema_exist(index_name)
            
            # 即使schema不存在，只要有相关键也执行删除
            if keys:
                self.redis_client.delete(*keys)
            
            if schema_exists:
                self.redis_client.execute_command(f"FT.DROPINDEX {index_name} DD")
        except Exception as e:
            raise RuntimeError(f"删除向量数据库失败: {str(e)}")
