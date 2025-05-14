import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_redis import RedisVectorStore
from redis import Redis as RedisClient
from typing import Optional, List, Union, Dict, Any, Tuple
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain.schema import Document
from langchain.storage import InMemoryStore

class VectorDBManager:
    """向量数据库管理器，用于管理Redis向量数据库的文档加载、检索和删除操作"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", embedding_model: Optional[Any] = None, 
                 chunk_size: int = 500, chunk_overlap: int = 200):
        """
        初始化向量数据库管理器
        
        Args:
            redis_url (str): Redis连接URL
            embedding_model: 嵌入模型，默认使用OpenAIEmbeddings
            chunk_size (int): 文本块大小
            chunk_overlap (int): 文本块重叠大小
        """
        self.redis_url = redis_url
        self.redis_client = RedisClient.from_url(redis_url, decode_responses=False)
        self.base_embeddings = embedding_model or OpenAIEmbeddings()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 使用绝对路径创建缓存目录
        self.cache_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache"))
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.store = LocalFileStore(self.cache_dir)
        self.embeddings = CacheBackedEmbeddings.from_bytes_store(
            self.base_embeddings, 
            self.store, 
            namespace=self.base_embeddings.model
        )

    def _is_schema_exist(self, index_name: str) -> bool:
        """
        检查索引schema是否存在
        
        Args:
            index_name (str): 索引名称
            
        Returns:
            bool: 索引是否存在
        """
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
    
    def _get_docstore_path(self, index_name: str) -> str:
        """
        获取文档存储文件路径
        
        Args:
            index_name (str): 索引名称
            
        Returns:
            str: 文档存储文件路径
        """
        return os.path.join(self.cache_dir, f"{index_name}_docstore.json")
    
    def _save_docstore(self, docstore: InMemoryStore, index_name: str) -> None:
        """
        将文档存储保存到本地文件
        
        Args:
            docstore (InMemoryStore): 文档存储
            index_name (str): 索引名称
        """
        docstore_path = self._get_docstore_path(index_name)

        # 获取 store 的所有键值对
        all_data = docstore.mget(docstore.yield_keys())
        
        # 转为可序列化形式（只存文本内容）
        serializable = {
            k: {"page_content": v.page_content, "metadata": v.metadata}
            for k, v in zip(docstore.yield_keys(), all_data)
        }
        
        # 保存为 JSON 文件
        with open(docstore_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    
    def _load_docstore(self, index_name: str) -> InMemoryStore:
        """
        从本地文件加载文档存储
        
        Args:
            index_name (str): 索引名称
            
        Returns:
            InMemoryStore: 文档存储
        """
        docstore_path = self._get_docstore_path(index_name)
        docstore = InMemoryStore()
        
        if os.path.exists(docstore_path):
            # 加载JSON文件
            with open(docstore_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 写入docstore
            docstore.mset({
                k: Document(page_content=v["page_content"], metadata=v["metadata"])
                for k, v in data.items()
            }.items())
        
        return docstore
    
    def _load_documents(self, document_path: str, file_glob: str) -> List[Document]:
        """
        加载文档
        
        Args:
            document_path (str): 文档路径
            file_glob (str): 文件匹配模式
            
        Returns:
            List[Document]: 文档列表
            
        Raises:
            ValueError: 当文档路径不存在时
        """
        if not os.path.exists(document_path):
            raise ValueError(f"文档路径 {document_path} 不存在")
            
        loader = DirectoryLoader(document_path, glob=file_glob)
        documents = loader.load()
        
        return documents
    
    def _get_text_splitters(self) -> Tuple[RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitter]:
        """
        获取文本分割器
        
        Returns:
            Tuple[RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitter]: 父文档分割器和子文档分割器
        """
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 2,
            chunk_overlap=self.chunk_overlap
        )
        
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        return parent_splitter, child_splitter
    
    def _create_parent_retriever(self, vectorstore, docstore, parent_splitter, child_splitter, k: int = 3) -> ParentDocumentRetriever:
        """
        创建父子文档检索器
        
        Args:
            vectorstore: 向量存储
            docstore: 文档存储
            parent_splitter: 父文档分割器
            child_splitter: 子文档分割器
            k (int): 返回的文档数量
            
        Returns:
            ParentDocumentRetriever: 父子文档检索器
        """
        return ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": k}
        )
    
    def _create_compression_retriever(self, base_retriever, similarity_threshold: float) -> ContextualCompressionRetriever:
        """
        创建压缩检索器
        
        Args:
            base_retriever: 基础检索器
            similarity_threshold (float): 相似性阈值
            
        Returns:
            ContextualCompressionRetriever: 压缩检索器
        """
        redundant_filter = EmbeddingsRedundantFilter(
            embeddings=self.base_embeddings, 
            similarity_threshold=similarity_threshold
        )
        embeddings_filter = EmbeddingsFilter(
            embeddings=self.base_embeddings, 
            similarity_threshold=similarity_threshold
        )
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[redundant_filter, embeddings_filter]
        )
        return ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, 
            base_retriever=base_retriever
        )
    
    def _clean_up_on_failure(self, index_name: str) -> None:
        """
        失败时清理资源
        
        Args:
            index_name (str): 索引名称
        """
        # 检查是否有部分键被创建但操作失败，进行清理
        keys = self.redis_client.keys(f"{index_name}:*")
        schema_exists = self._is_schema_exist(index_name)

        if keys:
            self.redis_client.delete(*keys)
        if schema_exists:
            self.redis_client.execute_command(f"FT.DROPINDEX {index_name} DD")

        # 删除可能存在的文档存储文件
        docstore_path = self._get_docstore_path(index_name)
        if os.path.exists(docstore_path):
            os.remove(docstore_path)
    
    def load_documents_to_vector_db(self, document_path: str, index_name: str, 
                                   file_glob: str = "**/*.pdf") -> str:
        """
        加载文档并转换为向量存入Redis数据库
        
        Args:
            document_path (str): 文档路径
            index_name (str): 索引名称
            file_glob (str): 文件匹配模式
            
        Returns:
            str: 操作结果信息
            
        Raises:
            ValueError: 当文档路径不存在时
            RuntimeError: 当加载文档到向量数据库失败时
        """
        try:
            # 检查是否存在相关键但schema不存在的情况
            self._check_and_clean_inconsistent_data(index_name)
            
            # 加载文档
            documents = self._load_documents(document_path, file_glob)
            
            if not documents:
                return "没有找到文档"
            
            # 获取文本分割器
            parent_splitter, child_splitter = self._get_text_splitters()
            
            # 创建Redis向量存储
            vectorstore = RedisVectorStore.from_documents(
                documents=[],  # 先创建空的向量存储
                embedding=self.embeddings,
                redis_url=self.redis_url,
                index_name=index_name
            )
            
            # 创建Redis文档存储
            docstore = InMemoryStore()
            
            # 创建父子文档检索器
            retriever = self._create_parent_retriever(
                vectorstore=vectorstore,
                docstore=docstore,
                parent_splitter=parent_splitter,
                child_splitter=child_splitter
            )
            
            # 添加文档到检索器
            retriever.add_documents(documents)
            
            # 保存文档存储到本地
            self._save_docstore(docstore, index_name)
            
            return f"成功加载 {len(documents)} 个文档到向量数据库"
                
        except Exception as e:
            self._clean_up_on_failure(index_name)
            raise RuntimeError(f"加载文档到向量数据库失败: {str(e)}")
    
    def append_documents_to_vector_db(self, document_path: str, index_name: str, 
                                     file_glob: str = "**/*.pdf") -> str:
        """
        追加文档到现有的向量数据库
        
        Args:
            document_path (str): 文档路径
            index_name (str): 索引名称
            file_glob (str): 文件匹配模式
            
        Returns:
            str: 操作结果信息
            
        Raises:
            ValueError: 当文档路径不存在或索引状态不一致时
            RuntimeError: 当追加文档失败时
        """
        try:
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
            documents = self._load_documents(document_path, file_glob)
            
            if not documents:
                return "没有找到文档"
            
            # 获取文本分割器
            parent_splitter, child_splitter = self._get_text_splitters()
            
            # 加载现有的Redis向量库
            vectorstore = RedisVectorStore.from_existing_index(
                embedding=self.embeddings,
                index_name=index_name,
                redis_url=self.redis_url
            )
            
            # 从本地加载现有的文档存储
            docstore = self._load_docstore(index_name)
            
            # 创建父子文档检索器
            retriever = self._create_parent_retriever(
                vectorstore=vectorstore,
                docstore=docstore,
                parent_splitter=parent_splitter,
                child_splitter=child_splitter
            )
            
            # 添加文档到检索器
            retriever.add_documents(documents)
            
            # 保存更新后的文档存储到本地
            self._save_docstore(docstore, index_name)
            
            return f"成功追加 {len(documents)} 个文档到向量数据库"
            
        except Exception as e:
            raise RuntimeError(f"追加文档到向量数据库失败: {str(e)}")
    
    def _get_all_documents(self, index_name: str) -> List[Document]:
        """
        获取索引中的所有文档
        
        Args:
            index_name (str): 索引名称
            
        Returns:
            List[Document]: 文档列表
        """
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
        
        return all_docs
    
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
            vectorstore = RedisVectorStore.from_existing_index(
                embedding=self.embeddings,
                index_name=index_name,
                redis_url=self.redis_url
            )
            
            # 从本地加载文档存储
            docstore = self._load_docstore(index_name)
            
            # 获取文本分割器
            parent_splitter, child_splitter = self._get_text_splitters()
            
            # 创建父子文档检索器
            parent_retriever = self._create_parent_retriever(
                vectorstore=vectorstore,
                docstore=docstore,
                parent_splitter=parent_splitter,
                child_splitter=child_splitter,
                k=k
            )
            
            # 创建压缩检索器
            compression_retriever = self._create_compression_retriever(
                parent_retriever, 
                similarity_threshold
            )
            
            if not use_ensemble:
                return compression_retriever
            else:
                # 获取所有文档以创建BM25检索器
                all_docs = self._get_all_documents(index_name)
                
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
    
    def delete_vector_db(self, index_name: str) -> str:
        """
        删除向量数据库
        
        Args:
            index_name (str): 索引名称
            
        Returns:
            str: 操作结果信息
            
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
                
            # 删除本地文档存储文件
            docstore_path = self._get_docstore_path(index_name)
            if os.path.exists(docstore_path):
                os.remove(docstore_path)
                
            return f"成功删除索引 {index_name}"
                
        except Exception as e:
            raise RuntimeError(f"删除向量数据库失败: {str(e)}")
