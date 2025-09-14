import os
import sys
from typing import Optional, Any
from langchain_redis import RedisVectorStore
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag.VectorDBManagerBase import VectorDBManagerBase


class ParentDocumentRetrieverManager(VectorDBManagerBase):
    """
    父文档检索器管理器类，负责基于父子文档结构的检索
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", embedding_model: Optional[Any] = None):
        """
        初始化父文档检索器管理器

        Args:
            redis_url: Redis服务器URL
            embedding_model: 可选的嵌入模型，默认使用SiliconFlowEmbeddings
        """
        super().__init__(redis_url, embedding_model)

        # 设置缓存目录和文件存储
        self.cache_dir = os.path.abspath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "vectors",
            "parent_child"
        ))

        os.makedirs(self.cache_dir, exist_ok=True)
        self.store = LocalFileStore(self.cache_dir)

        # 创建带缓存的嵌入模型
        self.embeddings = CacheBackedEmbeddings.from_bytes_store(
            self.embeddings,
            self.store,
            namespace=self.embeddings.model,
            key_encoder="sha256"
        )

    async def append_documents_to_vector_db(self, document_path: str, index_name: str,
                                            chunk_size: int = 400, chunk_overlap: int = 100,
                                            file_glob: str = "**/*.pdf", ttl: int = None):
        """
        向现有向量数据库追加文档

        Args:
            document_path: 文档路径
            index_name: 索引名称
            chunk_size: 文本分块大小
            chunk_overlap: 文本分块重叠大小
            file_glob: 文件匹配模式
            ttl: 存活时间
        """
        # 加载并预处理文档
        documents = await VectorDBManagerBase.load_documents(document_path, file_glob)
        if not documents:
            raise RuntimeError(f"documents load failed: {document_path}")

        # 标准化文档内容
        for doc in documents:
            doc.page_content = VectorDBManagerBase.normalize_newlines(doc.page_content)

        # 创建文档分割器
        parent_splitter, child_splitter = VectorDBManagerBase.get_text_splitter(chunk_size, chunk_overlap,
                                                            mode="ParentDocumentRetriever")

        # 加载现有索引和文档存储
        vectorstore = RedisVectorStore.from_existing_index(
            embedding=self.embeddings,
            index_name=index_name,
            redis_url=self.redis_url,
            ttl=ttl  # 添加这一行来使用传入的ttl参数
        )

        doc_store = await VectorDBManagerBase.load_docStore(self.cache_dir, index_name)

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
                doc_hash = VectorDBManagerBase.get_doc_hash(doc)
                if doc_hash not in existing_keys:
                    new_documents.append(doc)

            if new_documents:
                await retriever.aadd_documents(
                    new_documents,
                    ids=[VectorDBManagerBase.get_doc_hash(doc) for doc in new_documents]
                )
                await VectorDBManagerBase.save_docStore(self.cache_dir, doc_store, index_name)

        except Exception as e:
            raise RuntimeError("Failed to load documents: " + str(e))

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

            doc_store = await VectorDBManagerBase.load_docStore(self.cache_dir, index_name)

            parent_splitter, child_splitter = VectorDBManagerBase.get_text_splitter(chunk_size, chunk_overlap,
                                                                mode="ParentDocumentRetriever")

            parent_retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=doc_store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                search_kwargs={"k": k}
            )

            compression_retriever = VectorDBManagerBase.create_compression_retriever(
                self.embeddings,
                parent_retriever,
                threshold1,
                threshold2
            )

            if not use_ensemble:
                return compression_retriever
            else:
                all_docs = await VectorDBManagerBase.get_all_documents(self.redis_client, index_name)
                bm25_retriever = BM25Retriever.from_documents(all_docs)
                bm25_retriever.k = 2
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, compression_retriever],
                    weights=[0.3, 0.7]
                )
                return ensemble_retriever

        except Exception as e:
            raise RuntimeError(f"Failed to get retriever: {str(e)}")