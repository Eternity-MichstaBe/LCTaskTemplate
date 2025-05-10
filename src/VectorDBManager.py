from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_redis import RedisVectorStore
from redis import Redis as RedisClient
import time
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VectorDBManager")


class VectorDBManager:
    """向量数据库管理器"""
    
    def __init__(self, redis_url="redis://localhost:6379/0", embedding_model=None):
        """
        初始化向量数据库管理器
        
        Args:
            redis_url (str): Redis连接URL
            embedding_model: 嵌入模型，默认使用OpenAIEmbeddings
        """
        self.redis_url = redis_url
        self.redis_client = RedisClient.from_url(redis_url, decode_responses=True)
        self.embeddings = embedding_model or OpenAIEmbeddings()
        logger.info("向量数据库管理器初始化完成")

    def _is_schema_exist(self, index_name: str) -> bool:
        try:
            self.redis_client.execute_command(f"FT.INFO {index_name}")
            return True
        except:
            return False
    
    def _check_and_clean_inconsistent_data(self, index_name):
        """
        检查并清理不一致的数据
        
        Args:
            index_name (str): 索引名称
            
        """
        keys = self.redis_client.keys(f"{index_name}:*")
        schema = self._is_schema_exist(index_name)
        
        if keys and not schema:
            self.redis_client.delete(*keys)
            logger.warning(f"清理了索引 {index_name} 的不一致数据")
    
    def load_documents_to_vector_db(self, document_path, index_name, chunk_size=1000, chunk_overlap=200, file_glob="**/*.pdf"):
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
                logger.warning(f"在路径 {document_path} 中没有找到匹配 {file_glob} 的文档")
                return f"在路径 {document_path} 中没有找到匹配的文档"
                
            logger.info(f"成功加载了 {len(documents)} 个文档")
                
            # 文本分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
            logger.info(f"文档已分割为 {len(chunks)} 个文本块")
                
            # 创建Redis向量数据库
            vector_db = RedisVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                redis_url=self.redis_url,
                index_name=index_name
            )
            
            # 验证schema是否成功创建
            time.sleep(2)
            if not self._is_schema_exist(index_name):
                logger.warning(f"警告: 索引 {index_name} 的schema未成功创建，可能是由于Redis写入延迟或权限问题")
                # 检查其他键是否存在
                other_keys = self.redis_client.keys(f"{index_name}:*")
                if other_keys:
                    logger.warning(f"发现 {len(other_keys)} 个相关键，但schema不存在，这可能导致数据不一致")
                
                
            return f"成功加载 {len(documents)} 个文档到向量数据库，共 {len(chunks)} 个文本块"
        except Exception as e:
            # 检查是否有部分键被创建但操作失败
            keys = self.redis_client.keys(f"{index_name}:*")
            schema_exists = self._is_schema_exist(index_name)

            if keys:
                self.redis_client.delete(*keys)
            if schema_exists:
                self.redis_client.execute_command(f"FT.DROPINDEX {index_name} DD")

            logger.exception(f"加载文档到向量数据库失败: {str(e)}")
            raise RuntimeError(f"加载文档到向量数据库失败: {str(e)}")
    
    def append_documents_to_vector_db(self, document_path, index_name, chunk_size=1000, chunk_overlap=200, file_glob="**/*.pdf"):
        """
        追加文档到现有的向量数据库
        
        Args:
            document_path (str): 文档路径
            index_name (str): 索引名称
            chunk_size (int): 文本块大小
            chunk_overlap (int): 文本块重叠大小
            file_glob (str): 文件匹配模式
            
        Returns:
            str: 操作结果信息
        """
        try:
            # 检查文档路径是否存在
            if not os.path.exists(document_path):
                raise ValueError(f"文档路径 {document_path} 不存在")
                
            # 检查索引状态
            keys = self.redis_client.keys(f"{index_name}:*")
            schema_exists = self.redis_client.exists(f"{index_name}:schema")
            
            # 如果有相关键但schema不存在，先清理这些键并提示重新创建
            if keys and not schema_exists:
                self.redis_client.delete(*keys)
                logger.warning(f"索引 {index_name} 数据不一致已清理")
                raise ValueError(f"索引 {index_name} 数据不一致已清理，请重新创建索引")
            
            # 检查索引是否存在
            if not schema_exists:
                raise ValueError(f"索引 {index_name} 不存在，请先创建索引")
            
            # 加载文档
            loader = DirectoryLoader(document_path, glob=file_glob)
            documents = loader.load()
            
            if not documents:
                logger.warning(f"在路径 {document_path} 中没有找到匹配 {file_glob} 的文档")
                return f"在路径 {document_path} 中没有找到匹配的文档"
                
            logger.info(f"成功加载了 {len(documents)} 个文档")
            
            # 文本分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
            logger.info(f"文档已分割为 {len(chunks)} 个文本块")
            
            # 加载现有的Redis向量库
            vector_db = RedisVectorStore.from_existing_index(
                embedding=self.embeddings,
                index_name=index_name,
                redis_url=self.redis_url
            )
            
            # 添加新文档
            vector_db.add_documents(chunks)
            return f"成功追加 {len(documents)} 个文档到向量数据库，共 {len(chunks)} 个文本块"
        except Exception as e:
            logger.exception(f"追加文档到向量数据库失败: {str(e)}")
            raise RuntimeError(f"追加文档到向量数据库失败: {str(e)}")
    
    def get_retriever(self, index_name, k=5):
        """
        获取向量检索器
        
        Args:
            index_name (str): 索引名称
            k (int): 返回的文档数量
            
        Returns:
            Retriever: 向量检索器
        """
        try:
            # 检查索引状态
            keys, schema_exists, cleaned = self._check_and_clean_inconsistent_data(index_name)
            
            # 如果有相关键但schema不存在，先清理这些键
            if cleaned:
                raise ValueError(f"索引 {index_name} 数据不一致已清理，请重新创建索引")
            
            # 检查索引是否存在
            if not schema_exists:
                raise ValueError(f"索引 {index_name} 不存在")
            
            # 加载现有的Redis向量库
            vector_db = RedisVectorStore.from_existing_index(
                embedding=self.embeddings,
                index_name=index_name,
                redis_url=self.redis_url
            )
            
            return vector_db.as_retriever(search_kwargs={"k": k})
        except Exception as e:
            logger.exception(f"获取向量检索器失败: {str(e)}")
            raise RuntimeError(f"获取向量检索器失败: {str(e)}")
    
    def delete_vector_db(self, index_name):
        """
        删除向量数据库
        
        Args:
            index_name (str): 索引名称
            
        Returns:
            str: 删除结果信息
        """
        try:
            # 获取所有与该索引相关的键
            keys = self.redis_client.keys(f"{index_name}:*")
            
            # 检查索引是否存在
            schema_exists = self.redis_client.exists(f"{index_name}:schema")
            
            # 即使schema不存在，只要有相关键也执行删除
            if keys:
                deleted_count = self.redis_client.delete(*keys)
                logger.info(f"删除了 {deleted_count} 个与索引 {index_name} 相关的键")
                if not schema_exists:
                    return f"删除向量数据库 {index_name} 的不一致数据成功，共删除 {deleted_count} 个键"
                return f"删除向量数据库 {index_name} 成功，共删除 {deleted_count} 个键"
            elif not schema_exists:
                return f"索引 {index_name} 不存在"
            
            return f"没有找到索引 {index_name} 相关的数据"
        except Exception as e:
            logger.exception(f"删除向量数据库失败: {str(e)}")
            raise RuntimeError(f"删除向量数据库失败: {str(e)}")
        
    def delete_all_schema_keys(self):
        """
        删除所有schema键及其关联数据
        
        Returns:
            str: 操作结果信息
        """
        try:
            total_deleted = 0
            schema_keys = []
            cursor = 0
            pattern = "*:schema"
            
            # 先收集所有schema键
            while True:
                cursor, keys = self.redis_client.scan(cursor=cursor, match=pattern, count=100)
                if keys:
                    schema_keys.extend(keys)
                if cursor == 0:
                    break
                    
            logger.info(f"找到 {len(schema_keys)} 个schema键")
            
            # 对每个schema键，删除相关的所有键
            for schema_key in schema_keys:
                index_name = schema_key.split(':')[0]
                all_keys = self.redis_client.keys(f"{index_name}:*")
                if all_keys:
                    deleted = self.redis_client.delete(*all_keys)
                    total_deleted += deleted
                    logger.info(f"删除了索引 {index_name} 的 {deleted} 个键")
            
            return f"已删除所有向量数据库，共 {len(schema_keys)} 个索引，{total_deleted} 个键"
        except Exception as e:
            logger.exception(f"删除所有schema键失败: {str(e)}")
            raise RuntimeError(f"删除所有schema键失败: {str(e)}")
