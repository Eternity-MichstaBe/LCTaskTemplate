import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

from src.VectorDBManager import VectorDBManager

# 初始化向量数据库管理器
vectorDBManager = VectorDBManager()
redis_client = vectorDBManager.redis_client

print(vectorDBManager.load_documents_to_vector_db(os.path.join(root, "knowledge"), 'Test'))
# print(vectorDBManager.append_documents_to_vector_db(os.path.join(root, "test"), 'Test'))
# print(vectorDBManager.delete_vector_db('Test'))
