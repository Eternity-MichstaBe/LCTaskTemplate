import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

from src.VectorDBManager import VectorDBManager

# 初始化向量数据库管理器
vectorDBManager = VectorDBManager()

print(vectorDBManager.load_documents_to_vector_db(os.path.join(root, "knowledge"), 'Test'))
# print(vectorDBManager.append_documents_to_vector_db(os.path.join(root, "knowledge"), 'Test'))
# vectorDBManager.delete_vector_db('pdf_qa_index')
