import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

# from src.VectorDBManager import VectorDBManager
from src.ParentVectorDBManager import VectorDBManager

# 初始化向量数据库管理器
vectorDBManager = VectorDBManager()

vectorDBManager.load_documents_to_vector_db(os.path.join(root, "knowledge"), 'Government_Work_Report_2025', "**/*.txt")
# vectorDBManager.append_documents_to_vector_db(os.path.join(root, "knowledge"), 'Test')
# vectorDBManager.delete_vector_db('Test_ParentVectorDBManager')
