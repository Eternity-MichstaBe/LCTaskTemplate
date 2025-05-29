import os
import sys
import asyncio

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

from src.rag.ParentDocumentRetriever import VectorDBManager

# 初始化向量数据库管理器
vectorDBManager = VectorDBManager()

asyncio.run(vectorDBManager.load_documents_to_vector_db(
    os.path.join(root, "files"),
    'ParentDocumentRetrieverDB',
    file_glob="*.txt"
))

# asyncio.run(vectorDBManager.append_documents_to_vector_db(
#     os.path.join(root, "files"),
#     'ParentDocumentRetrieverDB'
# ))

# asyncio.run(vectorDBManager.delete_vector_db('Test'))
