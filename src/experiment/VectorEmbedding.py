import os
import sys
import asyncio

from src.rag.VectorStoreRetrieverManager import VectorStoreRetrieverManager
from src.rag.ParentDocumentRetrieverManager import ParentDocumentRetrieverManager

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

# 原始向量嵌入
# vectorDBManager1 = VectorStoreRetrieverManager()
# vectorDBManager1.init_vector_db('VectorStoreRetrieverDB')
#
# asyncio.run(vectorDBManager1.append_documents_to_vector_db(
#     os.path.join(root, "files"),
#     'VectorStoreRetrieverDB',
#     file_glob="*.txt"
# ))

# asyncio.run(vectorDBManager1.delete_vector_db('VectorStoreRetrieverDB'))


# 父子文档向量嵌入
# vectorDBManager2 = ParentDocumentRetrieverManager()
#
# vectorDBManager2.init_vector_db('ParentDocumentRetrieverDB')
#
# asyncio.run(vectorDBManager2.append_documents_to_vector_db(
#     os.path.join(root, "files"),
#     'ParentDocumentRetrieverDB',
#     file_glob="*.txt"
# ))

# asyncio.run(vectorDBManager2.delete_vector_db('ParentDocumentRetrieverDB'))
