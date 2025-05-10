"""
向量数据库删除问题分析

问题描述：
当我们尝试删除向量数据库时，可能会遇到schema键不存在但其他相关键存在的情况。
这是因为在VectorDBManager.delete_vector_db方法中，首先会检查schema键是否存在，
如果不存在就会抛出异常，导致无法删除其他相关键。

问题原因：
1. 数据不一致：Redis数据库中的键可能因为某些操作（如手动删除、部分失败的操作等）
   导致数据不一致，schema键被删除但其他键仍然存在。
   
2. 代码设计限制：VectorDBManager.delete_vector_db方法设计上要求schema键必须存在，
   这是为了确保索引确实存在，避免误删除。但这也导致了当数据不一致时无法完全清理。
   
3. 可能的并发问题：多个进程或线程同时操作同一个索引，导致数据状态不一致。

解决方案：
下面的代码提供了两种方式删除向量数据库中的数据：
1. 删除特定索引的所有数据
2. 删除Redis数据库中的所有数据
"""

import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

from src.VectorDBManager import VectorDBManager

# 初始化向量数据库管理器
vectorDBManager = VectorDBManager()
redis_client = vectorDBManager.redis_client

index_list = redis_client.execute_command("FT._LIST")
print("list\n", index_list)

# 查看 schema 信息
info = redis_client.execute_command(f"FT.INFO LLM")
print(info)

# print(redis_client.exists("llm_event_extract:schema"))  # 应返回 1
# redis_client.delete("llm_event_extract:schema")


# print(vectorDBManager.load_documents_to_vector_db(os.path.join(root, "knowledge"), 'Test'))
# print(vectorDBManager.append_documents_to_vector_db(os.path.join(root, "test"), 'llm_event_extract'))
# print(vectorDBManager.delete_vector_db('llm_event_extract'))
# vectorDBManager.delete_all_schema_keys()
