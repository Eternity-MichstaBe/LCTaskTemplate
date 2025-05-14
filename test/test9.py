from langchain_core.output_parsers import PydanticToolsParser
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from typing import Optional
from typing import Union, List
from langchain_core.tools import BaseTool, tool
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain.output_parsers import RetryOutputParser
from langchain.output_parsers import OutputFixingParser
import asyncio

from langchain_community.document_loaders import PyPDFLoader

from langchain_community.document_loaders import WebBaseLoader

page_url = "https://python.langchain.com/docs/how_to/chatbots_memory/"

loader = WebBaseLoader(web_paths=[page_url])
docs = []

for doc in loader.lazy_load():
    docs.append(doc)

assert len(docs) == 1
doc = docs[0]

print(f"{doc}\n")
