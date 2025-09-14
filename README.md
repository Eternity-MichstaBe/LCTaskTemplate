## 基于LangChain的LLM通用任务处理框架

- [项目结构]
  - [1 环境配置] config目录
  - [2 上下文示例管理] example目录
  - [3 核心代码] src目录
    - [3.1 回调管理] src.callbacks目录
    - [3.2 核心类] src.llm目录
      - [3.2.1 LLM配置管理] LLMConfig.py 管理LLM的加载参数 
      - [3.2.2 提示词管理] PromptManager.py 管理系统提示词
      - [3.2.3 模板构建管理] PromptBuilder.py 融合Agent支持、上下文记忆支持、多模态支持、示例选择支持的封装模板
      - [3.2.4 LLM链管理] LLMChainBuilder.py 非Agent（多模态）推理链快速构建
      - [3.2.5 Agent链管理] LLMAgentBuilder.py Agent（多模态）推理链快速构建
    - [3.3 RAG实现类] src.rag目录
    - [3.4 工具类] src.utils目录
  - [4 演示代码] src下的ChatAgentWithPDR.py  ChatAgentWithVSR.py
  - [5 非核心部分] files（测试文件）、pdr_cache（父子文档检索器生成的向量空间）、vsr_cache（常规检索器生成的向量空间）、test、my-app（LangServe测试）