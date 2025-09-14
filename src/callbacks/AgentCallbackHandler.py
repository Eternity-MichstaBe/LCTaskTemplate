from typing import Any
from langchain.schema import AgentAction
from langchain_core.callbacks import BaseCallbackHandler


class AgentCallbackHandler(BaseCallbackHandler):
    """自定义Agent回调处理器"""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        # print(f"\n🧠 Agent 计划调用工具: {action.tool}")
        # print(f"📦 工具输入: {action.tool_input}")
        #
        # choice = input("❓是否继续调用工具？(y/n): ").strip().lower()
        # if choice != "y":
        #     raise "⛔ 用户取消了工具调用"
        pass