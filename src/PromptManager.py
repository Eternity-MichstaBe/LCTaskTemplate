"""
提示词统一管理
"""

class PromptManager:
    """提示词管理器"""

    @staticmethod
    def get_task_prompt() -> str:
        return """
            任务描述
        """

    @staticmethod
    def get_task_validator_prompt() -> str:
        return """
            验证逻辑设置
        """

    