import os
import sys

# 添加项目根目录到路径
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)


def custom_trim_messages(messages, k_rounds=3):
    """
    自定义实现：保留最近的K轮对话
    确保起始是human消息，结尾是ai或tool消息

    Args:
        messages: 消息列表
        k_rounds: 保留的对话轮数

    Returns:
        修剪后的消息列表
    """
    if not messages:
        return []

    # 从后往前找，找到k轮完整的对话
    trimmed_messages = []
    round_count = 0
    i = len(messages) - 1

    # 从最后一条消息开始向前遍历
    while i >= 0 and round_count < k_rounds:
        current_msg = messages[i]
        msg_type = current_msg.type

        # 如果是AI或tool消息，添加到结果中
        if msg_type in ["ai", "tool"]:
            trimmed_messages.append(current_msg)
            i -= 1

            # 继续向前查找对应的human消息
            while i >= 0:
                prev_msg = messages[i]
                if prev_msg.type == "human":
                    trimmed_messages.append(prev_msg)
                    round_count += 1
                    i -= 1
                    break
                elif prev_msg.type in ["ai", "tool"]:
                    # 如果前面还是AI/tool消息，也添加进来
                    trimmed_messages.append(prev_msg)
                    i -= 1
                else:
                    i -= 1
        else:
            i -= 1

    # 反转消息顺序，因为我们是从后往前添加的
    trimmed_messages.reverse()

    # 确保第一条消息是human类型
    while trimmed_messages and trimmed_messages[0].type != "human":
        trimmed_messages.pop(0)

    # 确保最后一条消息是ai或tool类型
    while trimmed_messages and trimmed_messages[-1].type not in ["ai", "tool"]:
        trimmed_messages.pop()

    # 如果需要包含system消息，则添加第一条system消息（如果存在）
    if messages and hasattr(messages[0], 'type') and messages[0].type == "system":
        # 检查是否已经包含了system消息
        has_system = any(msg.type == "system" for msg in trimmed_messages)
        if not has_system:
            trimmed_messages.insert(0, messages[0])

    return trimmed_messages

